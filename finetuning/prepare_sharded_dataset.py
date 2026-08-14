#!/usr/bin/env python3
"""Resumably encode and length-filter a raw MOSS-TTS training manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable

import torch
from transformers import AutoConfig, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from finetuning.common import (
    format_timestamp,
    load_jsonl,
    resolve_record_audio_paths,
    select_rank_shard,
    shard_output_path,
)
from finetuning.dataset import MossTTSNanoSFTDataset
from finetuning.prepare_data import encode_audio_paths, load_codec

PREPARATION_SCHEMA_VERSION = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--codec-path", required=True)
    parser.add_argument("--model-path", required=True, help="Extended model/tokenizer directory.")
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prefix", required=True, help="Output basename, for example train or val.")
    parser.add_argument("--num-shards", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--n-vq", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def atomic_dump_jsonl(records: Iterable[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def atomic_dump_json(value: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def file_fingerprint(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "sha256": digest.hexdigest(),
    }


def shard_paths(output_dir: Path, prefix: str, rank: int, world_size: int) -> tuple[Path, Path, Path]:
    accepted = shard_output_path(output_dir / f"{prefix}.jsonl", rank, world_size)
    rejected = accepted.with_name(accepted.name.removesuffix(".jsonl") + ".rejected.jsonl")
    report = accepted.with_name(accepted.name.removesuffix(".jsonl") + ".report.json")
    return accepted, rejected, report


def signature_for(
    *,
    input_fingerprint: dict[str, Any],
    model_path: str,
    codec_path: str,
    max_length: int,
    n_vq: int | None,
    rank: int,
    num_shards: int,
) -> dict[str, Any]:
    model_root = Path(model_path).expanduser().resolve()
    codec_root = Path(codec_path).expanduser()
    return {
        "schema_version": PREPARATION_SCHEMA_VERSION,
        "input": input_fingerprint,
        "model_path": str(model_root),
        "model_config": file_fingerprint(model_root / "config.json"),
        "tokenizer_model": file_fingerprint(model_root / "tokenizer.model"),
        "codec_path": (
            str(codec_root.resolve())
            if codec_root.exists()
            else codec_path
        ),
        "codec_config": (
            file_fingerprint(codec_root.resolve() / "config.json")
            if codec_root.exists()
            else None
        ),
        "max_length": max_length,
        "n_vq": n_vq,
        "rank": rank,
        "num_shards": num_shards,
    }


def completed_shard(report_path: Path, accepted_path: Path, rejected_path: Path, signature: dict[str, Any]) -> bool:
    if not (report_path.is_file() and accepted_path.is_file() and rejected_path.is_file()):
        return False
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        report.get("status") == "complete"
        and report.get("signature") == signature
        and report.get("input_records")
        == report.get("accepted_records", 0) + report.get("rejected_records", 0)
    )


def main() -> None:
    args = parse_args()
    if args.num_shards <= 0:
        raise ValueError("--num-shards must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.max_length <= 8:
        raise ValueError("--max-length must be greater than 8")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise EnvironmentError("CUDA is required for codec preparation but is not available")

    input_path = args.input_jsonl.expanduser().resolve()
    raw_records = [
        resolve_record_audio_paths(record, base_dir=input_path.parent)
        for record in load_jsonl(input_path)
    ]
    if not raw_records:
        raise ValueError(f"no records found in {input_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True, local_files_only=True
    )
    model_config = AutoConfig.from_pretrained(
        args.model_path, trust_remote_code=True, local_files_only=True
    )
    if len(tokenizer) != int(model_config.vocab_size):
        raise RuntimeError(
            f"tokenizer vocab={len(tokenizer)} does not match config vocab={model_config.vocab_size}"
        )
    input_fingerprint = file_fingerprint(input_path)
    pending: list[tuple[int, list[dict[str, Any]], Path, Path, Path, dict[str, Any]]] = []
    total_input = 0

    for rank in range(args.num_shards):
        shard_records = select_rank_shard(raw_records, args.num_shards, rank)
        accepted_path, rejected_path, report_path = shard_paths(
            args.output_dir, args.prefix, rank, args.num_shards
        )
        signature = signature_for(
            input_fingerprint=input_fingerprint,
            model_path=args.model_path,
            codec_path=args.codec_path,
            max_length=args.max_length,
            n_vq=args.n_vq,
            rank=rank,
            num_shards=args.num_shards,
        )
        total_input += len(shard_records)
        if not args.force and completed_shard(
            report_path, accepted_path, rejected_path, signature
        ):
            print(
                f"[{format_timestamp()}] [prepare_sharded] skip completed "
                f"rank={rank}/{args.num_shards} output={accepted_path}",
                flush=True,
            )
            continue
        pending.append(
            (rank, shard_records, accepted_path, rejected_path, report_path, signature)
        )

    if not pending:
        print(
            f"[{format_timestamp()}] [prepare_sharded] all shards already complete "
            f"records={total_input} output_dir={args.output_dir}",
            flush=True,
        )
        return

    codec = load_codec(args.codec_path, device=args.device)
    total_accepted = 0
    total_rejected = 0
    for rank, shard_records, accepted_path, rejected_path, report_path, signature in pending:
        target_paths: list[str] = []
        for local_index, record in enumerate(shard_records):
            audio_path = record.get("audio")
            if not isinstance(audio_path, str) or not audio_path:
                raise ValueError(f"shard {rank} record {local_index} has no valid audio path")
            target_paths.append(audio_path)

        encoded = encode_audio_paths(
            codec,
            target_paths,
            batch_size=args.batch_size,
            n_vq=args.n_vq,
        )
        for record in shard_records:
            record["audio_codes"] = encoded[str(record["audio"])]

        packed_dataset = MossTTSNanoSFTDataset(
            shard_records,
            tokenizer=tokenizer,
            model_config=model_config,
            max_length=args.max_length,
        )
        accepted_records: list[dict[str, Any]] = []
        rejected_records: list[dict[str, Any]] = []
        packed_lengths: list[int] = []
        for local_index, record in enumerate(shard_records):
            try:
                example = packed_dataset[local_index]
            except ValueError as error:
                message = str(error)
                is_overlength = (
                    "packed length" in message and "exceeds max_length" in message
                ) or (
                    "prompt length" in message and ">= max_length" in message
                )
                if not is_overlength:
                    raise ValueError(
                        f"shard {rank} record {local_index} failed packing: {message}"
                    ) from error
                rejected = dict(record)
                rejected.pop("audio_codes", None)
                rejected["rejection_reason"] = message
                rejected_records.append(rejected)
                continue
            packed_lengths.append(int(example["seq_len"].item()))
            accepted_records.append(record)

        report = {
            "status": "complete",
            "completed_at": format_timestamp(),
            "signature": signature,
            "input_records": len(shard_records),
            "accepted_records": len(accepted_records),
            "rejected_records": len(rejected_records),
            "packed_length_min": min(packed_lengths) if packed_lengths else None,
            "packed_length_max": max(packed_lengths) if packed_lengths else None,
            "accepted_jsonl": str(accepted_path.resolve()),
            "rejected_jsonl": str(rejected_path.resolve()),
        }
        atomic_dump_jsonl(accepted_records, accepted_path)
        atomic_dump_jsonl(rejected_records, rejected_path)
        atomic_dump_json(report, report_path)
        total_accepted += len(accepted_records)
        total_rejected += len(rejected_records)
        print(
            f"[{format_timestamp()}] [prepare_sharded] rank={rank}/{args.num_shards} "
            f"input={len(shard_records)} accepted={len(accepted_records)} "
            f"rejected={len(rejected_records)} output={accepted_path}",
            flush=True,
        )

    print(
        f"[{format_timestamp()}] [prepare_sharded] processed_pending={len(pending)} "
        f"accepted={total_accepted} rejected={total_rejected}",
        flush=True,
    )


if __name__ == "__main__":
    main()
