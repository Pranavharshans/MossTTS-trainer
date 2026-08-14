#!/usr/bin/env python3
"""Append deterministic Malayalam BPE pieces to a MOSS SentencePiece model.

Existing piece IDs are preserved exactly. Malayalam pieces are appended with
positive BPE priorities so they are preferred over the model's UTF-8 byte
fallback while byte fallback remains available for unseen text.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import tempfile
import unicodedata
from pathlib import Path
from typing import Iterable

import sentencepiece as spm
from sentencepiece import sentencepiece_model_pb2 as model_pb2


MALAYALAM_START = ord("\u0d00")
MALAYALAM_END = ord("\u0d7f")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-tokenizer-model", type=Path, required=True)
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--output-model", type=Path, required=True)
    parser.add_argument("--report-path", type=Path, required=True)
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--num-new-tokens", type=int, default=2048)
    parser.add_argument("--candidate-vocab-size", type=int, default=4096)
    return parser.parse_args()


def normalize_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).split())


def read_texts(path: Path, text_column: str) -> list[str]:
    texts: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            text = value.get(text_column)
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"{path}:{line_number} has no non-empty {text_column!r}")
            texts.append(normalize_text(text))
    if not texts:
        raise ValueError(f"no text records found in {path}")
    return texts


def contains_malayalam(piece: str) -> bool:
    return any(MALAYALAM_START <= ord(character) <= MALAYALAM_END for character in piece)


def load_proto(path: Path) -> model_pb2.ModelProto:
    model = model_pb2.ModelProto()
    model.ParseFromString(path.read_bytes())
    return model


def train_candidate_model(
    texts: Iterable[str], *, vocab_size: int, output_dir: Path
) -> model_pb2.ModelProto:
    corpus_path = output_dir / "malayalam_corpus.txt"
    with corpus_path.open("w", encoding="utf-8") as handle:
        for text in texts:
            handle.write(text + "\n")

    model_prefix = output_dir / "malayalam_candidate"
    spm.SentencePieceTrainer.train(
        input=str(corpus_path),
        model_prefix=str(model_prefix),
        model_type="bpe",
        vocab_size=vocab_size,
        character_coverage=1.0,
        normalization_rule_name="identity",
        byte_fallback=False,
        unk_id=0,
        bos_id=-1,
        eos_id=-1,
        pad_id=-1,
        hard_vocab_limit=False,
        shuffle_input_sentence=False,
        num_threads=1,
        minloglevel=1,
    )
    return load_proto(model_prefix.with_suffix(".model"))


def select_new_pieces(
    base_model: model_pb2.ModelProto,
    candidate_model: model_pb2.ModelProto,
    *,
    count: int,
) -> list[str]:
    existing = {piece.piece for piece in base_model.pieces}
    selected: list[str] = []
    for piece in candidate_model.pieces:
        if piece.type != model_pb2.ModelProto.SentencePiece.NORMAL:
            continue
        if piece.piece in existing or not contains_malayalam(piece.piece):
            continue
        selected.append(piece.piece)
        existing.add(piece.piece)
        if len(selected) == count:
            break
    if len(selected) != count:
        raise ValueError(
            f"candidate tokenizer yielded {len(selected)} unique Malayalam pieces; "
            f"requested {count}. Increase --candidate-vocab-size or inspect the corpus."
        )
    return selected


def append_pieces(
    base_model: model_pb2.ModelProto, pieces: list[str]
) -> model_pb2.ModelProto:
    merged = model_pb2.ModelProto()
    merged.CopyFrom(base_model)
    # The base tokenizer is BPE. Positive priorities make Malayalam merges win
    # over zero-scored byte pieces while affecting no other Unicode script.
    for rank, piece_text in enumerate(pieces):
        piece = merged.pieces.add()
        piece.piece = piece_text
        piece.score = float(len(pieces) - rank)
        piece.type = model_pb2.ModelProto.SentencePiece.NORMAL
    merged.trainer_spec.vocab_size = len(merged.pieces)
    return merged


def percentile(values: list[int], percentile_value: float) -> int:
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(percentile_value * len(ordered)) - 1))
    return int(ordered[index])


def summarize_lengths(values: list[int]) -> dict[str, float | int]:
    return {
        "min": min(values),
        "median": float(statistics.median(values)),
        "p95": percentile(values, 0.95),
        "max": max(values),
        "mean": round(float(statistics.fmean(values)), 3),
    }


def verify_existing_ids(
    base_processor: spm.SentencePieceProcessor,
    extended_processor: spm.SentencePieceProcessor,
) -> None:
    for token_id in range(base_processor.vocab_size()):
        if base_processor.id_to_piece(token_id) != extended_processor.id_to_piece(token_id):
            raise RuntimeError(f"base token ID {token_id} changed during extension")


def verify_base_piece_records(
    base_model: model_pb2.ModelProto,
    extended_model: model_pb2.ModelProto,
) -> None:
    for token_id, base_piece in enumerate(base_model.pieces):
        if base_piece.SerializeToString() != extended_model.pieces[token_id].SerializeToString():
            raise RuntimeError(f"base SentencePiece record {token_id} changed during extension")


def main() -> None:
    args = parse_args()
    if args.num_new_tokens <= 0:
        raise ValueError("--num-new-tokens must be positive")
    if args.candidate_vocab_size <= args.num_new_tokens:
        raise ValueError("--candidate-vocab-size must be greater than --num-new-tokens")

    texts = read_texts(args.train_jsonl, args.text_column)
    base_bytes = args.base_tokenizer_model.read_bytes()
    base_model = load_proto(args.base_tokenizer_model)
    base_processor = spm.SentencePieceProcessor(model_proto=base_bytes)
    with tempfile.TemporaryDirectory(prefix="moss-ml-tokenizer-") as temp_dir:
        candidate_model = train_candidate_model(
            texts,
            vocab_size=args.candidate_vocab_size,
            output_dir=Path(temp_dir),
        )

    selected = select_new_pieces(base_model, candidate_model, count=args.num_new_tokens)
    extended_model = append_pieces(base_model, selected)
    extended_bytes = extended_model.SerializeToString()
    extended_processor = spm.SentencePieceProcessor(model_proto=extended_bytes)
    verify_base_piece_records(base_model, extended_model)
    verify_existing_ids(base_processor, extended_processor)

    before_lengths = [len(base_processor.encode(text)) for text in texts]
    after_lengths = [len(extended_processor.encode(text)) for text in texts]
    if sum(after_lengths) >= sum(before_lengths):
        raise RuntimeError("extended tokenizer did not reduce aggregate Malayalam token count")
    for text in texts[: min(1000, len(texts))]:
        expected = base_processor.decode(base_processor.encode(text))
        if extended_processor.decode(extended_processor.encode(text)) != expected:
            raise RuntimeError(f"extended tokenizer failed round-trip validation for: {text!r}")

    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    args.output_model.write_bytes(extended_bytes)
    report = {
        "base_tokenizer_model": str(args.base_tokenizer_model.resolve()),
        "train_jsonl": str(args.train_jsonl.resolve()),
        "records": len(texts),
        "base_vocab_size": base_processor.vocab_size(),
        "new_tokens": len(selected),
        "extended_vocab_size": extended_processor.vocab_size(),
        "before": summarize_lengths(before_lengths),
        "after": summarize_lengths(after_lengths),
        "aggregate_token_reduction_ratio": round(
            1.0 - (sum(after_lengths) / sum(before_lengths)), 6
        ),
        "selected_pieces": selected,
    }
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        "[moss_tts_ml.tokenizer] "
        f"base_vocab={report['base_vocab_size']} extended_vocab={report['extended_vocab_size']} "
        f"median={report['before']['median']}->{report['after']['median']} "
        f"p95={report['before']['p95']}->{report['after']['p95']} "
        f"reduction={report['aggregate_token_reduction_ratio']:.2%}",
        flush=True,
    )


if __name__ == "__main__":
    main()
