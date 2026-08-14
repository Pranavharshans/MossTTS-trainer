#!/usr/bin/env python3
"""Generate a matched qualitative suite for one or more MOSS checkpoints."""

from __future__ import annotations

import argparse
import glob
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from finetuning.prepare_data import load_codec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--prompts-jsonl", type=Path, required=True)
    parser.add_argument("--codec-path", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prompt-audio-path", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--max-new-frames", type=int, default=500)
    parser.add_argument("--do-sample", type=int, choices=(0, 1), default=1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def resolve_checkpoints(specs: list[str]) -> list[Path]:
    paths: list[Path] = []
    for spec in specs:
        matches = sorted(glob.glob(spec)) if any(char in spec for char in "*?[]") else [spec]
        paths.extend(Path(match).expanduser().resolve() for match in matches)
    resolved: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        if path in seen:
            continue
        if not (path / "config.json").is_file():
            raise FileNotFoundError(f"checkpoint has no config.json: {path}")
        seen.add(path)
        resolved.append(path)
    if not resolved:
        raise ValueError("no checkpoints matched --checkpoints")
    return resolved


def load_prompts(path: Path) -> list[dict[str, str]]:
    prompts: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value.get("id"), str) or not isinstance(value.get("text"), str):
                raise ValueError(f"{path}:{line_number} requires string id and text")
            prompts.append({"id": value["id"], "text": value["text"]})
    if not prompts:
        raise ValueError(f"no prompts found in {path}")
    return prompts


def safe_name(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-.")
    return normalized or "checkpoint"


def dtype_from_name(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def checkpoint_label(path: Path) -> str:
    if path.name.startswith("checkpoint-"):
        return safe_name(path.name)
    return safe_name(path.name + "-base")


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise EnvironmentError("CUDA evaluation requested but CUDA is unavailable")
    checkpoints = resolve_checkpoints(args.checkpoints)
    prompts = load_prompts(args.prompts_jsonl.expanduser().resolve())
    prompt_audio = (
        str(args.prompt_audio_path.expanduser().resolve())
        if args.prompt_audio_path is not None
        else None
    )
    if prompt_audio is not None and not Path(prompt_audio).is_file():
        raise FileNotFoundError(prompt_audio)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    codec = load_codec(args.codec_path, device=str(device))
    model_dtype = dtype_from_name(args.dtype)
    results: list[dict[str, Any]] = []

    for checkpoint in checkpoints:
        label = checkpoint_label(checkpoint)
        checkpoint_output = args.output_dir / label
        checkpoint_output.mkdir(parents=True, exist_ok=True)
        tokenizer = AutoTokenizer.from_pretrained(
            checkpoint, trust_remote_code=True, local_files_only=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint,
            trust_remote_code=True,
            local_files_only=True,
            torch_dtype=model_dtype,
        ).to(device)
        if hasattr(model, "_set_attention_implementation"):
            model._set_attention_implementation("sdpa" if device.type == "cuda" else "eager")
        model.eval()

        for prompt_index, prompt in enumerate(prompts):
            seed = args.seed + prompt_index
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            output_path = checkpoint_output / f"{safe_name(prompt['id'])}.wav"
            started = time.perf_counter()
            generated = model.inference(
                text=prompt["text"],
                output_audio_path=output_path,
                mode="voice_clone" if prompt_audio else "continuation",
                prompt_audio_path=prompt_audio,
                text_tokenizer=tokenizer,
                audio_tokenizer=codec,
                device=device,
                max_new_frames=args.max_new_frames,
                do_sample=bool(args.do_sample),
                use_kv_cache=True,
            )
            elapsed = time.perf_counter() - started
            waveform = generated["waveform"]
            sample_rate = int(generated["sample_rate"])
            duration = float(waveform.shape[-1]) / float(sample_rate)
            record = {
                "checkpoint": str(checkpoint),
                "checkpoint_label": label,
                "prompt_id": prompt["id"],
                "text": prompt["text"],
                "seed": seed,
                "audio_path": str(output_path.resolve()),
                "sample_rate": sample_rate,
                "duration_seconds": round(duration, 6),
                "generation_seconds": round(elapsed, 6),
                "audio_token_frames": int(generated["audio_token_ids"].shape[0]),
            }
            results.append(record)
            print(
                f"[moss_tts_ml.eval] checkpoint={label} prompt={prompt['id']} "
                f"duration={duration:.2f}s generation={elapsed:.2f}s output={output_path}",
                flush=True,
            )

        del model, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = {
        "prompt_audio_path": prompt_audio,
        "codec_path": args.codec_path,
        "checkpoints": [str(path) for path in checkpoints],
        "prompts_jsonl": str(args.prompts_jsonl.resolve()),
        "results": results,
    }
    (args.output_dir / "evaluation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"[moss_tts_ml.eval] complete checkpoints={len(checkpoints)} "
        f"prompts={len(prompts)} output={args.output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
