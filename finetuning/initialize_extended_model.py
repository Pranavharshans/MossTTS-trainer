#!/usr/bin/env python3
"""Resize MOSS text embeddings for an appended SentencePiece vocabulary."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import sentencepiece as spm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_SUPPORT_FILES = (
    "__init__.py",
    "configuration_moss_tts_nano.py",
    "gpt2_decoder.py",
    "modeling_moss_tts_nano.py",
    "prompting.py",
    "tokenization_moss_tts_nano.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model-path", type=Path, required=True)
    parser.add_argument("--extended-tokenizer-model", type=Path, required=True)
    parser.add_argument("--tokenizer-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def resize_embeddings(model, new_vocab_size: int) -> None:
    try:
        model.resize_token_embeddings(new_vocab_size, mean_resizing=False)
    except TypeError:
        model.resize_token_embeddings(new_vocab_size)
    model.config.vocab_size = new_vocab_size
    model.config.gpt2_config.vocab_size = new_vocab_size
    model.tie_weights()


def initialize_appended_rows(
    model,
    *,
    base_processor: spm.SentencePieceProcessor,
    extended_processor: spm.SentencePieceProcessor,
    base_vocab_size: int,
) -> None:
    embedding = model.get_input_embeddings().weight
    with torch.no_grad():
        base_weight = embedding[:base_vocab_size].detach().clone()
        for token_id in range(base_vocab_size, extended_processor.vocab_size()):
            surface = extended_processor.decode([token_id])
            source_ids = [
                token
                for token in base_processor.encode(surface)
                if 0 <= token < base_vocab_size
            ]
            if not source_ids:
                raise RuntimeError(
                    f"new token {token_id} {extended_processor.id_to_piece(token_id)!r} "
                    "has no base-token decomposition"
                )
            embedding[token_id].copy_(base_weight[source_ids].mean(dim=0))
    model.tie_weights()


def verify_tied_weights(model) -> None:
    if model.text_lm_head.weight.data_ptr() != model.get_input_embeddings().weight.data_ptr():
        raise RuntimeError("text_lm_head is not tied to transformer.wte after resize")


def main() -> None:
    args = parse_args()
    base_tokenizer_model = args.base_model_path / "tokenizer.model"
    if not base_tokenizer_model.is_file():
        raise FileNotFoundError(base_tokenizer_model)

    base_processor = spm.SentencePieceProcessor(model_file=str(base_tokenizer_model))
    extended_processor = spm.SentencePieceProcessor(
        model_file=str(args.extended_tokenizer_model)
    )
    base_vocab_size = base_processor.vocab_size()
    new_vocab_size = extended_processor.vocab_size()
    if new_vocab_size <= base_vocab_size:
        raise ValueError("extended tokenizer is not larger than the base tokenizer")
    for token_id in range(base_vocab_size):
        if base_processor.id_to_piece(token_id) != extended_processor.id_to_piece(token_id):
            raise RuntimeError(f"base token ID {token_id} changed")

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model_path, trust_remote_code=True, local_files_only=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    if model.get_input_embeddings().num_embeddings != base_vocab_size:
        raise RuntimeError("base model embedding size does not match its tokenizer")

    resize_embeddings(model, new_vocab_size)
    initialize_appended_rows(
        model,
        base_processor=base_processor,
        extended_processor=extended_processor,
        base_vocab_size=base_vocab_size,
    )
    verify_tied_weights(model)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir, safe_serialization=False)
    tokenizer.save_pretrained(args.output_dir)
    shutil.copy2(args.extended_tokenizer_model, args.output_dir / "tokenizer.model")
    for filename in MODEL_SUPPORT_FILES:
        source = args.base_model_path / filename
        if source.is_file():
            shutil.copy2(source, args.output_dir / filename)

    reloaded_tokenizer = AutoTokenizer.from_pretrained(
        args.output_dir, trust_remote_code=True, local_files_only=True
    )
    if len(reloaded_tokenizer) != new_vocab_size:
        raise RuntimeError(
            f"saved tokenizer vocab={len(reloaded_tokenizer)} does not match model vocab={new_vocab_size}"
        )

    metadata = {
        "base_model_path": str(args.base_model_path.resolve()),
        "base_vocab_size": base_vocab_size,
        "extended_vocab_size": new_vocab_size,
        "new_tokens": new_vocab_size - base_vocab_size,
        "initialization": "mean_of_base_byte_decomposition",
        "base_token_ids_preserved": True,
        "text_lm_head_tied": True,
        "tokenizer_report": json.loads(args.tokenizer_report.read_text(encoding="utf-8")),
    }
    (args.output_dir / "malayalam_tokenizer_extension.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"[moss_tts_ml.initialize] base_vocab={base_vocab_size} "
        f"extended_vocab={new_vocab_size} output={args.output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
