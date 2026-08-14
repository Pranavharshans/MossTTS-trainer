#!/usr/bin/env python3
"""Download a pinned Hugging Face model snapshot into a local directory."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=args.repo_id,
        revision=args.revision,
        local_dir=output_dir,
    )
    print(
        f"[moss_tts_ml.download] repo={args.repo_id} revision={args.revision} "
        f"output={output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
