"""Download the pre-quantized MLX ColQwen2.5 model into the local HF cache.

The model is already 4-bit quantized for MLX: no separate conversion or
quantization step needed. Running this script once is enough; subsequent
loads will hit the cache.

Usage:
    uv run python scripts/fetch_model.py
    uv run python scripts/fetch_model.py --repo qnguyen3/colqwen2_5-v0.2-mlx-8bit
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from huggingface_hub import snapshot_download

DEFAULT_REPO = "qnguyen3/colqwen2_5-v0.2-mlx-4bit"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo",
        default=DEFAULT_REPO,
        help=f"HF repo id (default: {DEFAULT_REPO})",
    )
    args = parser.parse_args()

    print(f"Fetching {args.repo} into the local HuggingFace cache ...", flush=True)
    t0 = time.perf_counter()
    local_path = snapshot_download(repo_id=args.repo)
    elapsed = time.perf_counter() - t0

    p = Path(local_path)
    n_files = sum(1 for _ in p.rglob("*") if _.is_file())
    total_mb = sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1024 / 1024

    print(f"Done in {elapsed:.1f}s.")
    print(f"  local path: {local_path}")
    print(f"  files:      {n_files}")
    print(f"  total size: {total_mb:.1f} MB")


if __name__ == "__main__":
    main()
