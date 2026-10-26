#!/usr/bin/env python3
"""Build the frozen symmetric modular-addition raster dataset."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from image_reasoning.symmetric_segment_mod23 import build_dataset


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split-seed", type=int, required=True)
    parser.add_argument("--variants", type=int, default=4)
    args = parser.parse_args()
    manifest = build_dataset(
        args.output.absolute(), split_seed=args.split_seed, variants=args.variants
    )
    print(
        json.dumps(
            {
                "pair_counts": manifest["pair_counts"],
                "dataset_sha256": digest(args.output / "dataset.npz"),
                "manifest_sha256": digest(args.output / "manifest.json"),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
