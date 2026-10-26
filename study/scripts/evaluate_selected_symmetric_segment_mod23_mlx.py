#!/usr/bin/env python3
"""Open the sealed test split once for the development-selected checkpoint."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from image_reasoning.symmetric_segment_mod23 import LEFT_CROP, RIGHT_CROP, SymmetricSegmentRasterCompleter
from train_symmetric_segment_mod23_mlx import evaluate, write


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--development-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    development = json.loads(args.development_summary.read_text())
    assert config["status"] == "frozen-before-training"
    assert development["complete"] and development["test_evaluations"] == 0
    assert digest(args.checkpoint) == development["checkpoint_sha256"]
    assert digest(args.dataset / "dataset.npz") == config["dataset"]["sha256"]
    assert not args.output.exists() and not args.output.is_symlink()
    data = np.load(args.dataset / "dataset.npz")
    inputs = data["inputs"].astype(np.float32)
    targets = data["targets"].astype(np.float32)
    pairs = data["pairs"]
    splits = data["splits"]
    variants = data["variants"]
    prototypes = data["prototypes"].astype(np.float32)
    test_indices = np.flatnonzero((splits == 2) & (variants == 0))
    assert len(test_indices) == config["dataset"]["test_pair_count"] == 46
    model_config = config["model"]
    model = SymmetricSegmentRasterCompleter(
        width=model_config["width"],
        layers=model_config["layers"],
        heads=model_config["heads"],
        mlp_ratio=model_config["mlp_ratio"],
    )
    model.load_weights(str(args.checkpoint))
    args.output.mkdir(parents=True, mode=0o700)
    result = evaluate(
        model,
        inputs[test_indices],
        targets[test_indices],
        pairs[test_indices],
        prototypes,
        config["evaluation"]["batch_size"],
    )
    swapped_inputs = inputs[test_indices].copy()
    swapped_inputs[:, :, LEFT_CROP[0] : LEFT_CROP[1], :] = inputs[
        test_indices, :, RIGHT_CROP[0] : RIGHT_CROP[1], :
    ]
    swapped_inputs[:, :, RIGHT_CROP[0] : RIGHT_CROP[1], :] = inputs[
        test_indices, :, LEFT_CROP[0] : LEFT_CROP[1], :
    ]
    swapped = evaluate(
        model,
        swapped_inputs,
        targets[test_indices],
        pairs[test_indices],
        prototypes,
        config["evaluation"]["batch_size"],
    )
    summary = {
        "complete": True,
        "selected_seed_index": development["seed_index"],
        "selected_step": development["selected_step"],
        "checkpoint_sha256": development["checkpoint_sha256"],
        "test_pair_count": len(test_indices),
        "test": {key: value for key, value in result.items() if np.isscalar(value)},
        "swapped_test": {key: value for key, value in swapped.items() if np.isscalar(value)},
        "swap_maximum_pixel_difference": float(
            np.max(np.abs(result["outputs"] - swapped["outputs"]))
        ),
        "test_evaluations": 1,
    }
    np.savez_compressed(
        args.output / "test-outputs.npz",
        pairs=pairs[test_indices],
        targets=targets[test_indices].astype(np.float16),
        outputs=result["outputs"].astype(np.float16),
        swapped_outputs=swapped["outputs"].astype(np.float16),
        predictions=result["predictions"],
        swapped_predictions=swapped["predictions"],
    )
    write(args.output / "summary.json", summary)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
