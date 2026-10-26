#!/usr/bin/env python3
"""Train one registered seed and evaluate the untouched test split once."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten

from image_reasoning.segment_pixel_grok import classify
from image_reasoning.symmetric_segment_mod23 import (
    LEFT_CROP,
    RIGHT_CROP,
    SymmetricSegmentRasterCompleter,
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value: object) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "x", opener=lambda name, flags: os.open(name, flags, 0o600)) as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def append(path: Path, value: object) -> None:
    with open(path, "a", opener=lambda name, flags: os.open(name, flags, 0o600)) as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def health(root: Path, floor: int) -> dict:
    result = subprocess.run(
        ["/usr/bin/memory_pressure"], capture_output=True, text=True, timeout=10
    )
    match = re.search(r"System-wide memory free percentage: (\d+)%", result.stdout)
    if result.returncode != 0 or match is None:
        raise RuntimeError("memory-pressure-read")
    row = {
        "memory_free_percent": int(match.group(1)),
        "disk_free_bytes": shutil.disk_usage(root).free,
        "unix": time.time(),
    }
    if row["memory_free_percent"] < 15 or row["disk_free_bytes"] < floor:
        raise RuntimeError("resource-floor")
    return row


def evaluate(model, inputs, targets, pairs, prototypes, batch_size):
    chunks = []
    for start in range(0, len(inputs), batch_size):
        output = model(mx.array(inputs[start : start + batch_size]).astype(mx.bfloat16))
        mx.eval(output)
        chunks.append(np.asarray(output.astype(mx.float32)))
    outputs = np.concatenate(chunks)
    predictions = classify(outputs, prototypes)
    labels = (pairs[:, 0].astype(int) + pairs[:, 1].astype(int)) % 23
    binary = outputs < 0.5
    truth = targets < 0.5
    return {
        "accuracy": float(np.mean(predictions == labels)),
        "exact_raster": float(np.mean(np.all(binary == truth, axis=(1, 2, 3)))),
        "pixel_accuracy": float(np.mean(binary == truth)),
        "mse": float(np.mean((outputs - targets) ** 2)),
        "predictions": predictions,
        "outputs": outputs,
    }


def main() -> None:
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed-index", type=int, required=True)
    parser.add_argument("--defer-test", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    assert config["schema"] == 1 and config["status"] == "frozen-before-training"
    assert sys.platform == "darwin" and os.geteuid() == 502
    assert 0 <= args.seed_index < len(config["seeds"]) and 3 <= len(config["seeds"]) <= 10
    assert not args.output.exists() and not args.output.is_symlink()
    dataset_path = args.dataset / "dataset.npz"
    manifest_path = args.dataset / "manifest.json"
    assert digest(dataset_path) == config["dataset"]["sha256"]
    assert digest(manifest_path) == config["dataset"]["manifest_sha256"]
    data = np.load(dataset_path)
    inputs = data["inputs"].astype(np.float32)
    targets = data["targets"].astype(np.float32)
    pairs = data["pairs"]
    splits = data["splits"]
    variants = data["variants"]
    prototypes = data["prototypes"].astype(np.float32)
    manifest = json.loads(manifest_path.read_text())
    assert manifest["pair_counts"] == {"development": 46, "test": 46, "train": 184}
    seed = config["seeds"][args.seed_index]
    args.output.mkdir(parents=True, mode=0o700)
    started = time.monotonic()
    test_evaluations = 0
    error_kind = None
    try:
        admission = health(args.output, config["resources"]["free_floor_bytes"])
        write(
            args.output / "started.json",
            {
                "config_sha256": digest(args.config),
                "dataset_sha256": digest(dataset_path),
                "seed_index": args.seed_index,
                "model_seed": seed["model"],
                "training_seed": seed["training"],
                "admission": admission,
                "interpreter": sys.executable,
            },
        )
        model_config = config["model"]
        mx.random.seed(seed["model"])
        model = SymmetricSegmentRasterCompleter(
            width=model_config["width"],
            layers=model_config["layers"],
            heads=model_config["heads"],
            mlp_ratio=model_config["mlp_ratio"],
        )
        parameter_count = sum(value.size for _, value in tree_flatten(model.parameters()))
        assert parameter_count == model_config["parameter_count"]
        optimizer = optim.AdamW(
            config["optimization"]["learning_rate"],
            betas=config["optimization"]["betas"],
            eps=config["optimization"]["epsilon"],
            weight_decay=config["optimization"]["weight_decay"],
            bias_correction=True,
        )
        optimizer.init(model.trainable_parameters())
        mx.eval(optimizer.state)
        rng = np.random.default_rng(seed["training"])
        train_pairs = [tuple(pair) for pair in manifest["pairs"]["train"]]
        train_indices = [
            np.flatnonzero(
                (splits == 0) & (pairs[:, 0] == pair[0]) & (pairs[:, 1] == pair[1])
            )
            for pair in train_pairs
        ]
        assert len(train_indices) == 184 and all(len(indices) == 4 for indices in train_indices)
        canonical = variants == 0
        train_eval = np.flatnonzero((splits == 0) & canonical)
        development_eval = np.flatnonzero((splits == 1) & canonical)
        test_eval = np.flatnonzero((splits == 2) & canonical)
        assert (len(train_eval), len(development_eval), len(test_eval)) == (184, 46, 46)
        ink_weight = config["optimization"]["ink_weight"]

        def loss_fn(batch_inputs, batch_targets):
            output = model(batch_inputs)
            target = batch_targets.astype(mx.float32)
            weights = 1 + ink_weight * (1 - target)
            epsilon = 1e-6
            bce = -(target * mx.log(output + epsilon) + (1 - target) * mx.log(1 - output + epsilon))
            weighted_bce = mx.sum(weights * bce) / mx.sum(weights)
            predicted_ink = 1 - output
            target_ink = 1 - target
            intersection = mx.sum(predicted_ink * target_ink, axis=(1, 2, 3))
            mass = mx.sum(predicted_ink, axis=(1, 2, 3)) + mx.sum(target_ink, axis=(1, 2, 3))
            dice = mx.mean(1 - (2 * intersection + epsilon) / (mass + epsilon))
            return dice + config["optimization"]["bce_weight"] * weighted_bce

        value_and_grad = nn.value_and_grad(model, loss_fn)
        best_score = None
        best_step = None
        test_opened = False
        max_steps = config["optimization"]["max_steps"]
        evaluation_every = config["evaluation"]["every_steps"]
        for step in range(1, max_steps + 1):
            batch = np.asarray([indices[rng.integers(len(indices))] for indices in train_indices])
            value, gradients = value_and_grad(
                mx.array(inputs[batch]).astype(mx.bfloat16),
                mx.array(targets[batch]).astype(mx.bfloat16),
            )
            gradients, norm = optim.clip_grad_norm(
                gradients, config["optimization"]["gradient_clip"]
            )
            optimizer.update(model, gradients)
            mx.eval(value, norm, model.parameters(), optimizer.state)
            if step == 1 or step % config["execution"]["metrics_every_steps"] == 0:
                append(
                    args.output / "metrics.jsonl",
                    {
                        "step": step,
                        "loss": value.item(),
                        "gradient_norm": norm.item(),
                        "elapsed_seconds": time.monotonic() - started,
                        "peak_memory_gib": mx.get_peak_memory() / 2**30,
                    },
                )
            if step % evaluation_every == 0:
                train_result = evaluate(
                    model,
                    inputs[train_eval],
                    targets[train_eval],
                    pairs[train_eval],
                    prototypes,
                    config["evaluation"]["batch_size"],
                )
                development_result = evaluate(
                    model,
                    inputs[development_eval],
                    targets[development_eval],
                    pairs[development_eval],
                    prototypes,
                    config["evaluation"]["batch_size"],
                )
                row = {
                    "step": step,
                    **{f"train_{key}": value for key, value in train_result.items() if np.isscalar(value)},
                    **{
                        f"development_{key}": value
                        for key, value in development_result.items()
                        if np.isscalar(value)
                    },
                    "health": health(args.output, config["resources"]["free_floor_bytes"]),
                }
                append(args.output / "development-history.jsonl", row)
                score = (
                    development_result["exact_raster"],
                    development_result["accuracy"],
                    train_result["exact_raster"],
                    -development_result["mse"],
                )
                if best_score is None or score > best_score:
                    temporary = args.output / "selected.tmp.safetensors"
                    model.save_weights(str(temporary))
                    os.replace(temporary, args.output / "selected.safetensors")
                    best_score = score
                    best_step = step
                    write(
                        args.output / "selection.json",
                        {
                            "step": step,
                            "rule": "highest development exact raster, then development accuracy, then training exact raster, then lowest development MSE",
                            "train": {key: value for key, value in train_result.items() if np.isscalar(value)},
                            "development": {
                                key: value
                                for key, value in development_result.items()
                                if np.isscalar(value)
                            },
                        },
                    )
        assert not test_opened and test_evaluations == 0
        model.load_weights(str(args.output / "selected.safetensors"))
        if args.defer_test:
            summary = {
                "complete": True,
                "seed_index": args.seed_index,
                "model_seed": seed["model"],
                "training_seed": seed["training"],
                "parameter_count": parameter_count,
                "training_pairs": len(train_eval),
                "development_pairs": len(development_eval),
                "selected_step": best_step,
                "selection": json.loads((args.output / "selection.json").read_text()),
                "test_evaluations": test_evaluations,
                "checkpoint_sha256": digest(args.output / "selected.safetensors"),
                "elapsed_seconds": time.monotonic() - started,
            }
            write(args.output / "development-summary.json", summary)
            print(json.dumps(summary, sort_keys=True), flush=True)
            return
        test_opened = True
        test_result = evaluate(
            model,
            inputs[test_eval],
            targets[test_eval],
            pairs[test_eval],
            prototypes,
            config["evaluation"]["batch_size"],
        )
        test_evaluations += 1
        swapped_inputs = inputs[test_eval].copy()
        swapped_inputs[:, :, LEFT_CROP[0] : LEFT_CROP[1], :] = inputs[
            test_eval, :, RIGHT_CROP[0] : RIGHT_CROP[1], :
        ]
        swapped_inputs[:, :, RIGHT_CROP[0] : RIGHT_CROP[1], :] = inputs[
            test_eval, :, LEFT_CROP[0] : LEFT_CROP[1], :
        ]
        swapped_result = evaluate(
            model,
            swapped_inputs,
            targets[test_eval],
            pairs[test_eval],
            prototypes,
            config["evaluation"]["batch_size"],
        )
        assert test_evaluations == 1
        summary = {
            "complete": True,
            "seed_index": args.seed_index,
            "model_seed": seed["model"],
            "training_seed": seed["training"],
            "parameter_count": parameter_count,
            "training_pairs": len(train_eval),
            "development_pairs": len(development_eval),
            "test_pairs": len(test_eval),
            "selected_step": best_step,
            "selection": json.loads((args.output / "selection.json").read_text()),
            "test": {key: value for key, value in test_result.items() if np.isscalar(value)},
            "swapped_test": {
                key: value for key, value in swapped_result.items() if np.isscalar(value)
            },
            "swap_maximum_pixel_difference": float(
                np.max(np.abs(test_result["outputs"] - swapped_result["outputs"]))
            ),
            "test_evaluations": test_evaluations,
            "checkpoint_sha256": digest(args.output / "selected.safetensors"),
            "elapsed_seconds": time.monotonic() - started,
        }
        np.savez_compressed(
            args.output / "test-outputs.npz",
            pairs=pairs[test_eval],
            targets=targets[test_eval].astype(np.float16),
            outputs=test_result["outputs"].astype(np.float16),
            swapped_outputs=swapped_result["outputs"].astype(np.float16),
            predictions=test_result["predictions"],
            swapped_predictions=swapped_result["predictions"],
        )
        write(args.output / "summary.json", summary)
        print(json.dumps(summary, sort_keys=True), flush=True)
    except BaseException as error:
        error_kind = type(error).__name__
        raise
    finally:
        write(
            args.output / "terminal.json",
            {
                "error_kind": error_kind,
                "test_evaluations": test_evaluations,
                "finished_unix": time.time(),
            },
        )


if __name__ == "__main__":
    main()
