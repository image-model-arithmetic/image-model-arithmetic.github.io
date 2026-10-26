#!/usr/bin/env python3
"""Evaluate generated-answer chains across separately trained modulo-23 models."""

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


def choose_second_rows(pairs: np.ndarray) -> list[int]:
    tuples = [tuple(map(int, pair)) for pair in pairs]
    chosen = []
    for value in range(23):
        candidates = [index for index, pair in enumerate(tuples) if value in pair]
        assert candidates
        chosen.append(min(candidates, key=lambda index: tuples[index]))
    return chosen


def load_model(spec: dict, checkpoint: Path):
    model = SymmetricSegmentRasterCompleter(
        width=spec["width"],
        layers=spec["layers"],
        heads=spec["heads"],
        mlp_ratio=spec["mlp_ratio"],
    )
    model.load_weights(str(checkpoint))
    return model


def main() -> None:
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-config", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.evaluation_config.read_text(encoding="utf-8"))
    model_config = json.loads(args.model_config.read_text(encoding="utf-8"))
    assert config["status"] == "frozen-before-evaluation"
    assert digest(args.model_config) == config["model_config_sha256"]
    assert digest(args.dataset / "dataset.npz") == config["dataset_sha256"]
    assert not args.output.exists() and not args.output.is_symlink()
    seeds = config["selection"]["seed_indices"]
    assert seeds == [0, 1, 2]
    checkpoints = {}
    for seed in seeds:
        run = args.runs_root / f"seed-{seed}"
        summary = json.loads((run / "development-summary.json").read_text(encoding="utf-8"))
        correct = round(summary["selection"]["development"]["accuracy"] * 46)
        assert correct == config["selection"]["development_correct"][seeds.index(seed)]
        assert summary["test_evaluations"] == 0
        checkpoint = run / "selected.safetensors"
        assert digest(checkpoint) == config["checkpoint_sha256"][str(seed)]
        checkpoints[seed] = checkpoint

    data = np.load(args.dataset / "dataset.npz", allow_pickle=False)
    rows = np.flatnonzero((data["splits"] == 2) & (data["variants"] == 0))
    assert len(rows) == 46
    test_pairs = data["pairs"][rows].astype(np.int64)
    test_inputs = data["inputs"][rows].astype(np.float32)
    test_targets = data["targets"][rows].astype(np.float32)
    prototypes = data["prototypes"].astype(np.float32)
    first_answers = test_pairs.sum(axis=1) % 23
    chosen_by_value = choose_second_rows(test_pairs)

    target_inputs, second_targets, second_pairs, second_rows, replacement_crops = [], [], [], [], []
    for value in first_answers:
        second_index = chosen_by_value[int(value)]
        pair = test_pairs[second_index]
        crop = LEFT_CROP if int(pair[0]) == int(value) else RIGHT_CROP
        clean = test_inputs[second_index].copy()
        clean[:, crop[0] : crop[1], :] = prototypes[int(value)]
        target_inputs.append(clean)
        second_targets.append(test_targets[second_index])
        second_pairs.append(pair)
        second_rows.append(second_index)
        replacement_crops.append(crop)
    target_inputs = np.stack(target_inputs)
    second_targets = np.stack(second_targets)
    second_pairs = np.stack(second_pairs)
    expected_second = second_pairs.sum(axis=1) % 23

    first_results = {}
    canonical_results = {}
    for seed in seeds:
        model = load_model(model_config["model"], checkpoints[seed])
        first_results[seed] = evaluate(
            model, test_inputs, test_targets, test_pairs, prototypes, model_config["evaluation"]["batch_size"]
        )
        canonical_results[seed] = evaluate(
            model, target_inputs, second_targets, second_pairs, prototypes, model_config["evaluation"]["batch_size"]
        )

    matrix = []
    chain_predictions = np.empty((len(seeds), len(seeds), 46), dtype=np.int64)
    chain_outputs = np.empty((len(seeds), len(seeds), 46, 32, 28, 1), dtype=np.float16)
    for source_position, source_seed in enumerate(seeds):
        generated_inputs = target_inputs.copy()
        source_outputs = first_results[source_seed]["outputs"]
        for index, crop in enumerate(replacement_crops):
            generated_inputs[index, :, crop[0] : crop[1], :] = source_outputs[index]
        source_correct = first_results[source_seed]["predictions"] == first_answers
        for receiver_position, receiver_seed in enumerate(seeds):
            receiver = load_model(model_config["model"], checkpoints[receiver_seed])
            result = evaluate(
                receiver,
                generated_inputs,
                second_targets,
                second_pairs,
                prototypes,
                model_config["evaluation"]["batch_size"],
            )
            generated_correct = result["predictions"] == expected_second
            canonical_correct = canonical_results[receiver_seed]["predictions"] == expected_second
            both = source_correct & generated_correct
            readable_failure = source_correct & ~generated_correct & canonical_correct
            readable_repair = source_correct & generated_correct & ~canonical_correct
            matrix.append(
                {
                    "source_seed": source_seed,
                    "receiver_seed": receiver_seed,
                    "separately_trained": source_seed != receiver_seed,
                    "case_count": 46,
                    "first_correct": int(np.sum(source_correct)),
                    "canonical_second_correct": int(np.sum(canonical_correct)),
                    "generated_second_correct": int(np.sum(generated_correct)),
                    "both_steps_correct": int(np.sum(both)),
                    "readable_generated_image_failure": int(np.sum(readable_failure)),
                    "readable_generated_image_repair": int(np.sum(readable_repair)),
                    "generated_second_exact_raster": int(round(result["exact_raster"] * 46)),
                }
            )
            chain_predictions[source_position, receiver_position] = result["predictions"]
            chain_outputs[source_position, receiver_position] = result["outputs"].astype(np.float16)

    cross = [row for row in matrix if row["separately_trained"]]
    diagonal = [row for row in matrix if not row["separately_trained"]]
    summary = {
        "complete": True,
        "seed_indices": seeds,
        "test_pair_count": 46,
        "unique_second_pair_count": len(set(map(tuple, second_pairs.tolist()))),
        "all_second_pairs_from_test_split": True,
        "first_correct_by_source": {
            str(seed): int(np.sum(first_results[seed]["predictions"] == first_answers)) for seed in seeds
        },
        "canonical_second_correct_by_receiver": {
            str(seed): int(np.sum(canonical_results[seed]["predictions"] == expected_second)) for seed in seeds
        },
        "matrix": matrix,
        "cross_model": {
            key: sum(row[key] for row in cross)
            for key in (
                "case_count",
                "both_steps_correct",
                "readable_generated_image_failure",
                "readable_generated_image_repair",
            )
        },
        "same_model": {
            key: sum(row[key] for row in diagonal)
            for key in (
                "case_count",
                "both_steps_correct",
                "readable_generated_image_failure",
                "readable_generated_image_repair",
            )
        },
        "test_evaluations_per_model": 1,
    }
    args.output.mkdir(parents=True, mode=0o700)
    np.savez_compressed(
        args.output / "cross-model-chain-outputs.npz",
        seeds=np.asarray(seeds, dtype=np.int8),
        first_pairs=test_pairs.astype(np.int8),
        second_pairs=second_pairs.astype(np.int8),
        second_test_rows=np.asarray(second_rows, dtype=np.int8),
        first_predictions=np.stack([first_results[seed]["predictions"] for seed in seeds]),
        first_outputs=np.stack([first_results[seed]["outputs"] for seed in seeds]).astype(np.float16),
        canonical_second_predictions=np.stack([canonical_results[seed]["predictions"] for seed in seeds]),
        chain_predictions=chain_predictions,
        chain_outputs=chain_outputs,
    )
    write(args.output / "summary.json", summary)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
