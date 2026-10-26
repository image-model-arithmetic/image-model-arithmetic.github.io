#!/usr/bin/env python3
"""Pass saved modulo-23 test outputs into a second held-out addition."""

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


def test_rows(data) -> np.ndarray:
    return np.flatnonzero((data["splits"] == 2) & (data["variants"] == 0))


def second_pair_rows(pairs: np.ndarray) -> list[int]:
    """Choose from pair identities only, before looking at model results."""
    rows = []
    tuples = [tuple(map(int, pair)) for pair in pairs]
    for value in range(23):
        candidates = [index for index, pair in enumerate(tuples) if value in pair]
        assert candidates
        rows.append(min(candidates, key=lambda index: tuples[index]))
    return rows


def main() -> None:
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chain-config", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--first-step", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    chain_config = json.loads(args.chain_config.read_text(encoding="utf-8"))
    model_config = json.loads(args.model_config.read_text(encoding="utf-8"))
    assert chain_config["status"] == "frozen-before-chain-evaluation"
    assert digest(args.dataset / "dataset.npz") == chain_config["source_dataset_sha256"]
    assert digest(args.checkpoint) == chain_config["source_checkpoint_sha256"]
    assert not args.output.exists() and not args.output.is_symlink()

    data = np.load(args.dataset / "dataset.npz", allow_pickle=False)
    rows = test_rows(data)
    test_pairs = data["pairs"][rows].astype(np.int64)
    test_inputs = data["inputs"][rows].astype(np.float32)
    test_targets = data["targets"][rows].astype(np.float32)
    prototypes = data["prototypes"].astype(np.float32)
    with np.load(args.first_step, allow_pickle=False) as first:
        first_pairs = first["pairs"].astype(np.int64)
        first_targets = first["targets"].astype(np.float32)
        first_outputs = first["outputs"].astype(np.float32)
        first_predictions = first["predictions"].astype(np.int64)
    assert np.array_equal(first_pairs, test_pairs)
    assert np.array_equal(first_targets, test_targets)

    chosen_by_value = second_pair_rows(test_pairs)
    generated_inputs = []
    target_inputs = []
    second_targets = []
    second_pairs = []
    second_rows = []
    first_answers = (test_pairs[:, 0] + test_pairs[:, 1]) % 23
    for first_index, value in enumerate(first_answers):
        second_index = chosen_by_value[int(value)]
        pair = test_pairs[second_index]
        generated = test_inputs[second_index].copy()
        clean = test_inputs[second_index].copy()
        crop = LEFT_CROP if int(pair[0]) == int(value) else RIGHT_CROP
        generated[:, crop[0] : crop[1], :] = first_outputs[first_index]
        clean[:, crop[0] : crop[1], :] = first_targets[first_index]
        generated_inputs.append(generated)
        target_inputs.append(clean)
        second_targets.append(test_targets[second_index])
        second_pairs.append(pair)
        second_rows.append(second_index)

    model_spec = model_config["model"]
    model = SymmetricSegmentRasterCompleter(
        width=model_spec["width"],
        layers=model_spec["layers"],
        heads=model_spec["heads"],
        mlp_ratio=model_spec["mlp_ratio"],
    )
    model.load_weights(str(args.checkpoint))
    generated_result = evaluate(
        model,
        np.stack(generated_inputs),
        np.stack(second_targets),
        np.stack(second_pairs),
        prototypes,
        model_config["evaluation"]["batch_size"],
    )
    target_result = evaluate(
        model,
        np.stack(target_inputs),
        np.stack(second_targets),
        np.stack(second_pairs),
        prototypes,
        model_config["evaluation"]["batch_size"],
    )
    first_correct = first_predictions == first_answers
    expected_second = np.asarray(second_pairs).sum(axis=1) % 23
    second_correct = generated_result["predictions"] == expected_second
    both_correct = first_correct & second_correct
    summary = {
        "complete": True,
        "chain_count": len(test_pairs),
        "unique_second_pair_count": len(set(map(tuple, np.asarray(second_pairs).tolist()))),
        "all_second_pairs_from_test_split": True,
        "first_step_accuracy": float(np.mean(first_correct)),
        "second_step_with_generated_operand": {
            key: value for key, value in generated_result.items() if np.isscalar(value)
        },
        "second_step_with_target_operand": {
            key: value for key, value in target_result.items() if np.isscalar(value)
        },
        "both_steps_correct": float(np.mean(both_correct)),
        "both_steps_correct_count": int(np.sum(both_correct)),
        "checkpoint_sha256": digest(args.checkpoint),
        "first_step_output_sha256": digest(args.first_step),
        "followup_evaluations": 1,
    }
    args.output.mkdir(parents=True, mode=0o700)
    np.savez_compressed(
        args.output / "chain-outputs.npz",
        first_pairs=test_pairs.astype(np.int8),
        first_outputs=first_outputs.astype(np.float16),
        second_pairs=np.asarray(second_pairs, dtype=np.int8),
        second_test_rows=np.asarray(second_rows, dtype=np.int8),
        generated_inputs=np.stack(generated_inputs).astype(np.float16),
        target_inputs=np.stack(target_inputs).astype(np.float16),
        targets=np.stack(second_targets).astype(np.float16),
        outputs=generated_result["outputs"].astype(np.float16),
        predictions=generated_result["predictions"],
        target_operand_predictions=target_result["predictions"],
        both_steps_correct=both_correct,
    )
    write(args.output / "summary.json", summary)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
