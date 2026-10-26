#!/usr/bin/env python3
"""Build the source-receiver and paired-interface tables used in the paper."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np

MODELS = [
    "sana",
    "mlp-20260924",
    "mlp-20260925",
    "mlp-20260926",
    "pixel-2026091701",
    "pixel-2026091702",
    "pixel-2026091703",
]
DISPLAY = ["SANA", "MLP 1", "MLP 2", "MLP 3", "CNN 1", "CNN 2", "CNN 3"]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def family(model: str) -> str:
    if model == "sana":
        return "SANA"
    if model.startswith("mlp-"):
        return "MLP"
    if model.startswith("pixel-"):
        return "CNN"
    raise ValueError(model)


def main() -> None:
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interchange-csv", type=Path, required=True)
    parser.add_argument("--paired-json", type=Path, required=True)
    parser.add_argument("--mod23-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists() and not args.output.is_symlink()
    with args.interchange_csv.open(newline="", encoding="utf-8") as stream:
        binary_rows = list(csv.DictReader(stream))
    assert len(binary_rows) == 98
    assert {(row["writer"], row["receiver"], row["transform"]) for row in binary_rows} == {
        (writer, receiver, transform)
        for writer in MODELS
        for receiver in MODELS
        for transform in ("raw", "binary-128")
    }
    paired = json.loads(args.paired_json.read_text(encoding="utf-8"))
    assert len(paired["pairs"]) == 784

    binary_matrix_rows = []
    binary_summary = {}
    for transform in ("raw", "binary-128"):
        rows = [row for row in binary_rows if row["transform"] == transform]
        for row in rows:
            binary_matrix_rows.append(
                {
                    "source": row["writer"],
                    "receiver": row["receiver"],
                    "source_label": DISPLAY[MODELS.index(row["writer"])],
                    "receiver_label": DISPLAY[MODELS.index(row["receiver"])],
                    "source_family": family(row["writer"]),
                    "receiver_family": family(row["receiver"]),
                    "transform": transform,
                    "correct": int(row["consensus_correct"]),
                    "total": int(row["count"]),
                    "same_model": row["writer"] == row["receiver"],
                }
            )
        self_rows = [row for row in rows if row["writer"] == row["receiver"]]
        cross_rows = [row for row in rows if row["writer"] != row["receiver"]]
        binary_summary[transform] = {
            "all_correct": sum(int(row["consensus_correct"]) for row in rows),
            "all_total": sum(int(row["count"]) for row in rows),
            "same_model_correct": sum(int(row["consensus_correct"]) for row in self_rows),
            "same_model_total": sum(int(row["count"]) for row in self_rows),
            "cross_model_correct": sum(int(row["consensus_correct"]) for row in cross_rows),
            "cross_model_total": sum(int(row["count"]) for row in cross_rows),
        }

    result_npz = args.mod23_result / "cross-model-chain-outputs.npz"
    result_summary = json.loads((args.mod23_result / "summary.json").read_text(encoding="utf-8"))
    assert result_summary["complete"]
    with np.load(result_npz, allow_pickle=False) as data:
        seeds = data["seeds"].astype(int)
        first_pairs = data["first_pairs"].astype(int)
        second_pairs = data["second_pairs"].astype(int)
        first_predictions = data["first_predictions"].astype(int)
        canonical_predictions = data["canonical_second_predictions"].astype(int)
        chain_predictions = data["chain_predictions"].astype(int)
    first_truth = first_pairs.sum(axis=1) % 23
    second_truth = second_pairs.sum(axis=1) % 23
    matrix = []
    failures = []
    for source_position, source in enumerate(seeds):
        for receiver_position, receiver in enumerate(seeds):
            first_correct = first_predictions[source_position] == first_truth
            canonical_correct = canonical_predictions[receiver_position] == second_truth
            generated_correct = chain_predictions[source_position, receiver_position] == second_truth
            opportunity = first_correct & canonical_correct
            interface_failure = opportunity & ~generated_correct
            interface_success = opportunity & generated_correct
            repair = first_correct & ~canonical_correct & generated_correct
            row = {
                "source_run": int(source + 1),
                "receiver_run": int(receiver + 1),
                "same_model": bool(source == receiver),
                "first_correct": int(first_correct.sum()),
                "canonical_second_correct": int(canonical_correct.sum()),
                "generated_second_correct": int(generated_correct.sum()),
                "both_steps_correct": int((first_correct & generated_correct).sum()),
                "paired_opportunities": int(opportunity.sum()),
                "interface_successes": int(interface_success.sum()),
                "interface_failures": int(interface_failure.sum()),
                "generated_repairs": int(repair.sum()),
                "total": 46,
            }
            matrix.append(row)
            for index in np.flatnonzero(interface_failure):
                failures.append(
                    {
                        "source_run": int(source + 1),
                        "receiver_run": int(receiver + 1),
                        "same_model": bool(source == receiver),
                        "case_index": int(index),
                        "first_left": int(first_pairs[index, 0]),
                        "first_right": int(first_pairs[index, 1]),
                        "first_answer": int(first_truth[index]),
                        "second_left": int(second_pairs[index, 0]),
                        "second_right": int(second_pairs[index, 1]),
                        "second_answer": int(second_truth[index]),
                        "canonical_prediction": int(canonical_predictions[receiver_position, index]),
                        "generated_prediction": int(chain_predictions[source_position, receiver_position, index]),
                    }
                )
    cross = [row for row in matrix if not row["same_model"]]
    diagonal = [row for row in matrix if row["same_model"]]
    mod23_summary = {
        "cross_model_cases": sum(row["total"] for row in cross),
        "cross_model_both_steps_correct": sum(row["both_steps_correct"] for row in cross),
        "cross_model_paired_opportunities": sum(row["paired_opportunities"] for row in cross),
        "cross_model_interface_successes": sum(row["interface_successes"] for row in cross),
        "cross_model_interface_failures": sum(row["interface_failures"] for row in cross),
        "same_model_cases": sum(row["total"] for row in diagonal),
        "same_model_paired_opportunities": sum(row["paired_opportunities"] for row in diagonal),
        "same_model_interface_failures": sum(row["interface_failures"] for row in diagonal),
    }
    assert mod23_summary["cross_model_interface_successes"] == 210
    assert mod23_summary["cross_model_interface_failures"] == 5

    args.output.mkdir(parents=True, mode=0o700)
    with (args.output / "binary-transfer-matrix.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(binary_matrix_rows[0]))
        writer.writeheader()
        writer.writerows(binary_matrix_rows)
    with (args.output / "mod23-cross-model-matrix.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(matrix[0]))
        writer.writeheader()
        writer.writerows(matrix)
    with (args.output / "mod23-interface-failures.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(failures[0]))
        writer.writeheader()
        writer.writerows(failures)
    summary = {
        "schema": 1,
        "complete": True,
        "sources": {
            "interchange_csv_sha256": digest(args.interchange_csv),
            "paired_json_sha256": digest(args.paired_json),
            "mod23_npz_sha256": digest(result_npz),
            "mod23_summary_sha256": digest(args.mod23_result / "summary.json"),
        },
        "binary": binary_summary,
        "black_white_pair_changes": paired["totals"]["consensus"],
        "mod23": mod23_summary,
    }
    write_path = args.output / "interface-summary.json"
    write_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
