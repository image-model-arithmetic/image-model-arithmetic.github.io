#!/usr/bin/env python3
"""Plot development accuracy for the five fixed modulo-23 training runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists() and not args.output.is_symlink()

    figure, axis = plt.subplots(figsize=(6.9, 2.65))
    colors = ["#3b75af", "#d17529", "#4b9a55", "#a14f91", "#777777"]
    final_scores = []
    for seed in range(5):
        run = args.runs / f"seed-{seed}"
        rows = [json.loads(line) for line in (run / "development-history.jsonl").read_text().splitlines()]
        assert len(rows) == 100
        steps = [int(row["step"]) for row in rows]
        accuracy = [46 * float(row["development_accuracy"]) for row in rows]
        assert steps == list(range(1000, 100001, 1000))
        selection = json.loads((run / "selection.json").read_text())
        selected_step = int(selection["step"])
        selected_score = 46 * float(selection["development"]["accuracy"])
        final_scores.append(round(selected_score))
        axis.plot(steps, accuracy, color=colors[seed], linewidth=1.25, alpha=0.9, label=f"Run {seed + 1}: {round(selected_score)}/46")
        axis.scatter([selected_step], [selected_score], color=colors[seed], s=22, edgecolor="white", linewidth=0.5, zorder=4)

    assert final_scores == [42, 44, 40, 9, 11]
    axis.set_xlim(0, 100000)
    axis.set_ylim(0, 47)
    axis.set_xlabel("Training update")
    axis.set_ylabel("Development pairs correct (of 46)")
    axis.grid(axis="y", color="#d8d8d8", linewidth=0.6)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(ncol=3, frameon=False, fontsize=7.5, loc="upper center")
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, bbox_inches="tight")
    plt.close(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
