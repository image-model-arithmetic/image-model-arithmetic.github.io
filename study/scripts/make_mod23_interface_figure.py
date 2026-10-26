#!/usr/bin/env python3
"""Draw the paired modulo-23 interface matrix and one failure example."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-csv", type=Path, required=True)
    parser.add_argument("--outputs", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists() and not args.output.is_symlink()
    with args.matrix_csv.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 9
    successes = np.zeros((3, 3), dtype=int)
    opportunities = np.zeros((3, 3), dtype=int)
    for row in rows:
        source = int(row["source_run"]) - 1
        receiver = int(row["receiver_run"]) - 1
        successes[source, receiver] = int(row["interface_successes"])
        opportunities[source, receiver] = int(row["paired_opportunities"])
    assert successes.sum() == 317 and opportunities.sum() == 323
    with np.load(args.outputs, allow_pickle=False) as data:
        first_outputs = data["first_outputs"].astype(np.float32)
        chain_outputs = data["chain_outputs"].astype(np.float32)
    with np.load(args.dataset, allow_pickle=False) as data:
        prototypes = data["prototypes"].astype(np.float32)

    figure = plt.figure(figsize=(7.0, 3.15))
    grid = figure.add_gridspec(1, 2, width_ratios=[1.0, 1.5], wspace=0.32)
    ax = figure.add_subplot(grid[0, 0])
    rates = successes / opportunities
    ax.imshow(rates, cmap="Blues", vmin=0.88, vmax=1.0)
    labels = ["Run 1", "Run 2", "Run 3"]
    for source in range(3):
        for receiver in range(3):
            color = "white" if rates[source, receiver] > 0.95 else "#202020"
            ax.text(
                receiver,
                source,
                f"{successes[source, receiver]}/{opportunities[source, receiver]}",
                ha="center",
                va="center",
                fontsize=9,
                color=color,
            )
            if source == receiver:
                ax.add_patch(plt.Rectangle((receiver - 0.48, source - 0.48), 0.96, 0.96, fill=False, edgecolor="#e39d31", linewidth=1.5))
    ax.set_xticks(range(3), labels, fontsize=8)
    ax.set_yticks(range(3), labels, fontsize=8)
    ax.set_xlabel("Receiving model", fontsize=8.5)
    ax.set_ylabel("Source model", fontsize=8.5)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("Generated / clean", fontsize=9, pad=9)
    ax.text(1.0, 3.05, "Separate models: 210/215", ha="center", va="center", fontsize=8, weight="bold")

    right = grid[0, 1].subgridspec(2, 4, height_ratios=[0.25, 1.0], wspace=0.14)
    title = figure.add_subplot(right[0, :])
    title.axis("off")
    title.text(0.5, 0.80, "A readable image can still change the computation", ha="center", va="center", fontsize=9, weight="bold")
    title.text(0.5, 0.08, "Run 2 draws 0 for 9 + 14; Run 1 then computes 0 + 8", ha="center", va="center", fontsize=7.5)
    images = [
        (prototypes[0, :, :, 0], "Clean 0", "receiver gives 8"),
        (first_outputs[1, 28, :, :, 0], "Generated 0", "reader gives 0"),
        (prototypes[8, :, :, 0], "Target answer", "8"),
        (chain_outputs[1, 0, 28, :, :, 0], "Receiver output", "read as 18"),
    ]
    for position, (image, heading, detail) in enumerate(images):
        image_ax = figure.add_subplot(right[1, position])
        image_ax.imshow(image, cmap="gray_r", vmin=0, vmax=1, interpolation="nearest")
        image_ax.set_title(f"{heading}\n{detail}", fontsize=7.2, color="#a33a32" if position == 3 else "#202020", pad=3)
        image_ax.set_xticks([])
        image_ax.set_yticks([])
        for spine in image_ax.spines.values():
            spine.set_linewidth(0.8)
            spine.set_edgecolor("#a33a32" if position == 3 else "#3f4a54")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, bbox_inches="tight")
    plt.close(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
