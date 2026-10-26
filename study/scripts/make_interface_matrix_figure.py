#!/usr/bin/env python3
"""Draw the binary source-receiver matrix and matched image-change results."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch


def box(ax, x, text, color):
    ax.add_patch(
        FancyBboxPatch(
            (x, 0.28), 0.19, 0.50,
            boxstyle="round,pad=0.012,rounding_size=0.025",
            facecolor=color, edgecolor="#3f4a54", linewidth=0.9,
        )
    )
    ax.text(x + 0.095, 0.53, text, ha="center", va="center", fontsize=9, weight="bold")


def annotate_matrix(ax, values, labels, *, delta=False):
    if delta:
        image = ax.imshow(values, cmap="RdBu", vmin=-2, vmax=2)
    else:
        image = ax.imshow(values, cmap="Blues", vmin=12, vmax=16)
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            value = int(values[row, col])
            text = f"{value:+d}" if delta else f"{value}/16"
            color = "white" if (not delta and value >= 15) or (delta and abs(value) >= 2) else "#202020"
            ax.text(col, row, text, ha="center", va="center", fontsize=6.6, color=color)
            if row == col:
                ax.add_patch(plt.Rectangle((col - 0.48, row - 0.48), 0.96, 0.96, fill=False, edgecolor="#e39d31", linewidth=1.4))
    ax.set_xticks(np.arange(len(labels)), labels, rotation=40, ha="right", fontsize=7)
    ax.set_yticks(np.arange(len(labels)), labels, fontsize=7)
    ax.set_xlabel("Receiving model", fontsize=8)
    ax.set_ylabel("Source model", fontsize=8)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return image


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists() and not args.output.is_symlink()
    with args.matrix_csv.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    models = []
    for row in rows:
        if row["source_label"] not in models:
            models.append(row["source_label"])
    assert len(models) == 7
    raw = np.zeros((7, 7), dtype=int)
    converted = np.zeros((7, 7), dtype=int)
    for row in rows:
        target = raw if row["transform"] == "raw" else converted
        target[models.index(row["source_label"]), models.index(row["receiver_label"])] = int(row["correct"])
    assert raw.sum() == 773 and converted.sum() == 769

    figure = plt.figure(figsize=(7.0, 4.55))
    grid = figure.add_gridspec(2, 2, height_ratios=[0.76, 3.5], hspace=0.32, wspace=0.33)
    flow = figure.add_subplot(grid[0, :])
    flow.set_xlim(0, 1)
    flow.set_ylim(0, 1)
    flow.axis("off")
    box(flow, 0.03, "Source model", "#d9e8f5")
    box(flow, 0.30, "Generated digit", "#f4e4bd")
    box(flow, 0.57, "Receiving model", "#d9efdf")
    box(flow, 0.81, "New answer", "#e5dff0")
    for left, right in ((0.22, 0.30), (0.49, 0.57), (0.76, 0.81)):
        flow.annotate("", xy=(right, 0.53), xytext=(left, 0.53), arrowprops={"arrowstyle": "->", "lw": 1.1})
    flow.text(0.395, 0.11, "Readers score the digit after the model calls", ha="center", va="center", fontsize=7.5, color="#444444")
    flow.annotate("", xy=(0.395, 0.27), xytext=(0.395, 0.17), arrowprops={"arrowstyle": "->", "lw": 0.8, "linestyle": ":", "color": "#666666"})

    raw_ax = figure.add_subplot(grid[1, 0])
    annotate_matrix(raw_ax, raw, models)
    raw_ax.set_title("Generated digits: correct answers", fontsize=9, pad=8)
    delta_ax = figure.add_subplot(grid[1, 1])
    annotate_matrix(delta_ax, converted - raw, models, delta=True)
    delta_ax.set_title("Change after making digits black and white", fontsize=9, pad=8)
    figure.text(0.26, 0.01, "Orange boxes are same-model controls", ha="center", fontsize=7.2, color="#555555")
    figure.text(0.74, 0.01, "Blue adds correct answers; red removes them", ha="center", fontsize=7.2, color="#555555")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, bbox_inches="tight")
    plt.close(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
