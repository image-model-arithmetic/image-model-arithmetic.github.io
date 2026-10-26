#!/usr/bin/env python3
"""Train registered restarts, select on development pairs, and test once."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import run_symmetric_segment_study_mlx as base


def run(root: Path) -> None:
    config = base.verify(root)
    assert config["selection_across_seeds"]["enabled"] is True
    assert not (root / "started.json").exists() and not (root / "terminal.json").exists()
    admission = base.admission(root, config)
    deadline = time.time() + 3 * 3600
    base.write(root / "started.json", {"admission": admission, "deadline_unix": deadline})
    caffeinate = subprocess.Popen(
        ["/usr/bin/caffeinate", "-dimsu", "-w", str(os.getpid())],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    completed = []
    error_kind = None
    try:
        (root / "runs").mkdir(mode=0o700)
        (root / "executions").mkdir(mode=0o700)
        for seed_index in config["execution"]["seed_order"]:
            base.write(root / "progress.json", {"phase": "training", "seed_index": seed_index})
            output = root / "runs" / f"seed-{seed_index}"
            base.execute(
                [
                    sys.executable,
                    "-B",
                    str(root / "source/train_symmetric_segment_mod23_mlx.py"),
                    "--config",
                    str(root / "config.json"),
                    "--dataset",
                    str(root / "dataset"),
                    "--output",
                    str(output),
                    "--seed-index",
                    str(seed_index),
                    "--defer-test",
                ],
                root / "executions" / f"seed-{seed_index}",
                deadline,
            )
            summary = json.loads((output / "development-summary.json").read_text())
            assert summary["complete"] and summary["test_evaluations"] == 0
            completed.append(summary)
        assert len(completed) == len(config["seeds"])

        def score(row):
            selection = row["selection"]
            return (
                selection["development"]["exact_raster"],
                selection["development"]["accuracy"],
                selection["train"]["exact_raster"],
                -selection["development"]["mse"],
            )

        selected = max(completed, key=score)
        selected_index = selected["seed_index"]
        base.write(
            root / "selected-seed.json",
            {
                "seed_index": selected_index,
                "rule": config["selection_across_seeds"]["rule"],
                "development_scores": [
                    {
                        "seed_index": row["seed_index"],
                        "selected_step": row["selected_step"],
                        "accuracy": row["selection"]["development"]["accuracy"],
                        "exact_raster": row["selection"]["development"]["exact_raster"],
                    }
                    for row in completed
                ],
            },
        )
        base.write(root / "progress.json", {"phase": "sealed-test", "seed_index": selected_index})
        selected_root = root / "runs" / f"seed-{selected_index}"
        base.execute(
            [
                sys.executable,
                "-B",
                str(root / "source/evaluate_selected_symmetric_segment_mod23_mlx.py"),
                "--config",
                str(root / "config.json"),
                "--dataset",
                str(root / "dataset"),
                "--checkpoint",
                str(selected_root / "selected.safetensors"),
                "--development-summary",
                str(selected_root / "development-summary.json"),
                "--output",
                str(root / "sealed-test"),
            ],
            root / "executions" / "sealed-test",
            deadline,
        )
        test = json.loads((root / "sealed-test/summary.json").read_text())
        assert test["complete"] and test["test_evaluations"] == 1
        summary = {
            "complete": True,
            "registered_seed_count": len(config["seeds"]),
            "completed_seed_count": len(completed),
            "selected_seed_index": selected_index,
            "development_accuracies": [
                row["selection"]["development"]["accuracy"] for row in completed
            ],
            "development_exact_rasters": [
                row["selection"]["development"]["exact_raster"] for row in completed
            ],
            "test": test["test"],
            "swapped_test": test["swapped_test"],
            "test_evaluations": test["test_evaluations"],
            "passes_registered_threshold": test["test"]["accuracy"]
            >= config["reporting"]["success_accuracy"],
            "config_sha256": base.digest(root / "config.json"),
            "dataset_sha256": base.digest(root / "dataset/dataset.npz"),
        }
        base.write(root / "summary.json", summary)
        base.write(root / "progress.json", {"phase": "complete", "seed_index": selected_index})
    except BaseException as error:
        error_kind = type(error).__name__
        raise
    finally:
        if caffeinate.poll() is None:
            caffeinate.terminate()
            try:
                caffeinate.wait(timeout=10)
            except subprocess.TimeoutExpired:
                caffeinate.kill()
                caffeinate.wait(timeout=10)
        base.write(
            root / "terminal.json",
            {
                "error_kind": error_kind,
                "completed_development_seed_count": len(completed),
                "caffeinate_quiescent": caffeinate.poll() is not None,
                "finished_unix": time.time(),
                "deadline_unix": deadline,
            },
        )


def launch(root: Path) -> None:
    base.verify(root)
    assert not (root / "launch.json").exists()
    stdout = (root / "controller.stdout").open("xb")
    stderr = (root / "controller.stderr").open("xb")
    try:
        child = subprocess.Popen(
            [sys.executable, "-B", str(Path(__file__).resolve()), "run", "--root", str(root)],
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            cwd=root,
            start_new_session=True,
            umask=0o077,
            env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(root / "source")),
        )
    finally:
        stdout.close()
        stderr.close()
    base.write(
        root / "launch.json",
        {
            "pid": child.pid,
            "controller_sha256": base.digest(Path(__file__).resolve()),
            "launched_unix": time.time(),
        },
    )
    print(json.dumps({"launched": True, "pid": child.pid}))


def main() -> None:
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("launch", "run", "verify"):
        command = commands.add_parser(name)
        command.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.command == "launch":
        launch(root)
    elif args.command == "run":
        run(root)
    else:
        config = base.verify(root)
        print(
            json.dumps(
                {
                    "verified": True,
                    "registered_seed_count": len(config["seeds"]),
                    "config_sha256": base.digest(root / "config.json"),
                    "dataset_sha256": base.digest(root / "dataset/dataset.npz"),
                },
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
