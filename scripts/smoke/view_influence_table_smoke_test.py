#!/usr/bin/env python3
"""LOCAL-SAFE smoke test for PR12 view influence table generation."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def _make_condition(data_root: Path) -> Path:
    root = data_root / "viewtrust-mini" / "nerf_synthetic" / "chair" / "corrupt_occluder"
    _write_json(
        root / "corruption_summary.json",
        {
            "scene": "chair",
            "source_condition": "clean",
            "output_condition": "corrupt_occluder",
            "corruption_type": "occluder",
            "seed": 20260706,
            "selected_train_view_count": 1,
            "corrupted_image_count": 1,
            "selected_train_views": ["train_000"],
        },
    )
    _write_csv(
        root / "corruption_manifest.csv",
        [
            {"split": "train", "view_name": "train_000", "was_corrupted": "true", "corruption_type": "occluder"},
            {"split": "train", "view_name": "train_001", "was_corrupted": "false", "corruption_type": ""},
        ],
    )
    return root


def _make_run(run_dir: Path, condition_root: Path, *, missing_view: bool = False, missing_source: bool = False) -> None:
    _write_json(
        run_dir / "metadata.json",
        {
            "run_id": run_dir.name,
            "scene": "chair",
            "condition": "corrupt_occluder",
            "trainer": "gaussian-splatting",
            "prepared_scene_root": str(condition_root),
        },
    )
    _write_json(run_dir / "summary.json", {"returncode": 0, "observation_only": True})
    _write_json(run_dir / "training_events_summary.json", {"invalid_training_event_rows": 0})
    _write_json(run_dir / "gaussian_lifecycle_summary.json", {"invariant_violations": 0})
    _write_csv(
        run_dir / "tables" / "training_events.csv",
        [
            {
                "iteration": 10,
                "event_type": "iteration_metrics",
                "view_name": "" if missing_view else "train_000",
                "view_split": "train",
                "loss": 0.3,
                "l1_loss": 0.2,
                "ssim": 0.8,
                "gaussian_count": 100,
                "visible_gaussian_count": 80,
                "visibility_ratio": 0.8,
                "radii_nonzero_count": 81,
                "densification_triggered": "true",
            },
            {
                "iteration": 20,
                "event_type": "iteration_metrics",
                "view_name": "train_001",
                "view_split": "train",
                "loss": 0.4,
                "l1_loss": 0.25,
                "ssim": 0.75,
                "gaussian_count": 105,
                "visible_gaussian_count": 70,
                "visibility_ratio": 0.666,
                "radii_nonzero_count": 70,
                "densification_triggered": "false",
            },
        ],
    )
    _write_csv(
        run_dir / "tables" / "densification_events.csv",
        [{"iteration": 10, "gaussian_count_before": 100, "gaussian_count_after": 102, "gaussian_count_delta": 2}],
    )
    _write_csv(
        run_dir / "tables" / "gaussian_lifecycle_events.csv",
        [
            {
                "iteration": 10,
                "source_iteration": 10,
                "source_view_name": "" if missing_source else "train_000",
                "source_view_split": "train",
                "event_type": "clone_birth",
                "gaussian_id": 100,
            },
            {
                "iteration": 10,
                "source_iteration": 10,
                "source_view_name": "" if missing_source else "train_000",
                "source_view_split": "train",
                "event_type": "prune_death",
                "gaussian_id": 3,
            },
        ],
    )
    _write_csv(
        run_dir / "tables" / "gaussian_lifecycle_final.csv",
        [
            {"gaussian_id": 100, "alive": "true"},
            {"gaussian_id": 3, "alive": "false"},
        ],
    )
    _write_json(run_dir / "view_metrics_summary.json", {"view_count_total": 2})
    _write_csv(
        run_dir / "tables" / "view_metrics.csv",
        [
            {"split": "train", "image_name": "train_000.png", "psnr": 20.0, "ssim": 0.8, "l1_mean": 0.1},
            {"split": "train", "image_name": "train_001.png", "psnr": 22.0, "ssim": 0.85, "l1_mean": 0.08},
        ],
    )


def main() -> int:
    project_root = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory(prefix="viewtrust-view-influence-") as tmp:
        root = Path(tmp)
        data_root = root / "data"
        condition_root = _make_condition(data_root)
        run_dir = root / "run"
        _make_run(run_dir, condition_root)
        output_dir = root / "out"
        completed = _run(
            [
                sys.executable,
                str(project_root / "scripts" / "measure" / "build_view_influence_table.py"),
                "--run-dir",
                str(run_dir),
                "--data-root",
                str(data_root),
                "--scene",
                "chair",
                "--condition",
                "corrupt_occluder",
                "--output-dir",
                str(output_dir),
                "--require-view-identity",
                "--require-source-view",
                "--write-markdown",
            ]
        )
        if completed.returncode != 0:
            raise RuntimeError(completed.stderr or completed.stdout)
        for name in (
            "view_influence_summary.json",
            "view_influence.csv",
            "view_lifecycle_attribution.csv",
            "view_iteration_events.csv",
            "view_influence_artifact_manifest.csv",
            "view_influence_report.md",
        ):
            if not (output_dir / name).exists():
                raise FileNotFoundError(output_dir / name)
        summary = json.loads((output_dir / "view_influence_summary.json").read_text())
        if summary["corrupted_view_count"] != 1:
            raise ValueError("corrupted view count mismatch")
        rows = list(csv.DictReader((output_dir / "view_influence.csv").open(newline="", encoding="utf-8")))
        by_view = {row["view_name"]: row for row in rows}
        if by_view["train_000"]["was_corrupted"] != "true":
            raise ValueError("corrupted view was not labeled")
        if by_view["train_000"]["birth_event_count_after_view"] != "1":
            raise ValueError("birth count mismatch")
        if by_view["train_000"]["prune_death_count_after_view"] != "1":
            raise ValueError("prune count mismatch")
        if float(by_view["train_000"]["birth_survival_ratio_after_view"]) != 1.0:
            raise ValueError("birth survival ratio mismatch")

        missing_view_run = root / "missing-view"
        _make_run(missing_view_run, condition_root, missing_view=True)
        missing_view = _run(
            [
                sys.executable,
                str(project_root / "scripts" / "measure" / "build_view_influence_table.py"),
                "--run-dir",
                str(missing_view_run),
                "--data-root",
                str(data_root),
                "--condition",
                "corrupt_occluder",
                "--output-dir",
                str(root / "missing-view-out"),
                "--require-view-identity",
            ]
        )
        if missing_view.returncode == 0:
            raise ValueError("missing view identity should fail")

        missing_source_run = root / "missing-source"
        _make_run(missing_source_run, condition_root, missing_source=True)
        missing_source = _run(
            [
                sys.executable,
                str(project_root / "scripts" / "measure" / "build_view_influence_table.py"),
                "--run-dir",
                str(missing_source_run),
                "--data-root",
                str(data_root),
                "--condition",
                "corrupt_occluder",
                "--output-dir",
                str(root / "missing-source-out"),
                "--require-source-view",
            ]
        )
        if missing_source.returncode == 0:
            raise ValueError("missing lifecycle source view should fail")

    print("view influence table smoke test ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
