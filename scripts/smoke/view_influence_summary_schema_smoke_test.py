#!/usr/bin/env python3
"""LOCAL-SAFE smoke test for PR12.1 view influence summary schema."""

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
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    project_root = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory(prefix="viewtrust-view-influence-schema-") as tmp:
        root = Path(tmp)
        run_dir = root / "run"
        output_dir = root / "out"
        _write_json(
            run_dir / "metadata.json",
            {"metadata": {"run_id": "schema", "scene": "chair", "condition": "clean"}},
        )
        _write_json(
            run_dir / "summary.json",
            {"summary": {"returncode": 0, "observation_only": True}},
        )
        _write_json(
            run_dir / "training_events_summary.json",
            {"invalid_training_event_rows": 0, "observation_only": True},
        )
        _write_json(
            run_dir / "gaussian_lifecycle_summary.json",
            {"invariant_violations": 0, "observation_only": True},
        )
        _write_json(run_dir / "view_metrics_summary.json", {"view_count_total": 3})
        _write_csv(
            run_dir / "tables" / "training_events.csv",
            [
                {
                    "iteration": 1,
                    "event_type": "iteration_metrics",
                    "view_name": "train_000",
                    "view_split": "train",
                    "loss": 0.1,
                    "l1_loss": 0.05,
                    "ssim": 0.9,
                    "gaussian_count": 100,
                    "visible_gaussian_count": 40,
                    "visibility_ratio": 0.4,
                    "radii_nonzero_count": 40,
                    "densification_triggered": "false",
                },
                {
                    "iteration": 2,
                    "event_type": "iteration_metrics",
                    "view_name": "test_000",
                    "view_split": "test",
                    "loss": 0.2,
                    "l1_loss": 0.1,
                    "ssim": 0.8,
                    "gaussian_count": 100,
                    "visible_gaussian_count": 30,
                    "visibility_ratio": 0.3,
                    "radii_nonzero_count": 30,
                    "densification_triggered": "false",
                },
                {
                    "iteration": 3,
                    "event_type": "iteration_metrics",
                    "view_name": "target_000",
                    "view_split": "target",
                    "loss": 0.3,
                    "l1_loss": 0.15,
                    "ssim": 0.7,
                    "gaussian_count": 100,
                    "visible_gaussian_count": 20,
                    "visibility_ratio": 0.2,
                    "radii_nonzero_count": 20,
                    "densification_triggered": "false",
                },
            ],
        )
        _write_csv(
            run_dir / "tables" / "densification_events.csv",
            [{"iteration": 1, "gaussian_count_before": 100, "gaussian_count_after": 101, "gaussian_count_delta": 1}],
        )
        _write_csv(
            run_dir / "tables" / "gaussian_lifecycle_final.csv",
            [{"gaussian_id": 1, "alive": "true"}],
        )
        _write_csv(
            run_dir / "tables" / "gaussian_lifecycle_events.csv",
            [
                {
                    "iteration": 1,
                    "source_iteration": 1,
                    "source_view_name": "train_000",
                    "source_view_split": "train",
                    "event_type": "clone_birth",
                    "gaussian_id": 1,
                }
            ],
        )
        _write_csv(
            run_dir / "tables" / "view_metrics.csv",
            [
                {"split": "train", "image_name": "train_000.png", "psnr": 20, "ssim": 0.8, "l1_mean": 0.1},
                {"split": "test", "image_name": "test_000.png", "psnr": 21, "ssim": 0.81, "l1_mean": 0.09},
                {"split": "target", "image_name": "target_000.png", "psnr": 22, "ssim": 0.82, "l1_mean": 0.08},
            ],
        )

        completed = subprocess.run(
            [
                sys.executable,
                str(project_root / "scripts" / "measure" / "build_view_influence_table.py"),
                "--run-dir",
                str(run_dir),
                "--condition",
                "clean",
                "--output-dir",
                str(output_dir),
                "--quiet",
            ],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if completed.returncode != 0:
            raise RuntimeError(completed.stderr or completed.stdout)
        summary = json.loads((output_dir / "view_influence_summary.json").read_text(encoding="utf-8"))
        if summary["observation_only"] is not True:
            raise ValueError("observation_only should propagate from nested summary sources")
        if summary["observation_only_sources"]["summary"] is not True:
            raise ValueError("nested summary observation_only source was not recorded")
        if summary["sampled_train_view_count"] != 1:
            raise ValueError("sampled train count mismatch")
        if summary["sampled_test_view_count"] != 1:
            raise ValueError("sampled test count mismatch")
        if summary["sampled_target_view_count"] != 1:
            raise ValueError("sampled target count mismatch")
        if summary["unexpected_non_train_sampled_view_count"] != 2:
            raise ValueError("unexpected non-train sampled count mismatch")
        if not summary["warnings"]:
            raise ValueError("non-train sampled views should emit a warning")

    print("view influence summary schema smoke test ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
