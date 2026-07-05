#!/usr/bin/env python3
"""LOCAL-SAFE smoke test for PR6 view metrics extraction."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image


def _bootstrap_project_imports() -> Path:
    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root))
    return project_root


def _write_png(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color).save(path)


def _write_render_summary(run_dir: Path, counts: dict[str, int]) -> None:
    path = run_dir / "view_evaluation" / "view_render_summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "preflight": {
                    "expected_view_count_by_split": counts,
                }
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    project_root = _bootstrap_project_imports()

    with tempfile.TemporaryDirectory(prefix="viewtrust-view-metrics-") as tmp:
        run_dir = Path(tmp) / "run"
        roots = [
            run_dir / "view_evaluation" / "render_models" / "train_test_model" / "train" / "ours_500",
            run_dir / "view_evaluation" / "render_models" / "train_test_model" / "test" / "ours_500",
            run_dir / "view_evaluation" / "render_models" / "target_model" / "test" / "ours_500",
        ]
        for index, root in enumerate(roots):
            _write_png(root / "renders" / "00000.png", (10 + index, 20, 30))
            _write_png(root / "gt" / "00000.png", (12 + index, 20, 31))
        _write_render_summary(run_dir, {"train": 1, "test": 1, "target": 1})

        completed = subprocess.run(
            [
                sys.executable,
                str(project_root / "scripts" / "measure" / "extract_view_metrics.py"),
                "--run-dir",
                str(run_dir),
                "--scene",
                "chair",
                "--condition",
                "clean",
                "--iteration",
                "500",
                "--require-renders",
            ],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if completed.returncode != 0:
            raise RuntimeError(completed.stderr or completed.stdout)

        summary_path = run_dir / "view_metrics_summary.json"
        metrics_path = run_dir / "tables" / "view_metrics.csv"
        artifacts_path = run_dir / "tables" / "view_render_artifacts.csv"
        for path in (summary_path, metrics_path, artifacts_path):
            if not path.exists():
                raise FileNotFoundError(path)

        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary["view_count_total"] != 3:
            raise ValueError("view_count_total mismatch")
        if summary["view_count_by_split"] != {"target": 1, "test": 1, "train": 1}:
            raise ValueError("view_count_by_split mismatch")
        if summary["expected_view_count_by_split"] != {"target": 1, "test": 1, "train": 1}:
            raise ValueError("expected_view_count_by_split mismatch")
        if summary["rendered_splits"] != ["train", "test", "target"]:
            raise ValueError("rendered_splits mismatch")

        rows = _read_csv_rows(metrics_path)
        if len(rows) != 3:
            raise ValueError("view_metrics.csv row count mismatch")
        for row in rows:
            if row["status"] != "ok":
                raise ValueError(f"metric row status is not ok: {row}")
            float(row["l1_mean"])
            float(row["psnr"])

        artifact_rows = _read_csv_rows(artifacts_path)
        if len(artifact_rows) != 6:
            raise ValueError("view_render_artifacts.csv row count mismatch")

        missing_run_dir = Path(tmp) / "missing-run"
        train_root = (
            missing_run_dir
            / "view_evaluation"
            / "render_models"
            / "train_test_model"
            / "train"
            / "ours_500"
        )
        _write_png(train_root / "renders" / "00000.png", (10, 20, 30))
        _write_png(train_root / "gt" / "00000.png", (12, 20, 31))
        _write_render_summary(missing_run_dir, {"train": 1, "test": 1, "target": 1})
        missing_completed = subprocess.run(
            [
                sys.executable,
                str(project_root / "scripts" / "measure" / "extract_view_metrics.py"),
                "--run-dir",
                str(missing_run_dir),
                "--scene",
                "chair",
                "--condition",
                "clean",
                "--iteration",
                "500",
                "--require-renders",
            ],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if missing_completed.returncode == 0:
            raise ValueError("missing requested splits should fail under --require-renders")
        if "requested split test has zero render/gt pairs" not in missing_completed.stderr:
            raise ValueError(missing_completed.stderr or missing_completed.stdout)

    print("view metrics extraction smoke test ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
