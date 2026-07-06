#!/usr/bin/env python3
"""LOCAL-SAFE smoke test for PR9 Priority 0 report building."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_run(run_dir: Path) -> None:
    _write_json(
        run_dir / "metadata.json",
        {
            "run_id": "observed",
            "scene": "chair",
            "condition": "clean",
            "trainer": "gaussian-splatting",
            "iterations": 700,
            "command": ["python", "train.py", "--iterations", "700"],
        },
    )
    _write_json(
        run_dir / "summary.json",
        {"summary": {"returncode": 0, "elapsed_s": 36.0, "observation_only": True}},
    )
    _write_json(run_dir / "stats.json", {"stats": {}})
    _write_json(run_dir / "config_snapshot.json", {"observation_only": True})
    _write_json(
        run_dir / "training_events_summary.json",
        {
            "requested_iterations": 700,
            "scene": "chair",
            "condition": "clean",
            "trainer": "gaussian-splatting",
            "final_gaussian_count": 92955,
            "invalid_training_event_rows": 0,
        },
    )
    _write_json(
        run_dir / "gaussian_lifecycle_summary.json",
        {
            "final_gaussian_count": 92955,
            "known_gaussian_count": 101437,
            "alive_final_count": 92955,
            "dead_final_count": 8482,
            "invariant_violations": 0,
        },
    )
    _write_json(
        run_dir / "view_metrics_summary.json",
        {"view_count_total": 28, "view_count_by_split": {"train": 20, "test": 5, "target": 3}},
    )
    (run_dir / "stdout.log").write_text("ok\n", encoding="utf-8")
    (run_dir / "stderr.log").write_text("", encoding="utf-8")
    (run_dir / "trainer_output").mkdir(parents=True, exist_ok=True)
    tables = run_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    for name in (
        "command_summary.csv",
        "gpu_memory_samples.csv",
        "training_events.csv",
        "densification_events.csv",
        "gaussian_count_timeseries.csv",
        "gaussian_lifecycle_events.csv",
        "gaussian_lifecycle_final.csv",
        "view_metrics.csv",
        "view_render_artifacts.csv",
    ):
        (tables / name).write_text("status\nok\n", encoding="utf-8")


def _run_report(project_root: Path, run_dir: Path, output: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts" / "measure" / "build_priority0_report.py"),
            "--run-dir",
            str(run_dir),
            "--output-dir",
            str(output),
            "--include-view-metrics",
            "--include-training-events",
            "--include-gaussian-lifecycle",
            "--require-priority0-complete",
            "--write-markdown",
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def main() -> int:
    project_root = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory(prefix="viewtrust-priority0-report-") as tmp:
        tmp_root = Path(tmp)
        run_dir = tmp_root / "observed"
        _write_run(run_dir)
        output = tmp_root / "report"
        completed = _run_report(project_root, run_dir, output)
        if completed.returncode != 0:
            raise RuntimeError(completed.stderr or completed.stdout)
        for name in (
            "priority0_report_summary.json",
            "priority0_report.md",
            "priority0_artifact_manifest.csv",
        ):
            if not (output / name).exists():
                raise FileNotFoundError(output / name)
        summary = json.loads((output / "priority0_report_summary.json").read_text())
        if summary["priority0_complete"] is not True:
            raise ValueError("priority0_complete should be true")
        if summary["missing_required_artifacts"]:
            raise ValueError("missing_required_artifacts should be empty")

        missing_run = tmp_root / "missing"
        _write_run(missing_run)
        (missing_run / "tables" / "gaussian_lifecycle_final.csv").unlink()
        missing_completed = _run_report(project_root, missing_run, tmp_root / "missing-report")
        if missing_completed.returncode == 0:
            raise ValueError("missing required artifact should fail")

    print("priority0 report smoke test ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
