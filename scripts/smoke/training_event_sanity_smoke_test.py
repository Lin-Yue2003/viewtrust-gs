#!/usr/bin/env python3
"""LOCAL-SAFE smoke test for PR7.2 training event sanity inspection."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_run(run_dir: Path, *, invalid: bool) -> None:
    (run_dir / "tables").mkdir(parents=True)
    training_rows = [
        {
            "iteration": 1,
            "event_type": "iteration_metrics",
            "gaussian_count": 100,
            "visible_gaussian_count": 25 if not invalid else 125,
            "visibility_ratio": 0.25 if not invalid else 1.25,
            "status": "ok",
        }
    ]
    _write_csv(
        run_dir / "tables" / "training_events.csv",
        [
            "iteration",
            "event_type",
            "gaussian_count",
            "visible_gaussian_count",
            "visibility_ratio",
            "status",
        ],
        training_rows,
    )
    _write_csv(
        run_dir / "tables" / "densification_events.csv",
        ["iteration", "status"],
        [],
    )
    _write_csv(
        run_dir / "tables" / "gaussian_count_timeseries.csv",
        ["iteration", "stage", "gaussian_count"],
        [{"iteration": 1, "stage": "iteration_end", "gaussian_count": 100}],
    )
    summary = {
        "schema_name": "viewtrust.training_events.summary",
        "schema_version": 1,
        "requested_iterations": 1,
        "logged_iteration_count": 1,
        "training_event_rows": 1,
        "invalid_training_event_rows": 0,
        "observation_only": True,
    }
    (run_dir / "training_events_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _run_inspector(project_root: Path, run_dir: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts" / "measure" / "inspect_training_events.py"),
            "--run-dir",
            str(run_dir),
            "--require-events",
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def main() -> int:
    project_root = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory(prefix="viewtrust-event-sanity-") as tmp:
        tmp_root = Path(tmp)
        valid_run = tmp_root / "valid"
        invalid_run = tmp_root / "invalid"
        _write_run(valid_run, invalid=False)
        _write_run(invalid_run, invalid=True)

        valid = _run_inspector(project_root, valid_run)
        if valid.returncode != 0:
            raise RuntimeError(valid.stderr or valid.stdout)
        valid_report = json.loads(valid.stdout)
        if valid_report["invalid_training_event_rows"] != 0:
            raise ValueError("valid run reported invalid training event rows")

        invalid = _run_inspector(project_root, invalid_run)
        if invalid.returncode == 0:
            raise ValueError("invalid run unexpectedly passed sanity inspection")
        invalid_report = json.loads(invalid.stdout)
        if invalid_report["invalid_training_event_rows"] != 1:
            raise ValueError("invalid row count was not detected")
        if "invalid scalar rows" not in invalid.stderr:
            raise ValueError("invalid sanity failure did not print a clear error")

    print("training event sanity smoke test ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
