#!/usr/bin/env python3
"""LOCAL-SAFE smoke test for PR8 lifecycle invariant inspection."""

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


def _write_valid_run(run_dir: Path) -> None:
    summary = {
        "schema_name": "viewtrust.gaussian_lifecycle.summary",
        "schema_version": 1,
        "run_id": "valid",
        "observation_only": True,
        "enabled": True,
        "initial_gaussian_count": 3,
        "final_gaussian_count": 2,
        "known_gaussian_count": 3,
        "alive_final_count": 2,
        "dead_final_count": 1,
        "birth_event_count": 0,
        "prune_death_count": 1,
        "invariant_violations": 0,
        "warnings": [],
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "gaussian_lifecycle_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_csv(run_dir / "tables" / "gaussian_lifecycle_events.csv", ["event_type"], [])
    _write_csv(
        run_dir / "tables" / "gaussian_lifecycle_final.csv",
        [
            "gaussian_id",
            "alive",
            "final_index",
            "death_iteration",
            "lifetime_iterations",
        ],
        [
            {"gaussian_id": 0, "alive": "true", "final_index": 0, "death_iteration": "", "lifetime_iterations": 5},
            {"gaussian_id": 1, "alive": "false", "final_index": "", "death_iteration": 3, "lifetime_iterations": 3},
            {"gaussian_id": 2, "alive": "true", "final_index": 1, "death_iteration": "", "lifetime_iterations": 5},
        ],
    )


def _run_inspector(project_root: Path, run_dir: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts" / "measure" / "inspect_gaussian_lifecycle.py"),
            "--run-dir",
            str(run_dir),
            "--require-lifecycle",
            "--require-no-invariant-violations",
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def main() -> int:
    project_root = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory(prefix="viewtrust-lifecycle-invariant-") as tmp:
        tmp_root = Path(tmp)
        valid_run = tmp_root / "valid"
        missing_run = tmp_root / "missing"
        invalid_run = tmp_root / "invalid"
        _write_valid_run(valid_run)
        _write_valid_run(invalid_run)

        valid = _run_inspector(project_root, valid_run)
        if valid.returncode != 0:
            raise RuntimeError(valid.stderr or valid.stdout)

        missing = _run_inspector(project_root, missing_run)
        if missing.returncode == 0:
            raise ValueError("missing lifecycle outputs unexpectedly passed")

        final_csv = invalid_run / "tables" / "gaussian_lifecycle_final.csv"
        rows = list(csv.DictReader(final_csv.open(newline="", encoding="utf-8")))
        rows[2]["final_index"] = "0"
        _write_csv(final_csv, list(rows[0]), rows)
        invalid = _run_inspector(project_root, invalid_run)
        if invalid.returncode == 0:
            raise ValueError("duplicate final_index unexpectedly passed")
        report = json.loads(invalid.stdout)
        if report["invariant_violations"] <= 0:
            raise ValueError("duplicate final_index was not reported")

    print("gaussian lifecycle invariant smoke test ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
