#!/usr/bin/env python3
"""LOCAL-SAFE smoke test for PR12 view influence comparison."""

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


def _make_dir(root: Path, *, condition: str, corrupt: bool) -> None:
    _write_json(root / "view_influence_summary.json", {"run_id": condition, "scene": "chair", "condition": condition})
    _write_csv(
        root / "view_influence.csv",
        [
            {
                "view_name": "train_000",
                "was_corrupted": "true" if corrupt else "false",
                "times_sampled": 4 if corrupt else 3,
                "birth_event_count_after_view": 5 if corrupt else 2,
                "prune_death_count_after_view": 9 if corrupt else 1,
                "birth_survival_ratio_after_view": 0.4 if corrupt else 0.8,
                "mean_visibility_ratio": 0.7 if corrupt else 0.9,
            },
            {
                "view_name": "train_001",
                "was_corrupted": "false",
                "times_sampled": 3,
                "birth_event_count_after_view": 1,
                "prune_death_count_after_view": 2,
                "birth_survival_ratio_after_view": 1.0,
                "mean_visibility_ratio": 0.8,
            },
        ],
    )


def main() -> int:
    project_root = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory(prefix="viewtrust-view-influence-compare-") as tmp:
        root = Path(tmp)
        clean_dir = root / "clean"
        corrupt_dir = root / "corrupt"
        output_dir = root / "out"
        _make_dir(clean_dir, condition="clean", corrupt=False)
        _make_dir(corrupt_dir, condition="corrupt_occluder", corrupt=True)
        completed = subprocess.run(
            [
                sys.executable,
                str(project_root / "scripts" / "measure" / "compare_view_influence_tables.py"),
                "--clean-view-influence-dir",
                str(clean_dir),
                "--corrupt-view-influence-dir",
                str(corrupt_dir),
                "--output-dir",
                str(output_dir),
                "--write-markdown",
            ],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if completed.returncode != 0:
            raise RuntimeError(completed.stderr or completed.stdout)
        for name in (
            "view_influence_comparison_summary.json",
            "view_influence_comparison.csv",
            "view_influence_comparison_report.md",
        ):
            if not (output_dir / name).exists():
                raise FileNotFoundError(output_dir / name)
        summary = json.loads((output_dir / "view_influence_comparison_summary.json").read_text())
        if summary["joined_view_count"] != 2:
            raise ValueError("joined view count mismatch")
        rows = list(csv.DictReader((output_dir / "view_influence_comparison.csv").open(newline="", encoding="utf-8")))
        by_view = {row["view_name"]: row for row in rows}
        if float(by_view["train_000"]["prune_death_count_delta"]) != 8.0:
            raise ValueError("prune death delta mismatch")

    print("view influence comparison smoke test ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
