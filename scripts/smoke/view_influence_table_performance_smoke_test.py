#!/usr/bin/env python3
"""LOCAL-SAFE smoke test for PR12.1 view influence timing/progress fields."""

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
    with tempfile.TemporaryDirectory(prefix="viewtrust-view-influence-perf-") as tmp:
        root = Path(tmp)
        run_dir = root / "run"
        output_dir = root / "out"
        _write_json(run_dir / "metadata.json", {"metadata": {"run_id": "perf"}})
        _write_json(run_dir / "summary.json", {"summary": {"returncode": 0, "observation_only": True}})
        _write_json(run_dir / "training_events_summary.json", {"invalid_training_event_rows": 0, "observation_only": True})
        _write_json(run_dir / "gaussian_lifecycle_summary.json", {"invariant_violations": 0, "observation_only": True})
        _write_json(run_dir / "view_metrics_summary.json", {"view_count_total": 2})
        _write_csv(
            run_dir / "tables" / "training_events.csv",
            [
                {
                    "iteration": i,
                    "event_type": "iteration_metrics",
                    "view_name": f"train_{i % 2:03d}",
                    "view_split": "train",
                    "loss": 0.1,
                    "l1_loss": 0.05,
                    "ssim": 0.9,
                    "gaussian_count": 100,
                    "visible_gaussian_count": 50,
                    "visibility_ratio": 0.5,
                    "radii_nonzero_count": 50,
                    "densification_triggered": "false",
                }
                for i in range(1, 21)
            ],
        )
        _write_csv(
            run_dir / "tables" / "densification_events.csv",
            [{"iteration": 10, "gaussian_count_before": 100, "gaussian_count_after": 120, "gaussian_count_delta": 20}],
        )
        _write_csv(
            run_dir / "tables" / "gaussian_lifecycle_final.csv",
            [{"gaussian_id": i, "alive": "true"} for i in range(200)],
        )
        _write_csv(
            run_dir / "tables" / "gaussian_lifecycle_events.csv",
            [
                {
                    "iteration": i,
                    "source_iteration": i,
                    "source_view_name": f"train_{i % 2:03d}",
                    "source_view_split": "train",
                    "event_type": "clone_birth" if i % 3 else "prune_death",
                    "gaussian_id": i,
                }
                for i in range(200)
            ],
        )
        _write_csv(
            run_dir / "tables" / "view_metrics.csv",
            [
                {"split": "train", "image_name": "train_000.png", "psnr": 20, "ssim": 0.8, "l1_mean": 0.1},
                {"split": "train", "image_name": "train_001.png", "psnr": 21, "ssim": 0.82, "l1_mean": 0.09},
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
                "--progress-interval-rows",
                "10",
                "--quiet",
            ],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if completed.returncode != 0:
            raise RuntimeError(completed.stderr or completed.stdout)
        if completed.stderr.strip():
            raise ValueError("--quiet should suppress lifecycle progress logs")
        summary = json.loads((output_dir / "view_influence_summary.json").read_text(encoding="utf-8"))
        if summary["input_rows"]["lifecycle_event_rows"] != 200:
            raise ValueError("lifecycle input row count mismatch")
        if "runtime_s" not in summary or "timing" not in summary or "throughput" not in summary:
            raise ValueError("summary missing PR12.1 timing fields")
        for key in (
            "load_training_events_s",
            "load_densification_events_s",
            "load_final_lifecycle_s",
            "stream_lifecycle_events_s",
            "load_view_metrics_s",
            "load_corruption_manifest_s",
            "write_outputs_s",
        ):
            if key not in summary["timing"]:
                raise ValueError(f"missing timing field: {key}")

    print("view influence table performance smoke test ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
