#!/usr/bin/env python3
"""LOCAL-SAFE smoke test for PR11.2 corruption manifest linking."""

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


def _make_run(run_dir: Path, *, condition: str, psnr: float) -> None:
    _write_json(
        run_dir / "metadata.json",
        {"run_id": run_dir.name, "scene": "chair", "condition": condition, "trainer": "gaussian-splatting", "iterations": 700},
    )
    _write_json(run_dir / "summary.json", {"returncode": 0, "observation_only": True})
    _write_json(run_dir / "training_events_summary.json", {"invalid_training_event_rows": 0, "final_gaussian_count": 10})
    _write_json(
        run_dir / "gaussian_lifecycle_summary.json",
        {
            "invariant_violations": 0,
            "alive_final_count": 10,
            "dead_final_count": 0,
            "known_gaussian_count": 10,
            "final_gaussian_count": 10,
        },
    )
    _write_json(run_dir / "view_metrics_summary.json", {"condition": condition, "view_count_total": 1, "metrics": {"train": {"psnr_mean": psnr, "ssim_mean": 0.9}}})
    _write_csv(run_dir / "tables" / "training_events.csv", [{"iteration": 1, "visible_gaussian_count": 1, "visibility_ratio": 0.1}])
    _write_csv(run_dir / "tables" / "densification_events.csv", [{"iteration": 1}])
    _write_csv(run_dir / "tables" / "gaussian_count_timeseries.csv", [{"iteration": 1, "gaussian_count": 10}])
    _write_csv(run_dir / "tables" / "gaussian_lifecycle_events.csv", [{"event_type": "clone_birth"}])
    _write_csv(run_dir / "tables" / "gaussian_lifecycle_final.csv", [{"gaussian_id": 1, "alive": "true"}])
    _write_csv(
        run_dir / "tables" / "view_metrics.csv",
        [{"split": "train", "image_name": "train_000.png", "psnr": psnr, "ssim": 0.9, "l1_mean": 0.1}],
    )


def main() -> int:
    project_root = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory(prefix="viewtrust-corruption-link-") as tmp:
        root = Path(tmp)
        data_root = root / "data"
        condition_root = data_root / "viewtrust-mini" / "nerf_synthetic" / "chair" / "corrupt_occluder"
        _write_json(
            condition_root / "corruption_summary.json",
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
        _write_json(condition_root / "corruption_manifest.json", {"corruptions": []})
        _write_json(condition_root / "manifest.json", {"condition": "corrupt_occluder"})
        _write_csv(
            condition_root / "corruption_manifest.csv",
            [{"split": "train", "view_name": "train_000", "was_corrupted": "true", "corruption_type": "occluder"}],
        )
        clean_run = root / "clean"
        corrupt_run = root / "corrupt"
        _make_run(clean_run, condition="clean", psnr=30.0)
        _make_run(corrupt_run, condition="corrupt_occluder", psnr=25.0)
        output_dir = root / "report"
        completed = subprocess.run(
            [
                sys.executable,
                str(project_root / "scripts" / "measure" / "compare_clean_corrupt_observations.py"),
                "--clean-run-dir",
                str(clean_run),
                "--corrupt-run-dir",
                str(corrupt_run),
                "--data-root",
                str(data_root),
                "--scene",
                "chair",
                "--corruption-condition",
                "corrupt_occluder",
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
        summary = json.loads((output_dir / "clean_vs_corrupt_summary.json").read_text())
        if summary["corruption_summary_available"] is not True:
            raise ValueError("corruption summary was not linked")
        if summary["corruption_summary"]["corrupted_image_count"] != 1:
            raise ValueError("corruption summary content mismatch")
        effects_path = output_dir / "view_corruption_effects.csv"
        if not effects_path.exists():
            raise FileNotFoundError(effects_path)
        effects = list(csv.DictReader(effects_path.open(newline="", encoding="utf-8")))
        if effects[0]["was_corrupted"] != "true":
            raise ValueError("view corruption effect label mismatch")
        if float(effects[0]["psnr_delta"]) != -5.0:
            raise ValueError("view corruption effect delta mismatch")

    print("corruption manifest linking smoke test ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
