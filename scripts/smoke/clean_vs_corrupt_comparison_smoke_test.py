#!/usr/bin/env python3
"""LOCAL-SAFE smoke test for PR11 clean-vs-corrupt comparison."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.smoke.natural_corruption_generation_smoke_test import _make_fake_clean


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
    return subprocess.run(
        command,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _make_observed_run(
    run_dir: Path,
    *,
    condition: str,
    prepared_scene_root: Path | None,
    final_count: int,
    birth_count: int,
    invalid_training_rows: int = 0,
    include_view_metrics: bool = True,
) -> None:
    _write_json(
        run_dir / "metadata.json",
        {
            "run_id": run_dir.name,
            "scene": "chair",
            "condition": condition,
            "trainer": "gaussian-splatting",
            "iterations": 700,
            "prepared_scene_root": str(prepared_scene_root) if prepared_scene_root else "",
        },
    )
    _write_json(
        run_dir / "summary.json",
        {
            "returncode": 0,
            "elapsed_s": 100.0 if condition == "clean" else 112.5,
            "observation_only": True,
        },
    )
    _write_json(
        run_dir / "training_events_summary.json",
        {
            "scene": "chair",
            "condition": condition,
            "trainer": "gaussian-splatting",
            "requested_iterations": 700,
            "training_event_rows": 70,
            "invalid_training_event_rows": invalid_training_rows,
            "densification_event_rows": 2,
            "densification_trigger_count": 2,
            "opacity_reset_count": 1,
            "initial_gaussian_count": 100000,
            "final_gaussian_count": final_count,
            "max_visible_gaussian_count": 50000,
            "max_visibility_ratio": 0.5,
        },
    )
    _write_json(
        run_dir / "gaussian_lifecycle_summary.json",
        {
            "scene": "chair",
            "condition": condition,
            "trainer": "gaussian-splatting",
            "final_gaussian_count": final_count,
            "known_gaussian_count": final_count + 100,
            "birth_event_count": birth_count,
            "clone_birth_count": birth_count // 2,
            "split_birth_count": birth_count - birth_count // 2,
            "densification_birth_count": birth_count,
            "prune_death_count": 100,
            "alive_final_count": final_count,
            "dead_final_count": 100,
            "final_lifecycle_rows": final_count + 100,
            "lifecycle_event_rows": birth_count + 100,
            "invariant_violations": 0,
        },
    )
    _write_csv(
        run_dir / "tables" / "training_events.csv",
        [
            {
                "iteration": 10,
                "gaussian_count": 100000,
                "visible_gaussian_count": 40000,
                "visibility_ratio": 0.4,
            }
        ],
    )
    _write_csv(run_dir / "tables" / "densification_events.csv", [{"iteration": 500, "triggered": "true"}])
    _write_csv(
        run_dir / "tables" / "gaussian_count_timeseries.csv",
        [{"iteration": 700, "gaussian_count": final_count}],
    )
    _write_csv(
        run_dir / "tables" / "gaussian_lifecycle_events.csv",
        [{"event_type": "clone_birth", "iteration": 500}],
    )
    _write_csv(
        run_dir / "tables" / "gaussian_lifecycle_final.csv",
        [{"lifecycle_id": 1, "status": "alive", "final_index": 0}],
    )
    _write_csv(
        run_dir / "tables" / "gpu_memory_samples.csv",
        [{"sample_index": 0, "memory_used_mb": 1234 if condition == "clean" else 1300}],
    )
    if include_view_metrics:
        _write_json(
            run_dir / "view_metrics_summary.json",
            {
                "scene": "chair",
                "condition": condition,
                "view_count_total": 3,
                "view_count_by_split": {"train": 1, "test": 1, "target": 1},
                "metrics": {
                    "train": {"psnr_mean": 30.0, "ssim_mean": 0.9},
                    "test": {"psnr_mean": 28.0, "ssim_mean": 0.88},
                    "target": {"psnr_mean": 27.0, "ssim_mean": 0.86},
                },
            },
        )
        _write_csv(
            run_dir / "tables" / "view_metrics.csv",
            [
                {
                    "split": "train",
                    "image_name": "00000.png",
                    "psnr": 30.0,
                    "ssim": 0.9,
                    "lpips": "",
                }
            ],
        )
        _write_csv(
            run_dir / "tables" / "view_render_artifacts.csv",
            [{"split": "train", "kind": "render", "relative_path": "fake.png"}],
        )


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="viewtrust-clean-corrupt-") as tmp:
        root = Path(tmp)
        data_root = root / "data"
        _make_fake_clean(data_root)
        generate = _run(
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "data" / "generate_natural_corruptions.py"),
                "--data-root",
                str(data_root),
                "--output-condition",
                "corrupt_occluder",
                "--corruption-type",
                "occluder",
                "--num-corrupt-train-views",
                "2",
                "--copy-mode",
                "copy",
                "--overwrite",
            ]
        )
        if generate.returncode != 0:
            raise RuntimeError(generate.stderr or generate.stdout)
        condition_root = (
            data_root / "viewtrust-mini" / "nerf_synthetic" / "chair" / "corrupt_occluder"
        )
        if not (condition_root / "manifest.json").exists():
            raise FileNotFoundError(condition_root / "manifest.json")
        inspect = _run(
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "measure" / "inspect_natural_corruption_dataset.py"),
                "--data-root",
                str(data_root),
                "--condition",
                "corrupt_occluder",
                "--require-valid",
                "--require-corrupted-count",
                "2",
            ]
        )
        if inspect.returncode != 0:
            raise RuntimeError(inspect.stderr or inspect.stdout)
        inspect_report = json.loads(inspect.stdout)
        if inspect_report["has_manifest"] is not True:
            raise ValueError("PR10.1 manifest was not inspected")

        clean_run = root / "runs" / "clean"
        corrupt_run = root / "runs" / "corrupt"
        _make_observed_run(
            clean_run,
            condition="clean",
            prepared_scene_root=None,
            final_count=92979,
            birth_count=1437,
        )
        _make_observed_run(
            corrupt_run,
            condition="corrupt_occluder",
            prepared_scene_root=condition_root,
            final_count=93012,
            birth_count=1600,
        )
        output_dir = root / "report"
        compare = _run(
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "measure" / "compare_clean_corrupt_observations.py"),
                "--clean-run-dir",
                str(clean_run),
                "--corrupt-run-dir",
                str(corrupt_run),
                "--corruption-condition",
                "corrupt_occluder",
                "--output-dir",
                str(output_dir),
                "--require-observation-invariants",
                "--write-markdown",
            ]
        )
        if compare.returncode != 0:
            raise RuntimeError(compare.stderr or compare.stdout)
        for name in (
            "clean_vs_corrupt_summary.json",
            "clean_vs_corrupt_report.md",
            "clean_vs_corrupt_metrics.csv",
            "clean_vs_corrupt_artifact_manifest.csv",
        ):
            if not (output_dir / name).exists():
                raise FileNotFoundError(output_dir / name)
        summary = json.loads((output_dir / "clean_vs_corrupt_summary.json").read_text())
        if summary["clean_success"] is not True or summary["corrupt_success"] is not True:
            raise ValueError("success flags mismatch")
        if summary["final_gaussian_count_delta"] != 33:
            raise ValueError("final Gaussian count delta mismatch")
        if summary["view_metrics_available"] is not True:
            raise ValueError("view metrics should be available")

        no_view_run = root / "runs" / "corrupt-no-view"
        _make_observed_run(
            no_view_run,
            condition="corrupt_occluder",
            prepared_scene_root=condition_root,
            final_count=93012,
            birth_count=1600,
            include_view_metrics=False,
        )
        no_view_output = root / "report-no-view"
        no_view = _run(
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "measure" / "compare_clean_corrupt_observations.py"),
                "--clean-run-dir",
                str(clean_run),
                "--corrupt-run-dir",
                str(no_view_run),
                "--corruption-condition",
                "corrupt_occluder",
                "--output-dir",
                str(no_view_output),
            ]
        )
        if no_view.returncode != 0:
            raise RuntimeError(no_view.stderr or no_view.stdout)
        no_view_summary = json.loads((no_view_output / "clean_vs_corrupt_summary.json").read_text())
        if no_view_summary["view_metrics_available"] is not False:
            raise ValueError("missing view metrics should be non-fatal")
        if "view metrics unavailable for one or both runs" not in no_view_summary["warnings"]:
            raise ValueError("missing view metrics warning unavailable")

        bad_run = root / "runs" / "bad-corrupt"
        _make_observed_run(
            bad_run,
            condition="corrupt_occluder",
            prepared_scene_root=condition_root,
            final_count=93012,
            birth_count=1600,
            invalid_training_rows=1,
        )
        bad = _run(
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "measure" / "compare_clean_corrupt_observations.py"),
                "--clean-run-dir",
                str(clean_run),
                "--corrupt-run-dir",
                str(bad_run),
                "--corruption-condition",
                "corrupt_occluder",
                "--output-dir",
                str(root / "report-bad"),
                "--require-observation-invariants",
            ]
        )
        if bad.returncode == 0:
            raise ValueError("invariant failure should fail under --require-observation-invariants")

    print("clean vs corrupt comparison smoke test ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
