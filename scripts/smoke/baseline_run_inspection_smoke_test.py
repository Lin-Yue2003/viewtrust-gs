#!/usr/bin/env python3
"""LOCAL-SAFE smoke test for baseline run inspection."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


def _bootstrap_project_imports() -> Path:
    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root))
    return project_root


def main() -> int:
    project_root = _bootstrap_project_imports()

    from scripts.measure.inspect_baseline_run import inspect_baseline_run

    with tempfile.TemporaryDirectory(prefix="viewtrust-baseline-inspect-") as tmp:
        run_dir = Path(tmp) / "outputs" / "baseline" / "chair_clean_gaussian_splatting" / "mock-run"
        (run_dir / "tables").mkdir(parents=True)
        (run_dir / "trainer_output" / "point_cloud").mkdir(parents=True)

        (run_dir / "summary.json").write_text(
            json.dumps(
                {
                    "summary": {
                        "label": "chair_clean_gaussian_splatting",
                        "returncode": 0,
                        "elapsed_s": 16.850461,
                        "gpu_sample_count": 2,
                        "observation_only": True,
                    }
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        (run_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "label": "chair_clean_gaussian_splatting",
                    "command": ["python", "train.py", "--iterations", "500"],
                    "training_behavior_modified": False,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        (run_dir / "stdout.log").write_text(
            "Training progress: iteration 500 complete\n", encoding="utf-8"
        )
        (run_dir / "stderr.log").write_text("", encoding="utf-8")
        (run_dir / "tables" / "command_summary.csv").write_text(
            "label,returncode,elapsed_s\nchair_clean_gaussian_splatting,0,16.850461\n",
            encoding="utf-8",
        )
        (run_dir / "tables" / "gpu_memory_samples.csv").write_text(
            "elapsed_s,gpu_index,gpu_name,memory_used_mb,memory_total_mb,utilization_gpu_percent\n"
            "0.0,0,Mock GPU,100,1000,10\n"
            "1.0,0,Mock GPU,200,1000,20\n",
            encoding="utf-8",
        )
        (run_dir / "trainer_output" / "point_cloud" / "iteration_500.ply").write_text(
            "mock\n", encoding="utf-8"
        )

        report = inspect_baseline_run(run_dir)
        if report["missing_required_paths"]:
            raise ValueError(report["missing_required_paths"])
        if report["returncode"] != 0:
            raise ValueError("returncode mismatch")
        if report["elapsed_s"] != 16.850461:
            raise ValueError("elapsed_s mismatch")
        if not report["has_stdout"] or not report["has_stderr"]:
            raise ValueError("stdout/stderr detection failed")
        if not report["has_gpu_samples"] or report["gpu_sample_count"] != 2:
            raise ValueError("gpu sample detection failed")
        if not report["trainer_output_exists"] or report["trainer_output_file_count"] != 1:
            raise ValueError("trainer output detection failed")
        if report["detected_iterations"] != [500]:
            raise ValueError("iteration detection failed")
        if report["observation_only"] is not True:
            raise ValueError("observation_only mismatch")

        completed = subprocess.run(
            [
                sys.executable,
                str(project_root / "scripts" / "measure" / "inspect_baseline_run.py"),
                "--run-dir",
                str(run_dir),
                "--require-success",
            ],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if completed.returncode != 0:
            raise RuntimeError(completed.stderr or completed.stdout)

    print("baseline run inspection smoke test ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
