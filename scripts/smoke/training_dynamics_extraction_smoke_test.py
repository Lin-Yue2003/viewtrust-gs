#!/usr/bin/env python3
"""LOCAL-SAFE smoke test for post-hoc training dynamics extraction."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def _bootstrap_project_imports() -> Path:
    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root))
    return project_root


def _write_json(path: Path, data: dict[str, object]) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    project_root = _bootstrap_project_imports()

    with tempfile.TemporaryDirectory(prefix="viewtrust-training-dynamics-") as tmp:
        run_dir = Path(tmp) / "run"
        tables_dir = run_dir / "tables"
        trainer_output = run_dir / "trainer_output"
        point_cloud_dir = trainer_output / "point_cloud" / "iteration_500"
        tables_dir.mkdir(parents=True)
        point_cloud_dir.mkdir(parents=True)

        _write_json(
            run_dir / "summary.json",
            {
                "summary": {
                    "returncode": 0,
                    "elapsed_s": 16.850461,
                    "gpu_sample_count": 2,
                    "observation_only": True,
                }
            },
        )
        _write_json(
            run_dir / "metadata.json",
            {
                "label": "chair_clean_gaussian_splatting",
                "command": ["python", "train.py", "--iterations", "500"],
                "training_behavior_modified": False,
            },
        )
        _write_json(run_dir / "config_snapshot.json", {"experiment": {"iterations": 500}})
        (run_dir / "stdout.log").write_text("iteration 500 complete\n", encoding="utf-8")
        (run_dir / "stderr.log").write_text("", encoding="utf-8")
        (tables_dir / "command_summary.csv").write_text(
            "label,returncode,elapsed_s\nchair_clean_gaussian_splatting,0,16.850461\n",
            encoding="utf-8",
        )
        (tables_dir / "gpu_memory_samples.csv").write_text(
            "elapsed_s,gpu_index,gpu_name,memory_used_mb,memory_total_mb,utilization_gpu_percent\n"
            "0.0,0,Mock GPU,100,1000,10\n"
            "1.0,0,Mock GPU,200,1000,20\n",
            encoding="utf-8",
        )
        (trainer_output / "cfg_args").write_text("mock config\n", encoding="utf-8")
        (point_cloud_dir / "point_cloud.ply").write_text(
            "\n".join(
                [
                    "ply",
                    "format ascii 1.0",
                    "element vertex 3",
                    "property float x",
                    "property float y",
                    "property float z",
                    "end_header",
                    "0 0 0",
                    "1 0 0",
                    "0 1 0",
                    "",
                ]
            ),
            encoding="ascii",
        )

        completed = subprocess.run(
            [
                sys.executable,
                str(project_root / "scripts" / "measure" / "extract_training_dynamics.py"),
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

        summary_path = run_dir / "training_dynamics_summary.json"
        dynamics_path = tables_dir / "training_dynamics.csv"
        artifacts_path = tables_dir / "training_artifacts.csv"
        final_gaussian_path = tables_dir / "final_gaussian_summary.csv"
        for path in (summary_path, dynamics_path, artifacts_path, final_gaussian_path):
            if not path.exists():
                raise FileNotFoundError(path)

        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary["final_gaussian_count"] != 3:
            raise ValueError("final_gaussian_count mismatch")
        if summary["returncode"] != 0:
            raise ValueError("returncode mismatch")
        if summary["observation_only"] is not True:
            raise ValueError("observation_only mismatch")

        dynamics_text = dynamics_path.read_text(encoding="utf-8")
        if not dynamics_text.startswith("run_id,source,iteration,tag,value,wall_time"):
            raise ValueError("training_dynamics.csv header mismatch")

        final_rows = _read_csv_rows(final_gaussian_path)
        if final_rows[0]["gaussian_count"] != "3":
            raise ValueError("final_gaussian_summary.csv gaussian_count mismatch")

    print("training dynamics extraction smoke test ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
