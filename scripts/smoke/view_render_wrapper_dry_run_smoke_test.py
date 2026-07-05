#!/usr/bin/env python3
"""LOCAL-SAFE smoke test for PR6 render wrapper dry-run planning."""

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


def _write_json(path: Path, data: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    project_root = _bootstrap_project_imports()

    with tempfile.TemporaryDirectory(prefix="viewtrust-render-dry-run-") as tmp:
        tmp_root = Path(tmp)
        run_dir = tmp_root / "run"
        data_root = tmp_root / "data"
        third_party_root = tmp_root / "third_party"
        prepared_scene = data_root / "viewtrust-mini" / "nerf_synthetic" / "chair" / "clean"
        render_script = third_party_root / "gaussian-splatting" / "render.py"
        point_cloud = run_dir / "trainer_output" / "point_cloud" / "iteration_500" / "point_cloud.ply"

        _write_json(run_dir / "summary.json", {"summary": {"returncode": 0}})
        _write_json(run_dir / "metadata.json", {"label": "mock"})
        _write_json(run_dir / "config_snapshot.json", {"mock": True})
        (run_dir / "stdout.log").parent.mkdir(parents=True, exist_ok=True)
        (run_dir / "stdout.log").write_text("", encoding="utf-8")
        (run_dir / "stderr.log").write_text("", encoding="utf-8")
        (run_dir / "tables").mkdir(parents=True)
        (run_dir / "tables" / "command_summary.csv").write_text(
            "label,returncode,elapsed_s\nmock,0,1.0\n", encoding="utf-8"
        )
        (run_dir / "tables" / "gpu_memory_samples.csv").write_text(
            "elapsed_s,gpu_index,gpu_name,memory_used_mb,memory_total_mb,utilization_gpu_percent\n",
            encoding="utf-8",
        )
        point_cloud.parent.mkdir(parents=True)
        point_cloud.write_text("ply\nend_header\n", encoding="ascii")
        (run_dir / "trainer_output" / "cfg_args").write_text("mock config\n", encoding="utf-8")

        for name in ("transforms_train.json", "transforms_test.json", "transforms_target.json"):
            _write_json(prepared_scene / name, {"frames": []})
        (prepared_scene / "images").mkdir(parents=True)
        render_script.parent.mkdir(parents=True)
        render_script.write_text("print('fake render')\n", encoding="utf-8")

        completed = subprocess.run(
            [
                sys.executable,
                str(project_root / "scripts" / "evaluate" / "render_clean_views.py"),
                "--run-dir",
                str(run_dir),
                "--data-root",
                str(data_root),
                "--third-party-root",
                str(third_party_root),
                "--trainer",
                "gaussian-splatting",
                "--scene",
                "chair",
                "--condition",
                "clean",
                "--iteration",
                "500",
                "--splits",
                "train",
                "test",
                "target",
                "--gpu",
                "0",
                "--dry-run",
            ],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if completed.returncode != 0:
            raise RuntimeError(completed.stderr or completed.stdout)

        summary = json.loads(completed.stdout)
        if summary["executed"] is not False:
            raise ValueError("dry-run should not execute render commands")
        if summary["preflight"]["render_script"] != str(render_script.resolve()):
            raise ValueError("official render.py path was not resolved")
        command_names = {command["name"] for command in summary["commands"]}
        if command_names != {"train_test", "target"}:
            raise ValueError(f"unexpected render command plan: {command_names}")
        target_command = next(command for command in summary["commands"] if command["name"] == "target")
        if "target_as_test" not in target_command["source_path"]:
            raise ValueError("target eval scene was not planned")
        if (run_dir / "view_evaluation" / "render_models").exists():
            raise ValueError("dry-run should not create render model directories")

    print("view render wrapper dry-run smoke test ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
