#!/usr/bin/env python3
"""LOCAL-SAFE smoke test for PR12.1 split-correct training command defaults."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _parse_dry_run_report(stdout: str) -> dict[str, object]:
    end = stdout.rfind("}\n")
    if end == -1:
        raise ValueError(f"dry-run stdout did not contain JSON report:\n{stdout}")
    return json.loads(stdout[: end + 1])


def main() -> int:
    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root))

    from viewtrust.training.baseline import build_gaussian_splatting_command

    trainer_path = project_root / "third_party" / "gaussian-splatting" / "train.py"
    scene_root = project_root / "data" / "viewtrust-mini" / "nerf_synthetic" / "chair" / "clean"
    output_dir = project_root / "outputs" / "dry-run"

    default_command = build_gaussian_splatting_command(
        trainer_path=trainer_path,
        prepared_scene_root=scene_root,
        trainer_output_dir=output_dir,
        iterations=700,
    )
    if "--eval" not in default_command:
        raise ValueError("default Gaussian Splatting command must include --eval")

    eval_command = build_gaussian_splatting_command(
        trainer_path=trainer_path,
        prepared_scene_root=scene_root,
        trainer_output_dir=output_dir,
        iterations=700,
        eval_split=True,
    )
    if "--eval" not in eval_command:
        raise ValueError("eval_split=True command must include --eval")

    non_eval_command = build_gaussian_splatting_command(
        trainer_path=trainer_path,
        prepared_scene_root=scene_root,
        trainer_output_dir=output_dir,
        iterations=700,
        eval_split=False,
    )
    if "--eval" in non_eval_command:
        raise ValueError("eval_split=False command must not include --eval")

    with tempfile.TemporaryDirectory(prefix="viewtrust-split-protocol-") as tmp:
        root = Path(tmp)
        data_root = root / "data"
        third_party_root = root / "third_party"
        output_root = root / "outputs"
        scene_root = data_root / "viewtrust-mini" / "nerf_synthetic" / "chair" / "clean"
        _write_json(scene_root / "transforms_train.json", {"frames": []})
        _write_json(scene_root / "transforms_test.json", {"frames": []})
        _write_json(scene_root / "manifest.json", {"image_count": 1})
        (scene_root / "images").mkdir(parents=True)
        (scene_root / "images" / "mock.png").write_bytes(b"mock")

        fake_trainer = third_party_root / "gaussian-splatting" / "train.py"
        fake_trainer.parent.mkdir(parents=True)
        fake_trainer.write_text("raise SystemExit('dry-run should not execute trainer')\n", encoding="utf-8")

        base_command = [
            sys.executable,
            str(project_root / "scripts" / "train" / "run_clean_chair_baseline.py"),
            "--trainer",
            "gaussian-splatting",
            "--data-root",
            str(data_root),
            "--third-party-root",
            str(third_party_root),
            "--output-root",
            str(output_root),
            "--scene",
            "chair",
            "--condition",
            "clean",
            "--iterations",
            "700",
            "--gpu",
            "0",
            "--sample-interval-s",
            "1.0",
            "--dry-run",
        ]
        default_dry_run = subprocess.run(
            base_command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if default_dry_run.returncode != 0:
            raise RuntimeError(default_dry_run.stderr or default_dry_run.stdout)
        default_report = _parse_dry_run_report(default_dry_run.stdout)
        if default_report["eval_split"] is not True:
            raise ValueError("dry-run default eval_split should be true")
        if "--eval" not in default_report["command"]:
            raise ValueError("dry-run default command should include --eval")
        if default_report["training_split_protocol"] != "official_3dgs_eval_mode_train_only":
            raise ValueError("dry-run default split protocol mismatch")

        non_eval_dry_run = subprocess.run(
            [*base_command, "--no-eval-split"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if non_eval_dry_run.returncode != 0:
            raise RuntimeError(non_eval_dry_run.stderr or non_eval_dry_run.stdout)
        non_eval_report = _parse_dry_run_report(non_eval_dry_run.stdout)
        if non_eval_report["eval_split"] is not False:
            raise ValueError("dry-run --no-eval-split should report eval_split=false")
        if "--eval" in non_eval_report["command"]:
            raise ValueError("dry-run --no-eval-split command should not include --eval")
        if (
            non_eval_report["training_split_protocol"]
            != "official_3dgs_non_eval_mode_may_merge_test"
        ):
            raise ValueError("dry-run non-eval split protocol mismatch")

    print("training split protocol smoke test ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
