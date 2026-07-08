#!/usr/bin/env python3
"""LOCAL-SAFE smoke test for the clean chair baseline wrapper."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path


def _bootstrap_project_imports() -> Path:
    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root))
    return project_root


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    _bootstrap_project_imports()

    from viewtrust.training.baseline import (
        build_gaussian_splatting_command,
        resolve_prepared_scene_root,
        resolve_trainer_path,
        validate_prepared_scene,
    )

    with tempfile.TemporaryDirectory(prefix="viewtrust-training-wrapper-") as tmp:
        tmp_root = Path(tmp)
        data_root = tmp_root / "data"
        third_party_root = tmp_root / "third_party"
        output_root = tmp_root / "outputs"
        prepared_scene_root = (
            data_root / "viewtrust-mini" / "nerf_synthetic" / "chair" / "clean"
        )

        _write_json(prepared_scene_root / "transforms_train.json", {"frames": []})
        _write_json(prepared_scene_root / "transforms_test.json", {"frames": []})
        _write_json(prepared_scene_root / "manifest.json", {"image_count": 1})
        (prepared_scene_root / "images").mkdir(parents=True)
        (prepared_scene_root / "images" / "mock.png").write_bytes(b"mock")

        trainer_path = third_party_root / "gaussian-splatting" / "train.py"
        trainer_path.parent.mkdir(parents=True)
        trainer_path.write_text(
            "raise SystemExit('fake trainer should not execute in dry-run smoke test')\n",
            encoding="utf-8",
        )

        resolved_scene_root = resolve_prepared_scene_root(data_root, "chair", "clean")
        if resolved_scene_root != prepared_scene_root.resolve():
            raise ValueError("prepared scene root resolution mismatch")
        validate_prepared_scene(resolved_scene_root)
        resolved_trainer_path = resolve_trainer_path("gaussian-splatting", third_party_root)

        trainer_output_dir = (
            output_root
            / "baseline"
            / "chair_clean_gaussian_splatting"
            / "dry-run"
            / "trainer_output"
        )
        command = build_gaussian_splatting_command(
            trainer_path=resolved_trainer_path,
            prepared_scene_root=resolved_scene_root,
            trainer_output_dir=trainer_output_dir,
            iterations=500,
        )
        command_text = " ".join(command)
        if "train.py" not in command_text:
            raise ValueError("command does not contain train.py")
        if "-s" not in command or str(resolved_scene_root) not in command:
            raise ValueError("command does not contain prepared scene -s argument")
        if "-m" not in command or str(trainer_output_dir) not in command:
            raise ValueError("command does not contain trainer output -m argument")
        if "--iterations" not in command or "500" not in command:
            raise ValueError("command does not contain --iterations 500")
        if "--eval" not in command:
            raise ValueError("default command should include official 3DGS --eval")

        non_eval_command = build_gaussian_splatting_command(
            trainer_path=resolved_trainer_path,
            prepared_scene_root=resolved_scene_root,
            trainer_output_dir=trainer_output_dir,
            iterations=500,
            eval_split=False,
        )
        if "--eval" in non_eval_command:
            raise ValueError("eval_split=False command should not include --eval")

    print("training wrapper dry-run smoke test ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
