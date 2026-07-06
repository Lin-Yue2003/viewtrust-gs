#!/usr/bin/env python3
"""LOCAL-SAFE smoke test for PR10 natural corruption generation."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_png(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), color).save(path)


def _make_fake_clean(data_root: Path) -> Path:
    clean = data_root / "viewtrust-mini" / "nerf_synthetic" / "chair" / "clean"
    train_frames = []
    test_frames = []
    target_frames = []
    for idx in range(6):
        name = f"train_{idx:03d}"
        train_frames.append({"file_path": f"images/{name}", "transform_matrix": []})
        _write_png(clean / "images" / f"{name}.png", (20 + idx, 30, 40))
    for idx in range(2):
        name = f"test_{idx:03d}"
        test_frames.append({"file_path": f"images/{name}", "transform_matrix": []})
        _write_png(clean / "images" / f"{name}.png", (80 + idx, 90, 100))
    for idx in range(1):
        name = f"target_{idx:03d}"
        target_frames.append({"file_path": f"images/{name}", "transform_matrix": []})
        _write_png(clean / "images" / f"{name}.png", (120 + idx, 130, 140))
    _write_json(clean / "transforms_train.json", {"camera_angle_x": 0.7, "frames": train_frames})
    _write_json(clean / "transforms_test.json", {"camera_angle_x": 0.7, "frames": test_frames})
    _write_json(clean / "transforms_target.json", {"camera_angle_x": 0.7, "frames": target_frames})
    return clean


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def main() -> int:
    project_root = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory(prefix="viewtrust-corrupt-gen-") as tmp:
        data_root = Path(tmp) / "data"
        _make_fake_clean(data_root)
        script = project_root / "scripts" / "data" / "generate_natural_corruptions.py"

        dry = _run(
            [
                sys.executable,
                str(script),
                "--data-root",
                str(data_root),
                "--output-condition",
                "dry_occluder",
                "--corruption-type",
                "occluder",
                "--num-corrupt-train-views",
                "2",
                "--dry-run",
            ]
        )
        if dry.returncode != 0:
            raise RuntimeError(dry.stderr or dry.stdout)
        if (data_root / "viewtrust-mini" / "nerf_synthetic" / "chair" / "dry_occluder").exists():
            raise ValueError("dry-run should not write condition directory")

        for condition, corruption_type in (
            ("corrupt_occluder", "occluder"),
            ("corrupt_mixed", "mixed"),
        ):
            completed = _run(
                [
                    sys.executable,
                    str(script),
                    "--data-root",
                    str(data_root),
                    "--output-condition",
                    condition,
                    "--corruption-type",
                    corruption_type,
                    "--num-corrupt-train-views",
                    "2",
                    "--copy-mode",
                    "copy",
                    "--overwrite",
                ]
            )
            if completed.returncode != 0:
                raise RuntimeError(completed.stderr or completed.stdout)
            root = data_root / "viewtrust-mini" / "nerf_synthetic" / "chair" / condition
            for name in (
                "manifest.json",
                "corruption_manifest.json",
                "corruption_manifest.csv",
                "corruption_summary.json",
                "preview/preview_grid.png",
            ):
                if not (root / name).exists():
                    raise FileNotFoundError(root / name)
            training_manifest = json.loads((root / "manifest.json").read_text())
            if training_manifest["schema_name"] != "viewtrust.nerf_synthetic_subset.manifest":
                raise ValueError("training-compatible manifest schema mismatch")
            if training_manifest["condition"] != condition:
                raise ValueError("training-compatible manifest condition mismatch")
            if training_manifest["condition_type"] != "natural_corruption":
                raise ValueError("training-compatible manifest condition_type mismatch")
            summary = json.loads((root / "corruption_summary.json").read_text())
            if summary["corrupted_image_count"] != 2:
                raise ValueError("corrupted_image_count mismatch")
            if summary["uncorrupted_image_count"] != 7:
                raise ValueError("uncorrupted_image_count mismatch")
            if summary["transforms_extensionless_file_path"] is not True:
                raise ValueError("transforms should remain extensionless")
            manifest = json.loads((root / "corruption_manifest.json").read_text())
            if any(row["split"] in {"test", "target"} for row in manifest["corruptions"]):
                raise ValueError("test/target should remain uncorrupted by default")
            if corruption_type == "mixed":
                actual_types = {row["corruption_type"] for row in manifest["corruptions"]}
                if not actual_types.issubset({"occluder", "blur", "exposure", "color_shift", "noise"}):
                    raise ValueError("mixed condition recorded unsupported actual corruption")

    print("natural corruption generation smoke test ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
