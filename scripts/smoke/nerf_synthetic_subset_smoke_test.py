#!/usr/bin/env python3
"""LOCAL-SAFE smoke test for NeRF Synthetic mini subset preparation."""

from __future__ import annotations

import json
import struct
import sys
import tempfile
import zlib
from pathlib import Path


def _bootstrap_project_imports() -> Path:
    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root))
    return project_root


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + kind
        + data
        + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    )


def _write_png(path: Path, width: int, height: int, rgb: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = b"\x00" + bytes(rgb) * width
    raw = row * height
    payload = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", payload)
        + _png_chunk(b"IDAT", zlib.compress(raw))
        + _png_chunk(b"IEND", b"")
    )


def _frame(split: str, index: int) -> dict[str, object]:
    return {
        "file_path": f"{split}/r_{index:03d}",
        "transform_matrix": [
            [1.0, 0.0, 0.0, float(index)],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        "mock_extra_field": f"{split}-{index}",
    }


def _write_transforms(path: Path, split: str, count: int) -> None:
    document = {
        "camera_angle_x": 0.6911112070083618,
        "fl_x": 1111.0,
        "fl_y": 1111.0,
        "cx": 4.0,
        "cy": 4.0,
        "w": 8,
        "h": 8,
        "aabb_scale": 4,
        "frames": [_frame(split, index) for index in range(count)],
    }
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")


def _contains_absolute_server_path(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    return (
        '"raw_scene_root": "/trainingData' in text
        or '"output_scene_root": "/trainingData' in text
        or '"file_path": "/' in text
    )


def main() -> int:
    project_root = _bootstrap_project_imports()
    _ = project_root

    from viewtrust.datasets.nerf_synthetic import prepare_nerf_synthetic_subset

    with tempfile.TemporaryDirectory(prefix="viewtrust-nerf-synthetic-") as tmp:
        tmp_root = Path(tmp)
        raw_scene_root = tmp_root / "raw" / "nerf_synthetic" / "chair"
        output_root = tmp_root / "viewtrust-mini" / "nerf_synthetic" / "chair"

        for split, count in (("train", 6), ("test", 4)):
            for index in range(count):
                _write_png(
                    raw_scene_root / split / f"r_{index:03d}.png",
                    width=8,
                    height=8,
                    rgb=(index * 20 % 255, 40, 90),
                )
            _write_transforms(raw_scene_root / f"transforms_{split}.json", split, count)

        plan = prepare_nerf_synthetic_subset(
            raw_scene_root=raw_scene_root,
            output_root=output_root,
            scene="chair",
            condition="clean",
            max_train_views=3,
            max_test_views=2,
            max_target_views=1,
            max_image_width=8,
            copy_mode="copy",
            seed=0,
            dry_run=False,
            overwrite=True,
        )

        clean_root = output_root / "clean"
        required = [
            clean_root / "manifest.json",
            clean_root / "transforms_train.json",
            clean_root / "transforms_test.json",
            clean_root / "transforms_target.json",
            clean_root / "README.md",
        ]
        for path in required:
            if not path.exists():
                raise FileNotFoundError(path)

        images = sorted((clean_root / "images").glob("*.png"))
        if len(images) != 6:
            raise ValueError(f"expected 6 prepared images, found {len(images)}")

        manifest = json.loads((clean_root / "manifest.json").read_text(encoding="utf-8"))
        if len(manifest["selected_train_frames"]) != 3:
            raise ValueError("selected train count mismatch")
        if len(manifest["selected_test_frames"]) != 2:
            raise ValueError("selected test count mismatch")
        if len(manifest["selected_target_frames"]) != 1:
            raise ValueError("selected target count mismatch")
        if manifest["image_count"] != plan.image_count:
            raise ValueError("manifest image_count mismatch")

        for transform_name in (
            "transforms_train.json",
            "transforms_test.json",
            "transforms_target.json",
        ):
            transform_path = clean_root / transform_name
            transform = json.loads(transform_path.read_text(encoding="utf-8"))
            if "camera_angle_x" not in transform or "frames" not in transform:
                raise ValueError(f"{transform_name} did not preserve camera fields")
            for frame in transform["frames"]:
                file_path = Path(frame["file_path"])
                if file_path.is_absolute():
                    raise ValueError(f"{transform_name} contains absolute file_path")
                if not str(file_path).startswith("images/"):
                    raise ValueError(f"{transform_name} file_path is not under images/")
            if _contains_absolute_server_path(transform_path):
                raise ValueError(f"{transform_name} contains forbidden absolute path")

        if _contains_absolute_server_path(clean_root / "manifest.json"):
            raise ValueError("manifest contains forbidden absolute server/local path")

    print("nerf synthetic subset smoke test ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
