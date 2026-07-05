"""NeRF Synthetic subset preparation helpers."""

from __future__ import annotations

import json
import os
import shutil
import struct
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

NERF_SYNTHETIC_SUBSET_SCHEMA = "viewtrust.datasets.nerf_synthetic_subset"
NERF_SYNTHETIC_SUBSET_VERSION = 1
SUPPORTED_COPY_MODES = {"symlink", "hardlink", "copy"}
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")


@dataclass(frozen=True)
class PreparedFrame:
    index: int
    split: str
    source_file_path: str
    source_image_path: Path
    output_file_path: str
    output_image_path: Path
    resized: bool

    def as_manifest_entry(self, raw_scene_root: Path) -> dict[str, object]:
        try:
            source_image_relative_path = self.source_image_path.relative_to(
                raw_scene_root
            ).as_posix()
        except ValueError:
            source_image_relative_path = Path(self.source_file_path).as_posix()

        return {
            "index": self.index,
            "split": self.split,
            "source_file_path": self.source_file_path,
            "source_image_relative_path": source_image_relative_path,
            "output_file_path": self.output_file_path,
            "resized": self.resized,
        }


@dataclass(frozen=True)
class NerfSyntheticSubsetPlan:
    data_root: Path
    raw_scene_root: Path
    output_condition_root: Path
    scene: str
    condition: str
    max_train_views: int
    max_test_views: int
    max_target_views: int
    max_image_width: int | None
    copy_mode: str
    seed: int
    train_frames: tuple[PreparedFrame, ...]
    test_frames: tuple[PreparedFrame, ...]
    target_frames: tuple[PreparedFrame, ...]
    will_resize: bool

    @property
    def image_count(self) -> int:
        return len(
            {
                frame.output_file_path
                for frame in (*self.train_frames, *self.test_frames, *self.target_frames)
            }
        )


def _load_transforms(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"missing transforms file: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    frames = data.get("frames")
    if not isinstance(frames, list):
        raise ValueError(f"{path} does not contain a frames list")
    return data


def _uniform_indices(count: int, maximum: int) -> list[int]:
    if maximum < 0:
        raise ValueError("view maximums must be non-negative")
    if count <= maximum:
        return list(range(count))
    if maximum == 0:
        return []
    if maximum == 1:
        return [0]
    return [round(i * (count - 1) / (maximum - 1)) for i in range(maximum)]


def _resolve_image_path(raw_scene_root: Path, frame_file_path: str) -> Path:
    raw_path = Path(frame_file_path)
    candidates: list[Path] = []

    if raw_path.is_absolute():
        candidates.append(raw_path)
    else:
        normalized = str(frame_file_path).removeprefix("./")
        candidates.append(raw_scene_root / normalized)

    expanded: list[Path] = []
    for candidate in candidates:
        expanded.append(candidate)
        if candidate.suffix:
            continue
        expanded.extend(candidate.with_suffix(extension) for extension in IMAGE_EXTENSIONS)

    for candidate in expanded:
        if candidate.exists():
            return candidate.resolve()

    raise FileNotFoundError(
        f"could not resolve image path for frame file_path={frame_file_path!r} under {raw_scene_root}"
    )


def _frame_output_name(split: str, output_index: int, source_path: Path) -> str:
    suffix = source_path.suffix or ".png"
    return f"{split}_{output_index:03d}{suffix.lower()}"


def _selected_frames(
    *,
    raw_scene_root: Path,
    output_condition_root: Path,
    split: str,
    frames: list[dict[str, Any]],
    maximum: int,
    max_image_width: int | None,
    copy_mode: str,
) -> tuple[PreparedFrame, ...]:
    indices = _uniform_indices(len(frames), maximum)
    selected: list[PreparedFrame] = []
    for output_index, frame_index in enumerate(indices):
        frame = frames[frame_index]
        source_file_path = str(frame.get("file_path", ""))
        if not source_file_path:
            raise ValueError(f"{split} frame {frame_index} is missing file_path")
        source_image_path = _resolve_image_path(raw_scene_root, source_file_path)
        output_name = _frame_output_name(split, output_index, source_image_path)
        output_file_path = f"images/{output_name}"
        selected.append(
            PreparedFrame(
                index=frame_index,
                split=split,
                source_file_path=source_file_path,
                source_image_path=source_image_path,
                output_file_path=output_file_path,
                output_image_path=output_condition_root / output_file_path,
                resized=_will_resize(source_image_path, max_image_width)
                if copy_mode in SUPPORTED_COPY_MODES
                else False,
            )
        )
    return tuple(selected)


def _will_resize(source_image_path: Path, max_image_width: int | None) -> bool:
    if max_image_width is None:
        return False
    width, _height = _image_size(source_image_path)
    return width > max_image_width


def _image_size(source_image_path: Path) -> tuple[int, int]:
    if source_image_path.suffix.lower() == ".png":
        with source_image_path.open("rb") as handle:
            header = handle.read(24)
        if header.startswith(b"\x89PNG\r\n\x1a\n") and header[12:16] == b"IHDR":
            return struct.unpack(">II", header[16:24])

    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "Pillow is required to inspect non-PNG image size when max_image_width is set. "
            "Install Pillow or run with --max-image-width 0."
        ) from exc

    with Image.open(source_image_path) as image:
        return image.size


def _copy_or_link_image(
    source_image_path: Path,
    output_image_path: Path,
    *,
    copy_mode: str,
    max_image_width: int | None,
) -> bool:
    output_image_path.parent.mkdir(parents=True, exist_ok=True)

    if _will_resize(source_image_path, max_image_width):
        _resize_image(source_image_path, output_image_path, max_image_width)
        return True

    if copy_mode == "symlink":
        os.symlink(source_image_path, output_image_path)
    elif copy_mode == "hardlink":
        os.link(source_image_path, output_image_path)
    elif copy_mode == "copy":
        shutil.copy2(source_image_path, output_image_path)
    else:
        raise ValueError(f"unsupported copy_mode: {copy_mode}")
    return False


def _resize_image(source_image_path: Path, output_image_path: Path, max_image_width: int | None) -> None:
    if max_image_width is None:
        raise ValueError("max_image_width is required for resizing")
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "Pillow is required for resizing. Install Pillow or disable resizing with --max-image-width 0."
        ) from exc

    with Image.open(source_image_path) as image:
        width, height = image.size
        if width <= max_image_width:
            image.save(output_image_path)
            return
        scale = max_image_width / width
        target_height = max(1, round(height * scale))
        resized = image.resize((max_image_width, target_height))
        resized.save(output_image_path)


def _rewrite_transforms(
    original: dict[str, Any],
    selected: tuple[PreparedFrame, ...],
) -> dict[str, Any]:
    original_frames = original.get("frames", [])
    rewritten = {key: value for key, value in original.items() if key != "frames"}
    frames: list[dict[str, Any]] = []
    for prepared in selected:
        frame = dict(original_frames[prepared.index])
        frame["file_path"] = Path(prepared.output_file_path).as_posix()
        frames.append(frame)
    rewritten["frames"] = frames
    return rewritten


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _portable_path(path: Path, data_root: Path) -> dict[str, str]:
    try:
        relative_path = path.resolve().relative_to(data_root.resolve()).as_posix()
        return {
            "path": relative_path,
            "path_type": "relative_to_data_root",
        }
    except ValueError:
        return {
            "path": os.path.relpath(path.resolve(), data_root.resolve()),
            "path_type": "relative_to_data_root",
        }


def _manifest(plan: NerfSyntheticSubsetPlan) -> dict[str, Any]:
    return {
        "schema_name": NERF_SYNTHETIC_SUBSET_SCHEMA,
        "schema_version": NERF_SYNTHETIC_SUBSET_VERSION,
        "source_dataset": "nerf_synthetic",
        "source_scene": plan.scene,
        "condition": plan.condition,
        "seed": plan.seed,
        "max_train_views": plan.max_train_views,
        "max_test_views": plan.max_test_views,
        "max_target_views": plan.max_target_views,
        "max_image_width": plan.max_image_width,
        "copy_mode": plan.copy_mode,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_root": {
            "path": ".",
            "path_type": "data_root",
        },
        "raw_scene_root": _portable_path(plan.raw_scene_root, plan.data_root),
        "output_scene_root": _portable_path(plan.output_condition_root, plan.data_root),
        "selected_train_frames": [
            frame.as_manifest_entry(plan.raw_scene_root) for frame in plan.train_frames
        ],
        "selected_test_frames": [
            frame.as_manifest_entry(plan.raw_scene_root) for frame in plan.test_frames
        ],
        "selected_target_frames": [
            frame.as_manifest_entry(plan.raw_scene_root) for frame in plan.target_frames
        ],
        "image_count": plan.image_count,
        "notes": [
            "clean condition only",
            "deterministic uniform frame selection",
            "relative file_path entries are used in prepared transforms",
        ],
    }


def build_nerf_synthetic_subset_plan(
    *,
    data_root: Path,
    raw_scene_root: Path,
    output_root: Path,
    scene: str,
    condition: str,
    max_train_views: int,
    max_test_views: int,
    max_target_views: int,
    max_image_width: int | None,
    copy_mode: str,
    seed: int,
) -> tuple[NerfSyntheticSubsetPlan, dict[str, Any], dict[str, Any]]:
    """Build a deterministic subset plan without writing files."""

    if condition != "clean":
        raise ValueError("PR2 supports only condition=clean")
    if copy_mode not in SUPPORTED_COPY_MODES:
        raise ValueError(f"copy_mode must be one of {sorted(SUPPORTED_COPY_MODES)}")

    data_root = data_root.resolve()
    raw_scene_root = raw_scene_root.resolve()
    output_condition_root = (output_root / condition).resolve()
    train_transforms = _load_transforms(raw_scene_root / "transforms_train.json")
    test_transforms = _load_transforms(raw_scene_root / "transforms_test.json")

    train_frames = _selected_frames(
        raw_scene_root=raw_scene_root,
        output_condition_root=output_condition_root,
        split="train",
        frames=train_transforms["frames"],
        maximum=max_train_views,
        max_image_width=max_image_width,
        copy_mode=copy_mode,
    )
    test_frames = _selected_frames(
        raw_scene_root=raw_scene_root,
        output_condition_root=output_condition_root,
        split="test",
        frames=test_transforms["frames"],
        maximum=max_test_views,
        max_image_width=max_image_width,
        copy_mode=copy_mode,
    )
    target_frames = _selected_frames(
        raw_scene_root=raw_scene_root,
        output_condition_root=output_condition_root,
        split="target",
        frames=test_transforms["frames"],
        maximum=max_target_views,
        max_image_width=max_image_width,
        copy_mode=copy_mode,
    )

    plan = NerfSyntheticSubsetPlan(
        data_root=data_root,
        raw_scene_root=raw_scene_root,
        output_condition_root=output_condition_root,
        scene=scene,
        condition=condition,
        max_train_views=max_train_views,
        max_test_views=max_test_views,
        max_target_views=max_target_views,
        max_image_width=max_image_width,
        copy_mode=copy_mode,
        seed=seed,
        train_frames=train_frames,
        test_frames=test_frames,
        target_frames=target_frames,
        will_resize=any(
            frame.resized for frame in (*train_frames, *test_frames, *target_frames)
        ),
    )
    return plan, train_transforms, test_transforms


def prepare_nerf_synthetic_subset(
    *,
    data_root: Path,
    raw_scene_root: Path,
    output_root: Path,
    scene: str,
    condition: str,
    max_train_views: int,
    max_test_views: int,
    max_target_views: int,
    max_image_width: int | None,
    copy_mode: str,
    seed: int,
    dry_run: bool,
    overwrite: bool,
) -> NerfSyntheticSubsetPlan:
    """Prepare a NeRF Synthetic clean subset from an already downloaded scene."""

    plan, train_transforms, test_transforms = build_nerf_synthetic_subset_plan(
        data_root=data_root,
        raw_scene_root=raw_scene_root,
        output_root=output_root,
        scene=scene,
        condition=condition,
        max_train_views=max_train_views,
        max_test_views=max_test_views,
        max_target_views=max_target_views,
        max_image_width=max_image_width,
        copy_mode=copy_mode,
        seed=seed,
    )

    if dry_run:
        return plan

    if plan.output_condition_root.exists():
        if not overwrite:
            raise FileExistsError(
                f"output already exists: {plan.output_condition_root}. Use --overwrite to replace it."
            )
        shutil.rmtree(plan.output_condition_root)

    plan.output_condition_root.mkdir(parents=True, exist_ok=True)
    all_frames = (*plan.train_frames, *plan.test_frames, *plan.target_frames)
    written_images: set[str] = set()
    for frame in all_frames:
        if frame.output_file_path in written_images:
            continue
        _copy_or_link_image(
            frame.source_image_path,
            frame.output_image_path,
            copy_mode=plan.copy_mode,
            max_image_width=plan.max_image_width,
        )
        written_images.add(frame.output_file_path)

    _write_json(
        plan.output_condition_root / "transforms_train.json",
        _rewrite_transforms(train_transforms, plan.train_frames),
    )
    _write_json(
        plan.output_condition_root / "transforms_test.json",
        _rewrite_transforms(test_transforms, plan.test_frames),
    )
    _write_json(
        plan.output_condition_root / "transforms_target.json",
        _rewrite_transforms(test_transforms, plan.target_frames),
    )
    _write_json(plan.output_condition_root / "manifest.json", _manifest(plan))
    (plan.output_condition_root / "README.md").write_text(
        "\n".join(
            [
                f"# NeRF Synthetic {plan.scene} clean mini subset",
                "",
                "Prepared by ViewTrust-GS PR2 tooling.",
                "",
                "This subset is clean-only and intended for Priority 0 observation experiments.",
                "It does not contain generated corruptions, attacks, or defenses.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return plan


def plan_summary(plan: NerfSyntheticSubsetPlan, *, dry_run: bool) -> dict[str, object]:
    return {
        "raw_scene_root": str(plan.raw_scene_root),
        "output_condition_root": str(plan.output_condition_root),
        "selected_train_count": len(plan.train_frames),
        "selected_test_count": len(plan.test_frames),
        "selected_target_count": len(plan.target_frames),
        "copy_mode": plan.copy_mode,
        "will_resize": plan.will_resize,
        "estimated_output_file_count": plan.image_count + 5,
        "mode": "dry-run" if dry_run else "write",
    }
