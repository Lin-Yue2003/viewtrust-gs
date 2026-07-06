"""Natural corruption condition generation for prepared NeRF Synthetic mini data."""

from __future__ import annotations

import csv
import json
import math
import os
import random
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

NATURAL_CORRUPTION_MANIFEST_SCHEMA = "viewtrust.natural_corruption.manifest"
NATURAL_CORRUPTION_SUMMARY_SCHEMA = "viewtrust.natural_corruption.summary"
NATURAL_CORRUPTION_SCHEMA_VERSION = 1
SUPPORTED_CORRUPTIONS = {"occluder", "blur", "exposure", "color_shift", "noise", "mixed"}
SUPPORTED_COPY_MODES = {"copy", "symlink"}
DEFAULT_CONDITIONS = {
    "corrupt_occluder": "occluder",
    "corrupt_blur": "blur",
    "corrupt_exposure": "exposure",
    "corrupt_color_shift": "color_shift",
    "corrupt_noise": "noise",
    "corrupt_mixed": "mixed",
}


@dataclass(frozen=True)
class CorruptionResult:
    image: Any
    corruption_type: str
    parameters: dict[str, Any]


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _frames(path: Path) -> list[dict[str, Any]]:
    payload = _load_json(path)
    frames = payload.get("frames")
    if not isinstance(frames, list):
        raise ValueError(f"{path} does not contain frames")
    return frames


def _view_name(frame: dict[str, Any]) -> str:
    file_path = str(frame.get("file_path", ""))
    if not file_path:
        raise ValueError("frame missing file_path")
    return Path(file_path).name


def _is_extensionless(frame: dict[str, Any]) -> bool:
    return Path(str(frame.get("file_path", ""))).suffix == ""


def _image_path(condition_root: Path, frame: dict[str, Any]) -> Path:
    file_path = Path(str(frame["file_path"]))
    if file_path.suffix:
        return condition_root / file_path
    return condition_root / file_path.with_suffix(".png")


def _copy_or_symlink(source: Path, target: Path, copy_mode: str) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    if copy_mode == "copy":
        shutil.copy2(source, target)
        return "copy"
    if copy_mode == "symlink":
        os.symlink(source.resolve(), target)
        return "symlink"
    raise ValueError(f"unsupported copy_mode: {copy_mode}")


def _select_train_views(
    train_view_names: list[str],
    *,
    seed: int,
    num_corrupt_train_views: int | None,
    corrupt_train_fraction: float | None,
    corrupt_view_names: list[str] | None,
) -> list[str]:
    if corrupt_view_names:
        missing = sorted(set(corrupt_view_names) - set(train_view_names))
        if missing:
            raise ValueError(f"corrupt view names are not train views: {missing}")
        return list(corrupt_view_names)
    if num_corrupt_train_views is None:
        if corrupt_train_fraction is not None:
            num_corrupt_train_views = math.ceil(corrupt_train_fraction * len(train_view_names))
        else:
            num_corrupt_train_views = 4
    count = max(0, min(num_corrupt_train_views, len(train_view_names)))
    rng = random.Random(seed)
    return sorted(rng.sample(train_view_names, count))


def _pil_modules() -> tuple[Any, Any, Any]:
    try:
        from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
    except ImportError as exc:
        raise RuntimeError("Pillow is required for PR10 natural corruption generation.") from exc
    return ImageDraw, ImageEnhance, ImageFilter


def _apply_corruption(image: Any, corruption_type: str, rng: random.Random) -> CorruptionResult:
    ImageDraw, ImageEnhance, ImageFilter = _pil_modules()
    image = image.convert("RGB")
    width, height = image.size
    actual_type = corruption_type
    if corruption_type == "mixed":
        actual_type = rng.choice(["occluder", "blur", "exposure", "color_shift", "noise"])

    if actual_type == "occluder":
        side = min(width, height)
        size = round(side * rng.uniform(0.15, 0.35))
        x0 = rng.randint(0, max(0, width - size))
        y0 = rng.randint(0, max(0, height - size))
        color = [rng.randint(88, 168) for _ in range(3)]
        output = image.copy()
        ImageDraw.Draw(output).rectangle([x0, y0, x0 + size, y0 + size], fill=tuple(color))
        return CorruptionResult(
            output,
            actual_type,
            {
                "shape": "rectangle",
                "x0": x0,
                "y0": y0,
                "x1": x0 + size,
                "y1": y0 + size,
                "opacity": 1.0,
                "color": color,
            },
        )
    if actual_type == "blur":
        radius = rng.uniform(1.5, 4.0)
        return CorruptionResult(
            image.filter(ImageFilter.GaussianBlur(radius=radius)),
            actual_type,
            {"radius": radius},
        )
    if actual_type == "exposure":
        factor = rng.uniform(0.5, 1.6)
        return CorruptionResult(
            ImageEnhance.Brightness(image).enhance(factor),
            actual_type,
            {"factor": factor},
        )
    if actual_type == "color_shift":
        factors = [rng.uniform(0.75, 1.25) for _ in range(3)]
        pixels = image.load()
        output = image.copy()
        output_pixels = output.load()
        for y in range(height):
            for x in range(width):
                r, g, b = pixels[x, y]
                output_pixels[x, y] = tuple(
                    max(0, min(255, round(value * factor)))
                    for value, factor in zip((r, g, b), factors)
                )
        return CorruptionResult(
            output,
            actual_type,
            {
                "red_factor": factors[0],
                "green_factor": factors[1],
                "blue_factor": factors[2],
            },
        )
    if actual_type == "noise":
        std = rng.uniform(0.03, 0.10)
        pixels = image.load()
        output = image.copy()
        output_pixels = output.load()
        for y in range(height):
            for x in range(width):
                output_pixels[x, y] = tuple(
                    max(0, min(255, round(channel + rng.gauss(0, std * 255))))
                    for channel in pixels[x, y]
                )
        return CorruptionResult(output, actual_type, {"noise_std": std})
    raise ValueError(f"unsupported corruption_type: {corruption_type}")


def _diff_stats(clean: Any, corrupt: Any) -> dict[str, float]:
    clean_pixels = list(clean.convert("RGB").getdata())
    corrupt_pixels = list(corrupt.convert("RGB").getdata())
    total_values = max(1, len(clean_pixels) * 3)
    total_pixels = max(1, len(clean_pixels))
    abs_sum = 0
    max_abs = 0
    changed = 0
    for clean_pixel, corrupt_pixel in zip(clean_pixels, corrupt_pixels):
        pixel_changed = False
        for a, b in zip(clean_pixel, corrupt_pixel):
            diff = abs(a - b)
            abs_sum += diff
            max_abs = max(max_abs, diff)
            pixel_changed = pixel_changed or diff > 0
        changed += int(pixel_changed)
    return {
        "mean_abs_diff": abs_sum / (total_values * 255.0),
        "max_abs_diff": max_abs / 255.0,
        "changed_pixel_ratio": changed / total_pixels,
    }


def _write_preview(rows: list[dict[str, Any]], preview_path: Path, preview_count: int) -> None:
    if preview_count <= 0:
        return
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise RuntimeError("Pillow is required for preview generation.") from exc
    selected = [row for row in rows if row["was_corrupted"]][:preview_count]
    if not selected:
        return
    thumb_w, thumb_h = 120, 120
    grid = Image.new("RGB", (thumb_w * 2, thumb_h * len(selected)), "white")
    draw = ImageDraw.Draw(grid)
    for idx, row in enumerate(selected):
        y = idx * thumb_h
        with Image.open(row["clean_file"]) as clean, Image.open(row["output_file"]) as corrupt:
            grid.paste(clean.convert("RGB").resize((thumb_w, thumb_h)), (0, y))
            grid.paste(corrupt.convert("RGB").resize((thumb_w, thumb_h)), (thumb_w, y))
        draw.text((4, y + 4), row["view_name"], fill=(255, 0, 0))
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(preview_path)


def _write_manifest_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "scene",
        "source_condition",
        "output_condition",
        "split",
        "view_name",
        "was_corrupted",
        "corruption_type",
        "clean_file",
        "output_file",
        "parameter_json",
        "seed",
        "copy_mode",
        "mean_abs_diff",
        "max_abs_diff",
        "changed_pixel_ratio",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def generate_natural_corruption_condition(
    *,
    data_root: Path,
    scene: str,
    source_condition: str,
    output_condition: str,
    corruption_type: str,
    seed: int,
    num_corrupt_train_views: int | None,
    corrupt_train_fraction: float | None,
    corrupt_view_names: list[str] | None,
    copy_mode: str,
    overwrite: bool,
    dry_run: bool,
    preview_count: int,
) -> dict[str, Any]:
    if corruption_type not in SUPPORTED_CORRUPTIONS:
        raise ValueError(f"corruption_type must be one of {sorted(SUPPORTED_CORRUPTIONS)}")
    if copy_mode not in SUPPORTED_COPY_MODES:
        raise ValueError(f"copy_mode must be one of {sorted(SUPPORTED_COPY_MODES)}")

    data_root = data_root.resolve()
    scene_root = data_root / "viewtrust-mini" / "nerf_synthetic" / scene
    source_root = scene_root / source_condition
    output_root = scene_root / output_condition
    train_transforms = _load_json(source_root / "transforms_train.json")
    test_transforms = _load_json(source_root / "transforms_test.json")
    target_transforms = _load_json(source_root / "transforms_target.json")
    train_frames = train_transforms["frames"]
    test_frames = test_transforms["frames"]
    target_frames = target_transforms["frames"]
    train_names = [_view_name(frame) for frame in train_frames]
    selected = _select_train_views(
        train_names,
        seed=seed,
        num_corrupt_train_views=num_corrupt_train_views,
        corrupt_train_fraction=corrupt_train_fraction,
        corrupt_view_names=corrupt_view_names,
    )
    selected_set = set(selected)
    summary_base = {
        "scene": scene,
        "source_condition": source_condition,
        "output_condition": output_condition,
        "corruption_type": corruption_type,
        "seed": seed,
        "copy_mode": copy_mode,
        "source_condition_path": str(source_root),
        "output_condition_path": str(output_root),
        "train_view_count": len(train_frames),
        "test_view_count": len(test_frames),
        "target_view_count": len(target_frames),
        "selected_train_view_count": len(selected),
        "selected_train_views": selected,
        "estimated_output_image_count": len(train_frames) + len(test_frames) + len(target_frames),
        "estimated_new_corrupt_image_count": len(selected),
        "estimated_link_or_copy_image_count": (
            len(train_frames) + len(test_frames) + len(target_frames) - len(selected)
        ),
        "will_resize": False,
        "preview_count": preview_count,
        "mode": "dry-run" if dry_run else "write",
    }
    if dry_run:
        return summary_base

    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"output already exists: {output_root}")
        shutil.rmtree(output_root)
    (output_root / "images").mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    corruptions: list[dict[str, Any]] = []
    uncorrupted: list[dict[str, Any]] = []
    image_size: tuple[int, int] | None = None

    def process_frame(split: str, frame: dict[str, Any]) -> None:
        nonlocal image_size
        view_name = _view_name(frame)
        clean_file = _image_path(source_root, frame)
        output_file = _image_path(output_root, frame)
        was_corrupted = split == "train" and view_name in selected_set
        params: dict[str, Any] = {}
        actual_type = ""
        diff = {"mean_abs_diff": "", "max_abs_diff": "", "changed_pixel_ratio": ""}
        if was_corrupted:
            try:
                from PIL import Image
            except ImportError as exc:
                raise RuntimeError("Pillow is required for corruption generation.") from exc
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with Image.open(clean_file) as clean_image:
                clean_rgb = clean_image.convert("RGB")
                image_size = image_size or clean_rgb.size
                result = _apply_corruption(clean_rgb, corruption_type, rng)
                result.image.save(output_file)
                diff = _diff_stats(clean_rgb, result.image)
                params = result.parameters
                actual_type = result.corruption_type
            corruptions.append(
                {
                    "split": split,
                    "view_name": view_name,
                    "clean_file": str(clean_file),
                    "corrupt_file": str(output_file),
                    "was_corrupted": True,
                    "corruption_type": actual_type,
                    "parameters": params,
                    **diff,
                }
            )
        else:
            link_or_copy = _copy_or_symlink(clean_file, output_file, copy_mode)
            uncorrupted.append(
                {
                    "split": split,
                    "view_name": view_name,
                    "source_file": str(clean_file),
                    "output_file": str(output_file),
                    "link_or_copy": link_or_copy,
                }
            )
            if image_size is None:
                try:
                    from PIL import Image
                    with Image.open(clean_file) as image:
                        image_size = image.size
                except ImportError:
                    pass
        rows.append(
            {
                "scene": scene,
                "source_condition": source_condition,
                "output_condition": output_condition,
                "split": split,
                "view_name": view_name,
                "was_corrupted": str(was_corrupted).lower(),
                "corruption_type": actual_type if was_corrupted else "",
                "clean_file": str(clean_file),
                "output_file": str(output_file),
                "parameter_json": json.dumps(params, sort_keys=True),
                "seed": seed,
                "copy_mode": copy_mode,
                **diff,
            }
        )

    for split, frames in (("train", train_frames), ("test", test_frames), ("target", target_frames)):
        for frame in frames:
            process_frame(split, frame)

    shutil.copy2(source_root / "transforms_train.json", output_root / "transforms_train.json")
    shutil.copy2(source_root / "transforms_test.json", output_root / "transforms_test.json")
    shutil.copy2(source_root / "transforms_target.json", output_root / "transforms_target.json")
    _write_manifest_csv(output_root / "corruption_manifest.csv", rows)
    transforms_extensionless = all(
        _is_extensionless(frame)
        for frame in [*train_frames, *test_frames, *target_frames]
    )
    width, height = image_size or (None, None)
    manifest = {
        "schema_name": NATURAL_CORRUPTION_MANIFEST_SCHEMA,
        "schema_version": NATURAL_CORRUPTION_SCHEMA_VERSION,
        **summary_base,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "image_width": width,
        "image_height": height,
        "corruptions": corruptions,
        "uncorrupted_views": uncorrupted,
    }
    summary = {
        "schema_name": NATURAL_CORRUPTION_SUMMARY_SCHEMA,
        "schema_version": NATURAL_CORRUPTION_SCHEMA_VERSION,
        **summary_base,
        "image_width": width,
        "image_height": height,
        "corrupted_image_count": len(corruptions),
        "uncorrupted_image_count": len(uncorrupted),
        "total_image_count": len(rows),
        "transforms_extensionless_file_path": transforms_extensionless,
        "warnings": [],
    }
    _write_json(output_root / "corruption_manifest.json", manifest)
    _write_json(output_root / "corruption_summary.json", summary)
    _write_preview(rows, output_root / "preview" / "preview_grid.png", preview_count)
    return summary
