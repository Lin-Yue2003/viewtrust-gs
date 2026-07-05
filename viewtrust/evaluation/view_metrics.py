"""CPU-only view metric extraction for PR6 clean baseline evaluation."""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from viewtrust.analysis.tables import write_csv_table

VIEW_METRICS_SCHEMA = "viewtrust.view_metrics.summary"
VIEW_METRICS_SCHEMA_VERSION = 1
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")

VIEW_METRICS_FIELDS = [
    "run_id",
    "scene",
    "condition",
    "split",
    "iteration",
    "view_index",
    "image_name",
    "render_relative_path",
    "gt_relative_path",
    "width",
    "height",
    "l1_mean",
    "mse",
    "psnr",
    "ssim",
    "ssim_method",
    "residual_mean",
    "residual_median",
    "residual_p95",
    "residual_p99",
    "residual_max",
    "status",
    "warning",
]

VIEW_RENDER_ARTIFACT_FIELDS = [
    "run_id",
    "split",
    "iteration",
    "kind",
    "relative_path",
    "size_bytes",
    "width",
    "height",
    "exists",
]


@dataclass(frozen=True)
class ViewMetricsConfig:
    run_dir: Path
    scene: str = "chair"
    condition: str = "clean"
    iteration: int = 500
    require_renders: bool = False


def _split_roots(run_dir: Path, iteration: int) -> list[tuple[str, Path]]:
    base = run_dir / "view_evaluation" / "render_models"
    return [
        ("train", base / "train_test_model" / "train" / f"ours_{iteration}"),
        ("test", base / "train_test_model" / "test" / f"ours_{iteration}"),
        ("target", base / "target_model" / "test" / f"ours_{iteration}"),
    ]


def _image_files(path: Path) -> list[Path]:
    if not path.is_dir():
        return []
    return sorted(
        child
        for child in path.iterdir()
        if child.is_file() and child.suffix.lower() in IMAGE_EXTENSIONS
    )


def discover_render_pairs(run_dir: Path, iteration: int) -> list[dict[str, Any]]:
    run_dir = run_dir.resolve()
    pairs: list[dict[str, Any]] = []
    for split, root in _split_roots(run_dir, iteration):
        render_files = _image_files(root / "renders")
        gt_files = _image_files(root / "gt")
        if not render_files and not gt_files:
            continue
        if len(render_files) != len(gt_files):
            raise ValueError(
                f"render/gt image count mismatch for split {split}: "
                f"renders={len(render_files)}, gt={len(gt_files)}"
            )
        for view_index, (render_path, gt_path) in enumerate(zip(render_files, gt_files)):
            pairs.append(
                {
                    "split": split,
                    "iteration": iteration,
                    "view_index": view_index,
                    "image_name": render_path.name,
                    "render_path": render_path,
                    "gt_path": gt_path,
                }
            )
    return pairs


def load_rgb_image(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        return np.asarray(rgb, dtype=np.float64) / 255.0


def _ssim(render: np.ndarray, gt: np.ndarray) -> tuple[float | None, str]:
    try:
        from skimage.metrics import structural_similarity
    except ImportError:
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        x = render.reshape(-1, 3)
        y = gt.reshape(-1, 3)
        mu_x = float(np.mean(x))
        mu_y = float(np.mean(y))
        sigma_x = float(np.var(x))
        sigma_y = float(np.var(y))
        sigma_xy = float(np.mean((x - mu_x) * (y - mu_y)))
        denominator = (mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2)
        if denominator == 0:
            return None, "unavailable"
        value = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / denominator
        return float(value), "global_numpy_fallback"

    try:
        return (
            float(
                structural_similarity(
                    gt,
                    render,
                    channel_axis=2,
                    data_range=1.0,
                )
            ),
            "skimage",
        )
    except Exception:
        return None, "unavailable"


def compute_pair_metrics(render_path: Path, gt_path: Path) -> dict[str, Any]:
    render = load_rgb_image(render_path)
    gt = load_rgb_image(gt_path)
    if render.shape != gt.shape:
        raise ValueError(f"image shape mismatch: render={render.shape}, gt={gt.shape}")

    diff = render - gt
    residual = np.abs(diff).reshape(-1)
    mse = float(np.mean(diff**2))
    l1_mean = float(np.mean(residual))
    psnr = 100.0 if mse == 0 else float(20.0 * math.log10(1.0 / math.sqrt(mse)))
    ssim_value, ssim_method = _ssim(render, gt)
    height, width = render.shape[:2]
    return {
        "width": width,
        "height": height,
        "l1_mean": l1_mean,
        "mse": mse,
        "psnr": psnr,
        "ssim": ssim_value if ssim_value is not None else "",
        "ssim_method": ssim_method,
        "residual_mean": float(np.mean(residual)),
        "residual_median": float(np.median(residual)),
        "residual_p95": float(np.percentile(residual, 95)),
        "residual_p99": float(np.percentile(residual, 99)),
        "residual_max": float(np.max(residual)),
        "status": "ok",
        "warning": "",
    }


def _artifact_row(
    *,
    run_dir: Path,
    run_id: str,
    split: str,
    iteration: int,
    kind: str,
    path: Path,
) -> dict[str, Any]:
    exists = path.exists()
    width = ""
    height = ""
    if exists:
        with Image.open(path) as image:
            width, height = image.size
    return {
        "run_id": run_id,
        "split": split,
        "iteration": iteration,
        "kind": kind,
        "relative_path": path.relative_to(run_dir).as_posix(),
        "size_bytes": path.stat().st_size if exists else "",
        "width": width,
        "height": height,
        "exists": exists,
    }


def write_view_metrics(run_dir: Path, rows: list[dict[str, Any]]) -> Path:
    primary = run_dir / "tables" / "view_metrics.csv"
    mirror = run_dir / "view_evaluation" / "tables" / "view_metrics.csv"
    write_csv_table(primary, rows, VIEW_METRICS_FIELDS)
    write_csv_table(mirror, rows, VIEW_METRICS_FIELDS)
    return primary


def write_view_render_artifacts(run_dir: Path, rows: list[dict[str, Any]]) -> Path:
    primary = run_dir / "tables" / "view_render_artifacts.csv"
    mirror = run_dir / "view_evaluation" / "tables" / "view_render_artifacts.csv"
    write_csv_table(primary, rows, VIEW_RENDER_ARTIFACT_FIELDS)
    write_csv_table(mirror, rows, VIEW_RENDER_ARTIFACT_FIELDS)
    return primary


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def summarize_view_metrics(
    run_dir: Path,
    rows: list[dict[str, Any]],
    *,
    scene: str,
    condition: str,
    iteration: int,
) -> dict[str, Any]:
    run_id = run_dir.name
    warnings = sorted({str(row["warning"]) for row in rows if row.get("warning")})
    missing_fields: list[str] = []
    view_count_by_split = {
        split: sum(1 for row in rows if row.get("split") == split and row.get("status") == "ok")
        for split in ("train", "test", "target")
    }
    metrics: dict[str, dict[str, float | None]] = {}
    for split in ("train", "test", "target"):
        split_rows = [
            row for row in rows if row.get("split") == split and row.get("status") == "ok"
        ]
        psnr_values = [float(row["psnr"]) for row in split_rows if row.get("psnr") != ""]
        l1_values = [float(row["l1_mean"]) for row in split_rows if row.get("l1_mean") != ""]
        ssim_values = [float(row["ssim"]) for row in split_rows if row.get("ssim") != ""]
        if split_rows and not ssim_values:
            missing_fields.append(f"{split}.ssim")
        metrics[split] = {
            "psnr_mean": _mean(psnr_values),
            "psnr_min": min(psnr_values) if psnr_values else None,
            "psnr_max": max(psnr_values) if psnr_values else None,
            "l1_mean": _mean(l1_values),
            "ssim_mean": _mean(ssim_values),
        }
    summary = {
        "schema_name": VIEW_METRICS_SCHEMA,
        "schema_version": VIEW_METRICS_SCHEMA_VERSION,
        "run_dir": str(run_dir.resolve()),
        "run_id": run_id,
        "scene": scene,
        "condition": condition,
        "iteration": iteration,
        "observation_only": True,
        "rendered_splits": [
            split for split, count in view_count_by_split.items() if count > 0
        ],
        "view_count_total": sum(view_count_by_split.values()),
        "view_count_by_split": view_count_by_split,
        "metrics": metrics,
        "warnings": warnings,
        "missing_fields": sorted(set(missing_fields)),
    }
    path = run_dir / "view_metrics_summary.json"
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    mirror = run_dir / "view_evaluation" / "view_metrics_summary.json"
    mirror.parent.mkdir(parents=True, exist_ok=True)
    mirror.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def extract_view_metrics(config: ViewMetricsConfig) -> dict[str, Any]:
    run_dir = config.run_dir.resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"run_dir does not exist: {run_dir}")
    pairs = discover_render_pairs(run_dir, config.iteration)
    if config.require_renders and not pairs:
        raise FileNotFoundError(f"no rendered image pairs found for iteration {config.iteration}")

    run_id = run_dir.name
    metric_rows: list[dict[str, Any]] = []
    artifact_rows: list[dict[str, Any]] = []
    for pair in pairs:
        render_path = pair["render_path"]
        gt_path = pair["gt_path"]
        base_row = {
            "run_id": run_id,
            "scene": config.scene,
            "condition": config.condition,
            "split": pair["split"],
            "iteration": pair["iteration"],
            "view_index": pair["view_index"],
            "image_name": pair["image_name"],
            "render_relative_path": render_path.relative_to(run_dir).as_posix(),
            "gt_relative_path": gt_path.relative_to(run_dir).as_posix(),
        }
        try:
            metrics = compute_pair_metrics(render_path, gt_path)
        except Exception as exc:
            metrics = {
                "width": "",
                "height": "",
                "l1_mean": "",
                "mse": "",
                "psnr": "",
                "ssim": "",
                "ssim_method": "unavailable",
                "residual_mean": "",
                "residual_median": "",
                "residual_p95": "",
                "residual_p99": "",
                "residual_max": "",
                "status": "error",
                "warning": str(exc),
            }
        metric_rows.append({**base_row, **metrics})
        artifact_rows.append(
            _artifact_row(
                run_dir=run_dir,
                run_id=run_id,
                split=pair["split"],
                iteration=pair["iteration"],
                kind="render",
                path=render_path,
            )
        )
        artifact_rows.append(
            _artifact_row(
                run_dir=run_dir,
                run_id=run_id,
                split=pair["split"],
                iteration=pair["iteration"],
                kind="gt",
                path=gt_path,
            )
        )

    write_view_metrics(run_dir, metric_rows)
    write_view_render_artifacts(run_dir, artifact_rows)
    return summarize_view_metrics(
        run_dir,
        metric_rows,
        scene=config.scene,
        condition=config.condition,
        iteration=config.iteration,
    )
