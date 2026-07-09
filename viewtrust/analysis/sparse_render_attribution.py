"""Offline sparse residual and Gaussian attribution helpers for PR20.0."""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from viewtrust.analysis.clean_prior_normalization import load_csv_rows, load_json, normalize_bool
from viewtrust.analysis.offline_signals import safe_float


PR200_OUTPUT_FILES = [
    "pr200_sparse_render_attribution_summary.json",
    "pr200_selected_views.csv",
    "pr200_view_residual_summary.csv",
    "pr200_sparse_pixel_residuals.csv",
    "pr200_pixel_gaussian_contributions.csv",
    "pr200_gaussian_residual_attribution.csv",
    "pr200_view_group_residual_attribution.csv",
    "pr200_direct_collateral_residual_overlap.csv",
    "pr200_train013_residual_control.csv",
    "pr200_attribution_quality_audit.csv",
    "pr200_missing_inputs.csv",
    "pr200_report.md",
    "artifact_manifest.csv",
]

PR193_REQUIRED_FILES = [
    "pr193_view_group_map.csv",
    "gaussian_identity_table_grouped.csv",
    "view_gaussian_event_attribution_grouped.csv",
    "gaussian_support_summary_grouped.csv",
]

PR195_REQUIRED_FILES = [
    "pr195_attribution_semantics_summary.json",
    "pr195_required_attribution_field_gap.csv",
    "pr195_pr20_readiness_assessment.csv",
]

GROUPS = ["direct_corrupted", "co_visible_collateral", "clean_prior_demoted", "other_clean"]
GROUP_PREFIX = {
    "direct_corrupted": "direct_corrupted",
    "co_visible_collateral": "collateral",
    "clean_prior_demoted": "clean_prior",
    "other_clean": "other_clean",
}

SELECTED_VIEW_FIELDS = [
    "scene",
    "condition",
    "subset_name",
    "view_name",
    "view_group",
    "was_corrupted",
    "selection_reason",
    "image_path",
    "camera_id",
    "included",
]

VIEW_RESIDUAL_FIELDS = [
    "scene",
    "condition",
    "subset_name",
    "view_name",
    "view_group",
    "residual_metric",
    "image_width",
    "image_height",
    "evaluated_pixel_count",
    "selected_pixel_count",
    "mean_residual",
    "median_residual",
    "max_residual",
    "top_pixel_residual_mean",
    "residual_entropy",
    "residual_concentration",
    "artifact_mask_mode",
    "evidence_quality",
]

SPARSE_PIXEL_FIELDS = [
    "scene",
    "condition",
    "subset_name",
    "view_name",
    "view_group",
    "pixel_x",
    "pixel_y",
    "residual_l1",
    "residual_l2",
    "rendered_r",
    "rendered_g",
    "rendered_b",
    "gt_r",
    "gt_g",
    "gt_b",
    "artifact_region_flag",
    "target_region_flag",
    "selection_rank",
    "evidence_quality",
]

PIXEL_GAUSSIAN_FIELDS = [
    "scene",
    "condition",
    "subset_name",
    "view_name",
    "view_group",
    "pixel_x",
    "pixel_y",
    "gaussian_id",
    "root_gaussian_id",
    "parent_gaussian_id",
    "contribution_rank",
    "splat_weight",
    "alpha_contribution",
    "color_contribution_r",
    "color_contribution_g",
    "color_contribution_b",
    "residual_l1",
    "residual_weighted_splat",
    "residual_weighted_alpha",
    "evidence_quality",
    "attribution_method",
    "warnings",
]

GAUSSIAN_ATTR_FIELDS = [
    "scene",
    "condition",
    "subset_name",
    "gaussian_id",
    "root_gaussian_id",
    "parent_gaussian_id",
    "direct_corrupted_residual_weight",
    "collateral_residual_weight",
    "clean_prior_residual_weight",
    "other_clean_residual_weight",
    "direct_corrupted_pixel_count",
    "collateral_pixel_count",
    "clean_prior_pixel_count",
    "other_clean_pixel_count",
    "corrupted_plus_collateral_residual_ratio",
    "clean_prior_residual_ratio",
    "dominant_residual_view_group",
    "residual_source_entropy",
    "residual_source_concentration",
    "mean_residual_weighted_splat",
    "max_residual_weighted_splat",
    "evidence_quality",
    "attribution_method",
]

GROUP_ATTR_FIELDS = [
    "scene",
    "condition",
    "subset_name",
    "view_group",
    "selected_view_count",
    "selected_pixel_count",
    "unique_gaussian_count",
    "total_residual_weight",
    "mean_residual_weight",
    "top_gaussian_count",
    "residual_entropy",
    "residual_concentration",
    "evidence_quality",
]

DIRECT_COLLATERAL_FIELDS = [
    "scene",
    "condition",
    "subset_name",
    "direct_corrupted_view_names",
    "collateral_view_names",
    "direct_residual_gaussian_count",
    "collateral_residual_gaussian_count",
    "residual_overlap_gaussian_count",
    "residual_overlap_jaccard",
    "residual_overlap_ratio_over_direct",
    "residual_overlap_ratio_over_collateral",
    "residual_weighted_overlap_sum",
    "nontrivial_residual_overlap_supported",
    "evidence_quality",
    "notes",
]

TRAIN013_FIELDS = [
    "scene",
    "condition",
    "subset_name",
    "train013_present",
    "train013_view_group",
    "train013_residual_gaussian_count",
    "direct_collateral_residual_gaussian_count",
    "train013_overlap_with_direct_collateral_count",
    "train013_overlap_ratio",
    "train013_residual_weight_sum",
    "direct_collateral_residual_weight_sum",
    "train013_residual_control_supported",
    "evidence_quality",
    "reason",
    "notes",
]

QUALITY_FIELDS = ["criterion", "passed", "evidence", "blocker", "recommended_action"]
MISSING_FIELDS = ["input_name", "path", "exists", "required", "details"]
MANIFEST_FIELDS = ["relative_path", "path", "exists", "file_type", "size_bytes", "required", "artifact_group"]


def write_csv_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: "" if row.get(field) is None else row.get(field) for field in fieldnames})


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _truth(value: Any) -> bool:
    return normalize_bool(value) is True


def _bool_text(value: bool) -> str:
    return str(bool(value)).lower()


def _number(value: Any, default: float = 0.0) -> float:
    parsed = safe_float(value)
    return default if parsed is None else parsed


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _entropy(weights: list[float]) -> tuple[float, float]:
    total = sum(value for value in weights if value > 0)
    if total <= 0:
        return 0.0, 0.0
    probabilities = [value / total for value in weights if value > 0]
    entropy = -sum(p * math.log(p) for p in probabilities)
    max_entropy = math.log(len(probabilities)) if len(probabilities) > 1 else 1.0
    concentration = 1.0 - (entropy / max_entropy if max_entropy else 0.0)
    return entropy, max(0.0, min(1.0, concentration))


def _view_index(view_name: str) -> int | str:
    token = str(view_name).rsplit("_", 1)[-1]
    try:
        return int(token)
    except ValueError:
        return ""


def _load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float64) / 255.0


def _image_roots(run_dir: Path) -> list[tuple[str, Path]]:
    base = run_dir / "view_evaluation" / "render_models"
    return [
        ("train", base / "train_test_model" / "train" / "ours_700"),
        ("test", base / "train_test_model" / "test" / "ours_700"),
        ("target", base / "target_model" / "test" / "ours_700"),
        ("train", base / "train_test_model" / "train" / "ours_500"),
        ("test", base / "train_test_model" / "test" / "ours_500"),
        ("target", base / "target_model" / "test" / "ours_500"),
    ]


def _candidate_names(view_name: str) -> list[str]:
    index = _view_index(view_name)
    names = [view_name, f"{view_name}.png", f"{view_name}.jpg"]
    if index != "":
        names.extend([f"{int(index):05d}.png", f"{int(index):05d}.jpg", f"{int(index):03d}.png", f"r_{int(index):03d}.png"])
    return names


def _find_render_pair(run_dir: Path, view_name: str) -> tuple[Path | None, Path | None]:
    for split, root in _image_roots(run_dir):
        if not view_name.startswith(split):
            continue
        for name in _candidate_names(view_name):
            render = root / "renders" / name
            gt = root / "gt" / name
            if render.is_file() and gt.is_file():
                return render, gt
    for _, root in _image_roots(run_dir):
        for name in _candidate_names(view_name):
            render = root / "renders" / name
            gt = root / "gt" / name
            if render.is_file() and gt.is_file():
                return render, gt
    return None, None


def _select_views(view_rows: list[dict[str, str]], explicit_views: list[str] | None, max_views: int) -> list[dict[str, Any]]:
    by_name = {row.get("view_name", ""): row for row in view_rows if row.get("view_name")}
    selected: list[dict[str, Any]] = []
    if explicit_views:
        for view in explicit_views:
            row = by_name.get(view, {"view_name": view, "view_group": "other_clean", "was_corrupted": ""})
            selected.append({**row, "selection_reason": "explicit"})
    else:
        priority = [
            ("direct_corrupted", "direct_corrupted"),
            ("co_visible_collateral", "co_visible_collateral"),
            ("clean_prior_demoted", "train013_control"),
            ("other_clean", "other_clean_context"),
        ]
        for group, reason in priority:
            rows = [row for row in view_rows if row.get("view_group") == group]
            if group == "clean_prior_demoted":
                rows = sorted(rows, key=lambda row: row.get("view_name") != "train_013")
            for row in rows:
                if row.get("view_name") and row.get("view_name") not in {item.get("view_name") for item in selected}:
                    selected.append({**row, "selection_reason": reason})
                if len(selected) >= max_views:
                    break
            if len(selected) >= max_views:
                break
    return selected[:max_views]


def _top_residual_pixels(
    render: np.ndarray,
    gt: np.ndarray,
    *,
    top_pixels: int,
    residual_metric: str,
    downsample: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if render.shape != gt.shape:
        raise ValueError(f"render/gt shape mismatch: render={render.shape}, gt={gt.shape}")
    step = max(1, int(downsample))
    render_ds = render[::step, ::step]
    gt_ds = gt[::step, ::step]
    diff = render_ds - gt_ds
    residual_l1 = np.mean(np.abs(diff), axis=2)
    residual_l2 = np.sqrt(np.mean(diff**2, axis=2))
    if residual_metric in {"l1", "abs_rgb", "alpha_weighted_l1"}:
        metric = residual_l1
    elif residual_metric == "l2":
        metric = residual_l2
    else:
        raise ValueError(f"unsupported residual metric: {residual_metric}")
    flat = metric.reshape(-1)
    count = min(int(top_pixels), flat.size)
    if count <= 0:
        return [], {}
    indices = np.argsort(flat)[::-1][:count]
    rows = []
    height, width = metric.shape
    selected_values = []
    for rank, flat_index in enumerate(indices, start=1):
        y, x = divmod(int(flat_index), width)
        selected_values.append(float(metric[y, x]))
        rows.append(
            {
                "pixel_x": int(x * step),
                "pixel_y": int(y * step),
                "residual_l1": float(residual_l1[y, x]),
                "residual_l2": float(residual_l2[y, x]),
                "rendered_r": float(render_ds[y, x, 0]),
                "rendered_g": float(render_ds[y, x, 1]),
                "rendered_b": float(render_ds[y, x, 2]),
                "gt_r": float(gt_ds[y, x, 0]),
                "gt_g": float(gt_ds[y, x, 1]),
                "gt_b": float(gt_ds[y, x, 2]),
                "artifact_region_flag": True,
                "target_region_flag": "",
                "selection_rank": rank,
            }
        )
    entropy, concentration = _entropy([float(value) for value in flat])
    summary = {
        "image_width": int(render.shape[1]),
        "image_height": int(render.shape[0]),
        "evaluated_pixel_count": int(flat.size),
        "selected_pixel_count": len(rows),
        "mean_residual": float(np.mean(flat)),
        "median_residual": float(np.median(flat)),
        "max_residual": float(np.max(flat)),
        "top_pixel_residual_mean": _mean(selected_values),
        "residual_entropy": entropy,
        "residual_concentration": concentration,
    }
    return rows, summary


def _gaussian_meta(identity_rows: list[dict[str, str]], support_rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    output: dict[str, dict[str, str]] = {}
    for row in [*identity_rows, *support_rows]:
        gid = str(row.get("gaussian_id", ""))
        if not gid:
            continue
        current = output.setdefault(gid, {"gaussian_id": gid})
        for key in ("root_gaussian_id", "parent_gaussian_id"):
            if current.get(key) in ("", None) and row.get(key) not in ("", None):
                current[key] = str(row.get(key))
            elif key not in current:
                current[key] = str(row.get(key, ""))
    return output


def _view_gaussian_candidates(rows: list[dict[str, str]], top_k: int) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        view = str(row.get("view_name", ""))
        gid = str(row.get("gaussian_id", ""))
        if not view or not gid:
            continue
        grouped[view][gid] += max(_number(row.get("contribution_value"), 1.0), 1.0)
    return {
        view: [
            {"gaussian_id": gid, "weight": weight}
            for gid, weight in counter.most_common(top_k)
        ]
        for view, counter in grouped.items()
    }


def _aggregate_gaussian_rows(
    *,
    contribution_rows: list[dict[str, Any]],
    scene: str,
    condition: str,
    subset_name: str,
    evidence_quality: str,
    attribution_method: str,
) -> list[dict[str, Any]]:
    by_gid: dict[str, dict[str, Any]] = {}
    values_by_gid: dict[str, list[float]] = defaultdict(list)
    for row in contribution_rows:
        gid = str(row["gaussian_id"])
        group = str(row["view_group"])
        prefix = GROUP_PREFIX.get(group, "other_clean")
        item = by_gid.setdefault(
            gid,
            {
                "scene": scene,
                "condition": condition,
                "subset_name": subset_name,
                "gaussian_id": gid,
                "root_gaussian_id": row.get("root_gaussian_id", ""),
                "parent_gaussian_id": row.get("parent_gaussian_id", ""),
                "direct_corrupted_residual_weight": 0.0,
                "collateral_residual_weight": 0.0,
                "clean_prior_residual_weight": 0.0,
                "other_clean_residual_weight": 0.0,
                "direct_corrupted_pixel_count": 0,
                "collateral_pixel_count": 0,
                "clean_prior_pixel_count": 0,
                "other_clean_pixel_count": 0,
                "evidence_quality": evidence_quality,
                "attribution_method": attribution_method,
            },
        )
        weight = _number(row.get("residual_weighted_splat"))
        item[f"{prefix}_residual_weight"] += weight
        item[f"{prefix}_pixel_count"] += 1
        values_by_gid[gid].append(weight)
    for gid, item in by_gid.items():
        weights = [
            item["direct_corrupted_residual_weight"],
            item["collateral_residual_weight"],
            item["clean_prior_residual_weight"],
            item["other_clean_residual_weight"],
        ]
        total = sum(weights)
        entropy, concentration = _entropy(weights)
        dominant = max(
            [
                ("direct_corrupted", item["direct_corrupted_residual_weight"]),
                ("co_visible_collateral", item["collateral_residual_weight"]),
                ("clean_prior_demoted", item["clean_prior_residual_weight"]),
                ("other_clean", item["other_clean_residual_weight"]),
            ],
            key=lambda value: (value[1], value[0]),
        )[0] if total else ""
        item.update(
            {
                "corrupted_plus_collateral_residual_ratio": (weights[0] + weights[1]) / total if total else 0.0,
                "clean_prior_residual_ratio": weights[2] / total if total else 0.0,
                "dominant_residual_view_group": dominant,
                "residual_source_entropy": entropy,
                "residual_source_concentration": concentration,
                "mean_residual_weighted_splat": _mean(values_by_gid[gid]),
                "max_residual_weighted_splat": max(values_by_gid[gid]) if values_by_gid[gid] else 0.0,
            }
        )
    return list(by_gid.values())


def _group_rows(
    *,
    contribution_rows: list[dict[str, Any]],
    scene: str,
    condition: str,
    subset_name: str,
    evidence_quality: str,
) -> list[dict[str, Any]]:
    rows = []
    for group in GROUPS:
        group_rows = [row for row in contribution_rows if row.get("view_group") == group]
        weights = [_number(row.get("residual_weighted_splat")) for row in group_rows]
        entropy, concentration = _entropy(weights)
        gaussian_ids = {row.get("gaussian_id") for row in group_rows if row.get("gaussian_id")}
        view_names = {row.get("view_name") for row in group_rows if row.get("view_name")}
        rows.append(
            {
                "scene": scene,
                "condition": condition,
                "subset_name": subset_name,
                "view_group": group,
                "selected_view_count": len(view_names),
                "selected_pixel_count": len({(row.get("view_name"), row.get("pixel_x"), row.get("pixel_y")) for row in group_rows}),
                "unique_gaussian_count": len(gaussian_ids),
                "total_residual_weight": sum(weights),
                "mean_residual_weight": _mean(weights),
                "top_gaussian_count": len(gaussian_ids),
                "residual_entropy": entropy,
                "residual_concentration": concentration,
                "evidence_quality": evidence_quality,
            }
        )
    return rows


def _direct_collateral_overlap(
    gaussian_rows: list[dict[str, Any]],
    selected_views: list[dict[str, Any]],
    *,
    scene: str,
    condition: str,
    subset_name: str,
    evidence_quality: str,
) -> dict[str, Any]:
    direct = {row["gaussian_id"] for row in gaussian_rows if _number(row.get("direct_corrupted_residual_weight")) > 0}
    collateral = {row["gaussian_id"] for row in gaussian_rows if _number(row.get("collateral_residual_weight")) > 0}
    overlap = direct & collateral
    union = direct | collateral
    overlap_weight = sum(
        _number(row.get("direct_corrupted_residual_weight")) + _number(row.get("collateral_residual_weight"))
        for row in gaussian_rows
        if row["gaussian_id"] in overlap
    )
    supported = bool(overlap) and len(overlap) / len(union) < 0.95 if union else False
    return {
        "scene": scene,
        "condition": condition,
        "subset_name": subset_name,
        "direct_corrupted_view_names": ";".join(sorted(row["view_name"] for row in selected_views if row.get("view_group") == "direct_corrupted" and _truth(row.get("included")))),
        "collateral_view_names": ";".join(sorted(row["view_name"] for row in selected_views if row.get("view_group") == "co_visible_collateral" and _truth(row.get("included")))),
        "direct_residual_gaussian_count": len(direct),
        "collateral_residual_gaussian_count": len(collateral),
        "residual_overlap_gaussian_count": len(overlap),
        "residual_overlap_jaccard": len(overlap) / len(union) if union else 0.0,
        "residual_overlap_ratio_over_direct": len(overlap) / len(direct) if direct else 0.0,
        "residual_overlap_ratio_over_collateral": len(overlap) / len(collateral) if collateral else 0.0,
        "residual_weighted_overlap_sum": overlap_weight,
        "nontrivial_residual_overlap_supported": _bool_text(supported),
        "evidence_quality": evidence_quality,
        "notes": "" if supported else "no nontrivial residual overlap or overlap remains degenerate",
    }


def _train013_control(
    gaussian_rows: list[dict[str, Any]],
    selected_views: list[dict[str, Any]],
    *,
    scene: str,
    condition: str,
    subset_name: str,
    evidence_quality: str,
) -> dict[str, Any]:
    train_present = any(row.get("view_name") == "train_013" for row in selected_views)
    train_group = next((row.get("view_group", "") for row in selected_views if row.get("view_name") == "train_013"), "")
    train_ids = {row["gaussian_id"] for row in gaussian_rows if _number(row.get("clean_prior_residual_weight")) > 0}
    direct_collateral = {
        row["gaussian_id"]
        for row in gaussian_rows
        if _number(row.get("direct_corrupted_residual_weight")) > 0 or _number(row.get("collateral_residual_weight")) > 0
    }
    overlap = train_ids & direct_collateral
    ratio = len(overlap) / len(train_ids) if train_ids else 0.0
    train_weight = sum(_number(row.get("clean_prior_residual_weight")) for row in gaussian_rows if row["gaussian_id"] in train_ids)
    dc_weight = sum(
        _number(row.get("direct_corrupted_residual_weight")) + _number(row.get("collateral_residual_weight"))
        for row in gaussian_rows
        if row["gaussian_id"] in direct_collateral
    )
    supported = train_present and train_group == "clean_prior_demoted" and bool(train_ids) and ratio < 0.10
    reason = "clean_prior_low_residual_overlap" if supported else "train013 residual support overlaps direct/collateral or is unavailable"
    return {
        "scene": scene,
        "condition": condition,
        "subset_name": subset_name,
        "train013_present": _bool_text(train_present),
        "train013_view_group": train_group,
        "train013_residual_gaussian_count": len(train_ids),
        "direct_collateral_residual_gaussian_count": len(direct_collateral),
        "train013_overlap_with_direct_collateral_count": len(overlap),
        "train013_overlap_ratio": ratio,
        "train013_residual_weight_sum": train_weight,
        "direct_collateral_residual_weight_sum": dc_weight,
        "train013_residual_control_supported": _bool_text(supported),
        "evidence_quality": evidence_quality,
        "reason": reason,
        "notes": "",
    }


def _quality_rows(
    *,
    render_pairs: bool,
    gaussian_candidates: bool,
    evidence_quality: str,
    pr195_summary: dict[str, Any],
) -> list[dict[str, Any]]:
    criteria = [
        ("render/gt residuals available", render_pairs, "render/gt images discovered", "missing rendered or gt images", "run PR6 render/evaluation first"),
        ("Gaussian candidate IDs available", gaussian_candidates, "PR19.3 view attribution rows provide gaussian_id", "missing view-Gaussian candidates", "run PR19.3/PR19.2 exact logs first"),
        ("exact splat contribution available", evidence_quality == "exact_render_contribution", "per-pixel splat contributors", "exact renderer contribution not available", "add sparse renderer contribution capture"),
        ("residual-weighted attribution emitted", render_pairs and gaussian_candidates, "residual_weighted_splat columns written", "missing residual or Gaussian candidates", "collect both residual pixels and Gaussian candidates"),
        ("PR19.5 intervention blocker respected", pr195_summary.get("pr20_ready_for_intervention") is False, "pr20_ready_for_intervention remains false", "", "keep PR20.0 observation-only"),
    ]
    return [
        {
            "criterion": criterion,
            "passed": _bool_text(passed),
            "evidence": evidence,
            "blocker": "" if passed else blocker,
            "recommended_action": action,
        }
        for criterion, passed, evidence, blocker, action in criteria
    ]


def _artifact_rows(items: list[tuple[str, Path, bool, str]]) -> list[dict[str, Any]]:
    rows = []
    for relative, path, required, group in items:
        rows.append(
            {
                "relative_path": relative,
                "path": str(path),
                "exists": _bool_text(path.exists()),
                "file_type": "directory" if path.is_dir() else path.suffix.lstrip("."),
                "size_bytes": path.stat().st_size if path.is_file() else "",
                "required": _bool_text(required),
                "artifact_group": group,
            }
        )
    return rows


def write_artifact_manifest(output_dir: Path, pr193_dir: Path, pr195_dir: Path, run_dir: Path, data_root: Path) -> None:
    items: list[tuple[str, Path, bool, str]] = [
        ("pr193_dir", pr193_dir, True, "input"),
        ("pr195_dir", pr195_dir, True, "input"),
        ("run_dir", run_dir, True, "input"),
        ("data_root", data_root, False, "input"),
    ]
    items.extend((name, output_dir / name, True, "output_pr200") for name in PR200_OUTPUT_FILES)
    manifest = output_dir / "artifact_manifest.csv"
    write_csv_rows(manifest, _artifact_rows(items), MANIFEST_FIELDS)
    write_csv_rows(manifest, _artifact_rows(items), MANIFEST_FIELDS)


def _write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# PR20.0 Sparse Render Attribution",
        "",
        "PR20.0 is observation only. It does not reject views, reweight losses, suppress updates, gate densification, or modify training/rendering behavior.",
        "",
        "## Summary",
        f"- Attribution method: `{summary.get('attribution_method')}`",
        f"- Evidence quality: `{summary.get('evidence_quality')}`",
        f"- Selected views: `{summary.get('selected_view_count')}`",
        f"- Selected pixels: `{summary.get('selected_pixel_count')}`",
        f"- Pixel-Gaussian rows: `{summary.get('total_pixel_gaussian_contribution_rows')}`",
        f"- Direct/collateral residual overlap supported: `{summary.get('direct_collateral_residual_overlap_supported')}`",
        f"- Train013 residual control supported: `{summary.get('train013_residual_control_supported')}`",
        f"- PR20 ready for intervention: `{summary.get('pr20_ready_for_intervention')}`",
        "",
        "Approximate proxy attribution must not be interpreted as exact renderer contribution.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def build_sparse_render_attribution(
    *,
    pr193_dir: Path,
    pr195_dir: Path,
    run_dir: Path,
    data_root: Path,
    output_dir: Path,
    scene: str,
    condition: str,
    subset_name: str,
    views: list[str] | None = None,
    max_views: int = 8,
    top_pixels: int = 512,
    top_gaussians_per_pixel: int = 16,
    residual_metric: str = "l1",
    artifact_mask_mode: str = "top_residual",
    external_mask_csv: Path | None = None,
    downsample: int = 1,
    write_markdown: bool = False,
    strict: bool = False,
    allow_missing: bool = False,
) -> tuple[dict[str, Any], int]:
    del external_mask_csv, write_markdown
    output_dir.mkdir(parents=True, exist_ok=True)
    missing_rows: list[dict[str, Any]] = []
    for name in PR193_REQUIRED_FILES:
        path = pr193_dir / name
        if not path.exists():
            missing_rows.append({"input_name": name, "path": str(path), "exists": "false", "required": "true", "details": "missing PR19.3 input"})
    for name in PR195_REQUIRED_FILES:
        path = pr195_dir / name
        if not path.exists():
            missing_rows.append({"input_name": name, "path": str(path), "exists": "false", "required": "true", "details": "missing PR19.5 input"})

    view_group_rows = load_csv_rows(pr193_dir / "pr193_view_group_map.csv")
    identity_rows = load_csv_rows(pr193_dir / "gaussian_identity_table_grouped.csv")
    attribution_rows = load_csv_rows(pr193_dir / "view_gaussian_event_attribution_grouped.csv")
    support_rows = load_csv_rows(pr193_dir / "gaussian_support_summary_grouped.csv")
    pr195_summary = load_json(pr195_dir / "pr195_attribution_semantics_summary.json")
    selected = _select_views(view_group_rows, views, max_views)
    meta = _gaussian_meta(identity_rows, support_rows)
    candidates_by_view = _view_gaussian_candidates(attribution_rows, top_gaussians_per_pixel)
    evidence_quality = "approximate_projected_gaussian" if candidates_by_view else "residual_only_no_gaussian"
    attribution_method = "view_event_weighted_gaussian_proxy" if candidates_by_view else "residual_only_no_gaussian"
    selected_rows: list[dict[str, Any]] = []
    view_summaries: list[dict[str, Any]] = []
    sparse_pixels: list[dict[str, Any]] = []
    contribution_rows: list[dict[str, Any]] = []
    warnings: list[str] = []

    for row in selected:
        view_name = str(row.get("view_name", ""))
        view_group = str(row.get("view_group", "other_clean"))
        render_path, gt_path = _find_render_pair(run_dir, view_name)
        included = render_path is not None and gt_path is not None
        selected_rows.append(
            {
                "scene": scene,
                "condition": condition,
                "subset_name": subset_name,
                "view_name": view_name,
                "view_group": view_group,
                "was_corrupted": row.get("was_corrupted", ""),
                "selection_reason": row.get("selection_reason", ""),
                "image_path": str(render_path or ""),
                "camera_id": _view_index(view_name),
                "included": _bool_text(included),
            }
        )
        if not included:
            warnings.append(f"missing render/gt pair for {view_name}")
            continue
        render = _load_rgb(render_path)
        gt = _load_rgb(gt_path)
        pixel_rows, residual_summary = _top_residual_pixels(
            render,
            gt,
            top_pixels=top_pixels,
            residual_metric=residual_metric,
            downsample=downsample,
        )
        view_summaries.append(
            {
                "scene": scene,
                "condition": condition,
                "subset_name": subset_name,
                "view_name": view_name,
                "view_group": view_group,
                "residual_metric": residual_metric,
                **residual_summary,
                "artifact_mask_mode": artifact_mask_mode,
                "evidence_quality": evidence_quality,
            }
        )
        candidates = candidates_by_view.get(view_name, [])
        total_candidate_weight = sum(_number(item.get("weight")) for item in candidates) or 1.0
        for pixel in pixel_rows:
            pixel_out = {
                "scene": scene,
                "condition": condition,
                "subset_name": subset_name,
                "view_name": view_name,
                "view_group": view_group,
                **pixel,
                "artifact_region_flag": _bool_text(bool(pixel.get("artifact_region_flag"))),
                "evidence_quality": evidence_quality,
            }
            sparse_pixels.append(pixel_out)
            for rank, candidate in enumerate(candidates, start=1):
                gid = str(candidate["gaussian_id"])
                splat_weight = _number(candidate["weight"]) / total_candidate_weight
                residual_l1 = _number(pixel["residual_l1"])
                contribution_rows.append(
                    {
                        "scene": scene,
                        "condition": condition,
                        "subset_name": subset_name,
                        "view_name": view_name,
                        "view_group": view_group,
                        "pixel_x": pixel["pixel_x"],
                        "pixel_y": pixel["pixel_y"],
                        "gaussian_id": gid,
                        "root_gaussian_id": meta.get(gid, {}).get("root_gaussian_id", gid),
                        "parent_gaussian_id": meta.get(gid, {}).get("parent_gaussian_id", ""),
                        "contribution_rank": rank,
                        "splat_weight": splat_weight,
                        "alpha_contribution": splat_weight,
                        "color_contribution_r": pixel["rendered_r"] * splat_weight,
                        "color_contribution_g": pixel["rendered_g"] * splat_weight,
                        "color_contribution_b": pixel["rendered_b"] * splat_weight,
                        "residual_l1": residual_l1,
                        "residual_weighted_splat": residual_l1 * splat_weight,
                        "residual_weighted_alpha": residual_l1 * splat_weight,
                        "evidence_quality": evidence_quality,
                        "attribution_method": attribution_method,
                        "warnings": "approximate event-weighted Gaussian proxy; not exact render contribution",
                    }
                )

    if not contribution_rows and sparse_pixels:
        warnings.append("residual pixels available but no Gaussian candidates; contribution table is residual-only")
    gaussian_rows = _aggregate_gaussian_rows(
        contribution_rows=contribution_rows,
        scene=scene,
        condition=condition,
        subset_name=subset_name,
        evidence_quality=evidence_quality,
        attribution_method=attribution_method,
    )
    group_rows = _group_rows(
        contribution_rows=contribution_rows,
        scene=scene,
        condition=condition,
        subset_name=subset_name,
        evidence_quality=evidence_quality,
    )
    direct_collateral = _direct_collateral_overlap(
        gaussian_rows,
        selected_rows,
        scene=scene,
        condition=condition,
        subset_name=subset_name,
        evidence_quality=evidence_quality,
    )
    train013 = _train013_control(
        gaussian_rows,
        selected_rows,
        scene=scene,
        condition=condition,
        subset_name=subset_name,
        evidence_quality=evidence_quality,
    )
    quality_rows = _quality_rows(
        render_pairs=bool(sparse_pixels),
        gaussian_candidates=bool(candidates_by_view),
        evidence_quality=evidence_quality,
        pr195_summary=pr195_summary,
    )
    summary = {
        "schema_name": "viewtrust.pr200.sparse_render_attribution.summary",
        "schema_version": 1,
        "created_at_utc": _utc_now(),
        "scene": scene,
        "condition": condition,
        "subset_name": subset_name,
        "pr193_dir": str(pr193_dir),
        "pr195_dir": str(pr195_dir),
        "run_dir": str(run_dir),
        "data_root": str(data_root),
        "output_dir": str(output_dir),
        "observation_only": True,
        "training_intervention": False,
        "defense_enabled": False,
        "view_rejection_enabled": False,
        "densification_gating_enabled": False,
        "third_party_modified": False,
        "training_behavior_modified": False,
        "rendering_behavior_modified_for_training": False,
        "selected_view_count": sum(1 for row in selected_rows if _truth(row.get("included"))),
        "selected_pixel_count": len(sparse_pixels),
        "total_pixel_gaussian_contribution_rows": len(contribution_rows),
        "attribution_method": attribution_method,
        "evidence_quality": evidence_quality,
        "residual_metric": residual_metric,
        "artifact_mask_mode": artifact_mask_mode,
        "top_pixels": top_pixels,
        "top_gaussians_per_pixel": top_gaussians_per_pixel,
        "direct_collateral_residual_overlap_supported": _truth(direct_collateral.get("nontrivial_residual_overlap_supported")),
        "train013_residual_control_supported": _truth(train013.get("train013_residual_control_supported")),
        "pr20_ready_for_intervention": False,
        "warnings": warnings,
    }

    write_json(output_dir / "pr200_sparse_render_attribution_summary.json", summary)
    write_csv_rows(output_dir / "pr200_selected_views.csv", selected_rows, SELECTED_VIEW_FIELDS)
    write_csv_rows(output_dir / "pr200_view_residual_summary.csv", view_summaries, VIEW_RESIDUAL_FIELDS)
    write_csv_rows(output_dir / "pr200_sparse_pixel_residuals.csv", sparse_pixels, SPARSE_PIXEL_FIELDS)
    write_csv_rows(output_dir / "pr200_pixel_gaussian_contributions.csv", contribution_rows, PIXEL_GAUSSIAN_FIELDS)
    write_csv_rows(output_dir / "pr200_gaussian_residual_attribution.csv", gaussian_rows, GAUSSIAN_ATTR_FIELDS)
    write_csv_rows(output_dir / "pr200_view_group_residual_attribution.csv", group_rows, GROUP_ATTR_FIELDS)
    write_csv_rows(output_dir / "pr200_direct_collateral_residual_overlap.csv", [direct_collateral], DIRECT_COLLATERAL_FIELDS)
    write_csv_rows(output_dir / "pr200_train013_residual_control.csv", [train013], TRAIN013_FIELDS)
    write_csv_rows(output_dir / "pr200_attribution_quality_audit.csv", quality_rows, QUALITY_FIELDS)
    write_csv_rows(output_dir / "pr200_missing_inputs.csv", missing_rows, MISSING_FIELDS)
    _write_report(output_dir / "pr200_report.md", summary)
    write_artifact_manifest(output_dir, pr193_dir, pr195_dir, run_dir, data_root)

    missing_required = [name for name in PR200_OUTPUT_FILES if not (output_dir / name).is_file()]
    if missing_required:
        raise RuntimeError(f"missing PR20.0 outputs: {missing_required}")
    if missing_rows and strict and not allow_missing:
        return summary, 1
    if missing_rows and not allow_missing:
        return summary, 1
    return summary, 0
