"""Offline PR20.1 proxy degeneracy diagnostics for PR20.0 outputs."""

from __future__ import annotations

import csv
import hashlib
import json
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from viewtrust.analysis.clean_prior_normalization import load_csv_rows, load_json, normalize_bool
from viewtrust.analysis.offline_signals import safe_float


REQUIRED_PR200_FILES = [
    "pr200_sparse_render_attribution_summary.json",
    "pr200_selected_views.csv",
    "pr200_view_residual_summary.csv",
    "pr200_pixel_gaussian_contributions.csv",
    "pr200_gaussian_residual_attribution.csv",
    "pr200_view_group_residual_attribution.csv",
    "pr200_direct_collateral_residual_overlap.csv",
    "pr200_train013_residual_control.csv",
    "pr200_attribution_quality_audit.csv",
]

PR201_OUTPUT_FILES = [
    "pr201_proxy_degeneracy_summary.json",
    "pr201_run_summary.csv",
    "pr201_pixel_candidate_reuse.csv",
    "pr201_view_candidate_pool.csv",
    "pr201_view_candidate_pool_overlap.csv",
    "pr201_candidate_weight_uniformity.csv",
    "pr201_group_candidate_pool_audit.csv",
    "pr201_direct_collateral_degeneracy.csv",
    "pr201_train013_proxy_control_audit.csv",
    "pr201_proxy_failure_cases.csv",
    "pr201_recommendations.json",
    "pr201_missing_inputs.csv",
    "pr201_report.md",
    "artifact_manifest.csv",
]

RUN_SUMMARY_FIELDS = [
    "run_id",
    "scene",
    "condition",
    "subset_name",
    "pr200_dir",
    "selected_view_count",
    "selected_pixel_count",
    "total_pixel_gaussian_contribution_rows",
    "evidence_quality",
    "attribution_method",
    "proxy_degeneracy_confirmed",
    "pixel_candidate_reuse_rate_mean",
    "candidate_weight_uniformity_rate",
    "splat_weight_mean",
    "splat_weight_std",
    "splat_weight_cv",
    "alpha_contribution_mean",
    "alpha_contribution_std",
    "alpha_contribution_cv",
    "direct_collateral_jaccard",
    "direct_collateral_degenerate",
    "train013_overlap_ratio",
    "train013_control_supported_in_pr200",
    "train013_control_interpretation",
    "warnings",
]

PIXEL_REUSE_FIELDS = [
    "scene",
    "condition",
    "subset_name",
    "view_name",
    "view_group",
    "selected_pixel_count",
    "unique_pixel_candidate_sets",
    "dominant_candidate_set_hash",
    "dominant_candidate_set_pixel_count",
    "dominant_candidate_set_pixel_ratio",
    "min_candidate_set_size",
    "max_candidate_set_size",
    "mean_candidate_set_size",
    "pixel_candidate_reuse_degenerate",
    "evidence_quality",
    "notes",
]

VIEW_POOL_FIELDS = [
    "scene",
    "condition",
    "subset_name",
    "view_name",
    "view_group",
    "candidate_pool_hash",
    "candidate_gaussian_ids_semicolon",
    "candidate_count",
    "selected_pixel_count",
    "dominant_candidate_set_pixel_ratio",
    "mean_splat_weight",
    "std_splat_weight",
    "cv_splat_weight",
    "mean_alpha_contribution",
    "std_alpha_contribution",
    "cv_alpha_contribution",
    "evidence_quality",
    "attribution_method",
]

PAIR_OVERLAP_FIELDS = [
    "scene",
    "condition",
    "subset_name",
    "view_a",
    "group_a",
    "view_b",
    "group_b",
    "candidate_count_a",
    "candidate_count_b",
    "overlap_count",
    "jaccard",
    "overlap_ratio_over_a",
    "overlap_ratio_over_b",
    "same_candidate_pool",
    "evidence_quality",
]

WEIGHT_UNIFORMITY_FIELDS = [
    "scene",
    "condition",
    "subset_name",
    "view_name",
    "view_group",
    "pixel_x",
    "pixel_y",
    "candidate_count",
    "mean_splat_weight",
    "std_splat_weight",
    "cv_splat_weight",
    "min_splat_weight",
    "max_splat_weight",
    "all_splat_weights_equal",
    "mean_alpha_contribution",
    "std_alpha_contribution",
    "cv_alpha_contribution",
    "all_alpha_contributions_equal",
    "residual_l1",
    "evidence_quality",
]

GROUP_AUDIT_FIELDS = [
    "scene",
    "condition",
    "subset_name",
    "view_group",
    "selected_view_count",
    "selected_pixel_count",
    "unique_candidate_pool_count",
    "dominant_candidate_pool_hash",
    "dominant_candidate_pool_view_count",
    "dominant_candidate_pool_pixel_count",
    "unique_gaussian_count",
    "total_residual_weight",
    "mean_residual_weight",
    "candidate_pool_reuse_rate",
    "weight_uniformity_rate",
    "evidence_quality",
    "proxy_group_interpretation",
]

DIRECT_COLLATERAL_FIELDS = [
    "scene",
    "condition",
    "subset_name",
    "direct_view_names",
    "collateral_view_names",
    "direct_candidate_pool_hashes",
    "collateral_candidate_pool_hashes",
    "direct_unique_gaussian_count",
    "collateral_unique_gaussian_count",
    "overlap_count",
    "jaccard",
    "degenerate_overlap",
    "pr200_nontrivial_residual_overlap_supported",
    "explanation",
    "evidence_quality",
]

TRAIN013_FIELDS = [
    "scene",
    "condition",
    "subset_name",
    "train013_present",
    "train013_view_group",
    "train013_candidate_pool_hash",
    "direct_collateral_candidate_pool_hashes",
    "train013_candidate_count",
    "direct_collateral_candidate_count",
    "overlap_count",
    "overlap_ratio",
    "pr200_train013_residual_control_supported",
    "proxy_pool_separation_detected",
    "interpretation",
    "evidence_quality",
]

FAILURE_FIELDS = [
    "scene",
    "condition",
    "subset_name",
    "failure_type",
    "view_name",
    "view_group",
    "related_view_name",
    "related_view_group",
    "metric_name",
    "metric_value",
    "threshold",
    "evidence_quality",
    "explanation",
]

MISSING_FIELDS = ["pr200_dir", "missing_file", "severity", "action"]
MANIFEST_FIELDS = ["relative_path", "path", "exists", "file_type", "size_bytes", "required", "artifact_group"]


@dataclass(frozen=True)
class PR201Config:
    top_k: int = 16
    weight_uniformity_tol: float = 1e-9
    reuse_threshold: float = 0.95
    degenerate_jaccard_threshold: float = 0.95
    max_report_rows: int = 50


@dataclass
class PR200Inputs:
    pr200_dir: Path
    run_id: str
    summary: dict[str, Any]
    selected_views: list[dict[str, str]]
    view_residual_summary: list[dict[str, str]]
    pixel_gaussian_contributions: list[dict[str, str]]
    gaussian_residual_attribution: list[dict[str, str]]
    view_group_residual_attribution: list[dict[str, str]]
    direct_collateral_overlap: list[dict[str, str]]
    train013_control: list[dict[str, str]]
    quality_audit: list[dict[str, str]]
    missing_files: list[str]


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


def _std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return statistics.pstdev(values)


def _cv(values: list[float]) -> float:
    mean = _mean(values)
    return _std(values) / mean if mean else 0.0


def _hash_ids(ids: list[str]) -> str:
    payload = ";".join(ids)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def _scene(inputs: PR200Inputs, override: str | None = None) -> str:
    return override or str(inputs.summary.get("scene") or "")


def _condition(inputs: PR200Inputs, override: str | None = None) -> str:
    return override or str(inputs.summary.get("condition") or "")


def _subset(inputs: PR200Inputs, override: str | None = None) -> str:
    return override or str(inputs.summary.get("subset_name") or "")


def load_pr200_inputs(pr200_dir: Path) -> PR200Inputs:
    missing = [name for name in REQUIRED_PR200_FILES if not (pr200_dir / name).is_file()]
    return PR200Inputs(
        pr200_dir=pr200_dir,
        run_id=pr200_dir.name,
        summary=load_json(pr200_dir / "pr200_sparse_render_attribution_summary.json"),
        selected_views=load_csv_rows(pr200_dir / "pr200_selected_views.csv"),
        view_residual_summary=load_csv_rows(pr200_dir / "pr200_view_residual_summary.csv"),
        pixel_gaussian_contributions=load_csv_rows(pr200_dir / "pr200_pixel_gaussian_contributions.csv"),
        gaussian_residual_attribution=load_csv_rows(pr200_dir / "pr200_gaussian_residual_attribution.csv"),
        view_group_residual_attribution=load_csv_rows(pr200_dir / "pr200_view_group_residual_attribution.csv"),
        direct_collateral_overlap=load_csv_rows(pr200_dir / "pr200_direct_collateral_residual_overlap.csv"),
        train013_control=load_csv_rows(pr200_dir / "pr200_train013_residual_control.csv"),
        quality_audit=load_csv_rows(pr200_dir / "pr200_attribution_quality_audit.csv"),
        missing_files=missing,
    )


def _pixel_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row.get("view_name", "")),
        str(row.get("view_group", "")),
        str(row.get("pixel_x", "")),
        str(row.get("pixel_y", "")),
    )


def _candidate_set(row_group: list[dict[str, str]]) -> list[str]:
    ordered = sorted(
        row_group,
        key=lambda row: (
            int(_number(row.get("contribution_rank"), 10**9)),
            str(row.get("gaussian_id", "")),
        ),
    )
    return [str(row.get("gaussian_id", "")) for row in ordered if row.get("gaussian_id")]


def _pixel_groups(rows: list[dict[str, str]]) -> dict[tuple[str, str, str, str], list[dict[str, str]]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[_pixel_key(row)].append(row)
    return grouped


def analyze_pixel_candidate_reuse(inputs: PR200Inputs, config: PR201Config, *, scene: str | None = None, condition: str | None = None, subset_name: str | None = None) -> list[dict[str, Any]]:
    by_view: dict[tuple[str, str], list[list[str]]] = defaultdict(list)
    for (view_name, view_group, _, _), rows in _pixel_groups(inputs.pixel_gaussian_contributions).items():
        by_view[(view_name, view_group)].append(_candidate_set(rows))
    output = []
    for (view_name, view_group), candidate_sets in sorted(by_view.items()):
        hashes = [_hash_ids(ids) for ids in candidate_sets]
        counts = Counter(hashes)
        dominant_hash, dominant_count = counts.most_common(1)[0]
        sizes = [len(ids) for ids in candidate_sets]
        ratio = dominant_count / len(candidate_sets) if candidate_sets else 0.0
        output.append(
            {
                "scene": _scene(inputs, scene),
                "condition": _condition(inputs, condition),
                "subset_name": _subset(inputs, subset_name),
                "view_name": view_name,
                "view_group": view_group,
                "selected_pixel_count": len(candidate_sets),
                "unique_pixel_candidate_sets": len(counts),
                "dominant_candidate_set_hash": dominant_hash,
                "dominant_candidate_set_pixel_count": dominant_count,
                "dominant_candidate_set_pixel_ratio": ratio,
                "min_candidate_set_size": min(sizes) if sizes else 0,
                "max_candidate_set_size": max(sizes) if sizes else 0,
                "mean_candidate_set_size": _mean([float(size) for size in sizes]),
                "pixel_candidate_reuse_degenerate": _bool_text(ratio >= config.reuse_threshold),
                "evidence_quality": inputs.summary.get("evidence_quality", ""),
                "notes": "" if ratio < config.reuse_threshold else "same proxy candidate set reused across most selected pixels",
            }
        )
    return output


def analyze_candidate_weight_uniformity(inputs: PR200Inputs, config: PR201Config, *, scene: str | None = None, condition: str | None = None, subset_name: str | None = None) -> list[dict[str, Any]]:
    output = []
    for (view_name, view_group, pixel_x, pixel_y), rows in sorted(_pixel_groups(inputs.pixel_gaussian_contributions).items()):
        splat = [_number(row.get("splat_weight")) for row in rows]
        alpha = [_number(row.get("alpha_contribution")) for row in rows]
        residual = _number(rows[0].get("residual_l1")) if rows else 0.0
        output.append(
            {
                "scene": _scene(inputs, scene),
                "condition": _condition(inputs, condition),
                "subset_name": _subset(inputs, subset_name),
                "view_name": view_name,
                "view_group": view_group,
                "pixel_x": pixel_x,
                "pixel_y": pixel_y,
                "candidate_count": len(rows),
                "mean_splat_weight": _mean(splat),
                "std_splat_weight": _std(splat),
                "cv_splat_weight": _cv(splat),
                "min_splat_weight": min(splat) if splat else 0.0,
                "max_splat_weight": max(splat) if splat else 0.0,
                "all_splat_weights_equal": _bool_text((max(splat) - min(splat) <= config.weight_uniformity_tol) if splat else False),
                "mean_alpha_contribution": _mean(alpha),
                "std_alpha_contribution": _std(alpha),
                "cv_alpha_contribution": _cv(alpha),
                "all_alpha_contributions_equal": _bool_text((max(alpha) - min(alpha) <= config.weight_uniformity_tol) if alpha else False),
                "residual_l1": residual,
                "evidence_quality": inputs.summary.get("evidence_quality", ""),
            }
        )
    return output


def analyze_view_candidate_pools(
    inputs: PR200Inputs,
    config: PR201Config,
    pixel_reuse_rows: list[dict[str, Any]],
    uniformity_rows: list[dict[str, Any]],
    *,
    scene: str | None = None,
    condition: str | None = None,
    subset_name: str | None = None,
) -> list[dict[str, Any]]:
    pixel_sets: dict[tuple[str, str], list[list[str]]] = defaultdict(list)
    pixel_weights: dict[tuple[str, str], list[float]] = defaultdict(list)
    pixel_alpha: dict[tuple[str, str], list[float]] = defaultdict(list)
    for (view_name, view_group, _, _), rows in _pixel_groups(inputs.pixel_gaussian_contributions).items():
        key = (view_name, view_group)
        pixel_sets[key].append(_candidate_set(rows))
        pixel_weights[key].extend(_number(row.get("splat_weight")) for row in rows)
        pixel_alpha[key].extend(_number(row.get("alpha_contribution")) for row in rows)
    reuse_by_view = {(row["view_name"], row["view_group"]): row for row in pixel_reuse_rows}
    output = []
    for key, candidate_sets in sorted(pixel_sets.items()):
        view_name, view_group = key
        dominant_ids = Counter(_hash_ids(ids) for ids in candidate_sets).most_common(1)[0][0]
        candidate_ids = next(ids for ids in candidate_sets if _hash_ids(ids) == dominant_ids)
        splat = pixel_weights[key]
        alpha = pixel_alpha[key]
        reuse = reuse_by_view.get(key, {})
        output.append(
            {
                "scene": _scene(inputs, scene),
                "condition": _condition(inputs, condition),
                "subset_name": _subset(inputs, subset_name),
                "view_name": view_name,
                "view_group": view_group,
                "candidate_pool_hash": dominant_ids,
                "candidate_gaussian_ids_semicolon": ";".join(candidate_ids),
                "candidate_count": len(candidate_ids),
                "selected_pixel_count": len(candidate_sets),
                "dominant_candidate_set_pixel_ratio": reuse.get("dominant_candidate_set_pixel_ratio", ""),
                "mean_splat_weight": _mean(splat),
                "std_splat_weight": _std(splat),
                "cv_splat_weight": _cv(splat),
                "mean_alpha_contribution": _mean(alpha),
                "std_alpha_contribution": _std(alpha),
                "cv_alpha_contribution": _cv(alpha),
                "evidence_quality": inputs.summary.get("evidence_quality", ""),
                "attribution_method": inputs.summary.get("attribution_method", ""),
            }
        )
    return output


def _ids_from_pool(row: dict[str, Any]) -> set[str]:
    return {item for item in str(row.get("candidate_gaussian_ids_semicolon", "")).split(";") if item}


def analyze_view_candidate_pool_overlap(view_pools: list[dict[str, Any]], config: PR201Config) -> list[dict[str, Any]]:
    del config
    output = []
    ordered = sorted(view_pools, key=lambda row: (row.get("scene", ""), row.get("condition", ""), row.get("subset_name", ""), row.get("view_name", "")))
    for index, left in enumerate(ordered):
        left_ids = _ids_from_pool(left)
        for right in ordered[index + 1 :]:
            if (left.get("scene"), left.get("condition"), left.get("subset_name")) != (
                right.get("scene"),
                right.get("condition"),
                right.get("subset_name"),
            ):
                continue
            right_ids = _ids_from_pool(right)
            overlap = left_ids & right_ids
            union = left_ids | right_ids
            output.append(
                {
                    "scene": left.get("scene", ""),
                    "condition": left.get("condition", ""),
                    "subset_name": left.get("subset_name", ""),
                    "view_a": left.get("view_name", ""),
                    "group_a": left.get("view_group", ""),
                    "view_b": right.get("view_name", ""),
                    "group_b": right.get("view_group", ""),
                    "candidate_count_a": len(left_ids),
                    "candidate_count_b": len(right_ids),
                    "overlap_count": len(overlap),
                    "jaccard": len(overlap) / len(union) if union else 0.0,
                    "overlap_ratio_over_a": len(overlap) / len(left_ids) if left_ids else 0.0,
                    "overlap_ratio_over_b": len(overlap) / len(right_ids) if right_ids else 0.0,
                    "same_candidate_pool": _bool_text(left.get("candidate_pool_hash") == right.get("candidate_pool_hash")),
                    "evidence_quality": left.get("evidence_quality", ""),
                }
            )
    return output


def analyze_group_candidate_pool_audit(
    inputs: PR200Inputs,
    view_pools: list[dict[str, Any]],
    uniformity_rows: list[dict[str, Any]],
    config: PR201Config,
) -> list[dict[str, Any]]:
    del config
    weights_by_group = {
        row.get("view_group", ""): _number(row.get("total_residual_weight"))
        for row in inputs.view_group_residual_attribution
    }
    output = []
    for group in ["direct_corrupted", "co_visible_collateral", "clean_prior_demoted", "other_clean"]:
        rows = [row for row in view_pools if row.get("view_group") == group]
        uniform = [row for row in uniformity_rows if row.get("view_group") == group]
        pool_counts = Counter(row.get("candidate_pool_hash", "") for row in rows if row.get("candidate_pool_hash"))
        dominant_hash, dominant_view_count = pool_counts.most_common(1)[0] if pool_counts else ("", 0)
        dominant_pixel_count = sum(int(_number(row.get("selected_pixel_count"))) for row in rows if row.get("candidate_pool_hash") == dominant_hash)
        gaussian_ids = set()
        for row in rows:
            gaussian_ids.update(_ids_from_pool(row))
        selected_pixel_count = sum(int(_number(row.get("selected_pixel_count"))) for row in rows)
        uniform_rate = (
            sum(1 for row in uniform if _truth(row.get("all_splat_weights_equal"))) / len(uniform)
            if uniform
            else 0.0
        )
        reuse_rate = dominant_view_count / len(rows) if rows else 0.0
        interpretation = "not selected"
        if group == "direct_corrupted" and rows:
            interpretation = "shared proxy candidate pool across direct views" if reuse_rate >= 0.5 else "multiple direct proxy pools"
        elif group == "co_visible_collateral" and rows:
            interpretation = "collateral proxy pool available; compare with direct overlap"
        elif group == "clean_prior_demoted" and rows:
            interpretation = "separate clean-prior proxy pool" if reuse_rate >= 0.5 else "mixed clean-prior proxy pools"
        elif group == "other_clean" and rows:
            interpretation = "other clean proxy candidates selected"
        output.append(
            {
                "scene": rows[0].get("scene", _scene(inputs)) if rows else _scene(inputs),
                "condition": rows[0].get("condition", _condition(inputs)) if rows else _condition(inputs),
                "subset_name": rows[0].get("subset_name", _subset(inputs)) if rows else _subset(inputs),
                "view_group": group,
                "selected_view_count": len(rows),
                "selected_pixel_count": selected_pixel_count,
                "unique_candidate_pool_count": len(pool_counts),
                "dominant_candidate_pool_hash": dominant_hash,
                "dominant_candidate_pool_view_count": dominant_view_count,
                "dominant_candidate_pool_pixel_count": dominant_pixel_count,
                "unique_gaussian_count": len(gaussian_ids),
                "total_residual_weight": weights_by_group.get(group, 0.0),
                "mean_residual_weight": weights_by_group.get(group, 0.0) / selected_pixel_count if selected_pixel_count else 0.0,
                "candidate_pool_reuse_rate": reuse_rate,
                "weight_uniformity_rate": uniform_rate,
                "evidence_quality": inputs.summary.get("evidence_quality", ""),
                "proxy_group_interpretation": interpretation,
            }
        )
    return output


def analyze_direct_collateral_degeneracy(
    inputs: PR200Inputs,
    view_pools: list[dict[str, Any]],
    overlaps: list[dict[str, Any]],
    pixel_reuse_rows: list[dict[str, Any]],
    uniformity_rows: list[dict[str, Any]],
    config: PR201Config,
) -> list[dict[str, Any]]:
    del overlaps
    direct = [row for row in view_pools if row.get("view_group") == "direct_corrupted"]
    collateral = [row for row in view_pools if row.get("view_group") == "co_visible_collateral"]
    direct_ids: set[str] = set()
    collateral_ids: set[str] = set()
    for row in direct:
        direct_ids.update(_ids_from_pool(row))
    for row in collateral:
        collateral_ids.update(_ids_from_pool(row))
    overlap = direct_ids & collateral_ids
    union = direct_ids | collateral_ids
    jaccard = len(overlap) / len(union) if union else 0.0
    reuse_high = _mean([_number(row.get("dominant_candidate_set_pixel_ratio")) for row in pixel_reuse_rows if row.get("view_group") in {"direct_corrupted", "co_visible_collateral"}]) >= config.reuse_threshold
    uniform_high = (
        sum(1 for row in uniformity_rows if row.get("view_group") in {"direct_corrupted", "co_visible_collateral"} and _truth(row.get("all_splat_weights_equal")))
        / max(1, sum(1 for row in uniformity_rows if row.get("view_group") in {"direct_corrupted", "co_visible_collateral"}))
    ) >= config.reuse_threshold
    degenerate = (
        jaccard >= config.degenerate_jaccard_threshold
        and inputs.summary.get("evidence_quality") == "approximate_projected_gaussian"
        and reuse_high
        and uniform_high
    )
    pr200_overlap = inputs.direct_collateral_overlap[0] if inputs.direct_collateral_overlap else {}
    return [
        {
            "scene": _scene(inputs),
            "condition": _condition(inputs),
            "subset_name": _subset(inputs),
            "direct_view_names": ";".join(sorted(row.get("view_name", "") for row in direct)),
            "collateral_view_names": ";".join(sorted(row.get("view_name", "") for row in collateral)),
            "direct_candidate_pool_hashes": ";".join(sorted({str(row.get("candidate_pool_hash", "")) for row in direct if row.get("candidate_pool_hash")})),
            "collateral_candidate_pool_hashes": ";".join(sorted({str(row.get("candidate_pool_hash", "")) for row in collateral if row.get("candidate_pool_hash")})),
            "direct_unique_gaussian_count": len(direct_ids),
            "collateral_unique_gaussian_count": len(collateral_ids),
            "overlap_count": len(overlap),
            "jaccard": jaccard,
            "degenerate_overlap": _bool_text(degenerate),
            "pr200_nontrivial_residual_overlap_supported": pr200_overlap.get("nontrivial_residual_overlap_supported", ""),
            "explanation": "Direct/collateral overlap is explained by reused view-level proxy candidate pools, not exact per-pixel render contributors." if degenerate else "Direct/collateral proxy overlap is not fully degenerate by configured thresholds.",
            "evidence_quality": inputs.summary.get("evidence_quality", ""),
        }
    ]


def analyze_train013_proxy_control(
    inputs: PR200Inputs,
    view_pools: list[dict[str, Any]],
    config: PR201Config,
) -> list[dict[str, Any]]:
    del config
    train = next((row for row in view_pools if row.get("view_name") == "train_013"), None)
    direct_collateral = [row for row in view_pools if row.get("view_group") in {"direct_corrupted", "co_visible_collateral"}]
    train_ids = _ids_from_pool(train) if train else set()
    dc_ids: set[str] = set()
    for row in direct_collateral:
        dc_ids.update(_ids_from_pool(row))
    overlap = train_ids & dc_ids
    pr200_train = inputs.train013_control[0] if inputs.train013_control else {}
    proxy_sep = bool(train) and bool(train_ids) and len(overlap) == 0 and _truth(pr200_train.get("train013_residual_control_supported"))
    return [
        {
            "scene": _scene(inputs),
            "condition": _condition(inputs),
            "subset_name": _subset(inputs),
            "train013_present": _bool_text(train is not None),
            "train013_view_group": train.get("view_group", "") if train else "",
            "train013_candidate_pool_hash": train.get("candidate_pool_hash", "") if train else "",
            "direct_collateral_candidate_pool_hashes": ";".join(sorted({str(row.get("candidate_pool_hash", "")) for row in direct_collateral if row.get("candidate_pool_hash")})),
            "train013_candidate_count": len(train_ids),
            "direct_collateral_candidate_count": len(dc_ids),
            "overlap_count": len(overlap),
            "overlap_ratio": len(overlap) / len(train_ids) if train_ids else 0.0,
            "pr200_train013_residual_control_supported": pr200_train.get("train013_residual_control_supported", ""),
            "proxy_pool_separation_detected": _bool_text(proxy_sep),
            "interpretation": "train013 control is proxy-pool separation, not exact render-level evidence" if proxy_sep else "train013 proxy control is unavailable or overlaps direct/collateral proxy pools",
            "evidence_quality": inputs.summary.get("evidence_quality", ""),
        }
    ]


def _failure_cases(
    pixel_reuse_rows: list[dict[str, Any]],
    uniformity_rows: list[dict[str, Any]],
    direct_rows: list[dict[str, Any]],
    train_rows: list[dict[str, Any]],
    config: PR201Config,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in pixel_reuse_rows:
        if _truth(row.get("pixel_candidate_reuse_degenerate")):
            rows.append(
                {
                    "scene": row.get("scene", ""),
                    "condition": row.get("condition", ""),
                    "subset_name": row.get("subset_name", ""),
                    "failure_type": "pixel_candidate_reuse",
                    "view_name": row.get("view_name", ""),
                    "view_group": row.get("view_group", ""),
                    "related_view_name": "",
                    "related_view_group": "",
                    "metric_name": "dominant_candidate_set_pixel_ratio",
                    "metric_value": row.get("dominant_candidate_set_pixel_ratio", ""),
                    "threshold": config.reuse_threshold,
                    "evidence_quality": row.get("evidence_quality", ""),
                    "explanation": "same candidate set reused across selected pixels",
                }
            )
    for row in uniformity_rows:
        if _truth(row.get("all_splat_weights_equal")):
            rows.append(
                {
                    "scene": row.get("scene", ""),
                    "condition": row.get("condition", ""),
                    "subset_name": row.get("subset_name", ""),
                    "failure_type": "uniform_splat_weights",
                    "view_name": row.get("view_name", ""),
                    "view_group": row.get("view_group", ""),
                    "related_view_name": "",
                    "related_view_group": "",
                    "metric_name": "cv_splat_weight",
                    "metric_value": row.get("cv_splat_weight", ""),
                    "threshold": config.weight_uniformity_tol,
                    "evidence_quality": row.get("evidence_quality", ""),
                    "explanation": "candidate splat weights are uniform within pixel",
                }
            )
            break
    for row in direct_rows:
        if _truth(row.get("degenerate_overlap")):
            rows.append(
                {
                    "scene": row.get("scene", ""),
                    "condition": row.get("condition", ""),
                    "subset_name": row.get("subset_name", ""),
                    "failure_type": "direct_collateral_pool_overlap",
                    "view_name": row.get("direct_view_names", ""),
                    "view_group": "direct_corrupted",
                    "related_view_name": row.get("collateral_view_names", ""),
                    "related_view_group": "co_visible_collateral",
                    "metric_name": "jaccard",
                    "metric_value": row.get("jaccard", ""),
                    "threshold": config.degenerate_jaccard_threshold,
                    "evidence_quality": row.get("evidence_quality", ""),
                    "explanation": row.get("explanation", ""),
                }
            )
    for row in train_rows:
        if _truth(row.get("proxy_pool_separation_detected")):
            rows.append(
                {
                    "scene": row.get("scene", ""),
                    "condition": row.get("condition", ""),
                    "subset_name": row.get("subset_name", ""),
                    "failure_type": "train013_proxy_pool_separation",
                    "view_name": "train_013",
                    "view_group": row.get("train013_view_group", ""),
                    "related_view_name": row.get("direct_collateral_candidate_pool_hashes", ""),
                    "related_view_group": "direct_corrupted;co_visible_collateral",
                    "metric_name": "overlap_ratio",
                    "metric_value": row.get("overlap_ratio", ""),
                    "threshold": 0.0,
                    "evidence_quality": row.get("evidence_quality", ""),
                    "explanation": row.get("interpretation", ""),
                }
            )
    return rows[: config.max_report_rows]


def _recommendations() -> dict[str, Any]:
    return {
        "recommended_next_step": "exact_sparse_render_contribution_attribution",
        "should_continue_proxy_for_intervention": False,
        "should_use_proxy_for_training_gating": False,
        "should_use_proxy_for_defense_claim": False,
        "safe_use_of_pr20_proxy": [
            "input/output pipeline validation",
            "residual selection sanity check",
            "candidate-pool diagnostic",
            "motivation for exact attribution",
        ],
        "unsafe_use_of_pr20_proxy": [
            "causal Gaussian artifact localization",
            "view rejection",
            "densification gating",
            "training intervention",
            "claiming exact render contribution",
        ],
        "suggested_pr21_goal": "gsplat feasibility and exact sparse pixel-to-Gaussian attribution replay",
    }


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


def write_artifact_manifest(output_dir: Path, pr200_dirs: list[Path]) -> None:
    items: list[tuple[str, Path, bool, str]] = [(f"pr200_dir_{index}", path, True, "input") for index, path in enumerate(pr200_dirs)]
    items.extend((name, output_dir / name, True, "output_pr201") for name in PR201_OUTPUT_FILES)
    manifest = output_dir / "artifact_manifest.csv"
    write_csv_rows(manifest, _artifact_rows(items), MANIFEST_FIELDS)
    write_csv_rows(manifest, _artifact_rows(items), MANIFEST_FIELDS)


def _write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# PR20.1 Proxy Degeneracy Diagnosis",
        "",
        "PR20.1 is observation-only. It does not implement defense, view rejection, update suppression, or densification gating. It does not provide exact render contribution. The current PR20.0 proxy should not be used for training intervention.",
        "",
        "## Summary",
        f"- Proxy degeneracy confirmed: `{summary.get('proxy_degeneracy_confirmed')}`",
        f"- Pixel candidate reuse degeneracy: `{summary.get('pixel_candidate_reuse_degeneracy_confirmed')}`",
        f"- Candidate weight uniformity: `{summary.get('candidate_weight_uniformity_confirmed')}`",
        f"- Direct/collateral overlap degenerate: `{summary.get('direct_collateral_overlap_degenerate')}`",
        f"- Train013 control is proxy-pool separation: `{summary.get('train013_control_is_proxy_pool_separation')}`",
        f"- Recommended next step: `{summary.get('recommended_next_step')}`",
        "",
        "## Recommendation for PR21",
        "",
        "Proceed to gsplat feasibility and exact sparse pixel-to-Gaussian attribution replay.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def analyze_pr201_proxy_degeneracy(
    *,
    pr200_dirs: list[Path],
    output_dir: Path,
    scene: str | None = None,
    condition: str | None = None,
    subset_name: str | None = None,
    config: PR201Config | None = None,
    write_markdown: bool = False,
    strict: bool = False,
    allow_missing: bool = False,
) -> tuple[dict[str, Any], int]:
    del write_markdown
    config = config or PR201Config()
    output_dir.mkdir(parents=True, exist_ok=True)
    all_run_summary: list[dict[str, Any]] = []
    all_pixel_reuse: list[dict[str, Any]] = []
    all_view_pools: list[dict[str, Any]] = []
    all_pool_overlaps: list[dict[str, Any]] = []
    all_uniformity: list[dict[str, Any]] = []
    all_group_audit: list[dict[str, Any]] = []
    all_direct: list[dict[str, Any]] = []
    all_train: list[dict[str, Any]] = []
    all_failures: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    valid_inputs: list[PR200Inputs] = []

    for pr200_dir in pr200_dirs:
        inputs = load_pr200_inputs(pr200_dir)
        if inputs.missing_files:
            for name in inputs.missing_files:
                missing_rows.append(
                    {
                        "pr200_dir": str(pr200_dir),
                        "missing_file": name,
                        "severity": "error",
                        "action": "rerun PR20.0 or use --allow-missing",
                    }
                )
            if not allow_missing:
                continue
        valid_inputs.append(inputs)
        pixel_reuse = analyze_pixel_candidate_reuse(inputs, config, scene=scene, condition=condition, subset_name=subset_name)
        uniformity = analyze_candidate_weight_uniformity(inputs, config, scene=scene, condition=condition, subset_name=subset_name)
        view_pools = analyze_view_candidate_pools(inputs, config, pixel_reuse, uniformity, scene=scene, condition=condition, subset_name=subset_name)
        overlaps = analyze_view_candidate_pool_overlap(view_pools, config)
        group_audit = analyze_group_candidate_pool_audit(inputs, view_pools, uniformity, config)
        direct = analyze_direct_collateral_degeneracy(inputs, view_pools, overlaps, pixel_reuse, uniformity, config)
        train = analyze_train013_proxy_control(inputs, view_pools, config)
        failures = _failure_cases(pixel_reuse, uniformity, direct, train, config)
        uniform_splat = [_number(row.get("mean_splat_weight")) for row in uniformity]
        uniform_alpha = [_number(row.get("mean_alpha_contribution")) for row in uniformity]
        splat_values = [_number(row.get("splat_weight")) for row in inputs.pixel_gaussian_contributions]
        alpha_values = [_number(row.get("alpha_contribution")) for row in inputs.pixel_gaussian_contributions]
        reuse_rate = _mean([_number(row.get("dominant_candidate_set_pixel_ratio")) for row in pixel_reuse])
        uniform_rate = sum(1 for row in uniformity if _truth(row.get("all_splat_weights_equal"))) / len(uniformity) if uniformity else 0.0
        direct_row = direct[0] if direct else {}
        train_row = train[0] if train else {}
        proxy_degenerate = (
            reuse_rate >= config.reuse_threshold
            and uniform_rate >= config.reuse_threshold
            and _truth(direct_row.get("degenerate_overlap"))
        )
        all_run_summary.append(
            {
                "run_id": inputs.run_id,
                "scene": _scene(inputs, scene),
                "condition": _condition(inputs, condition),
                "subset_name": _subset(inputs, subset_name),
                "pr200_dir": str(inputs.pr200_dir),
                "selected_view_count": inputs.summary.get("selected_view_count", ""),
                "selected_pixel_count": inputs.summary.get("selected_pixel_count", ""),
                "total_pixel_gaussian_contribution_rows": inputs.summary.get("total_pixel_gaussian_contribution_rows", ""),
                "evidence_quality": inputs.summary.get("evidence_quality", ""),
                "attribution_method": inputs.summary.get("attribution_method", ""),
                "proxy_degeneracy_confirmed": _bool_text(proxy_degenerate),
                "pixel_candidate_reuse_rate_mean": reuse_rate,
                "candidate_weight_uniformity_rate": uniform_rate,
                "splat_weight_mean": _mean(splat_values),
                "splat_weight_std": _std(splat_values),
                "splat_weight_cv": _cv(splat_values),
                "alpha_contribution_mean": _mean(alpha_values),
                "alpha_contribution_std": _std(alpha_values),
                "alpha_contribution_cv": _cv(alpha_values),
                "direct_collateral_jaccard": direct_row.get("jaccard", ""),
                "direct_collateral_degenerate": direct_row.get("degenerate_overlap", ""),
                "train013_overlap_ratio": train_row.get("overlap_ratio", ""),
                "train013_control_supported_in_pr200": train_row.get("pr200_train013_residual_control_supported", ""),
                "train013_control_interpretation": train_row.get("interpretation", ""),
                "warnings": ";".join(str(item) for item in inputs.summary.get("warnings", [])),
            }
        )
        all_pixel_reuse.extend(pixel_reuse)
        all_uniformity.extend(uniformity)
        all_view_pools.extend(view_pools)
        all_pool_overlaps.extend(overlaps)
        all_group_audit.extend(group_audit)
        all_direct.extend(direct)
        all_train.extend(train)
        all_failures.extend(failures)

    proxy_degeneracy = any(_truth(row.get("proxy_degeneracy_confirmed")) for row in all_run_summary)
    pixel_reuse_degenerate = any(_truth(row.get("pixel_candidate_reuse_degenerate")) for row in all_pixel_reuse)
    weight_uniform = bool(all_uniformity) and (
        sum(1 for row in all_uniformity if _truth(row.get("all_splat_weights_equal"))) / len(all_uniformity) >= config.reuse_threshold
    )
    direct_degenerate = any(_truth(row.get("degenerate_overlap")) for row in all_direct)
    train_proxy = any(_truth(row.get("proxy_pool_separation_detected")) for row in all_train)
    summary = {
        "schema_name": "viewtrust.pr201.proxy_degeneracy.summary",
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "observation_only": True,
        "training_intervention": False,
        "defense_enabled": False,
        "view_rejection_enabled": False,
        "densification_gating_enabled": False,
        "training_behavior_modified": False,
        "rendering_behavior_modified_for_training": False,
        "third_party_modified": False,
        "input_pr200_dir_count": len(pr200_dirs),
        "valid_pr200_dir_count": len(valid_inputs),
        "missing_pr200_dir_count": len(pr200_dirs) - len(valid_inputs),
        "proxy_degeneracy_confirmed": proxy_degeneracy,
        "pixel_candidate_reuse_degeneracy_confirmed": pixel_reuse_degenerate,
        "candidate_weight_uniformity_confirmed": weight_uniform,
        "direct_collateral_overlap_degenerate": direct_degenerate,
        "train013_control_is_proxy_pool_separation": train_proxy,
        "mean_pixel_candidate_reuse_rate": _mean([_number(row.get("pixel_candidate_reuse_rate_mean")) for row in all_run_summary]),
        "mean_view_candidate_pool_reuse_rate": _mean([_number(row.get("candidate_pool_reuse_rate")) for row in all_group_audit]),
        "mean_candidate_weight_cv": _mean([_number(row.get("splat_weight_cv")) for row in all_run_summary]),
        "mean_direct_collateral_jaccard": _mean([_number(row.get("jaccard")) for row in all_direct]),
        "mean_train013_overlap_with_direct_collateral": _mean([_number(row.get("overlap_ratio")) for row in all_train]),
        "current_evidence_quality": _first_nonempty([row.get("evidence_quality") for row in all_run_summary]),
        "current_attribution_method": _first_nonempty([row.get("attribution_method") for row in all_run_summary]),
        "exact_render_contribution_available": _first_nonempty([row.get("evidence_quality") for row in all_run_summary]) == "exact_render_contribution",
        "pr20_ready_for_intervention": False,
        "recommended_next_step": "exact_sparse_render_contribution_attribution",
        "warnings": ["proxy should not be used for intervention"] if proxy_degeneracy else [],
    }
    recommendations = _recommendations()

    write_json(output_dir / "pr201_proxy_degeneracy_summary.json", summary)
    write_csv_rows(output_dir / "pr201_run_summary.csv", all_run_summary, RUN_SUMMARY_FIELDS)
    write_csv_rows(output_dir / "pr201_pixel_candidate_reuse.csv", all_pixel_reuse, PIXEL_REUSE_FIELDS)
    write_csv_rows(output_dir / "pr201_view_candidate_pool.csv", all_view_pools, VIEW_POOL_FIELDS)
    write_csv_rows(output_dir / "pr201_view_candidate_pool_overlap.csv", all_pool_overlaps, PAIR_OVERLAP_FIELDS)
    write_csv_rows(output_dir / "pr201_candidate_weight_uniformity.csv", all_uniformity, WEIGHT_UNIFORMITY_FIELDS)
    write_csv_rows(output_dir / "pr201_group_candidate_pool_audit.csv", all_group_audit, GROUP_AUDIT_FIELDS)
    write_csv_rows(output_dir / "pr201_direct_collateral_degeneracy.csv", all_direct, DIRECT_COLLATERAL_FIELDS)
    write_csv_rows(output_dir / "pr201_train013_proxy_control_audit.csv", all_train, TRAIN013_FIELDS)
    write_csv_rows(output_dir / "pr201_proxy_failure_cases.csv", all_failures[: config.max_report_rows], FAILURE_FIELDS)
    write_json(output_dir / "pr201_recommendations.json", recommendations)
    write_csv_rows(output_dir / "pr201_missing_inputs.csv", missing_rows, MISSING_FIELDS)
    _write_report(output_dir / "pr201_report.md", summary)
    write_artifact_manifest(output_dir, pr200_dirs)

    missing_required = [name for name in PR201_OUTPUT_FILES if not (output_dir / name).is_file()]
    if missing_required:
        raise RuntimeError(f"missing PR20.1 outputs: {missing_required}")
    if missing_rows and strict and not allow_missing:
        return summary, 1
    if missing_rows and not allow_missing:
        return summary, 1
    return summary, 0


def _first_nonempty(values: list[Any]) -> str:
    for value in values:
        if value not in ("", None):
            return str(value)
    return ""
