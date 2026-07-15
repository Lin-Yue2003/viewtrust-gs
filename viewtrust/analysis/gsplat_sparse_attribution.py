"""PR21.1 exact sparse pixel-to-Gaussian attribution replay helpers."""

from __future__ import annotations

import csv
import inspect
import json
import math
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from viewtrust.analysis.clean_prior_normalization import load_csv_rows, load_json, normalize_bool, write_csv_rows, write_json
from viewtrust.analysis.gsplat_feasibility import audit_checkpoint_conversion, parse_ply_header, probe_dependencies
from viewtrust.analysis.offline_signals import safe_float


PR211_OUTPUT_FILES = [
    "pr211_exact_sparse_attribution_summary.json",
    "pr211_input_readiness_audit.csv",
    "pr211_checkpoint_activation_audit.csv",
    "pr211_selected_pixels.csv",
    "pr211_gsplat_metadata_audit.csv",
    "pr211_gsplat_contributor_api_audit.csv",
    "pr211_transmittance_audit.csv",
    "pr211_gsplat_rasterization_output_audit.csv",
    "pr211_gsplat_source_audit.csv",
    "pr211_internal_loop_shape_audit.csv",
    "pr211_internal_loop_attempts.csv",
    "pr211_contributor_path_decision.json",
    "pr211_contributor_path_attempts.csv",
    "pr211_exact_pixel_gaussian_contributions.csv",
    "pr211_gaussian_residual_attribution_exact.csv",
    "pr211_view_group_residual_attribution_exact.csv",
    "pr211_direct_collateral_exact_overlap.csv",
    "pr211_train013_exact_control.csv",
    "pr211_exact_vs_proxy_comparison.csv",
    "pr211_weight_nonuniformity_audit.csv",
    "pr211_missing_fields.csv",
    "pr211_blockers.csv",
    "pr211_recommendations.json",
    "pr211_report.md",
    "artifact_manifest.csv",
]

INPUT_FIELDS = ["input_name", "path", "exists", "required", "status", "notes"]
ACTIVATION_FIELDS = ["component", "source_fields", "activation_or_conversion", "supported", "tensor_shape", "dtype", "device", "notes"]
SELECTED_PIXEL_FIELDS = [
    "scene",
    "condition",
    "subset_name",
    "view_name",
    "view_group",
    "pixel_x",
    "pixel_y",
    "pixel_id",
    "residual_l1",
    "residual_source",
    "selected_rank_within_view",
    "official_render_path",
    "official_gt_path",
]
METADATA_FIELDS = ["metadata_key", "available", "type", "shape", "dtype", "device", "used_for", "notes"]
CONTRIBUTOR_API_FIELDS = [
    "api_name",
    "available",
    "signature",
    "required_arguments",
    "provided_arguments",
    "compatible",
    "dry_run_attempted",
    "dry_run_succeeded",
    "dry_run_error",
    "notes",
]
TRANSMITTANCE_FIELDS = [
    "candidate_source",
    "available",
    "type",
    "shape",
    "dtype",
    "device",
    "shape_compatible",
    "dry_run_attempted",
    "dry_run_succeeded",
    "selected_as_transmittance",
    "notes",
]
RASTER_OUTPUT_FIELDS = ["output_name", "available", "type", "shape", "dtype", "device", "notes"]
SOURCE_AUDIT_FIELDS = ["item", "path", "line_number", "symbol", "signature", "snippet", "notes"]
INTERNAL_LOOP_SHAPE_FIELDS = ["tensor_name", "expected_shape", "actual_shape", "dtype", "device", "shape_ok", "notes"]
INTERNAL_LOOP_ATTEMPT_FIELDS = [
    "attempt_name",
    "packed",
    "attempted",
    "succeeded",
    "means2d_shape",
    "conics_shape",
    "opacities_shape",
    "colors_shape",
    "isect_offsets_shape",
    "flatten_ids_shape",
    "render_alphas_shape",
    "transmittances_shape",
    "num_batches",
    "total_contributor_rows_before_filter",
    "selected_pixel_hit_count",
    "error",
    "notes",
]
PATH_ATTEMPT_FIELDS = [
    "path_name",
    "attempted",
    "succeeded",
    "evidence_quality_if_success",
    "transmittance_source",
    "output_tuple_shapes",
    "output_tuple_interpretation",
    "gaussian_id_mapping",
    "pixel_id_mapping",
    "image_id_mapping",
    "selected_pixel_hit_count",
    "exact_row_count",
    "error",
    "notes",
]
EXACT_FIELDS = [
    "scene",
    "condition",
    "subset_name",
    "view_name",
    "view_group",
    "pixel_x",
    "pixel_y",
    "pixel_id",
    "gaussian_id",
    "contributor_rank",
    "depth_order",
    "alpha_contribution",
    "transmittance_before",
    "splat_weight",
    "opacity_after_activation",
    "color_contribution_r",
    "color_contribution_g",
    "color_contribution_b",
    "residual_l1",
    "residual_weighted_splat",
    "evidence_quality",
    "attribution_method",
]
GAUSSIAN_FIELDS = [
    "scene",
    "condition",
    "subset_name",
    "gaussian_id",
    "exact_contribution_count",
    "selected_pixel_count",
    "contributing_view_count",
    "view_groups_semicolon",
    "residual_weighted_splat_sum",
    "residual_weighted_splat_mean",
    "mean_splat_weight",
    "max_splat_weight",
    "evidence_quality",
]
GROUP_FIELDS = [
    "scene",
    "condition",
    "subset_name",
    "view_group",
    "selected_view_count",
    "selected_pixel_count",
    "unique_exact_gaussian_count",
    "exact_contribution_row_count",
    "residual_weighted_splat_sum",
    "mean_contributors_per_pixel",
    "mean_splat_weight",
    "evidence_quality",
]
DIRECT_FIELDS = [
    "scene",
    "condition",
    "subset_name",
    "direct_view_names",
    "collateral_view_names",
    "direct_unique_exact_gaussian_count",
    "collateral_unique_exact_gaussian_count",
    "overlap_count",
    "jaccard",
    "overlap_ratio_over_direct",
    "overlap_ratio_over_collateral",
    "nontrivial_exact_overlap_supported",
    "evidence_quality",
    "interpretation",
]
TRAIN013_FIELDS = [
    "scene",
    "condition",
    "subset_name",
    "train013_present",
    "train013_unique_exact_gaussian_count",
    "direct_collateral_unique_exact_gaussian_count",
    "overlap_count",
    "overlap_ratio",
    "train013_exact_control_supported",
    "evidence_quality",
    "interpretation",
]
COMPARISON_FIELDS = [
    "scene",
    "condition",
    "subset_name",
    "view_name",
    "pixel_x",
    "pixel_y",
    "proxy_candidate_count",
    "exact_candidate_count",
    "overlap_count",
    "jaccard",
    "exact_not_in_proxy_count",
    "proxy_not_in_exact_count",
    "proxy_weight_uniform",
    "exact_weight_cv",
    "interpretation",
]
WEIGHT_FIELDS = [
    "scene",
    "condition",
    "subset_name",
    "view_name",
    "pixel_x",
    "pixel_y",
    "exact_candidate_count",
    "mean_splat_weight",
    "std_splat_weight",
    "cv_splat_weight",
    "min_splat_weight",
    "max_splat_weight",
    "weights_nonuniform",
    "evidence_quality",
]
MISSING_FIELDS = ["field_name", "available", "required_for", "impact", "fallback_used", "notes"]
BLOCKER_FIELDS = ["severity", "component", "blocker", "evidence", "recommended_action"]
MANIFEST_FIELDS = ["relative_path", "path", "exists", "file_type", "size_bytes", "required", "artifact_group"]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _bool_text(value: Any) -> str:
    return "true" if bool(value) else "false"


def _truth(value: Any) -> bool:
    return normalize_bool(value) is True


def _number(value: Any, default: float = 0.0) -> float:
    parsed = safe_float(value)
    return default if parsed is None else parsed


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def _std(values: list[float]) -> float | None:
    return statistics.pstdev(values) if len(values) > 1 else 0.0 if values else None


def _cv(values: list[float]) -> float | None:
    mean = _mean(values)
    std = _std(values)
    if mean in (None, 0) or std is None:
        return None
    return std / abs(mean)


def _path_exists(path: Path) -> str:
    return _bool_text(path.exists())


def _input_row(name: str, path: Path, required: bool, notes: str = "") -> dict[str, Any]:
    exists = path.exists()
    return {
        "input_name": name,
        "path": str(path),
        "exists": _bool_text(exists),
        "required": _bool_text(required),
        "status": "ok" if exists else "missing",
        "notes": notes,
    }


def _view_index(view_name: str) -> int | None:
    digits = "".join(ch for ch in str(view_name) if ch.isdigit())
    if not digits:
        return None
    return int(digits)


def _pixel_id(x: int, y: int, width: int) -> int:
    return y * width + x


def _load_pr210_selected_views(pr210_dir: Path, requested_views: list[str]) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    rows = load_csv_rows(pr210_dir / "pr210_selected_view_audit.csv")
    blockers: list[dict[str, Any]] = []
    by_view = {row.get("requested_view_name", ""): row for row in rows}
    selected = []
    for view in requested_views:
        row = by_view.get(view)
        if not row:
            blockers.append(_blocker("error", "selected_view_matching", f"missing selected-view audit row for {view}", str(pr210_dir), "rerun PR21.0a"))
            continue
        selected.append(row)
        if not (_truth(row.get("strict_match")) and _truth(row.get("split_consistent")) and _truth(row.get("valid_for_exact_attribution"))):
            blockers.append(
                _blocker(
                    "error",
                    "selected_view_matching",
                    f"selected view {view} is not valid for exact attribution",
                    f"strict={row.get('strict_match')} split={row.get('split_consistent')} valid={row.get('valid_for_exact_attribution')}",
                    "require strict PR21.0a selected-view matching",
                )
            )
    return selected, blockers


def _blocker(severity: str, component: str, blocker: str, evidence: str, action: str) -> dict[str, Any]:
    return {"severity": severity, "component": component, "blocker": blocker, "evidence": evidence, "recommended_action": action}


def _load_selected_pixels(
    *,
    pr200_dir: Path,
    selected_views: list[dict[str, str]],
    scene: str,
    condition: str,
    subset_name: str,
    top_pixels_per_view: int,
) -> tuple[list[dict[str, Any]], dict[tuple[str, int, int], list[dict[str, str]]]]:
    view_meta = {row.get("requested_view_name", ""): row for row in selected_views}
    sparse_rows = load_csv_rows(pr200_dir / "pr200_sparse_pixel_residuals.csv")
    residual_source = "pr200_sparse_pixel_residuals"
    if not sparse_rows:
        residual_source = "pr200_pixel_gaussian_contributions_dedup"
        seen: dict[tuple[str, str, str], dict[str, str]] = {}
        for row in load_csv_rows(pr200_dir / "pr200_pixel_gaussian_contributions.csv"):
            key = (row.get("view_name", ""), row.get("pixel_x", ""), row.get("pixel_y", ""))
            seen.setdefault(key, row)
        sparse_rows = list(seen.values())

    by_view: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in sparse_rows:
        view_name = row.get("view_name", "")
        if view_name in view_meta:
            by_view[view_name].append(row)

    selected_pixels: list[dict[str, Any]] = []
    for view_name in [row.get("requested_view_name", "") for row in selected_views]:
        meta = view_meta.get(view_name, {})
        width = int(_number(meta.get("image_width"), 400) or 400)
        rows = sorted(
            by_view.get(view_name, []),
            key=lambda row: (-_number(row.get("residual_l1"), -1.0), int(_number(row.get("pixel_y"), 0)), int(_number(row.get("pixel_x"), 0))),
        )[:top_pixels_per_view]
        for rank, row in enumerate(rows, start=1):
            x = int(_number(row.get("pixel_x"), 0))
            y = int(_number(row.get("pixel_y"), 0))
            selected_pixels.append(
                {
                    "scene": scene,
                    "condition": condition,
                    "subset_name": subset_name,
                    "view_name": view_name,
                    "view_group": row.get("view_group", ""),
                    "pixel_x": x,
                    "pixel_y": y,
                    "pixel_id": _pixel_id(x, y, width),
                    "residual_l1": row.get("residual_l1", ""),
                    "residual_source": residual_source if row.get("residual_l1", "") not in ("", None) else "missing_or_proxy_source",
                    "selected_rank_within_view": rank,
                    "official_render_path": meta.get("official_render_path", ""),
                    "official_gt_path": meta.get("official_gt_path", ""),
                }
            )

    proxy_by_pixel: dict[tuple[str, int, int], list[dict[str, str]]] = defaultdict(list)
    for row in load_csv_rows(pr200_dir / "pr200_pixel_gaussian_contributions.csv"):
        try:
            key = (str(row.get("view_name", "")), int(float(row.get("pixel_x", 0))), int(float(row.get("pixel_y", 0))))
        except ValueError:
            continue
        proxy_by_pixel[key].append(row)
    return selected_pixels, proxy_by_pixel


def _metadata_audit(meta: dict[str, Any]) -> list[dict[str, Any]]:
    if not meta:
        return [{"metadata_key": "", "available": "false", "type": "", "shape": "", "dtype": "", "device": "", "used_for": "", "notes": "no metadata available"}]
    rows = []
    for key in sorted(meta):
        value = meta[key]
        shape = getattr(value, "shape", "")
        dtype = getattr(value, "dtype", "")
        device = getattr(value, "device", "")
        used_for = ""
        if key in {"flatten_ids", "isect_offsets", "gaussian_ids", "camera_ids"}:
            used_for = "sparse contributor recovery"
        elif key in {"transmittances", "opacities", "alphas"}:
            used_for = "contribution weights if available"
        rows.append(
            {
                "metadata_key": key,
                "available": "true",
                "type": type(value).__name__,
                "shape": tuple(shape) if shape != "" else "",
                "dtype": str(dtype),
                "device": str(device),
                "used_for": used_for,
                "notes": "",
            }
        )
    return rows


def _audit_gsplat_source() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        import gsplat
    except Exception as exc:
        return [
            {
                "item": "gsplat import",
                "path": "",
                "line_number": "",
                "symbol": "gsplat",
                "signature": "",
                "snippet": "",
                "notes": f"gsplat import failed: {exc}",
            }
        ]

    symbols = [("gsplat", gsplat)]
    try:
        rendering = __import__("gsplat.rendering", fromlist=["rasterization"])
        symbols.append(("gsplat.rendering", rendering))
        symbols.append(("gsplat.rendering.rasterization", getattr(rendering, "rasterization", None)))
    except Exception as exc:
        rows.append({"item": "import", "path": "", "line_number": "", "symbol": "gsplat.rendering", "signature": "", "snippet": "", "notes": str(exc)})
    for name in ["rasterize_to_indices_in_range", "accumulate"]:
        obj = _find_gsplat_callable(name)
        symbols.append((name, obj))

    seen_paths: set[Path] = set()
    for symbol, obj in symbols:
        if obj is None:
            rows.append({"item": "symbol", "path": "", "line_number": "", "symbol": symbol, "signature": "", "snippet": "", "notes": "unavailable"})
            continue
        try:
            unwrapped = inspect.unwrap(obj)
        except Exception:
            unwrapped = obj
        try:
            source_path = Path(inspect.getsourcefile(unwrapped) or getattr(unwrapped, "__file__", "") or getattr(obj, "__file__", ""))
        except Exception:
            source_path = Path(str(getattr(obj, "__file__", "")))
        try:
            _, line_number = inspect.getsourcelines(unwrapped)
        except Exception:
            line_number = ""
        try:
            signature = str(inspect.signature(unwrapped)) if callable(unwrapped) else ""
        except Exception:
            signature = ""
        rows.append(
            {
                "item": "symbol",
                "path": str(source_path),
                "line_number": line_number,
                "symbol": symbol,
                "signature": signature,
                "snippet": "",
                "notes": "inspect.unwrap source audit",
            }
        )
        if source_path.exists():
            seen_paths.add(source_path)

    package_root = Path(getattr(gsplat, "__file__", "")).resolve().parent
    patterns = [
        "transmittance",
        "transmittances",
        "rasterize_to_indices_in_range",
        "isect_offsets",
        "flatten_ids",
        "accumulate",
        "render_alphas",
        "alphas",
        "batch_per_iter",
    ]
    source_paths = sorted(seen_paths | set(package_root.rglob("*.py")))
    for path in source_paths:
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except Exception:
            continue
        for index, line in enumerate(lines, start=1):
            for pattern in patterns:
                if pattern in line:
                    rows.append(
                        {
                            "item": f"pattern:{pattern}",
                            "path": str(path),
                            "line_number": index,
                            "symbol": "",
                            "signature": "",
                            "snippet": line.strip()[:240],
                            "notes": "source grep",
                        }
                    )
                    break
    return rows or [{"item": "gsplat source", "path": str(package_root), "line_number": "", "symbol": "", "signature": "", "snippet": "", "notes": "no source rows found"}]


def _source_supports_alpha_transmittance(source_rows: list[dict[str, Any]]) -> bool:
    text = "\n".join(str(row.get("snippet", "")) for row in source_rows).lower()
    return "render_alphas" in text or "alphas" in text or "1 -" in text


def _contributor_path_decision(
    *,
    summary: dict[str, Any],
    source_rows: list[dict[str, Any]],
    path_attempts: list[dict[str, Any]],
) -> dict[str, Any]:
    succeeded = [row for row in path_attempts if row.get("succeeded") == "true"]
    selected = str(summary.get("contributor_path_selected") or (succeeded[0].get("path_name") if succeeded else "source_level_failure"))
    rejected = [
        {
            "path_name": row.get("path_name"),
            "reason": row.get("error") or row.get("notes"),
        }
        for row in path_attempts
        if row.get("succeeded") != "true"
    ]
    attempted_names = {str(row.get("path_name", "")) for row in path_attempts}
    for path_name, reason in [
        ("source_verified_internal_loop", "not selected; source-verified gsplat loop did not succeed or was not required for this run"),
        ("path_b_lower_level_private_api", "not selected; PR21.1c uses installed public/wrapper symbols without modifying private gsplat code"),
        ("path_c_tile_intersection_footprint", "not selected; proxy or geometric footprint fallback is not exact evidence"),
        ("path_d_source_level_failure", "selected only when no valid contributor path succeeds"),
    ]:
        if path_name not in attempted_names and path_name != selected:
            rejected.append({"path_name": path_name, "reason": reason})
    evidence = [
        {
            "item": row.get("item"),
            "path": row.get("path"),
            "line_number": row.get("line_number"),
            "snippet": row.get("snippet"),
        }
        for row in source_rows[:50]
    ]
    return {
        "selected_path": selected,
        "selected_path_reason": summary.get("contributor_path_reason", ""),
        "rejected_paths": rejected,
        "source_evidence": evidence,
        "exact_contributor_ids_possible": bool(summary.get("exact_contributor_id_only_succeeded") or summary.get("exact_render_contribution_succeeded")),
        "exact_alpha_possible": bool(summary.get("exact_alpha_available")),
        "exact_transmittance_possible": bool(summary.get("exact_transmittance_available")),
        "requires_private_api": selected in {"path_b_lower_level_private_api", "path_c_tile_intersection_footprint"},
        "modifies_gsplat_or_third_party": False,
    }


def _value_type_shape(value: Any) -> tuple[str, str, str, str]:
    shape = getattr(value, "shape", "")
    dtype = getattr(value, "dtype", "")
    device = getattr(value, "device", "")
    return type(value).__name__, str(tuple(shape)) if shape != "" else "", str(dtype), str(device)


def _rasterization_output_audit(outputs: tuple[Any, ...], meta: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, value in enumerate(outputs):
        value_type, shape, dtype, device = _value_type_shape(value)
        rows.append(
            {
                "output_name": f"rasterization_output_{index}",
                "available": "true",
                "type": value_type,
                "shape": shape,
                "dtype": dtype,
                "device": device,
                "notes": "captured raw rasterization return value",
            }
        )
    for key in sorted(meta):
        value_type, shape, dtype, device = _value_type_shape(meta[key])
        rows.append(
            {
                "output_name": f"metadata.{key}",
                "available": "true",
                "type": value_type,
                "shape": shape,
                "dtype": dtype,
                "device": device,
                "notes": "captured rasterization metadata value",
            }
        )
    if not rows:
        rows.append({"output_name": "", "available": "false", "type": "", "shape": "", "dtype": "", "device": "", "notes": "no rasterization outputs captured"})
    return rows


def _find_gsplat_callable(name: str) -> Any | None:
    for module_name in ["gsplat", "gsplat.cuda", "gsplat.cuda._wrapper"]:
        try:
            module = __import__(module_name, fromlist=[name])
            obj = getattr(module, name, None)
            if obj is not None:
                return obj
        except Exception:
            continue
    return None


def _required_args(signature: inspect.Signature) -> list[str]:
    return [
        name
        for name, param in signature.parameters.items()
        if param.default is inspect.Parameter.empty
        and param.kind in {inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY}
    ]


def _call_rasterize_to_indices_in_range_safely(
    api: Any,
    *,
    range_start: int,
    range_end: int,
    transmittances: Any,
    means2d: Any,
    conics: Any,
    opacities: Any,
    image_width: int,
    image_height: int,
    tile_size: int,
    isect_offsets: Any,
    flatten_ids: Any,
) -> Any:
    if transmittances is None:
        raise ValueError("transmittances is required")
    return api(
        range_start=range_start,
        range_end=range_end,
        transmittances=transmittances,
        means2d=means2d,
        conics=conics,
        opacities=opacities,
        image_width=image_width,
        image_height=image_height,
        tile_size=tile_size,
        isect_offsets=isect_offsets,
        flatten_ids=flatten_ids,
    )


def _looks_like_tensor(value: Any) -> bool:
    return hasattr(value, "shape") and hasattr(value, "dtype") and hasattr(value, "device")


def _candidate_shape_compatible(candidate: Any, meta: dict[str, Any]) -> bool:
    if not _looks_like_tensor(candidate):
        return False
    means = meta.get("means2d")
    if _looks_like_tensor(means) and str(getattr(candidate, "device", "")) != str(getattr(means, "device", "")):
        return False
    shape = tuple(getattr(candidate, "shape", ()))
    return bool(shape)


def _api_audit_row(
    *,
    api_name: str,
    api: Any | None,
    provided_arguments: list[str],
    dry_run_attempted: bool,
    dry_run_succeeded: bool,
    dry_run_error: str,
    notes: str = "",
) -> dict[str, Any]:
    if api is None:
        return {
            "api_name": api_name,
            "available": "false",
            "signature": "",
            "required_arguments": "",
            "provided_arguments": ";".join(provided_arguments),
            "compatible": "false",
            "dry_run_attempted": _bool_text(dry_run_attempted),
            "dry_run_succeeded": _bool_text(dry_run_succeeded),
            "dry_run_error": dry_run_error,
            "notes": notes or "API unavailable",
        }
    try:
        signature = inspect.signature(api)
        required = _required_args(signature)
    except Exception as exc:
        signature = None
        required = []
        notes = notes or f"signature unavailable: {exc}"
    compatible = bool(signature) and set(required).issubset(set(provided_arguments))
    return {
        "api_name": api_name,
        "available": "true",
        "signature": str(signature) if signature else "",
        "required_arguments": ";".join(required),
        "provided_arguments": ";".join(provided_arguments),
        "compatible": _bool_text(compatible),
        "dry_run_attempted": _bool_text(dry_run_attempted),
        "dry_run_succeeded": _bool_text(dry_run_succeeded),
        "dry_run_error": dry_run_error,
        "notes": notes,
    }


def _transmittance_audit_row(
    *,
    source: str,
    value: Any,
    available: bool,
    shape_compatible: bool,
    dry_run_attempted: bool,
    dry_run_succeeded: bool,
    selected: bool,
    notes: str,
) -> dict[str, Any]:
    value_type, shape, dtype, device = _value_type_shape(value) if available else ("", "", "", "")
    return {
        "candidate_source": source,
        "available": _bool_text(available),
        "type": value_type,
        "shape": shape,
        "dtype": dtype,
        "device": device,
        "shape_compatible": _bool_text(shape_compatible),
        "dry_run_attempted": _bool_text(dry_run_attempted),
        "dry_run_succeeded": _bool_text(dry_run_succeeded),
        "selected_as_transmittance": _bool_text(selected),
        "notes": notes,
    }


def _normalize_api_result(result: Any) -> tuple[Any | None, Any | None, Any | None]:
    if not isinstance(result, (tuple, list)) or len(result) < 3:
        return None, None, None
    return result[0], result[1], result[2]


def _result_nonempty(result: Any) -> bool:
    gaussian_ids, pixel_ids, image_ids = _normalize_api_result(result)
    del pixel_ids, image_ids
    if gaussian_ids is None:
        return False
    try:
        return int(getattr(gaussian_ids, "numel")()) > 0
    except Exception:
        try:
            return len(gaussian_ids) > 0
        except Exception:
            return False


def _resolve_transmittance_source(
    *,
    api: Any,
    meta: dict[str, Any],
    raster_outputs: tuple[Any, ...],
    image_width: int,
    image_height: int,
    source_rows: list[dict[str, Any]],
) -> tuple[Any | None, str, list[dict[str, Any]], list[dict[str, Any]], bool, str]:
    required_meta = ["means2d", "conics", "opacities", "isect_offsets", "flatten_ids"]
    missing = [key for key in required_meta if key not in meta]
    tile_size = int(meta.get("tile_size", 16))
    provided_args = [
        "range_start",
        "range_end",
        "transmittances",
        "means2d",
        "conics",
        "opacities",
        "image_width",
        "image_height",
        "tile_size",
        "isect_offsets",
        "flatten_ids",
    ]
    api_rows: list[dict[str, Any]] = []
    trans_rows: list[dict[str, Any]] = []
    if api is None:
        api_rows.append(
            _api_audit_row(
                api_name="rasterize_to_indices_in_range",
                api=None,
                provided_arguments=provided_args,
                dry_run_attempted=False,
                dry_run_succeeded=False,
                dry_run_error="",
                notes="API unavailable",
            )
        )
        return None, "", trans_rows, api_rows, False, "rasterize_to_indices_in_range unavailable"
    if missing:
        api_rows.append(
            _api_audit_row(
                api_name="rasterize_to_indices_in_range",
                api=api,
                provided_arguments=[arg for arg in provided_args if arg != "transmittances"],
                dry_run_attempted=False,
                dry_run_succeeded=False,
                dry_run_error=f"missing metadata: {missing}",
                notes="metadata incomplete",
            )
        )
        return None, "", trans_rows, api_rows, False, f"missing metadata: {missing}"

    candidates: list[tuple[str, Any]] = []
    for key in ["transmittances", "transmittance", "T", "final_T", "render_transmittances"]:
        candidates.append((f"metadata.{key}", meta.get(key)))
    if len(raster_outputs) > 1:
        candidates.append(("rasterization_output_1", raster_outputs[1]))
        output_1 = raster_outputs[1]
        if hasattr(output_1, "squeeze"):
            try:
                squeezed = output_1.squeeze(-1)
                candidates.append(("rasterization_output_1_squeezed", squeezed))
                if _source_supports_alpha_transmittance(source_rows):
                    candidates.append(("one_minus_rasterization_output_1_squeezed", 1.0 - squeezed))
            except Exception:
                pass
    dry_run_end = min(max(1, image_width * image_height), 4096)
    selected_tensor = None
    selected_source = ""
    selected_success = False
    dry_run_error = ""
    for source, candidate in candidates:
        available = candidate is not None
        compatible = available and _candidate_shape_compatible(candidate, meta)
        attempted = bool(compatible)
        succeeded = False
        notes = ""
        if attempted:
            try:
                result = _call_rasterize_to_indices_in_range_safely(
                    api,
                    range_start=0,
                    range_end=dry_run_end,
                    transmittances=candidate,
                    means2d=meta["means2d"],
                    conics=meta["conics"],
                    opacities=meta["opacities"],
                    image_width=image_width,
                    image_height=image_height,
                    tile_size=tile_size,
                    isect_offsets=meta["isect_offsets"],
                    flatten_ids=meta["flatten_ids"],
                )
                succeeded = _result_nonempty(result)
                notes = "dry-run returned non-empty contributor tensors" if succeeded else "dry-run returned no contributors"
            except Exception as exc:
                dry_run_error = repr(exc)
                notes = dry_run_error
        elif not available:
            notes = "candidate unavailable"
        else:
            notes = "candidate tensor shape/device incompatible"
        select = selected_tensor is None and succeeded
        if select:
            selected_tensor = candidate
            selected_source = source
            selected_success = True
        trans_rows.append(
            _transmittance_audit_row(
                source=source,
                value=candidate,
                available=available,
                shape_compatible=bool(compatible),
                dry_run_attempted=attempted,
                dry_run_succeeded=succeeded,
                selected=select,
                notes=notes,
            )
        )
    if not candidates:
        trans_rows.append(
            _transmittance_audit_row(
                source="unavailable",
                value=None,
                available=False,
                shape_compatible=False,
                dry_run_attempted=False,
                dry_run_succeeded=False,
                selected=False,
                notes="no transmittance candidates",
            )
        )
    api_rows.append(
        _api_audit_row(
            api_name="rasterize_to_indices_in_range",
            api=api,
            provided_arguments=provided_args,
            dry_run_attempted=any(row["dry_run_attempted"] == "true" for row in trans_rows),
            dry_run_succeeded=selected_success,
            dry_run_error="" if selected_success else dry_run_error,
            notes=f"selected transmittance source: {selected_source}" if selected_success else "no valid transmittance source selected",
        )
    )
    accumulate = _find_gsplat_callable("accumulate")
    api_rows.append(
        _api_audit_row(
            api_name="accumulate",
            api=accumulate,
            provided_arguments=[],
            dry_run_attempted=False,
            dry_run_succeeded=False,
            dry_run_error="",
            notes="audited for future scalar reconstruction",
        )
    )
    return selected_tensor, selected_source, trans_rows, api_rows, selected_success, "" if selected_success else "unable to resolve valid transmittances tensor for rasterize_to_indices_in_range"


def _activation_audit(rows: list[dict[str, Any]], device: str) -> list[dict[str, Any]]:
    by_step = {row.get("conversion_step", ""): row for row in rows}
    return [
        {
            "component": "means",
            "source_fields": "x;y;z",
            "activation_or_conversion": "identity",
            "supported": by_step.get("positions", {}).get("supported", "false"),
            "tensor_shape": by_step.get("positions", {}).get("tensor_shape", ""),
            "dtype": "float32",
            "device": device,
            "notes": "official 3DGS positions",
        },
        {
            "component": "opacity",
            "source_fields": "opacity",
            "activation_or_conversion": "sigmoid",
            "supported": by_step.get("opacities", {}).get("supported", "false"),
            "tensor_shape": by_step.get("opacities", {}).get("tensor_shape", ""),
            "dtype": "float32",
            "device": device,
            "notes": "pre-activated official 3DGS opacity assumed",
        },
        {
            "component": "scales",
            "source_fields": "scale_0;scale_1;scale_2",
            "activation_or_conversion": "exp",
            "supported": by_step.get("scales", {}).get("supported", "false"),
            "tensor_shape": by_step.get("scales", {}).get("tensor_shape", ""),
            "dtype": "float32",
            "device": device,
            "notes": "pre-activated official 3DGS scales assumed",
        },
        {
            "component": "rotations",
            "source_fields": "rot_0;rot_1;rot_2;rot_3",
            "activation_or_conversion": "normalize quaternion",
            "supported": by_step.get("rotations", {}).get("supported", "false"),
            "tensor_shape": by_step.get("rotations", {}).get("tensor_shape", ""),
            "dtype": "float32",
            "device": device,
            "notes": "quaternion normalized before gsplat replay",
        },
        {
            "component": "colors",
            "source_fields": "f_dc_0;f_dc_1;f_dc_2;f_rest_*",
            "activation_or_conversion": "SH DC to RGB; f_rest retained only if future replay supports SH",
            "supported": by_step.get("colors / f_dc", {}).get("supported", "false"),
            "tensor_shape": by_step.get("colors / f_dc", {}).get("tensor_shape", ""),
            "dtype": "float32",
            "device": device,
            "notes": "PR21.1 contributor IDs do not depend on final color parity",
        },
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


def _write_manifest(output_dir: Path, run_dir: Path, pr200_dir: Path, pr210_dir: Path) -> None:
    items = [("run_dir", run_dir, True, "input"), ("pr200_dir", pr200_dir, True, "input"), ("pr210_dir", pr210_dir, True, "input")]
    items.extend((name, output_dir / name, True, "output_pr211") for name in PR211_OUTPUT_FILES)
    manifest = output_dir / "artifact_manifest.csv"
    write_csv_rows(manifest, _artifact_rows(items), MANIFEST_FIELDS)
    write_csv_rows(manifest, _artifact_rows(items), MANIFEST_FIELDS)


def _group_for_view(view_name: str, fallback: str = "") -> str:
    if view_name in {"train_004", "train_009", "train_012", "train_017"}:
        return "direct_corrupted"
    if view_name in {"train_014", "train_007"}:
        return "co_visible_collateral"
    if view_name == "train_013":
        return "clean_prior_demoted"
    return fallback or "other_clean"


def _aggregate_gaussian(rows: list[dict[str, Any]], scene: str, condition: str, subset_name: str) -> list[dict[str, Any]]:
    by_gaussian: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_gaussian[str(row.get("gaussian_id", ""))].append(row)
    out = []
    for gid, items in sorted(by_gaussian.items(), key=lambda item: int(item[0]) if item[0].isdigit() else item[0]):
        evidence = str(items[0].get("evidence_quality") or "exact_sparse_contributor_id_only")
        splats = [_number(row.get("splat_weight")) for row in items if row.get("splat_weight") not in ("", None)]
        residual = [_number(row.get("residual_weighted_splat")) for row in items if row.get("residual_weighted_splat") not in ("", None)]
        pixels = {(row.get("view_name"), row.get("pixel_x"), row.get("pixel_y")) for row in items}
        views = {str(row.get("view_name", "")) for row in items}
        groups = {str(row.get("view_group", "")) for row in items}
        out.append(
            {
                "scene": scene,
                "condition": condition,
                "subset_name": subset_name,
                "gaussian_id": gid,
                "exact_contribution_count": len(items),
                "selected_pixel_count": len(pixels),
                "contributing_view_count": len(views),
                "view_groups_semicolon": ";".join(sorted(groups)),
                "residual_weighted_splat_sum": sum(residual) if residual else "",
                "residual_weighted_splat_mean": _mean(residual),
                "mean_splat_weight": _mean(splats),
                "max_splat_weight": max(splats) if splats else "",
                "evidence_quality": evidence,
            }
        )
    return out


def _aggregate_groups(rows: list[dict[str, Any]], selected_pixels: list[dict[str, Any]], scene: str, condition: str, subset_name: str) -> list[dict[str, Any]]:
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    pixel_by_group: dict[str, set[tuple[Any, Any, Any]]] = defaultdict(set)
    view_by_group: dict[str, set[str]] = defaultdict(set)
    for pixel in selected_pixels:
        group = str(pixel.get("view_group") or _group_for_view(str(pixel.get("view_name", ""))))
        pixel_by_group[group].add((pixel.get("view_name"), pixel.get("pixel_x"), pixel.get("pixel_y")))
        view_by_group[group].add(str(pixel.get("view_name", "")))
    for row in rows:
        by_group[str(row.get("view_group", ""))].append(row)
    out = []
    for group in sorted(set(by_group) | set(pixel_by_group)):
        items = by_group.get(group, [])
        evidence = str(items[0].get("evidence_quality") or "") if items else ""
        gaussians = {str(row.get("gaussian_id", "")) for row in items}
        residual = [_number(row.get("residual_weighted_splat")) for row in items if row.get("residual_weighted_splat") not in ("", None)]
        splats = [_number(row.get("splat_weight")) for row in items if row.get("splat_weight") not in ("", None)]
        pixel_count = len(pixel_by_group.get(group, set()))
        out.append(
            {
                "scene": scene,
                "condition": condition,
                "subset_name": subset_name,
                "view_group": group,
                "selected_view_count": len(view_by_group.get(group, set())),
                "selected_pixel_count": pixel_count,
                "unique_exact_gaussian_count": len(gaussians),
                "exact_contribution_row_count": len(items),
                "residual_weighted_splat_sum": sum(residual) if residual else "",
                "mean_contributors_per_pixel": len(items) / pixel_count if pixel_count else "",
                "mean_splat_weight": _mean(splats),
                "evidence_quality": evidence,
            }
        )
    return out


def _direct_collateral(rows: list[dict[str, Any]], scene: str, condition: str, subset_name: str) -> dict[str, Any]:
    if not rows:
        return {
            "scene": scene,
            "condition": condition,
            "subset_name": subset_name,
            "direct_view_names": "",
            "collateral_view_names": "",
            "direct_unique_exact_gaussian_count": "",
            "collateral_unique_exact_gaussian_count": "",
            "overlap_count": "",
            "jaccard": "",
            "overlap_ratio_over_direct": "",
            "overlap_ratio_over_collateral": "",
            "nontrivial_exact_overlap_supported": "false",
            "evidence_quality": "failed_exact_sparse_replay",
            "interpretation": "exact unavailable due to failed sparse replay",
        }
    direct_views = sorted({str(row.get("view_name", "")) for row in rows if row.get("view_group") == "direct_corrupted"})
    evidence = str(rows[0].get("evidence_quality") or "exact_sparse_contributor_id_only")
    collateral_views = sorted({str(row.get("view_name", "")) for row in rows if row.get("view_group") == "co_visible_collateral"})
    direct = {str(row.get("gaussian_id", "")) for row in rows if row.get("view_group") == "direct_corrupted"}
    collateral = {str(row.get("gaussian_id", "")) for row in rows if row.get("view_group") == "co_visible_collateral"}
    overlap = direct & collateral
    union = direct | collateral
    return {
        "scene": scene,
        "condition": condition,
        "subset_name": subset_name,
        "direct_view_names": ";".join(direct_views),
        "collateral_view_names": ";".join(collateral_views),
        "direct_unique_exact_gaussian_count": len(direct),
        "collateral_unique_exact_gaussian_count": len(collateral),
        "overlap_count": len(overlap),
        "jaccard": len(overlap) / len(union) if union else None,
        "overlap_ratio_over_direct": len(overlap) / len(direct) if direct else None,
        "overlap_ratio_over_collateral": len(overlap) / len(collateral) if collateral else None,
        "nontrivial_exact_overlap_supported": _bool_text(bool(overlap)),
        "evidence_quality": evidence,
        "interpretation": "selected direct/collateral pixels share exact gsplat contributors" if overlap else "no selected-pixel direct/collateral exact overlap detected",
    }


def _train013_control(rows: list[dict[str, Any]], scene: str, condition: str, subset_name: str) -> dict[str, Any]:
    if not rows:
        return {
            "scene": scene,
            "condition": condition,
            "subset_name": subset_name,
            "train013_present": "",
            "train013_unique_exact_gaussian_count": "",
            "direct_collateral_unique_exact_gaussian_count": "",
            "overlap_count": "",
            "overlap_ratio": "",
            "train013_exact_control_supported": "false",
            "evidence_quality": "failed_exact_sparse_replay",
            "interpretation": "exact unavailable due to failed sparse replay",
        }
    train = {str(row.get("gaussian_id", "")) for row in rows if row.get("view_name") == "train_013"}
    evidence = str(rows[0].get("evidence_quality") or "exact_sparse_contributor_id_only")
    direct_collateral = {
        str(row.get("gaussian_id", ""))
        for row in rows
        if row.get("view_group") in {"direct_corrupted", "co_visible_collateral"}
    }
    overlap = train & direct_collateral
    return {
        "scene": scene,
        "condition": condition,
        "subset_name": subset_name,
        "train013_present": _bool_text(bool(train)),
        "train013_unique_exact_gaussian_count": len(train),
        "direct_collateral_unique_exact_gaussian_count": len(direct_collateral),
        "overlap_count": len(overlap),
        "overlap_ratio": len(overlap) / len(train) if train else None,
        "train013_exact_control_supported": _bool_text(bool(train) and not overlap),
        "evidence_quality": evidence,
        "interpretation": "exact replay supports selected-pixel train013 separation, not global innocence" if train and not overlap else "train013 exact selected-pixel overlap exists or train013 is absent",
    }


def _proxy_comparison(rows: list[dict[str, Any]], proxy_by_pixel: dict[tuple[str, int, int], list[dict[str, str]]], scene: str, condition: str, subset_name: str) -> list[dict[str, Any]]:
    exact_by_pixel: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        exact_by_pixel[(str(row.get("view_name", "")), int(row.get("pixel_x", 0)), int(row.get("pixel_y", 0)))].append(row)
    out = []
    for key in sorted(set(proxy_by_pixel) | set(exact_by_pixel)):
        view, x, y = key
        proxy_rows = proxy_by_pixel.get(key, [])
        exact_rows = exact_by_pixel.get(key, [])
        proxy_ids = {str(row.get("gaussian_id", "")) for row in proxy_rows}
        exact_ids = {str(row.get("gaussian_id", "")) for row in exact_rows}
        overlap = proxy_ids & exact_ids
        union = proxy_ids | exact_ids
        proxy_weights = [_number(row.get("splat_weight")) for row in proxy_rows if row.get("splat_weight") not in ("", None)]
        exact_weights = [_number(row.get("splat_weight")) for row in exact_rows if row.get("splat_weight") not in ("", None)]
        proxy_uniform = len({round(value, 12) for value in proxy_weights}) <= 1 if proxy_weights else ""
        exact_cv = _cv(exact_weights)
        exact_unavailable = not rows
        out.append(
            {
                "scene": scene,
                "condition": condition,
                "subset_name": subset_name,
                "view_name": view,
                "pixel_x": x,
                "pixel_y": y,
                "proxy_candidate_count": len(proxy_ids),
                "exact_candidate_count": len(exact_ids),
                "overlap_count": len(overlap),
                "jaccard": len(overlap) / len(union) if union else None,
                "exact_not_in_proxy_count": len(exact_ids - proxy_ids),
                "proxy_not_in_exact_count": len(proxy_ids - exact_ids),
                "proxy_weight_uniform": _bool_text(proxy_uniform) if proxy_uniform != "" else "",
                "exact_weight_cv": exact_cv,
                "interpretation": "exact unavailable due to failed sparse replay"
                if exact_unavailable
                else "exact differs from proxy"
                if exact_ids != proxy_ids
                else "exact and proxy IDs match for this pixel",
            }
        )
    return out


def _weight_audit(rows: list[dict[str, Any]], scene: str, condition: str, subset_name: str) -> list[dict[str, Any]]:
    by_pixel: dict[tuple[str, int, int], list[float]] = defaultdict(list)
    for row in rows:
        if row.get("splat_weight") not in ("", None):
            by_pixel[(str(row.get("view_name", "")), int(row.get("pixel_x", 0)), int(row.get("pixel_y", 0)))].append(_number(row.get("splat_weight")))
    out = []
    for (view, x, y), weights in sorted(by_pixel.items()):
        cv = _cv(weights)
        out.append(
            {
                "scene": scene,
                "condition": condition,
                "subset_name": subset_name,
                "view_name": view,
                "pixel_x": x,
                "pixel_y": y,
                "exact_candidate_count": len(weights),
                "mean_splat_weight": _mean(weights),
                "std_splat_weight": _std(weights),
                "cv_splat_weight": cv,
                "min_splat_weight": min(weights) if weights else "",
                "max_splat_weight": max(weights) if weights else "",
                "weights_nonuniform": _bool_text((cv or 0) > 1e-9),
                "evidence_quality": "exact_sparse_render_contribution",
            }
        )
    return out


def _missing_fields(exact_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    field_specs = [
        ("alpha_contribution", "exact contribution weights", "limits alpha-aware attribution"),
        ("transmittance_before", "exact contribution weights", "limits compositing-order audit"),
        ("splat_weight", "residual weighted attribution", "limits weighted residual aggregation"),
        ("color_contribution_r", "color contribution", "limits channel attribution"),
    ]
    rows = []
    for field, required_for, impact in field_specs:
        available = any(row.get(field) not in ("", None) for row in exact_rows)
        rows.append(
            {
                "field_name": field,
                "available": _bool_text(available),
                "required_for": required_for,
                "impact": "" if available else impact,
                "fallback_used": "false",
                "notes": "",
            }
        )
    return rows


def _recommendations(summary: dict[str, Any]) -> dict[str, Any]:
    succeeded = bool(summary.get("exact_attribution_succeeded"))
    if summary.get("exact_render_contribution_succeeded"):
        next_step = "Proceed to PR21.2 exact-vs-proxy weighted attribution comparison and failure analysis."
    elif summary.get("exact_contributor_id_only_succeeded"):
        next_step = "Proceed to PR21.2 exact-vs-proxy contributor-ID comparison and failure analysis."
    else:
        next_step = "Inspect gsplat lower-level CUDA wrapper or implement a verified selected-pixel contributor kernel outside training."
    return {
        "recommended_next_step": next_step,
        "should_proceed_to_pr212_exact_vs_proxy_comparison": succeeded,
        "should_proceed_to_intervention": False,
        "should_use_exact_rows_for_training_gating": False,
        "should_modify_official_rasterizer_now": False,
        "safe_next_actions": [
            "inspect exact contributor IDs and exact-vs-proxy comparison",
            "compare direct/collateral exact overlap and train013 selected-pixel control",
            "keep exact attribution offline-only",
        ],
        "unsafe_next_actions": [
            "use exact rows for training gating",
            "claim defense success",
            "modify official trainer or third_party rasterizer",
            "silently fall back to PR20 proxy rows as exact evidence",
        ],
    }


def _write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# PR21.1 Exact Sparse Attribution",
        "",
        "PR21.1 is observation-only. It does not implement defense, view rejection, update suppression, or densification gating.",
        "",
        "## Summary",
        f"- Exact attribution succeeded: `{summary.get('exact_attribution_succeeded')}`",
        f"- Evidence quality: `{summary.get('evidence_quality')}`",
        f"- Exact rows: `{summary.get('exact_contribution_row_count')}`",
        f"- Unique exact Gaussians: `{summary.get('unique_exact_gaussian_count')}`",
        f"- Ready for intervention: `{summary.get('ready_for_intervention')}`",
        f"- Recommended next step: `{summary.get('recommended_next_step')}`",
        "",
        "## Inputs",
        f"- Scene: `{summary.get('scene')}`",
        f"- Condition: `{summary.get('condition')}`",
        f"- Iteration: `{summary.get('iteration')}`",
        "",
        "## Readiness Checks",
        f"- PR21 ready for exact attribution: `{summary.get('pr21_ready_for_exact_attribution')}`",
        f"- Selected-view matching supported: `{summary.get('selected_view_matching_supported')}`",
        "",
        "## gsplat Metadata Path",
        f"- Rasterization succeeded: `{summary.get('gsplat_rasterization_succeeded')}`",
        f"- Contributor path: `{summary.get('contributor_path_selected')}`",
        f"- Internal loop attempted: `{summary.get('internal_loop_replay_attempted')}`",
        f"- Internal loop succeeded: `{summary.get('internal_loop_replay_succeeded')}`",
        f"- Internal loop packed mode: `{summary.get('packed_mode_for_internal_loop')}`",
        f"- Shape validation succeeded: `{summary.get('internal_loop_shape_validation_succeeded')}`",
        f"- Rasterize-to-indices succeeded: `{summary.get('rasterize_to_indices_call_succeeded')}`",
        f"- Accumulate updated render alphas: `{summary.get('accumulate_updated_render_alphas')}`",
        f"- Contributor rows before selected-pixel filtering: `{summary.get('total_contributor_rows_before_filter')}`",
        f"- Selected-pixel hit count: `{summary.get('selected_pixel_hit_count')}`",
        f"- Contributor IDs available: `{summary.get('exact_contributor_ids_available')}`",
        "",
        "## Limitations",
        "Missing alpha/transmittance/color fields are reported in `pr211_missing_fields.csv`. PR21.1 never labels proxy rows as exact.",
        "",
        "## Recommendation",
        str(summary.get("recommended_next_step", "")),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _empty_exact_outputs(scene: str, condition: str, subset_name: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    return (
        [],
        [],
        [],
        _direct_collateral([], scene, condition, subset_name),
        _train013_control([], scene, condition, subset_name),
        [],
        [],
    )


def _tensor_shape(value: Any) -> str:
    shape = getattr(value, "shape", "")
    return str(tuple(shape)) if shape != "" else ""


def _shape_tuple(value: Any) -> tuple[int, ...]:
    shape = getattr(value, "shape", ())
    try:
        return tuple(int(item) for item in shape)
    except Exception:
        return ()


def _tensor_numel(value: Any) -> int:
    if hasattr(value, "numel"):
        return int(value.numel())
    try:
        return len(value)
    except Exception:
        return 0


def _shape_text(shape: tuple[int, ...]) -> str:
    return str(tuple(shape))


def _internal_shape_row(tensor_name: str, expected: tuple[int, ...], value: Any, shape_ok: bool, notes: str = "") -> dict[str, Any]:
    _, _, dtype, device = _value_type_shape(value)
    return {
        "tensor_name": tensor_name,
        "expected_shape": _shape_text(expected),
        "actual_shape": _tensor_shape(value),
        "dtype": dtype,
        "device": device,
        "shape_ok": _bool_text(shape_ok),
        "notes": notes,
    }


def validate_internal_loop_shapes(
    *,
    means2d: Any,
    conics: Any,
    opacities: Any,
    colors: Any,
    isect_offsets: Any,
    render_alphas: Any,
    transmittances: Any,
    image_width: int,
    image_height: int,
) -> tuple[bool, list[dict[str, Any]], tuple[int, ...], int]:
    means_shape = _shape_tuple(means2d)
    image_dims = means_shape[:-2] if len(means_shape) >= 2 else ()
    n_gaussians = means_shape[-2] if len(means_shape) >= 2 else 0
    channels = _shape_tuple(colors)[-1] if _shape_tuple(colors) else 0
    offsets_shape = _shape_tuple(isect_offsets)
    tile_shape = offsets_shape[-2:] if len(offsets_shape) >= 2 else ()
    expected = {
        "means2d": image_dims + (n_gaussians, 2),
        "conics": image_dims + (n_gaussians, 3),
        "opacities": image_dims + (n_gaussians,),
        "colors": image_dims + (n_gaussians, channels),
        "isect_offsets": image_dims + tile_shape,
        "render_alphas": image_dims + (image_height, image_width, 1),
        "transmittances": image_dims + (image_height, image_width),
    }
    values = {
        "means2d": means2d,
        "conics": conics,
        "opacities": opacities,
        "colors": colors,
        "isect_offsets": isect_offsets,
        "render_alphas": render_alphas,
        "transmittances": transmittances,
    }
    rows = []
    ok = True
    for name, expected_shape in expected.items():
        actual = _shape_tuple(values[name])
        shape_ok = bool(expected_shape) and actual == expected_shape
        ok = ok and shape_ok
        rows.append(_internal_shape_row(name, expected_shape, values[name], shape_ok))
    return ok, rows, image_dims, n_gaussians


def _expand_colors_for_internal_loop(colors: Any, means2d: Any) -> Any:
    image_dims = _shape_tuple(means2d)[:-2]
    n_gaussians = _shape_tuple(means2d)[-2] if len(_shape_tuple(means2d)) >= 2 else 0
    color_shape = _shape_tuple(colors)
    if color_shape[:-2] == image_dims and len(color_shape) >= 2:
        return colors
    if len(color_shape) == 2 and color_shape[0] == n_gaussians and image_dims and hasattr(colors, "reshape") and hasattr(colors, "expand"):
        return colors.reshape((1,) * len(image_dims) + color_shape).expand(image_dims + color_shape)
    return colors


def _to_int_list(value: Any) -> list[int]:
    if value is None:
        return []
    item = value
    if hasattr(item, "detach"):
        item = item.detach()
    if hasattr(item, "cpu"):
        item = item.cpu()
    if hasattr(item, "numpy"):
        item = item.numpy()
    if hasattr(item, "tolist"):
        item = item.tolist()
    if isinstance(item, (int, float)):
        return [int(item)]
    return [int(part) for part in item]


def _make_exact_contributor_rows_from_indices(
    *,
    gaussian_ids: list[int],
    pixel_ids: list[int],
    image_ids: list[int],
    meta: dict[str, Any],
    packed: bool,
    selected_views: list[dict[str, str]],
    selected_pixels: list[dict[str, Any]],
    max_contributors_per_pixel: int,
    attribution_method: str,
) -> tuple[list[dict[str, Any]], bool, str]:
    compact_map = None
    mapping_mode = "direct gaussian ids"
    if packed and "gaussian_ids" in meta:
        compact_map = _to_int_list(meta["gaussian_ids"])
        mapping_mode = "metadata.gaussian_ids compact mapping"
    selected_by_pixel = {(row["view_name"], int(row["pixel_id"])): row for row in selected_pixels}
    view_by_index = [row.get("requested_view_name", "") for row in selected_views]
    rows: list[dict[str, Any]] = []
    counts: dict[tuple[str, int], int] = defaultdict(int)
    for index, gid in enumerate(gaussian_ids):
        raw_gid = int(gid)
        compact_mapping_used = False
        if compact_map is not None and 0 <= raw_gid < len(compact_map):
            raw_gid = int(compact_map[raw_gid])
            compact_mapping_used = True
        pixel_id = int(pixel_ids[index]) if index < len(pixel_ids) else 0
        camera_id = int(image_ids[index]) if index < len(image_ids) else 0
        if camera_id >= len(view_by_index):
            continue
        view_name = view_by_index[camera_id]
        key = (view_name, pixel_id)
        selected = selected_by_pixel.get(key)
        if not selected:
            continue
        if counts[key] >= max_contributors_per_pixel:
            continue
        counts[key] += 1
        rows.append(
            {
                "scene": selected.get("scene", ""),
                "condition": selected.get("condition", ""),
                "subset_name": selected.get("subset_name", ""),
                "view_name": view_name,
                "view_group": selected.get("view_group") or _group_for_view(view_name),
                "pixel_x": selected.get("pixel_x", ""),
                "pixel_y": selected.get("pixel_y", ""),
                "pixel_id": pixel_id,
                "gaussian_id": raw_gid,
                "contributor_rank": counts[key],
                "depth_order": counts[key],
                "alpha_contribution": "",
                "transmittance_before": "",
                "splat_weight": "",
                "opacity_after_activation": "",
                "color_contribution_r": "",
                "color_contribution_g": "",
                "color_contribution_b": "",
                "residual_l1": selected.get("residual_l1", ""),
                "residual_weighted_splat": "",
                "evidence_quality": "exact_sparse_contributor_id_only",
                "attribution_method": attribution_method,
                "_compact_gaussian_id_mapping_used": compact_mapping_used,
            }
        )
    return rows, any(row.get("_compact_gaussian_id_mapping_used") for row in rows), mapping_mode


def recover_contributors_by_gsplat_internal_loop(
    *,
    rasterize_api: Any,
    accumulate_api: Any,
    meta: dict[str, Any],
    colors: Any,
    image_width: int,
    image_height: int,
    selected_views: list[dict[str, str]],
    selected_pixels: list[dict[str, Any]],
    max_contributors_per_pixel: int,
    torch: Any,
    packed: bool,
    batch_per_iter: int = 100,
) -> dict[str, Any]:
    shape_rows: list[dict[str, Any]] = []
    attempt_row: dict[str, Any] = {
        "attempt_name": "source_verified_internal_loop",
        "packed": _bool_text(packed),
        "attempted": "true",
        "succeeded": "false",
        "means2d_shape": _tensor_shape(meta.get("means2d")),
        "conics_shape": _tensor_shape(meta.get("conics")),
        "opacities_shape": _tensor_shape(meta.get("opacities")),
        "colors_shape": _tensor_shape(colors),
        "isect_offsets_shape": _tensor_shape(meta.get("isect_offsets")),
        "flatten_ids_shape": _tensor_shape(meta.get("flatten_ids")),
        "render_alphas_shape": "",
        "transmittances_shape": "",
        "num_batches": 0,
        "total_contributor_rows_before_filter": 0,
        "selected_pixel_hit_count": 0,
        "error": "",
        "notes": "",
    }
    required = ["means2d", "conics", "opacities", "isect_offsets", "flatten_ids"]
    missing = [name for name in required if name not in meta]
    if missing:
        attempt_row["error"] = f"missing metadata: {missing}"
        return {"rows": [], "shape_rows": shape_rows, "attempt_row": attempt_row, "status": {"error": attempt_row["error"]}}
    if rasterize_api is None:
        attempt_row["error"] = "rasterize_to_indices_in_range unavailable"
        return {"rows": [], "shape_rows": shape_rows, "attempt_row": attempt_row, "status": {"error": attempt_row["error"]}}
    if accumulate_api is None:
        attempt_row["error"] = "accumulate unavailable"
        return {"rows": [], "shape_rows": shape_rows, "attempt_row": attempt_row, "status": {"error": attempt_row["error"]}}

    try:
        means2d = meta["means2d"]
        conics = meta["conics"]
        opacities = meta["opacities"]
        isect_offsets = meta["isect_offsets"]
        flatten_ids = meta["flatten_ids"]
        colors_for_accumulate = _expand_colors_for_internal_loop(colors, means2d)
        image_dims = _shape_tuple(means2d)[:-2]
        device = getattr(means2d, "device", None)
        render_alphas = torch.zeros(image_dims + (image_height, image_width, 1), device=device)
        transmittances = 1.0 - render_alphas[..., 0]
        attempt_row["colors_shape"] = _tensor_shape(colors_for_accumulate)
        attempt_row["render_alphas_shape"] = _tensor_shape(render_alphas)
        attempt_row["transmittances_shape"] = _tensor_shape(transmittances)
        shapes_ok, shape_rows, _, _ = validate_internal_loop_shapes(
            means2d=means2d,
            conics=conics,
            opacities=opacities,
            colors=colors_for_accumulate,
            isect_offsets=isect_offsets,
            render_alphas=render_alphas,
            transmittances=transmittances,
            image_width=image_width,
            image_height=image_height,
        )
        if not shapes_ok:
            attempt_row["error"] = "internal loop shape validation failed"
            return {
                "rows": [],
                "shape_rows": shape_rows,
                "attempt_row": attempt_row,
                "status": {
                    "internal_loop_shape_validation_succeeded": False,
                    "error": attempt_row["error"],
                },
            }
        block_size = int(meta.get("tile_size", 16)) * int(meta.get("tile_size", 16))
        n_isects = _tensor_numel(flatten_ids)
        try:
            sentinel = torch.tensor([n_isects], device=device, dtype=getattr(isect_offsets, "dtype", None))
        except TypeError:
            sentinel = torch.tensor([n_isects], device=device)
        isect_offsets_fl = torch.cat([isect_offsets.flatten(), sentinel])
        max_range = int((isect_offsets_fl[1:] - isect_offsets_fl[:-1]).max().item())
        num_batches = (max_range + block_size - 1) // block_size if block_size else 0
        attempt_row["num_batches"] = num_batches
        collected_gids: list[int] = []
        collected_pixels: list[int] = []
        collected_images: list[int] = []
        rasterize_call_succeeded = False
        accumulate_succeeded = False
        for step in range(0, num_batches, batch_per_iter):
            transmittances = 1.0 - render_alphas[..., 0]
            result = _call_rasterize_to_indices_in_range_safely(
                rasterize_api,
                range_start=step,
                range_end=step + batch_per_iter,
                transmittances=transmittances,
                means2d=means2d,
                conics=conics,
                opacities=opacities,
                image_width=image_width,
                image_height=image_height,
                tile_size=int(meta.get("tile_size", 16)),
                isect_offsets=isect_offsets,
                flatten_ids=flatten_ids,
            )
            rasterize_call_succeeded = True
            gs_ids, pixel_ids, image_ids = _normalize_api_result(result)
            gs_list = _to_int_list(gs_ids)
            if not gs_list:
                break
            pixel_list = _to_int_list(pixel_ids)
            image_list = _to_int_list(image_ids)
            collected_gids.extend(gs_list)
            collected_pixels.extend(pixel_list)
            collected_images.extend(image_list if image_list else [0] * len(gs_list))
            _, accs_step = accumulate_api(
                means2d,
                conics,
                opacities,
                colors_for_accumulate,
                gs_ids,
                pixel_ids,
                image_ids,
                image_width,
                image_height,
            )
            render_alphas = render_alphas + accs_step * transmittances[..., None]
            accumulate_succeeded = True
        rows, compact_used, mapping_mode = _make_exact_contributor_rows_from_indices(
            gaussian_ids=collected_gids,
            pixel_ids=collected_pixels,
            image_ids=collected_images,
            meta=meta,
            packed=packed,
            selected_views=selected_views,
            selected_pixels=selected_pixels,
            max_contributors_per_pixel=max_contributors_per_pixel,
            attribution_method="gsplat_internal_loop_contributor_id_replay",
        )
        selected_keys = {(row["view_name"], int(row["pixel_id"])) for row in selected_pixels}
        hit_keys = {(row["view_name"], int(row["pixel_id"])) for row in rows}
        attempt_row["total_contributor_rows_before_filter"] = len(collected_gids)
        attempt_row["selected_pixel_hit_count"] = len(hit_keys)
        attempt_row["succeeded"] = _bool_text(bool(rows))
        if not rows:
            attempt_row["error"] = "selected_pixel_filtering_failed" if collected_gids else "internal loop returned no contributors"
        return {
            "rows": rows,
            "shape_rows": shape_rows,
            "attempt_row": attempt_row,
            "status": {
                "internal_loop_shape_validation_succeeded": True,
                "internal_loop_num_batches": num_batches,
                "total_contributor_rows_before_filter": len(collected_gids),
                "selected_pixel_hit_count": len(hit_keys),
                "selected_pixel_no_hit_count": max(0, len(selected_keys - hit_keys)),
                "selected_pixel_hit_rate": len(hit_keys) / len(selected_keys) if selected_keys else 0.0,
                "rasterize_to_indices_call_succeeded": rasterize_call_succeeded,
                "accumulate_succeeded": accumulate_succeeded,
                "compact_gaussian_id_mapping_used": compact_used,
                "gaussian_id_mapping_mode": mapping_mode,
                "error": attempt_row["error"],
            },
        }
    except Exception as exc:
        attempt_row["error"] = repr(exc)
        return {"rows": [], "shape_rows": shape_rows, "attempt_row": attempt_row, "status": {"error": repr(exc)}}


def _try_server_gsplat_replay(
    *,
    run_dir: Path,
    selected_views: list[dict[str, str]],
    selected_pixels: list[dict[str, Any]],
    iteration: int,
    device: str,
    max_contributors_per_pixel: int,
) -> dict[str, Any]:
    blockers: list[dict[str, Any]] = []
    metadata_rows: list[dict[str, Any]] = []
    exact_rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    api_rows: list[dict[str, Any]] = []
    transmittance_rows: list[dict[str, Any]] = []
    raster_output_rows: list[dict[str, Any]] = []
    source_rows = _audit_gsplat_source()
    internal_loop_shape_rows: list[dict[str, Any]] = []
    internal_loop_attempt_rows: list[dict[str, Any]] = []
    path_attempt_rows: list[dict[str, Any]] = []
    status: dict[str, Any] = {
        "transmittance_source_selected": "",
        "gsplat_rasterization_succeeded": False,
        "source_audit_completed": bool(source_rows),
        "contributor_path_selected": "source_level_failure",
        "contributor_path_reason": "",
        "exact_contributor_id_only_succeeded": False,
        "exact_render_contribution_succeeded": False,
        "transmittance_squeeze_applied": False,
        "transmittance_inversion_applied": False,
        "transmittance_resolution_succeeded": False,
        "contributor_api_dry_run_succeeded": False,
        "rasterize_to_indices_call_succeeded": False,
        "rasterize_output_tuple_interpretation": "",
        "compact_gaussian_id_mapping_used": False,
        "gaussian_id_mapping_mode": "",
        "internal_loop_replay_attempted": False,
        "internal_loop_replay_succeeded": False,
        "packed_mode_for_internal_loop": "",
        "internal_loop_shape_validation_succeeded": False,
        "internal_loop_batch_per_iter": 100,
        "internal_loop_num_batches": 0,
        "accumulate_updated_render_alphas": False,
        "total_contributor_rows_before_filter": 0,
        "sparse_contributor_filter_succeeded": False,
        "selected_pixel_hit_count": 0,
        "selected_pixel_no_hit_count": len(selected_pixels),
        "selected_pixel_hit_rate": 0.0,
        "pixel_id_convention": "pixel_id = y * width + x",
        "image_id_mapping_supported": False,
        "exact_contributor_id_row_count": 0,
        "exact_render_contribution_row_count": 0,
    }
    try:
        import numpy as np
        import torch
        import gsplat
    except Exception as exc:
        blockers.append(_blocker("error", "dependency", "torch/gsplat import failed", str(exc), "run inside server viewtrust-p0 environment"))
        return {
            "exact_rows": exact_rows,
            "metadata_rows": metadata_rows,
            "missing_rows": missing_rows,
            "blockers": blockers,
            "api_rows": api_rows,
            "transmittance_rows": transmittance_rows,
            "raster_output_rows": raster_output_rows,
            "source_rows": source_rows,
            "internal_loop_shape_rows": internal_loop_shape_rows,
            "internal_loop_attempt_rows": internal_loop_attempt_rows,
            "path_attempt_rows": path_attempt_rows,
            "status": status,
        }

    try:
        ply_path = run_dir / "trainer_output" / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"
        tensors, activation_rows = _load_checkpoint_tensors(ply_path, device=device, np=np, torch=torch)
        del activation_rows
        cameras = _load_camera_tensors(run_dir / "trainer_output" / "cameras.json", selected_views, device=device, torch=torch)
        rasterization = getattr(gsplat, "rasterization", None)
        if rasterization is None:
            rendering = __import__("gsplat.rendering", fromlist=["rasterization"])
            rasterization = getattr(rendering, "rasterization")
        api_rows.append(
            _api_audit_row(
                api_name="gsplat.rendering.rasterization",
                api=rasterization,
                provided_arguments=["means", "quats", "scales", "opacities", "colors", "viewmats", "Ks", "width", "height", "packed"],
                dry_run_attempted=False,
                dry_run_succeeded=False,
                dry_run_error="",
                notes="main replay call",
            )
        )
        with torch.no_grad():
            raster_outputs = rasterization(
                means=tensors["means"],
                quats=tensors["quats"],
                scales=tensors["scales"],
                opacities=tensors["opacities"],
                colors=tensors["colors"],
                viewmats=cameras["viewmats"],
                Ks=cameras["Ks"],
                width=int(cameras["width"]),
                height=int(cameras["height"]),
                packed=True,
            )
        if not isinstance(raster_outputs, tuple) or len(raster_outputs) < 3:
            raise RuntimeError(f"unexpected rasterization return type: {type(raster_outputs)}")
        meta = raster_outputs[2]
        if not isinstance(meta, dict):
            raise RuntimeError(f"unexpected rasterization metadata type: {type(meta)}")
        metadata_rows = _metadata_audit(meta)
        raster_output_rows = _rasterization_output_audit(tuple(raster_outputs), meta)
        status["gsplat_rasterization_succeeded"] = True
        api = _find_gsplat_callable("rasterize_to_indices_in_range")
        transmittances, source, transmittance_rows, api_trans_rows, trans_ok, trans_error = _resolve_transmittance_source(
            api=api,
            meta=meta,
            raster_outputs=tuple(raster_outputs),
            image_width=int(cameras["width"]),
            image_height=int(cameras["height"]),
            source_rows=source_rows,
        )
        api_rows.extend(api_trans_rows)
        status["transmittance_source_selected"] = source
        status["transmittance_squeeze_applied"] = "squeezed" in source
        status["transmittance_inversion_applied"] = source.startswith("one_minus_")
        status["transmittance_resolution_succeeded"] = trans_ok
        status["contributor_api_dry_run_succeeded"] = trans_ok
        status["contributor_path_selected"] = "path_a_public_rasterize_to_indices_in_range" if trans_ok else "source_level_failure"
        status["contributor_path_reason"] = f"selected transmittance source {source}" if trans_ok else trans_error
        if not trans_ok:
            blockers.append(
                _blocker(
                    "error",
                    "gsplat_transmittance_resolution",
                    "unable to resolve valid transmittances tensor for rasterize_to_indices_in_range",
                    f"metadata_keys={sorted(meta.keys())}; raster_outputs={[row.get('shape', '') for row in raster_output_rows]}; error={trans_error}",
                    "inspect gsplat 1.5.3 API/source or use a lower-level gsplat contributor path",
                )
            )
        path_attempt_rows.append(
            {
                "path_name": "path_a_public_rasterize_to_indices_in_range",
                "attempted": "true",
                "succeeded": "false",
                "evidence_quality_if_success": "exact_sparse_contributor_id_only",
                "transmittance_source": source,
                "output_tuple_shapes": ";".join(str(row.get("shape", "")) for row in raster_output_rows if str(row.get("output_name", "")).startswith("rasterization_output")),
                "output_tuple_interpretation": "gaussian_ids;pixel_ids;image_ids" if trans_ok else "unavailable",
                "gaussian_id_mapping": "",
                "pixel_id_mapping": "pixel_id = y * width + x",
                "image_id_mapping": "image_id indexes selected view order",
                "selected_pixel_hit_count": 0,
                "exact_row_count": 0,
                "error": "" if trans_ok else trans_error,
                "notes": "legacy path retained as audit; PR21.1c uses source-verified internal loop",
            }
        )

        internal_api = _find_gsplat_callable("rasterize_to_indices_in_range")
        accumulate_api = _find_gsplat_callable("accumulate")
        internal_attempts: list[tuple[str, bool, dict[str, Any]]] = []
        with torch.no_grad():
            try:
                unpacked_outputs = rasterization(
                    means=tensors["means"],
                    quats=tensors["quats"],
                    scales=tensors["scales"],
                    opacities=tensors["opacities"],
                    colors=tensors["colors"],
                    viewmats=cameras["viewmats"],
                    Ks=cameras["Ks"],
                    width=int(cameras["width"]),
                    height=int(cameras["height"]),
                    packed=False,
                )
                if isinstance(unpacked_outputs, tuple) and len(unpacked_outputs) >= 3 and isinstance(unpacked_outputs[2], dict):
                    internal_attempts.append(("packed_false", False, unpacked_outputs[2]))
                    metadata_rows = _metadata_audit(unpacked_outputs[2])
                else:
                    internal_loop_attempt_rows.append(
                        {
                            "attempt_name": "source_verified_internal_loop",
                            "packed": "false",
                            "attempted": "true",
                            "succeeded": "false",
                            "means2d_shape": "",
                            "conics_shape": "",
                            "opacities_shape": "",
                            "colors_shape": "",
                            "isect_offsets_shape": "",
                            "flatten_ids_shape": "",
                            "render_alphas_shape": "",
                            "transmittances_shape": "",
                            "num_batches": 0,
                            "total_contributor_rows_before_filter": 0,
                            "selected_pixel_hit_count": 0,
                            "error": f"unexpected unpacked rasterization output type: {type(unpacked_outputs)}",
                            "notes": "packed=False rasterization did not return metadata",
                        }
                    )
            except Exception as exc:
                internal_loop_attempt_rows.append(
                    {
                        "attempt_name": "source_verified_internal_loop",
                        "packed": "false",
                        "attempted": "true",
                        "succeeded": "false",
                        "means2d_shape": "",
                        "conics_shape": "",
                        "opacities_shape": "",
                        "colors_shape": "",
                        "isect_offsets_shape": "",
                        "flatten_ids_shape": "",
                        "render_alphas_shape": "",
                        "transmittances_shape": "",
                        "num_batches": 0,
                        "total_contributor_rows_before_filter": 0,
                        "selected_pixel_hit_count": 0,
                        "error": repr(exc),
                        "notes": "packed=False rasterization failed before internal-loop replay",
                    }
                )
        internal_attempts.append(("packed_true", True, meta))
        internal_error = ""
        for attempt_name, packed, attempt_meta in internal_attempts:
            if exact_rows:
                break
            result = recover_contributors_by_gsplat_internal_loop(
                rasterize_api=internal_api,
                accumulate_api=accumulate_api,
                meta=attempt_meta,
                colors=tensors["colors"],
                image_width=int(cameras["width"]),
                image_height=int(cameras["height"]),
                selected_views=selected_views,
                selected_pixels=selected_pixels,
                max_contributors_per_pixel=max_contributors_per_pixel,
                torch=torch,
                packed=packed,
                batch_per_iter=int(status["internal_loop_batch_per_iter"]),
            )
            attempt_row = dict(result["attempt_row"])
            attempt_row["attempt_name"] = attempt_name
            internal_loop_attempt_rows.append(attempt_row)
            internal_loop_shape_rows.extend(result["shape_rows"])
            attempt_status = result["status"]
            status["internal_loop_replay_attempted"] = True
            internal_error = str(attempt_status.get("error", ""))
            if attempt_status.get("internal_loop_shape_validation_succeeded"):
                status["internal_loop_shape_validation_succeeded"] = True
            if attempt_row.get("succeeded") == "true":
                exact_rows.extend(result["rows"])
                status["contributor_path_selected"] = "source_verified_internal_loop"
                status["contributor_path_reason"] = "replayed gsplat _torch_impl-style loop with transmittances = 1 - render_alphas[..., 0]"
                status["internal_loop_replay_succeeded"] = True
                status["packed_mode_for_internal_loop"] = "packed" if packed else "unpacked"
                status["internal_loop_num_batches"] = attempt_status.get("internal_loop_num_batches", 0)
                status["total_contributor_rows_before_filter"] = attempt_status.get("total_contributor_rows_before_filter", 0)
                status["selected_pixel_hit_count"] = attempt_status.get("selected_pixel_hit_count", 0)
                status["selected_pixel_no_hit_count"] = attempt_status.get("selected_pixel_no_hit_count", len(selected_pixels))
                status["selected_pixel_hit_rate"] = attempt_status.get("selected_pixel_hit_rate", 0.0)
                status["rasterize_to_indices_call_succeeded"] = bool(attempt_status.get("rasterize_to_indices_call_succeeded"))
                status["accumulate_updated_render_alphas"] = bool(attempt_status.get("accumulate_succeeded"))
                status["rasterize_output_tuple_interpretation"] = "gaussian_ids;pixel_ids;image_ids"
                status["compact_gaussian_id_mapping_used"] = bool(attempt_status.get("compact_gaussian_id_mapping_used"))
                status["gaussian_id_mapping_mode"] = attempt_status.get("gaussian_id_mapping_mode", "direct gaussian ids")
                status["sparse_contributor_filter_succeeded"] = True
                status["image_id_mapping_supported"] = True
                status["exact_contributor_id_only_succeeded"] = True
                status["exact_render_contribution_succeeded"] = False
                status["exact_contributor_id_row_count"] = len(exact_rows)
                status["exact_render_contribution_row_count"] = 0
                path_attempt_rows.append(
                    {
                        "path_name": "source_verified_internal_loop",
                        "attempted": "true",
                        "succeeded": "true",
                        "evidence_quality_if_success": "exact_sparse_contributor_id_only",
                        "transmittance_source": "dynamic: 1.0 - render_alphas[..., 0]",
                        "output_tuple_shapes": ";".join(str(row.get("shape", "")) for row in raster_output_rows if str(row.get("output_name", "")).startswith("rasterization_output")),
                        "output_tuple_interpretation": "gaussian_ids;pixel_ids;image_ids",
                        "gaussian_id_mapping": status["gaussian_id_mapping_mode"],
                        "pixel_id_mapping": "pixel_id = y * width + x",
                        "image_id_mapping": "image_id indexes selected view order",
                        "selected_pixel_hit_count": status["selected_pixel_hit_count"],
                        "exact_row_count": len(exact_rows),
                        "error": "",
                        "notes": "ID-only evidence; alpha/transmittance/splat weights not claimed",
                    }
                )
        if not exact_rows and trans_ok and transmittances is not None:
            contributor_rows = _extract_contributors_with_gsplat_api(
                api=api,
                meta=meta,
                transmittances=transmittances,
                image_width=int(cameras["width"]),
                image_height=int(cameras["height"]),
                selected_views=selected_views,
                selected_pixels=selected_pixels,
                max_contributors_per_pixel=max_contributors_per_pixel,
                torch=torch,
            )
            exact_rows.extend(contributor_rows)
            status["rasterize_to_indices_call_succeeded"] = True
            status["rasterize_output_tuple_interpretation"] = "gaussian_ids;pixel_ids;image_ids"
            status["compact_gaussian_id_mapping_used"] = any(row.get("_compact_gaussian_id_mapping_used") for row in exact_rows)
            status["gaussian_id_mapping_mode"] = "metadata.gaussian_ids compact mapping" if status["compact_gaussian_id_mapping_used"] else "direct gaussian ids"
            selected_keys = {(row["view_name"], int(row["pixel_id"])) for row in selected_pixels}
            hit_keys = {(row["view_name"], int(row["pixel_id"])) for row in exact_rows}
            status["selected_pixel_hit_count"] = len(hit_keys)
            status["selected_pixel_no_hit_count"] = max(0, len(selected_keys - hit_keys))
            status["selected_pixel_hit_rate"] = len(hit_keys) / len(selected_keys) if selected_keys else 0.0
            status["sparse_contributor_filter_succeeded"] = bool(hit_keys)
            status["image_id_mapping_supported"] = bool(exact_rows)
            status["exact_contributor_id_only_succeeded"] = bool(exact_rows)
            status["exact_render_contribution_succeeded"] = False
            status["exact_contributor_id_row_count"] = len(exact_rows)
            status["exact_render_contribution_row_count"] = 0
            if exact_rows:
                path_attempt_rows[0]["succeeded"] = "true"
                path_attempt_rows[0]["gaussian_id_mapping"] = status["gaussian_id_mapping_mode"]
                path_attempt_rows[0]["selected_pixel_hit_count"] = status["selected_pixel_hit_count"]
                path_attempt_rows[0]["exact_row_count"] = len(exact_rows)
                path_attempt_rows[0]["error"] = ""
        if not exact_rows:
            blockers.append(
                _blocker(
                    "error",
                    "gsplat_sparse_contributors",
                    "source-verified internal loop did not recover selected-pixel contributor IDs",
                    f"packed_false_attempted=true; packed_true_attempted=true; internal_error={internal_error}",
                    "inspect pr211_internal_loop_shape_audit.csv and pr211_internal_loop_attempts.csv",
                )
            )
    except Exception as exc:
        blockers.append(_blocker("error", "gsplat_sparse_replay", "gsplat sparse replay failed", repr(exc), "fix tensor/camera/API conversion before PR21.2"))
    if not metadata_rows:
        metadata_rows = [{"metadata_key": "", "available": "false", "type": "", "shape": "", "dtype": "", "device": "", "used_for": "", "notes": "gsplat replay did not produce metadata"}]
    if not raster_output_rows:
        raster_output_rows = [{"output_name": "", "available": "false", "type": "", "shape": "", "dtype": "", "device": "", "notes": "gsplat replay did not produce rasterization outputs"}]
    if not transmittance_rows:
        transmittance_rows = [
            _transmittance_audit_row(
                source="unavailable",
                value=None,
                available=False,
                shape_compatible=False,
                dry_run_attempted=False,
                dry_run_succeeded=False,
                selected=False,
                notes="no transmittance candidates audited",
            )
        ]
    if exact_rows:
        for field in ("alpha_contribution", "transmittance_before", "splat_weight"):
            if not any(row.get(field) not in ("", None) for row in exact_rows):
                missing_rows.append(
                    {
                        "field_name": field,
                        "available": "false",
                        "required_for": "exact contribution weighting",
                        "impact": "exact contributor IDs are present but this contribution scalar is unavailable",
                        "fallback_used": "false",
                        "notes": "no proxy fallback used",
                    }
                )
    return {
        "exact_rows": exact_rows,
        "metadata_rows": metadata_rows,
        "missing_rows": missing_rows,
        "blockers": blockers,
        "api_rows": api_rows,
        "transmittance_rows": transmittance_rows,
        "raster_output_rows": raster_output_rows,
        "source_rows": source_rows,
        "internal_loop_shape_rows": internal_loop_shape_rows,
        "internal_loop_attempt_rows": internal_loop_attempt_rows,
        "path_attempt_rows": path_attempt_rows,
        "status": status,
    }


def _load_checkpoint_tensors(ply_path: Path, *, device: str, np: Any, torch: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    data = _read_ply_table(ply_path, np=np)
    means = np.stack([data["x"], data["y"], data["z"]], axis=1).astype("float32")
    quats = np.stack([data["rot_0"], data["rot_1"], data["rot_2"], data["rot_3"]], axis=1).astype("float32")
    quat_norm = np.linalg.norm(quats, axis=1, keepdims=True)
    quats = quats / np.maximum(quat_norm, 1e-12)
    scales = np.exp(np.stack([data["scale_0"], data["scale_1"], data["scale_2"]], axis=1)).astype("float32")
    opacities = (1.0 / (1.0 + np.exp(-data["opacity"]))).astype("float32")
    colors = np.clip(np.stack([data["f_dc_0"], data["f_dc_1"], data["f_dc_2"]], axis=1) * 0.28209479177387814 + 0.5, 0.0, 1.0).astype("float32")
    tensors = {
        "means": torch.as_tensor(means, dtype=torch.float32, device=device),
        "quats": torch.as_tensor(quats, dtype=torch.float32, device=device),
        "scales": torch.as_tensor(scales, dtype=torch.float32, device=device),
        "opacities": torch.as_tensor(opacities, dtype=torch.float32, device=device),
        "colors": torch.as_tensor(colors, dtype=torch.float32, device=device),
    }
    return tensors, []


def _read_ply_table(ply_path: Path, *, np: Any) -> Any:
    header = parse_ply_header(ply_path)
    properties = header.get("properties", [])
    names = [item["property_name"] for item in properties]
    dtype_map = {"float": "f4", "float32": "f4", "double": "f8", "uchar": "u1", "uint8": "u1", "int": "i4", "int32": "i4", "uint": "u4"}
    dtype = [(item["property_name"], dtype_map.get(str(item["property_type"]).lower(), "f4")) for item in properties]
    header_bytes = 0
    with ply_path.open("rb") as handle:
        while True:
            line = handle.readline()
            header_bytes += len(line)
            if line.strip() == b"end_header":
                break
    if "binary_little_endian" in str(header.get("format", "")):
        return np.fromfile(ply_path, dtype=np.dtype(dtype), count=int(header["vertex_count"]), offset=header_bytes)
    if "binary_big_endian" in str(header.get("format", "")):
        big_dtype = [(name, ">" + spec) for name, spec in dtype]
        return np.fromfile(ply_path, dtype=np.dtype(big_dtype), count=int(header["vertex_count"]), offset=header_bytes)
    raw = np.genfromtxt(ply_path, skip_header=len(header.get("header_lines", [])), max_rows=int(header["vertex_count"]))
    return np.core.records.fromarrays(raw.T, names=",".join(names))


def _load_camera_tensors(cameras_json_path: Path, selected_views: list[dict[str, str]], *, device: str, torch: Any) -> dict[str, Any]:
    cameras = json.loads(cameras_json_path.read_text(encoding="utf-8"))
    if isinstance(cameras, dict):
        cameras = cameras.get("cameras") or cameras.get("frames") or []
    by_name = {Path(str(row.get("img_name") or row.get("image_name") or row.get("file_path") or "")).stem: row for row in cameras}
    viewmats = []
    Ks = []
    width = None
    height = None
    for selected in selected_views:
        name = selected.get("matched_camera_img_name", "")
        camera = by_name[name]
        width = int(camera.get("width") or camera.get("w") or width or 400)
        height = int(camera.get("height") or camera.get("h") or height or 400)
        fx = float(camera.get("fx") or camera.get("fl_x") or width)
        fy = float(camera.get("fy") or camera.get("fl_y") or fx)
        cx = float(camera.get("cx") or width / 2)
        cy = float(camera.get("cy") or height / 2)
        R = camera.get("rotation") or camera.get("R") or [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        T = camera.get("position") or camera.get("translation") or camera.get("T") or [0, 0, 0]
        mat = torch.eye(4, dtype=torch.float32, device=device)
        mat[:3, :3] = torch.as_tensor(R, dtype=torch.float32, device=device)
        mat[:3, 3] = torch.as_tensor(T, dtype=torch.float32, device=device)
        viewmats.append(mat)
        Ks.append(torch.tensor([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=torch.float32, device=device))
    return {"viewmats": torch.stack(viewmats), "Ks": torch.stack(Ks), "width": width or 400, "height": height or 400}


def _extract_contributors_with_gsplat_api(
    *,
    api: Any,
    meta: dict[str, Any],
    transmittances: Any,
    image_width: int,
    image_height: int,
    selected_views: list[dict[str, str]],
    selected_pixels: list[dict[str, Any]],
    max_contributors_per_pixel: int,
    torch: Any,
) -> list[dict[str, Any]]:
    if api is None:
        return []
    tile_size = int(meta.get("tile_size", 16))
    result = _call_rasterize_to_indices_in_range_safely(
        api,
        range_start=0,
        range_end=image_width * image_height * max(1, len(selected_views)),
        transmittances=transmittances,
        means2d=meta["means2d"],
        conics=meta["conics"],
        opacities=meta["opacities"],
        image_width=image_width,
        image_height=image_height,
        tile_size=tile_size,
        isect_offsets=meta["isect_offsets"],
        flatten_ids=meta["flatten_ids"],
    )
    if not isinstance(result, (tuple, list)) or len(result) < 3:
        return []
    tensors = [item.detach().cpu() if hasattr(item, "detach") else item for item in result]
    arrays = [item.numpy() if hasattr(item, "numpy") else item for item in tensors]
    names = ["gaussian_ids", "pixel_ids", "camera_ids"]
    mapped = {name: arrays[index] for index, name in enumerate(names) if index < len(arrays)}
    if "gaussian_ids" not in mapped or "pixel_ids" not in mapped:
        return []
    compact_map = None
    if "gaussian_ids" in meta:
        meta_gids = meta["gaussian_ids"].detach().cpu().numpy() if hasattr(meta["gaussian_ids"], "detach") else meta["gaussian_ids"]
        compact_map = meta_gids
    selected_by_pixel = {(row["view_name"], int(row["pixel_id"])): row for row in selected_pixels}
    view_by_index = [row.get("requested_view_name", "") for row in selected_views]
    rows: list[dict[str, Any]] = []
    counts: dict[tuple[str, int], int] = defaultdict(int)
    for index, gid in enumerate(mapped["gaussian_ids"]):
        raw_gid = int(gid)
        compact_mapping_used = False
        if compact_map is not None and 0 <= raw_gid < len(compact_map):
            raw_gid = int(compact_map[raw_gid])
            compact_mapping_used = True
        pixel_id = int(mapped["pixel_ids"][index])
        camera_id = int(mapped.get("camera_ids", [0] * len(mapped["gaussian_ids"]))[index])
        if camera_id >= len(view_by_index):
            continue
        view_name = view_by_index[camera_id]
        key = (view_name, pixel_id)
        selected = selected_by_pixel.get(key)
        if not selected:
            continue
        if counts[key] >= max_contributors_per_pixel:
            continue
        counts[key] += 1
        rows.append(
            {
                "scene": selected.get("scene", ""),
                "condition": selected.get("condition", ""),
                "subset_name": selected.get("subset_name", ""),
                "view_name": view_name,
                "view_group": selected.get("view_group") or _group_for_view(view_name),
                "pixel_x": selected.get("pixel_x", ""),
                "pixel_y": selected.get("pixel_y", ""),
                "pixel_id": pixel_id,
                "gaussian_id": raw_gid,
                "contributor_rank": counts[key],
                "depth_order": counts[key],
                "alpha_contribution": "",
                "transmittance_before": "",
                "splat_weight": "",
                "opacity_after_activation": "",
                "color_contribution_r": "",
                "color_contribution_g": "",
                "color_contribution_b": "",
                "residual_l1": selected.get("residual_l1", ""),
                "residual_weighted_splat": "",
                "evidence_quality": "exact_sparse_contributor_id_only",
                "attribution_method": "gsplat_sparse_contributor_id_replay",
                "_compact_gaussian_id_mapping_used": compact_mapping_used,
            }
        )
    return rows


def build_pr211_exact_sparse_attribution(
    *,
    run_dir: Path,
    pr200_dir: Path,
    pr210_dir: Path,
    scene: str,
    condition: str,
    subset_name: str,
    iteration: int,
    split: str,
    views: list[str],
    output_dir: Path,
    device: str = "cuda:0",
    top_pixels_per_view: int = 128,
    max_contributors_per_pixel: int = 16,
    write_markdown: bool = False,
    synthetic_exact_rows: list[dict[str, Any]] | None = None,
    force_failure: bool = False,
) -> tuple[dict[str, Any], int]:
    del write_markdown
    output_dir.mkdir(parents=True, exist_ok=True)
    point_cloud = run_dir / "trainer_output" / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"
    cameras_json = run_dir / "trainer_output" / "cameras.json"
    pr210_summary = load_json(pr210_dir / "pr210_gsplat_feasibility_summary.json")
    dependency = probe_dependencies()
    input_rows = [
        _input_row("run_dir", run_dir, True),
        _input_row("pr200_dir", pr200_dir, True),
        _input_row("pr210_dir", pr210_dir, True),
        _input_row("point_cloud.ply", point_cloud, True),
        _input_row("cameras.json", cameras_json, True),
        _input_row("selected view audit", pr210_dir / "pr210_selected_view_audit.csv", True),
        _input_row("official render root", run_dir / "view_evaluation" / "render_models" / "train_test_model" / split / f"ours_{iteration}" / "renders", True),
        _input_row("official GT root", run_dir / "view_evaluation" / "render_models" / "train_test_model" / split / f"ours_{iteration}" / "gt", True),
    ]
    blockers = [
        _blocker("error", "input", row["input_name"] + " missing", row["path"], "provide required PR21.1 input")
        for row in input_rows
        if row["required"] == "true" and row["exists"] != "true"
    ]
    if not pr210_summary.get("pr21_ready_for_exact_attribution", False):
        blockers.append(_blocker("error", "pr210_readiness", "PR21.0a did not mark run ready for exact attribution", str(pr210_dir), "rerun/fix PR21.0a"))
    if not pr210_summary.get("selected_view_matching_supported", False):
        blockers.append(_blocker("error", "selected_view_matching", "PR21.0a selected-view matching unsupported", str(pr210_dir), "fix strict selected-view matching"))

    selected_views, view_blockers = _load_pr210_selected_views(pr210_dir, views)
    blockers.extend(view_blockers)
    selected_pixels, proxy_by_pixel = _load_selected_pixels(
        pr200_dir=pr200_dir,
        selected_views=selected_views,
        scene=scene,
        condition=condition,
        subset_name=subset_name,
        top_pixels_per_view=top_pixels_per_view,
    )
    conversion_rows, conversion_supported = audit_checkpoint_conversion(point_cloud, cameras_json, scene, device)
    activation_rows = _activation_audit(conversion_rows, device)
    camera_supported = bool(selected_views) and any(
        row.get("conversion_step") in {"camera intrinsics", "camera extrinsics"} and row.get("supported") == "true"
        for row in conversion_rows
    )
    exact_rows: list[dict[str, Any]] = []
    metadata_rows: list[dict[str, Any]] = []
    contributor_api_rows: list[dict[str, Any]] = []
    transmittance_rows: list[dict[str, Any]] = []
    raster_output_rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = [{"item": "local-safe", "path": "", "line_number": "", "symbol": "", "signature": "", "snippet": "", "notes": "source audit requires installed gsplat"}]
    internal_loop_shape_rows: list[dict[str, Any]] = []
    internal_loop_attempt_rows: list[dict[str, Any]] = []
    path_attempt_rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    replay_status: dict[str, Any] = {
        "transmittance_source_selected": "",
        "gsplat_rasterization_succeeded": False,
        "source_audit_completed": False,
        "contributor_path_selected": "source_level_failure",
        "contributor_path_reason": "",
        "exact_contributor_id_only_succeeded": False,
        "exact_render_contribution_succeeded": False,
        "transmittance_squeeze_applied": False,
        "transmittance_inversion_applied": False,
        "transmittance_resolution_succeeded": False,
        "contributor_api_dry_run_succeeded": False,
        "rasterize_to_indices_call_succeeded": False,
        "rasterize_output_tuple_interpretation": "",
        "compact_gaussian_id_mapping_used": False,
        "gaussian_id_mapping_mode": "",
        "internal_loop_replay_attempted": False,
        "internal_loop_replay_succeeded": False,
        "packed_mode_for_internal_loop": "",
        "internal_loop_shape_validation_succeeded": False,
        "internal_loop_batch_per_iter": 100,
        "internal_loop_num_batches": 0,
        "accumulate_updated_render_alphas": False,
        "total_contributor_rows_before_filter": 0,
        "sparse_contributor_filter_succeeded": False,
        "selected_pixel_hit_count": 0,
        "selected_pixel_no_hit_count": len(selected_pixels),
        "selected_pixel_hit_rate": 0.0,
        "pixel_id_convention": "pixel_id = y * width + x",
        "image_id_mapping_supported": False,
        "exact_contributor_id_row_count": 0,
        "exact_render_contribution_row_count": 0,
    }
    if synthetic_exact_rows is not None:
        exact_rows = [dict(row) for row in synthetic_exact_rows]
        synthetic_render_success = any(row.get("evidence_quality") == "exact_sparse_render_contribution" for row in exact_rows)
        selected_keys = {(row["view_name"], int(row["pixel_id"])) for row in exact_rows if row.get("pixel_id") not in ("", None)}
        metadata_rows = [{"metadata_key": "synthetic_contributors", "available": "true", "type": "list", "shape": len(exact_rows), "dtype": "", "device": "cpu", "used_for": "smoke test aggregation", "notes": "local-safe synthetic exact rows"}]
        contributor_api_rows = [
            {
                "api_name": "rasterize_to_indices_in_range",
                "available": "true",
                "signature": "synthetic",
                "required_arguments": "range_start;range_end;transmittances;means2d;conics;opacities;image_width;image_height;tile_size;isect_offsets;flatten_ids",
                "provided_arguments": "synthetic_exact_rows",
                "compatible": "true",
                "dry_run_attempted": "true",
                "dry_run_succeeded": "true",
                "dry_run_error": "",
                "notes": "local-safe synthetic success path",
            }
        ]
        transmittance_rows = [
            {
                "candidate_source": "synthetic.transmittances",
                "available": "true",
                "type": "list",
                "shape": len(exact_rows),
                "dtype": "",
                "device": "cpu",
                "shape_compatible": "true",
                "dry_run_attempted": "true",
                "dry_run_succeeded": "true",
                "selected_as_transmittance": "true",
                "notes": "local-safe synthetic success path",
            }
        ]
        raster_output_rows = [{"output_name": "synthetic_exact_rows", "available": "true", "type": "list", "shape": len(exact_rows), "dtype": "", "device": "cpu", "notes": "local-safe synthetic success path"}]
        internal_loop_shape_rows = [
            {
                "tensor_name": "synthetic",
                "expected_shape": "",
                "actual_shape": str(len(exact_rows)),
                "dtype": "",
                "device": "cpu",
                "shape_ok": "true",
                "notes": "local-safe synthetic exact rows bypass server-only internal loop",
            }
        ]
        internal_loop_attempt_rows = [
            {
                "attempt_name": "synthetic_exact_rows",
                "packed": "",
                "attempted": "false",
                "succeeded": "true",
                "means2d_shape": "",
                "conics_shape": "",
                "opacities_shape": "",
                "colors_shape": "",
                "isect_offsets_shape": "",
                "flatten_ids_shape": "",
                "render_alphas_shape": "",
                "transmittances_shape": "",
                "num_batches": 0,
                "total_contributor_rows_before_filter": len(exact_rows),
                "selected_pixel_hit_count": len(selected_keys),
                "error": "",
                "notes": "local-safe synthetic path",
            }
        ]
        replay_status.update(
            {
                "transmittance_source_selected": "synthetic.transmittances",
                "gsplat_rasterization_succeeded": True,
                "source_audit_completed": True,
                "contributor_path_selected": "synthetic_exact_rows",
                "contributor_path_reason": "local-safe synthetic exact rows",
                "exact_contributor_id_only_succeeded": True,
                "exact_render_contribution_succeeded": synthetic_render_success,
                "transmittance_resolution_succeeded": True,
                "contributor_api_dry_run_succeeded": True,
                "rasterize_to_indices_call_succeeded": True,
                "rasterize_output_tuple_interpretation": "synthetic",
                "gaussian_id_mapping_mode": "direct gaussian ids",
                "sparse_contributor_filter_succeeded": bool(exact_rows),
                "selected_pixel_hit_count": len(selected_keys),
                "selected_pixel_no_hit_count": max(0, len(selected_pixels) - len(selected_keys)),
                "selected_pixel_hit_rate": len(selected_keys) / len(selected_pixels) if selected_pixels else 0.0,
                "image_id_mapping_supported": True,
                "exact_contributor_id_row_count": len(exact_rows),
                "exact_render_contribution_row_count": len(exact_rows) if synthetic_render_success else 0,
            }
        )
        path_attempt_rows = [
            {
                "path_name": "synthetic_exact_rows",
                "attempted": "true",
                "succeeded": "true",
                "evidence_quality_if_success": "exact_sparse_render_contribution",
                "transmittance_source": "synthetic.transmittances",
                "output_tuple_shapes": str(len(exact_rows)),
                "output_tuple_interpretation": "synthetic",
                "gaussian_id_mapping": "direct gaussian ids",
                "pixel_id_mapping": "pixel_id = y * width + x",
                "image_id_mapping": "synthetic view names",
                "selected_pixel_hit_count": replay_status["selected_pixel_hit_count"],
                "exact_row_count": len(exact_rows),
                "error": "",
                "notes": "local-safe synthetic path",
            }
        ]
    elif force_failure or blockers:
        metadata_rows = [{"metadata_key": "", "available": "false", "type": "", "shape": "", "dtype": "", "device": "", "used_for": "", "notes": "skipped due to readiness blockers"}]
        contributor_api_rows = [
            _api_audit_row(
                api_name="rasterize_to_indices_in_range",
                api=None,
                provided_arguments=[],
                dry_run_attempted=False,
                dry_run_succeeded=False,
                dry_run_error="skipped",
                notes="skipped due to readiness blockers or forced failure",
            )
        ]
        transmittance_rows = [
            _transmittance_audit_row(
                source="unavailable",
                value=None,
                available=False,
                shape_compatible=False,
                dry_run_attempted=False,
                dry_run_succeeded=False,
                selected=False,
                notes="unable to resolve valid transmittances tensor for rasterize_to_indices_in_range",
            )
        ]
        raster_output_rows = [{"output_name": "", "available": "false", "type": "", "shape": "", "dtype": "", "device": "", "notes": "skipped due to readiness blockers or forced failure"}]
        internal_loop_shape_rows = [
            {
                "tensor_name": "",
                "expected_shape": "",
                "actual_shape": "",
                "dtype": "",
                "device": "",
                "shape_ok": "false",
                "notes": "skipped due to readiness blockers or forced failure",
            }
        ]
        internal_loop_attempt_rows = [
            {
                "attempt_name": "source_verified_internal_loop",
                "packed": "",
                "attempted": "false",
                "succeeded": "false",
                "means2d_shape": "",
                "conics_shape": "",
                "opacities_shape": "",
                "colors_shape": "",
                "isect_offsets_shape": "",
                "flatten_ids_shape": "",
                "render_alphas_shape": "",
                "transmittances_shape": "",
                "num_batches": 0,
                "total_contributor_rows_before_filter": 0,
                "selected_pixel_hit_count": 0,
                "error": "skipped",
                "notes": "forced or readiness failure path",
            }
        ]
        path_attempt_rows = [
            {
                "path_name": "source_level_failure",
                "attempted": "false",
                "succeeded": "false",
                "evidence_quality_if_success": "",
                "transmittance_source": "",
                "output_tuple_shapes": "",
                "output_tuple_interpretation": "",
                "gaussian_id_mapping": "",
                "pixel_id_mapping": "pixel_id = y * width + x",
                "image_id_mapping": "",
                "selected_pixel_hit_count": 0,
                "exact_row_count": 0,
                "error": "unable to resolve valid transmittances tensor for rasterize_to_indices_in_range",
                "notes": "forced or readiness failure path",
            }
        ]
        if force_failure and not blockers:
            blockers.append(
                _blocker(
                    "error",
                    "gsplat_transmittance_resolution",
                    "unable to resolve valid transmittances tensor for rasterize_to_indices_in_range",
                    "forced local-safe failure path",
                    "inspect gsplat 1.5.3 API/source or use a lower-level gsplat contributor path",
                )
            )
    else:
        replay_result = _try_server_gsplat_replay(
            run_dir=run_dir,
            selected_views=selected_views,
            selected_pixels=selected_pixels,
            iteration=iteration,
            device=device,
            max_contributors_per_pixel=max_contributors_per_pixel,
        )
        exact_rows = replay_result["exact_rows"]
        metadata_rows = replay_result["metadata_rows"]
        contributor_api_rows = replay_result["api_rows"]
        transmittance_rows = replay_result["transmittance_rows"]
        raster_output_rows = replay_result["raster_output_rows"]
        source_rows = replay_result["source_rows"]
        internal_loop_shape_rows = replay_result["internal_loop_shape_rows"]
        internal_loop_attempt_rows = replay_result["internal_loop_attempt_rows"]
        path_attempt_rows = replay_result["path_attempt_rows"]
        replay_status.update(replay_result["status"])
        replay_missing = replay_result["missing_rows"]
        replay_blockers = replay_result["blockers"]
        missing_rows.extend(replay_missing)
        blockers.extend(replay_blockers)
    if not exact_rows and not any(row.get("component") == "gsplat_sparse_contributors" for row in blockers):
        blockers.append(
            _blocker(
                "error",
                "gsplat_sparse_contributors",
                "exact contributor IDs unavailable",
                "no exact rows produced; no proxy fallback used",
                "fix gsplat sparse contributor extraction before PR21.2",
            )
        )
    if exact_rows:
        blockers = [row for row in blockers if row.get("component") not in {"gsplat_sparse_contributors", "gsplat_transmittance_resolution"}]
    legacy_path_succeeded = bool(
        replay_status.get("transmittance_resolution_succeeded")
        and replay_status.get("contributor_api_dry_run_succeeded")
        and replay_status.get("rasterize_to_indices_call_succeeded")
    )
    internal_loop_succeeded = bool(replay_status.get("internal_loop_replay_succeeded"))
    exact_succeeded = bool(exact_rows and (legacy_path_succeeded or internal_loop_succeeded) and replay_status.get("selected_pixel_hit_count", 0) > 0)
    if not exact_succeeded:
        exact_rows = []
    exact_render_success = bool(exact_succeeded and replay_status.get("exact_render_contribution_succeeded"))
    exact_id_only_success = bool(exact_succeeded and not exact_render_success)
    evidence_quality = (
        "exact_sparse_render_contribution"
        if exact_render_success
        else "exact_sparse_contributor_id_only"
        if exact_id_only_success
        else "failed_exact_sparse_replay"
    )
    for row in exact_rows:
        row.setdefault("scene", scene)
        row.setdefault("condition", condition)
        row.setdefault("subset_name", subset_name)
        row.setdefault("view_group", _group_for_view(str(row.get("view_name", ""))))
        row.setdefault("evidence_quality", evidence_quality)
        row.setdefault(
            "attribution_method",
            "gsplat_sparse_replay" if evidence_quality == "exact_sparse_render_contribution" else "gsplat_sparse_contributor_id_replay",
        )
    gaussian_rows = _aggregate_gaussian(exact_rows, scene, condition, subset_name)
    group_rows = _aggregate_groups(exact_rows, selected_pixels, scene, condition, subset_name)
    direct_row = _direct_collateral(exact_rows, scene, condition, subset_name)
    train_row = _train013_control(exact_rows, scene, condition, subset_name)
    comparison_rows = _proxy_comparison(exact_rows, proxy_by_pixel, scene, condition, subset_name)
    weight_rows = _weight_audit(exact_rows, scene, condition, subset_name)
    missing_rows.extend(_missing_fields(exact_rows))
    contributors_per_pixel = []
    by_pixel: dict[tuple[Any, Any, Any], int] = defaultdict(int)
    for row in exact_rows:
        by_pixel[(row.get("view_name"), row.get("pixel_x"), row.get("pixel_y"))] += 1
    contributors_per_pixel = list(by_pixel.values())
    exact_weights = [_number(row.get("splat_weight")) for row in exact_rows if row.get("splat_weight") not in ("", None)]
    exact_vs_proxy_jaccards = [_number(row.get("jaccard")) for row in comparison_rows if row.get("jaccard") not in ("", None)]
    summary = {
        "schema_name": "viewtrust.pr211.exact_sparse_attribution.summary",
        "schema_version": 1,
        "created_at_utc": _utc_now(),
        "observation_only": True,
        "training_intervention": False,
        "defense_enabled": False,
        "view_rejection_enabled": False,
        "densification_gating_enabled": False,
        "training_behavior_modified": False,
        "rendering_behavior_modified_for_training": False,
        "third_party_modified": False,
        "scene": scene,
        "condition": condition,
        "subset_name": subset_name,
        "iteration": iteration,
        "split": split,
        "gsplat_available": bool(dependency.get("gsplat_import_ok")) or synthetic_exact_rows is not None,
        "gsplat_version": dependency.get("gsplat_version"),
        "device": device,
        "pr20_input_available": pr200_dir.exists(),
        "pr21_ready_for_exact_attribution": bool(pr210_summary.get("pr21_ready_for_exact_attribution", False)),
        "selected_view_matching_supported": bool(pr210_summary.get("selected_view_matching_supported", False)),
        "selected_view_count_requested": len(views),
        "selected_view_count_valid": len(selected_views) - len(view_blockers),
        "selected_pixel_count": len(selected_pixels),
        "top_pixels_per_view": top_pixels_per_view,
        "max_contributors_per_pixel": max_contributors_per_pixel,
        "official_point_cloud_found": point_cloud.exists(),
        "official_cameras_json_found": cameras_json.exists(),
        "checkpoint_conversion_supported": conversion_supported,
        "camera_conversion_supported": camera_supported,
        "gsplat_rasterization_succeeded": bool(replay_status.get("gsplat_rasterization_succeeded")),
        "gsplat_metadata_available": bool(metadata_rows) and metadata_rows[0].get("available") == "true",
        "exact_contributor_ids_available": exact_succeeded,
        "exact_alpha_available": any(row.get("alpha_contribution") not in ("", None) for row in exact_rows),
        "exact_transmittance_available": any(row.get("transmittance_before") not in ("", None) for row in exact_rows),
        "exact_splat_weight_available": any(row.get("splat_weight") not in ("", None) for row in exact_rows),
        "source_audit_completed": bool(replay_status.get("source_audit_completed")),
        "contributor_path_selected": replay_status.get("contributor_path_selected", ""),
        "contributor_path_reason": replay_status.get("contributor_path_reason", ""),
        "exact_contributor_id_only_succeeded": exact_id_only_success,
        "exact_render_contribution_succeeded": exact_render_success,
        "transmittance_source_selected": replay_status.get("transmittance_source_selected", ""),
        "transmittance_squeeze_applied": bool(replay_status.get("transmittance_squeeze_applied")),
        "transmittance_inversion_applied": bool(replay_status.get("transmittance_inversion_applied")),
        "transmittance_resolution_succeeded": bool(replay_status.get("transmittance_resolution_succeeded")),
        "contributor_api_dry_run_succeeded": bool(replay_status.get("contributor_api_dry_run_succeeded")),
        "rasterize_to_indices_call_succeeded": bool(replay_status.get("rasterize_to_indices_call_succeeded")),
        "rasterize_output_tuple_interpretation": replay_status.get("rasterize_output_tuple_interpretation", ""),
        "compact_gaussian_id_mapping_used": bool(replay_status.get("compact_gaussian_id_mapping_used")),
        "gaussian_id_mapping_mode": replay_status.get("gaussian_id_mapping_mode", ""),
        "internal_loop_replay_attempted": bool(replay_status.get("internal_loop_replay_attempted")),
        "internal_loop_replay_succeeded": bool(replay_status.get("internal_loop_replay_succeeded")),
        "packed_mode_for_internal_loop": replay_status.get("packed_mode_for_internal_loop", ""),
        "internal_loop_shape_validation_succeeded": bool(replay_status.get("internal_loop_shape_validation_succeeded")),
        "internal_loop_batch_per_iter": replay_status.get("internal_loop_batch_per_iter", 100),
        "internal_loop_num_batches": replay_status.get("internal_loop_num_batches", 0),
        "accumulate_updated_render_alphas": bool(replay_status.get("accumulate_updated_render_alphas")),
        "total_contributor_rows_before_filter": replay_status.get("total_contributor_rows_before_filter", 0),
        "sparse_contributor_filter_succeeded": bool(replay_status.get("sparse_contributor_filter_succeeded")),
        "selected_pixel_hit_count": replay_status.get("selected_pixel_hit_count", 0),
        "selected_pixel_no_hit_count": replay_status.get("selected_pixel_no_hit_count", len(selected_pixels)),
        "selected_pixel_hit_rate": replay_status.get("selected_pixel_hit_rate", 0.0),
        "pixel_id_convention": replay_status.get("pixel_id_convention", "pixel_id = y * width + x"),
        "image_id_mapping_supported": bool(replay_status.get("image_id_mapping_supported")),
        "exact_attribution_succeeded": exact_succeeded,
        "evidence_quality": evidence_quality,
        "exact_contribution_row_count": len(exact_rows),
        "exact_contributor_id_row_count": len(exact_rows) if (exact_id_only_success or exact_render_success) else 0,
        "exact_render_contribution_row_count": len(exact_rows) if exact_render_success else 0,
        "unique_exact_gaussian_count": len({str(row.get("gaussian_id", "")) for row in exact_rows}),
        "mean_contributors_per_pixel": _mean(contributors_per_pixel),
        "median_contributors_per_pixel": _median(contributors_per_pixel),
        "direct_collateral_exact_jaccard": direct_row.get("jaccard"),
        "train013_exact_overlap_with_direct_collateral": train_row.get("overlap_ratio"),
        "exact_vs_proxy_mean_pixel_jaccard": _mean(exact_vs_proxy_jaccards),
        "exact_weight_cv_mean": _cv(exact_weights),
        "pr211_ready_for_pr212_comparison": exact_succeeded,
        "ready_for_intervention": False,
        "recommended_next_step": "Proceed to PR21.2 exact-vs-proxy weighted attribution comparison and failure analysis."
        if exact_render_success
        else "Proceed to PR21.2 exact-vs-proxy contributor-ID comparison and failure analysis."
        if exact_id_only_success
        else "Inspect gsplat lower-level CUDA wrapper or implement a verified selected-pixel contributor kernel outside training.",
        "blocker_count": len([row for row in blockers if row.get("severity") == "error"]),
        "warnings": [row["blocker"] for row in blockers if row.get("severity") in {"warning", "error"}],
    }
    recommendations = _recommendations(summary)
    contributor_decision = _contributor_path_decision(summary=summary, source_rows=source_rows, path_attempts=path_attempt_rows)
    write_json(output_dir / "pr211_exact_sparse_attribution_summary.json", summary)
    write_csv_rows(output_dir / "pr211_input_readiness_audit.csv", input_rows, INPUT_FIELDS)
    write_csv_rows(output_dir / "pr211_checkpoint_activation_audit.csv", activation_rows, ACTIVATION_FIELDS)
    write_csv_rows(output_dir / "pr211_selected_pixels.csv", selected_pixels, SELECTED_PIXEL_FIELDS)
    write_csv_rows(output_dir / "pr211_gsplat_metadata_audit.csv", metadata_rows, METADATA_FIELDS)
    write_csv_rows(output_dir / "pr211_gsplat_contributor_api_audit.csv", contributor_api_rows, CONTRIBUTOR_API_FIELDS)
    write_csv_rows(output_dir / "pr211_transmittance_audit.csv", transmittance_rows, TRANSMITTANCE_FIELDS)
    write_csv_rows(output_dir / "pr211_gsplat_rasterization_output_audit.csv", raster_output_rows, RASTER_OUTPUT_FIELDS)
    write_csv_rows(output_dir / "pr211_gsplat_source_audit.csv", source_rows, SOURCE_AUDIT_FIELDS)
    write_csv_rows(output_dir / "pr211_internal_loop_shape_audit.csv", internal_loop_shape_rows, INTERNAL_LOOP_SHAPE_FIELDS)
    write_csv_rows(output_dir / "pr211_internal_loop_attempts.csv", internal_loop_attempt_rows, INTERNAL_LOOP_ATTEMPT_FIELDS)
    write_json(output_dir / "pr211_contributor_path_decision.json", contributor_decision)
    write_csv_rows(output_dir / "pr211_contributor_path_attempts.csv", path_attempt_rows, PATH_ATTEMPT_FIELDS)
    write_csv_rows(output_dir / "pr211_exact_pixel_gaussian_contributions.csv", exact_rows, EXACT_FIELDS)
    write_csv_rows(output_dir / "pr211_gaussian_residual_attribution_exact.csv", gaussian_rows, GAUSSIAN_FIELDS)
    write_csv_rows(output_dir / "pr211_view_group_residual_attribution_exact.csv", group_rows, GROUP_FIELDS)
    write_csv_rows(output_dir / "pr211_direct_collateral_exact_overlap.csv", [direct_row], DIRECT_FIELDS)
    write_csv_rows(output_dir / "pr211_train013_exact_control.csv", [train_row], TRAIN013_FIELDS)
    write_csv_rows(output_dir / "pr211_exact_vs_proxy_comparison.csv", comparison_rows, COMPARISON_FIELDS)
    write_csv_rows(output_dir / "pr211_weight_nonuniformity_audit.csv", weight_rows, WEIGHT_FIELDS)
    write_csv_rows(output_dir / "pr211_missing_fields.csv", missing_rows, MISSING_FIELDS)
    write_csv_rows(output_dir / "pr211_blockers.csv", blockers, BLOCKER_FIELDS)
    write_json(output_dir / "pr211_recommendations.json", recommendations)
    _write_report(output_dir / "pr211_report.md", summary)
    _write_manifest(output_dir, run_dir, pr200_dir, pr210_dir)
    missing_required = [name for name in PR211_OUTPUT_FILES if not (output_dir / name).is_file()]
    if missing_required:
        raise RuntimeError(f"missing PR21.1 outputs: {missing_required}")
    return summary, 0
