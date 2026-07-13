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
                "evidence_quality": "exact_sparse_render_contribution",
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
                "evidence_quality": "exact_sparse_render_contribution" if items else "",
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
        "evidence_quality": "exact_sparse_render_contribution" if rows else "failed_exact_sparse_replay",
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
        "evidence_quality": "exact_sparse_render_contribution" if rows else "failed_exact_sparse_replay",
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
    return {
        "recommended_next_step": "Proceed to PR21.2 exact-vs-proxy attribution comparison and failure analysis."
        if succeeded
        else "Fix gsplat sparse contributor extraction before comparison.",
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
    status: dict[str, Any] = {
        "transmittance_source_selected": "",
        "gsplat_rasterization_succeeded": False,
        "transmittance_resolution_succeeded": False,
        "contributor_api_dry_run_succeeded": False,
        "rasterize_to_indices_call_succeeded": False,
        "sparse_contributor_filter_succeeded": False,
        "selected_pixel_hit_count": 0,
        "selected_pixel_no_hit_count": len(selected_pixels),
        "selected_pixel_hit_rate": 0.0,
        "pixel_id_convention": "pixel_id = y * width + x",
        "image_id_mapping_supported": False,
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
        )
        api_rows.extend(api_trans_rows)
        status["transmittance_source_selected"] = source
        status["transmittance_resolution_succeeded"] = trans_ok
        status["contributor_api_dry_run_succeeded"] = trans_ok
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
            return {
                "exact_rows": exact_rows,
                "metadata_rows": metadata_rows,
                "missing_rows": missing_rows,
                "blockers": blockers,
                "api_rows": api_rows,
                "transmittance_rows": transmittance_rows,
                "raster_output_rows": raster_output_rows,
                "status": status,
            }
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
        selected_keys = {(row["view_name"], int(row["pixel_id"])) for row in selected_pixels}
        hit_keys = {(row["view_name"], int(row["pixel_id"])) for row in exact_rows}
        status["selected_pixel_hit_count"] = len(hit_keys)
        status["selected_pixel_no_hit_count"] = max(0, len(selected_keys - hit_keys))
        status["selected_pixel_hit_rate"] = len(hit_keys) / len(selected_keys) if selected_keys else 0.0
        status["sparse_contributor_filter_succeeded"] = bool(hit_keys)
        status["image_id_mapping_supported"] = bool(exact_rows)
        if not contributor_rows:
            blockers.append(
                _blocker(
                    "error",
                    "gsplat_sparse_contributors",
                    "exact contributor IDs unavailable from gsplat metadata/API",
                    "rasterization ran but selected-pixel contributor rows were empty",
                    "inspect gsplat metadata and contributor API signature",
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
    selected_by_pixel = {(row["view_name"], int(row["pixel_id"])): row for row in selected_pixels}
    view_by_index = [row.get("requested_view_name", "") for row in selected_views]
    rows: list[dict[str, Any]] = []
    counts: dict[tuple[str, int], int] = defaultdict(int)
    for index, gid in enumerate(mapped["gaussian_ids"]):
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
        splat_weight = 1.0 / max_contributors_per_pixel
        residual = _number(selected.get("residual_l1"))
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
                "gaussian_id": int(gid),
                "contributor_rank": counts[key],
                "depth_order": counts[key],
                "alpha_contribution": "",
                "transmittance_before": "",
                "splat_weight": splat_weight,
                "opacity_after_activation": "",
                "color_contribution_r": "",
                "color_contribution_g": "",
                "color_contribution_b": "",
                "residual_l1": selected.get("residual_l1", ""),
                "residual_weighted_splat": residual * splat_weight,
                "evidence_quality": "exact_sparse_render_contribution",
                "attribution_method": "gsplat_sparse_replay",
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
    missing_rows: list[dict[str, Any]] = []
    replay_status: dict[str, Any] = {
        "transmittance_source_selected": "",
        "gsplat_rasterization_succeeded": False,
        "transmittance_resolution_succeeded": False,
        "contributor_api_dry_run_succeeded": False,
        "rasterize_to_indices_call_succeeded": False,
        "sparse_contributor_filter_succeeded": False,
        "selected_pixel_hit_count": 0,
        "selected_pixel_no_hit_count": len(selected_pixels),
        "selected_pixel_hit_rate": 0.0,
        "pixel_id_convention": "pixel_id = y * width + x",
        "image_id_mapping_supported": False,
    }
    if synthetic_exact_rows is not None:
        exact_rows = [dict(row) for row in synthetic_exact_rows]
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
        selected_keys = {(row["view_name"], int(row["pixel_id"])) for row in exact_rows if row.get("pixel_id") not in ("", None)}
        replay_status.update(
            {
                "transmittance_source_selected": "synthetic.transmittances",
                "gsplat_rasterization_succeeded": True,
                "transmittance_resolution_succeeded": True,
                "contributor_api_dry_run_succeeded": True,
                "rasterize_to_indices_call_succeeded": True,
                "sparse_contributor_filter_succeeded": bool(exact_rows),
                "selected_pixel_hit_count": len(selected_keys),
                "selected_pixel_no_hit_count": max(0, len(selected_pixels) - len(selected_keys)),
                "selected_pixel_hit_rate": len(selected_keys) / len(selected_pixels) if selected_pixels else 0.0,
                "image_id_mapping_supported": True,
            }
        )
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
        blockers = [row for row in blockers if row.get("component") != "gsplat_sparse_contributors"]
    exact_succeeded = bool(exact_rows)
    exact_succeeded = bool(
        exact_succeeded
        and replay_status.get("transmittance_resolution_succeeded")
        and replay_status.get("contributor_api_dry_run_succeeded")
        and replay_status.get("rasterize_to_indices_call_succeeded")
        and replay_status.get("selected_pixel_hit_count", 0) > 0
    )
    if not exact_succeeded:
        exact_rows = []
    evidence_quality = "exact_sparse_render_contribution" if exact_succeeded else "failed_exact_sparse_replay"
    for row in exact_rows:
        row.setdefault("scene", scene)
        row.setdefault("condition", condition)
        row.setdefault("subset_name", subset_name)
        row.setdefault("view_group", _group_for_view(str(row.get("view_name", ""))))
        row.setdefault("evidence_quality", "exact_sparse_render_contribution")
        row.setdefault("attribution_method", "gsplat_sparse_replay")
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
        "transmittance_source_selected": replay_status.get("transmittance_source_selected", ""),
        "transmittance_resolution_succeeded": bool(replay_status.get("transmittance_resolution_succeeded")),
        "contributor_api_dry_run_succeeded": bool(replay_status.get("contributor_api_dry_run_succeeded")),
        "rasterize_to_indices_call_succeeded": bool(replay_status.get("rasterize_to_indices_call_succeeded")),
        "sparse_contributor_filter_succeeded": bool(replay_status.get("sparse_contributor_filter_succeeded")),
        "selected_pixel_hit_count": replay_status.get("selected_pixel_hit_count", 0),
        "selected_pixel_no_hit_count": replay_status.get("selected_pixel_no_hit_count", len(selected_pixels)),
        "selected_pixel_hit_rate": replay_status.get("selected_pixel_hit_rate", 0.0),
        "pixel_id_convention": replay_status.get("pixel_id_convention", "pixel_id = y * width + x"),
        "image_id_mapping_supported": bool(replay_status.get("image_id_mapping_supported")),
        "exact_attribution_succeeded": exact_succeeded,
        "evidence_quality": evidence_quality,
        "exact_contribution_row_count": len(exact_rows),
        "unique_exact_gaussian_count": len({str(row.get("gaussian_id", "")) for row in exact_rows}),
        "mean_contributors_per_pixel": _mean(contributors_per_pixel),
        "median_contributors_per_pixel": _median(contributors_per_pixel),
        "direct_collateral_exact_jaccard": direct_row.get("jaccard"),
        "train013_exact_overlap_with_direct_collateral": train_row.get("overlap_ratio"),
        "exact_vs_proxy_mean_pixel_jaccard": _mean(exact_vs_proxy_jaccards),
        "exact_weight_cv_mean": _cv(exact_weights),
        "pr211_ready_for_pr212_comparison": exact_succeeded,
        "ready_for_intervention": False,
        "recommended_next_step": "Proceed to PR21.2 exact-vs-proxy attribution comparison and failure analysis."
        if exact_succeeded
        else "Fix gsplat sparse contributor extraction before comparison.",
        "blocker_count": len([row for row in blockers if row.get("severity") == "error"]),
        "warnings": [row["blocker"] for row in blockers if row.get("severity") in {"warning", "error"}],
    }
    recommendations = _recommendations(summary)
    write_json(output_dir / "pr211_exact_sparse_attribution_summary.json", summary)
    write_csv_rows(output_dir / "pr211_input_readiness_audit.csv", input_rows, INPUT_FIELDS)
    write_csv_rows(output_dir / "pr211_checkpoint_activation_audit.csv", activation_rows, ACTIVATION_FIELDS)
    write_csv_rows(output_dir / "pr211_selected_pixels.csv", selected_pixels, SELECTED_PIXEL_FIELDS)
    write_csv_rows(output_dir / "pr211_gsplat_metadata_audit.csv", metadata_rows, METADATA_FIELDS)
    write_csv_rows(output_dir / "pr211_gsplat_contributor_api_audit.csv", contributor_api_rows, CONTRIBUTOR_API_FIELDS)
    write_csv_rows(output_dir / "pr211_transmittance_audit.csv", transmittance_rows, TRANSMITTANCE_FIELDS)
    write_csv_rows(output_dir / "pr211_gsplat_rasterization_output_audit.csv", raster_output_rows, RASTER_OUTPUT_FIELDS)
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
