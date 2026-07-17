"""PR21.2d exact-pixel-anchored PR20 proxy repair export."""

from __future__ import annotations

import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from viewtrust.analysis.clean_prior_normalization import load_csv_rows, load_json, write_csv_rows, write_json


TARGET_COLUMNS = ["final_index", "alive_index", "compact_index", "current_index", "final_checkpoint_index"]
OPTIONAL_PROXY_FIELDS = ["splat_weight", "alpha_contribution", "color_contribution", "residual"]
GROUP_ORDER = ["direct_corrupted", "collateral", "clean_control", "other_clean"]

ANCHOR_FIELDS = [
    "scene",
    "condition",
    "subset_name",
    "view_name",
    "view_group",
    "pixel_x",
    "pixel_y",
    "exact_row_count",
    "exact_unique_id_count",
    "exact_ids_semicolon",
]

COVERAGE_FIELDS = [
    "view_name",
    "pixel_x",
    "pixel_y",
    "exact_row_count",
    "pr20_proxy_row_count",
    "pr20_proxy_unique_raw_id_count",
    "coverage_status",
    "notes",
]

MAPPING_AUDIT_FIELDS = [
    "mapping_source",
    "source_id_column",
    "target_index_column",
    "source_id_count",
    "target_index_count",
    "duplicate_source_id_count",
    "conflicting_source_id_count",
    "mapped_pr20_exact_pixel_proxy_row_count",
    "unmapped_pr20_exact_pixel_proxy_row_count",
    "mapped_unique_proxy_id_count",
    "unmapped_unique_proxy_id_count",
    "final_index_min",
    "final_index_max",
    "final_index_in_checkpoint_range",
    "mapping_confidence",
    "mapping_status",
    "notes",
]

REPAIRED_FIELDS = [
    "scene",
    "condition",
    "subset_name",
    "view_name",
    "view_group",
    "pixel_x",
    "pixel_y",
    "original_gaussian_id",
    "original_root_gaussian_id",
    "original_parent_gaussian_id",
    "contribution_rank",
    "verified_final_gaussian_index",
    "mapping_source",
    "mapping_source_id_column",
    "mapping_target_index_column",
    "mapping_confidence",
    "mapping_status",
    "repair_warning",
] + OPTIONAL_PROXY_FIELDS

PIXEL_FIELDS = [
    "scene",
    "condition",
    "subset_name",
    "view_name",
    "view_group",
    "pixel_x",
    "pixel_y",
    "exact_id_count",
    "repaired_proxy_id_count",
    "unmapped_proxy_row_count",
    "intersection_count",
    "exact_only_count",
    "repaired_proxy_only_count",
    "union_count",
    "jaccard",
    "exact_recall_by_repaired_proxy",
    "repaired_proxy_precision_against_exact",
    "comparison_status",
    "interpretation",
]

VIEW_FIELDS = [
    "scene",
    "condition",
    "subset_name",
    "view_name",
    "view_group",
    "exact_pixel_count",
    "exact_row_count",
    "exact_unique_id_count",
    "repaired_proxy_row_count",
    "repaired_proxy_unique_id_count",
    "unmapped_proxy_row_count",
    "intersection_unique_id_count",
    "exact_only_unique_id_count",
    "repaired_proxy_only_unique_id_count",
    "union_unique_id_count",
    "view_jaccard",
    "view_exact_recall_by_repaired_proxy",
    "view_repaired_proxy_precision_against_exact",
    "valid_pixel_count",
    "incomplete_pixel_count",
    "zero_overlap_pixel_count",
    "nonzero_overlap_pixel_count",
    "comparison_status",
    "interpretation",
]

GROUP_FIELDS = [
    "scene",
    "condition",
    "subset_name",
    "group_name",
    "exact_view_count",
    "exact_pixel_count",
    "exact_row_count",
    "exact_unique_id_count",
    "repaired_proxy_row_count",
    "repaired_proxy_unique_id_count",
    "unmapped_proxy_row_count",
    "intersection_unique_id_count",
    "exact_only_unique_id_count",
    "repaired_proxy_only_unique_id_count",
    "union_unique_id_count",
    "group_jaccard",
    "group_exact_recall_by_repaired_proxy",
    "group_repaired_proxy_precision_against_exact",
    "comparison_status",
    "interpretation",
    "caveat",
]

DELTA_FIELDS = ["metric", "pr212c_value", "pr212d_value", "delta", "interpretation"]
CLAIM_FIELDS = ["claim_id", "claim", "status", "evidence", "scope", "paper_safe_wording", "unsafe_wording_to_avoid", "recommended_next_step"]
MANIFEST_FIELDS = ["relative_path", "path", "exists", "file_type", "size_bytes", "required", "artifact_group"]

OUTPUT_FILES = [
    "pr212d_exact_pixel_anchored_proxy_repair_summary.json",
    "pr212d_exact_pixel_anchor.csv",
    "pr212d_pr20_proxy_rows_on_exact_pixels.csv",
    "pr212d_anchor_proxy_coverage.csv",
    "pr212d_verified_mapping_audit.csv",
    "pr212d_exact_pixel_anchored_repaired_proxy_rows.csv",
    "pr212d_complete_repaired_pixel_exact_vs_proxy.csv",
    "pr212d_complete_repaired_view_exact_vs_proxy.csv",
    "pr212d_complete_repaired_group_exact_vs_proxy.csv",
    "pr212d_pr212c_partial_vs_complete_delta.csv",
    "pr212d_claim_status_table.csv",
    "pr212d_exact_pixel_anchored_proxy_repair_report.md",
    "pr212d_paper_wording.md",
    "pr212d_next_step_decision_memo.md",
    "artifact_manifest.csv",
]


PixelKey = tuple[str, int, int]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _bool_text(value: Any) -> str:
    return "true" if bool(value) else "false"


def _truth(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def _safe_int(value: Any) -> int | None:
    try:
        if value in ("", None):
            return None
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return None


def _safe_float(value: Any) -> float | None:
    try:
        if value in ("", None):
            return None
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def _ratio(num: int, den: int) -> float | None:
    return num / den if den else None


def _norm_id(value: Any) -> str:
    value_int = _safe_int(value)
    return "" if value_int is None else str(value_int)


def _pixel_key(row: dict[str, Any]) -> PixelKey:
    return str(row.get("view_name", "")).strip(), _safe_int(row.get("pixel_x")) or 0, _safe_int(row.get("pixel_y")) or 0


def _normalize_group(value: Any, view_name: str) -> str:
    text = str(value or "").strip()
    if text in GROUP_ORDER:
        return text
    if text in {"co_visible_collateral", "collateral_view"}:
        return "collateral"
    if text in {"clean_prior_demoted", "clean_control_view"}:
        return "clean_control"
    if view_name in {"train_004", "train_009", "train_012", "train_017"}:
        return "direct_corrupted"
    if view_name == "train_014":
        return "collateral"
    if view_name == "train_013":
        return "clean_control"
    return "other_clean"


def _fmt_ids(values: set[str]) -> str:
    return ";".join(sorted(values, key=lambda item: (0, int(item)) if item.isdigit() else (1, item)))


def _id_set(rows: list[dict[str, str]], column: str) -> set[str]:
    return {gid for gid in (_norm_id(row.get(column)) for row in rows) if gid}


def _by_pixel(rows: list[dict[str, str]]) -> dict[PixelKey, list[dict[str, str]]]:
    out: dict[PixelKey, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        out[_pixel_key(row)].append(row)
    return dict(out)


def _compare(exact: set[str], repaired: set[str]) -> dict[str, Any]:
    inter = exact & repaired
    exact_only = exact - repaired
    repaired_only = repaired - exact
    union = exact | repaired
    return {
        "intersection": inter,
        "exact_only": exact_only,
        "repaired_only": repaired_only,
        "union": union,
        "jaccard": len(inter) / len(union) if union else None,
        "recall": _ratio(len(inter), len(exact)),
        "precision": _ratio(len(inter), len(repaired)),
    }


def _validate_readiness(pr212b: dict[str, Any], pr212cfix: dict[str, Any], pr211: dict[str, Any], exact_rows: list[dict[str, str]]) -> list[str]:
    warnings = []

    def matches(value: Any, expected: Any) -> bool:
        if isinstance(expected, bool):
            return _truth(value) is expected
        if isinstance(expected, float):
            return _safe_float(value) == expected
        if isinstance(expected, int):
            return _safe_int(value) == expected
        if isinstance(expected, str):
            return value == expected
        return value == expected

    expected_b = {
        "explicit_mapping_available": True,
        "mapping_confidence": "high",
        "all_pr20_proxy_rows_repair_feasible": True,
        "exact_available_proxy_rows_repair_feasible": True,
        "exact_available_mapping_coverage_rate": 1.0,
        "proxy_safe_for_intervention": False,
    }
    for key, expected in expected_b.items():
        value = pr212b.get(key)
        if value in ("", None):
            warnings.append(f"PR21.2b missing {key}")
        elif not matches(value, expected):
            warnings.append(f"PR21.2b {key}={value!r}, expected {expected!r}")
    expected_fix = {
        "exact_pixel_count": 43,
        "missing_pixel_count": 7,
        "exact_pixels_present_in_pr20_proxy_count": 43,
        "exact_pixels_present_in_pr212b_repaired_preview_count": 36,
        "coverage_problem_resolved": False,
    }
    for key, expected in expected_fix.items():
        value = pr212cfix.get(key)
        if value in ("", None):
            warnings.append(f"PR21.2c-fix missing {key}")
        elif not matches(value, expected):
            warnings.append(f"PR21.2c-fix {key}={value!r}, expected {expected!r}")
    failure_counts = pr212cfix.get("missing_pixel_failure_mode_counts", {})
    if "present_in_pr20_but_missing_from_pr212b_repaired_preview" not in str(failure_counts):
        warnings.append("PR21.2c-fix failure modes do not mention PR21.2b repaired preview gap")
    if not exact_rows:
        warnings.append("PR21.1e exact rows are missing")
    if pr211 and pr211.get("evidence_quality") != "exact_sparse_contributor_id_only":
        warnings.append("PR21.1e evidence_quality is not exact_sparse_contributor_id_only")
    if _truth(pr211.get("exact_render_contribution_succeeded")):
        warnings.append("PR21.1e unexpectedly reports exact render contribution")
    return warnings


def _anchor_rows(scene: str, condition: str, subset_name: str, exact_by_pixel: dict[PixelKey, list[dict[str, str]]]) -> list[dict[str, Any]]:
    rows = []
    for key in sorted(exact_by_pixel):
        exact_rows = exact_by_pixel[key]
        exact_ids = _id_set(exact_rows, "gaussian_id")
        group = _normalize_group(exact_rows[0].get("view_group", ""), key[0])
        rows.append(
            {
                "scene": scene,
                "condition": condition,
                "subset_name": subset_name,
                "view_name": key[0],
                "view_group": group,
                "pixel_x": key[1],
                "pixel_y": key[2],
                "exact_row_count": len(exact_rows),
                "exact_unique_id_count": len(exact_ids),
                "exact_ids_semicolon": _fmt_ids(exact_ids),
            }
        )
    return rows


def _extract_pr20_anchor_rows(
    scene: str,
    condition: str,
    subset_name: str,
    exact_by_pixel: dict[PixelKey, list[dict[str, str]]],
    pr20_by_pixel: dict[PixelKey, list[dict[str, str]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    proxy_rows = []
    coverage_rows = []
    for key in sorted(exact_by_pixel):
        exact_rows = exact_by_pixel[key]
        rows = pr20_by_pixel.get(key, [])
        raw_ids = _id_set(rows, "gaussian_id")
        coverage_rows.append(
            {
                "view_name": key[0],
                "pixel_x": key[1],
                "pixel_y": key[2],
                "exact_row_count": len(exact_rows),
                "pr20_proxy_row_count": len(rows),
                "pr20_proxy_unique_raw_id_count": len(raw_ids),
                "coverage_status": "pr20_proxy_present" if rows else "pr20_proxy_missing",
                "notes": "matched by stripped view_name and integer pixel coordinates; no coordinate transform applied",
            }
        )
        for row in rows:
            out = dict(row)
            out.update(
                {
                    "scene": scene,
                    "condition": condition,
                    "subset_name": subset_name,
                    "exact_anchor_present": "true",
                    "original_proxy_gaussian_id": row.get("gaussian_id", ""),
                    "original_root_gaussian_id": row.get("root_gaussian_id", ""),
                    "original_parent_gaussian_id": row.get("parent_gaussian_id", ""),
                    "extraction_source": "pr20_original_proxy_rows",
                    "anchor_match_status": "exact_pixel_anchor_match",
                }
            )
            proxy_rows.append(out)
    return proxy_rows, coverage_rows


def _candidate_mapping_paths(mapping_source: str, run_dir: Path, pr212b_dir: Path) -> list[Path]:
    candidates = []
    if mapping_source:
        src = Path(mapping_source)
        candidates.append(src)
        if not src.is_absolute():
            candidates.extend([run_dir / src, pr212b_dir / src, run_dir / "tables" / src.name])
    candidates.append(run_dir / "tables" / "gaussian_lifecycle_final.csv")
    out = []
    seen = set()
    for path in candidates:
        key = str(path)
        if key not in seen:
            out.append(path)
            seen.add(key)
    return out


def _load_mapping(
    *,
    pr212b_summary: dict[str, Any],
    pr212b_dir: Path,
    run_dir: Path,
    checkpoint_gaussian_count: int | None,
    proxy_rows_on_anchor: list[dict[str, Any]],
) -> tuple[dict[int, int], dict[str, Any]]:
    mapping_source = str(pr212b_summary.get("mapping_source") or "")
    mapping_path = next((path for path in _candidate_mapping_paths(mapping_source, run_dir, pr212b_dir) if path.is_file()), None)
    if mapping_path is None:
        return {}, {
            "mapping_source": mapping_source,
            "source_id_column": "gaussian_id",
            "target_index_column": "",
            "source_id_count": 0,
            "target_index_count": 0,
            "duplicate_source_id_count": 0,
            "conflicting_source_id_count": 0,
            "mapped_pr20_exact_pixel_proxy_row_count": 0,
            "unmapped_pr20_exact_pixel_proxy_row_count": len(proxy_rows_on_anchor),
            "mapped_unique_proxy_id_count": 0,
            "unmapped_unique_proxy_id_count": len(_id_set(proxy_rows_on_anchor, "original_proxy_gaussian_id")),
            "final_index_min": "",
            "final_index_max": "",
            "final_index_in_checkpoint_range": "false",
            "mapping_confidence": "none",
            "mapping_status": "mapping_source_missing",
            "notes": "mapping source file not found",
        }
    rows = load_csv_rows(mapping_path)
    columns = list(rows[0]) if rows else []
    source_col = "gaussian_id"
    target_col = next((col for col in TARGET_COLUMNS if col in columns), "")
    per_source: dict[int, set[int]] = defaultdict(set)
    for row in rows:
        source = _safe_int(row.get(source_col))
        target = _safe_int(row.get(target_col)) if target_col else None
        if source is None or target is None:
            continue
        per_source[source].add(target)
    conflicting = {source: targets for source, targets in per_source.items() if len(targets) > 1}
    mapping = {source: next(iter(targets)) for source, targets in per_source.items() if len(targets) == 1}
    source_counts = Counter(_safe_int(row.get(source_col)) for row in rows if _safe_int(row.get(source_col)) is not None)
    duplicate_sources = len([source for source, count in source_counts.items() if count > 1 and source not in conflicting])
    proxy_ids = [_safe_int(row.get("original_proxy_gaussian_id")) for row in proxy_rows_on_anchor]
    proxy_ids = [value for value in proxy_ids if value is not None]
    mapped_rows = [value for value in proxy_ids if value in mapping]
    unmapped_rows = [value for value in proxy_ids if value not in mapping]
    final_values = list(mapping.values())
    in_range = bool(
        final_values
        and all(value >= 0 for value in final_values)
        and (checkpoint_gaussian_count is None or all(value < checkpoint_gaussian_count for value in final_values))
    )
    confidence = str(pr212b_summary.get("mapping_confidence") or "none")
    complete = bool(proxy_ids and not unmapped_rows and not conflicting and in_range and confidence == "high")
    audit = {
        "mapping_source": str(mapping_path),
        "source_id_column": source_col,
        "target_index_column": target_col,
        "source_id_count": len(mapping),
        "target_index_count": len(set(mapping.values())),
        "duplicate_source_id_count": duplicate_sources,
        "conflicting_source_id_count": len(conflicting),
        "mapped_pr20_exact_pixel_proxy_row_count": len(mapped_rows),
        "unmapped_pr20_exact_pixel_proxy_row_count": len(unmapped_rows),
        "mapped_unique_proxy_id_count": len(set(mapped_rows)),
        "unmapped_unique_proxy_id_count": len(set(unmapped_rows)),
        "final_index_min": min(final_values) if final_values else "",
        "final_index_max": max(final_values) if final_values else "",
        "final_index_in_checkpoint_range": _bool_text(in_range),
        "mapping_confidence": confidence,
        "mapping_status": "verified_complete_for_exact_anchor" if complete else "incomplete_or_invalid_for_exact_anchor",
        "notes": "mapping loaded directly from verified lifecycle/identity source; no PR21.2b repaired preview used",
    }
    return mapping, audit


def _repaired_rows(
    *,
    proxy_rows_on_anchor: list[dict[str, Any]],
    mapping: dict[int, int],
    audit: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    for row in proxy_rows_on_anchor:
        source = _safe_int(row.get("original_proxy_gaussian_id"))
        mapped = mapping.get(source) if source is not None else None
        warning = "" if mapped is not None and audit.get("mapping_status") == "verified_complete_for_exact_anchor" else "unmapped_or_mapping_incomplete"
        out = {
            "scene": row.get("scene", ""),
            "condition": row.get("condition", ""),
            "subset_name": row.get("subset_name", ""),
            "view_name": str(row.get("view_name", "")).strip(),
            "view_group": _normalize_group(row.get("view_group", ""), str(row.get("view_name", "")).strip()),
            "pixel_x": _safe_int(row.get("pixel_x")) or 0,
            "pixel_y": _safe_int(row.get("pixel_y")) or 0,
            "original_gaussian_id": row.get("original_proxy_gaussian_id", ""),
            "original_root_gaussian_id": row.get("original_root_gaussian_id", ""),
            "original_parent_gaussian_id": row.get("original_parent_gaussian_id", ""),
            "contribution_rank": row.get("contribution_rank", ""),
            "verified_final_gaussian_index": "" if mapped is None else mapped,
            "mapping_source": audit.get("mapping_source", ""),
            "mapping_source_id_column": audit.get("source_id_column", ""),
            "mapping_target_index_column": audit.get("target_index_column", ""),
            "mapping_confidence": audit.get("mapping_confidence", ""),
            "mapping_status": audit.get("mapping_status", ""),
            "repair_warning": warning,
        }
        for field in OPTIONAL_PROXY_FIELDS:
            out[field] = row.get(field, "")
        rows.append(out)
    return rows


def _pixel_comparison_rows(
    *,
    scene: str,
    condition: str,
    subset_name: str,
    exact_by_pixel: dict[PixelKey, list[dict[str, str]]],
    repaired_by_pixel: dict[PixelKey, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    rows = []
    for key in sorted(exact_by_pixel):
        exact_rows = exact_by_pixel[key]
        repaired_rows = repaired_by_pixel.get(key, [])
        exact_ids = _id_set(exact_rows, "gaussian_id")
        mapped_rows = [row for row in repaired_rows if _norm_id(row.get("verified_final_gaussian_index")) and not row.get("repair_warning")]
        repaired_ids = _id_set(mapped_rows, "verified_final_gaussian_index")
        unmapped = len(repaired_rows) - len(mapped_rows)
        cmp = _compare(exact_ids, repaired_ids)
        if not repaired_rows:
            status = "missing_pr20_proxy_rows"
            interp = "complete_repaired_comparison_incomplete"
        elif unmapped:
            status = "unmapped_proxy_ids"
            interp = "complete_repaired_comparison_incomplete"
        else:
            status = "complete_repaired_comparison_valid"
            interp = "complete_repaired_nonzero_overlap" if cmp["intersection"] else "complete_repaired_zero_overlap"
        rows.append(
            {
                "scene": scene,
                "condition": condition,
                "subset_name": subset_name,
                "view_name": key[0],
                "view_group": _normalize_group(exact_rows[0].get("view_group", ""), key[0]),
                "pixel_x": key[1],
                "pixel_y": key[2],
                "exact_id_count": len(exact_ids),
                "repaired_proxy_id_count": len(repaired_ids),
                "unmapped_proxy_row_count": unmapped,
                "intersection_count": len(cmp["intersection"]),
                "exact_only_count": len(cmp["exact_only"]),
                "repaired_proxy_only_count": len(cmp["repaired_only"]),
                "union_count": len(cmp["union"]),
                "jaccard": cmp["jaccard"],
                "exact_recall_by_repaired_proxy": cmp["recall"],
                "repaired_proxy_precision_against_exact": cmp["precision"],
                "comparison_status": status,
                "interpretation": interp,
            }
        )
    return rows


def _aggregate_by_view(
    *,
    scene: str,
    condition: str,
    subset_name: str,
    exact_by_pixel: dict[PixelKey, list[dict[str, str]]],
    repaired_by_pixel: dict[PixelKey, list[dict[str, Any]]],
    pixel_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    exact_by_view: dict[str, set[str]] = defaultdict(set)
    repaired_by_view: dict[str, set[str]] = defaultdict(set)
    exact_rows_count: Counter[str] = Counter()
    exact_pixels_count: Counter[str] = Counter()
    repaired_rows_count: Counter[str] = Counter()
    unmapped_count: Counter[str] = Counter()
    status_counts: dict[str, Counter[str]] = defaultdict(Counter)
    group_by_view: dict[str, str] = {}
    for key, rows in exact_by_pixel.items():
        view = key[0]
        exact_pixels_count[view] += 1
        exact_rows_count[view] += len(rows)
        exact_by_view[view].update(_id_set(rows, "gaussian_id"))
        group_by_view.setdefault(view, _normalize_group(rows[0].get("view_group", ""), view))
    for key, rows in repaired_by_pixel.items():
        view = key[0]
        repaired_rows_count[view] += len(rows)
        repaired_by_view[view].update(_id_set([row for row in rows if not row.get("repair_warning")], "verified_final_gaussian_index"))
    for row in pixel_rows:
        view = str(row.get("view_name", ""))
        unmapped_count[view] += _safe_int(row.get("unmapped_proxy_row_count")) or 0
        status_counts[view][str(row.get("comparison_status", ""))] += 1
        status_counts[view][str(row.get("interpretation", ""))] += 1
    out = []
    for view in sorted(exact_by_view):
        exact = exact_by_view.get(view, set())
        repaired = repaired_by_view.get(view, set())
        cmp = _compare(exact, repaired)
        valid = status_counts[view]["complete_repaired_comparison_valid"]
        incomplete = exact_pixels_count[view] - valid
        out.append(
            {
                "scene": scene,
                "condition": condition,
                "subset_name": subset_name,
                "view_name": view,
                "view_group": group_by_view.get(view, _normalize_group("", view)),
                "exact_pixel_count": exact_pixels_count[view],
                "exact_row_count": exact_rows_count[view],
                "exact_unique_id_count": len(exact),
                "repaired_proxy_row_count": repaired_rows_count[view],
                "repaired_proxy_unique_id_count": len(repaired),
                "unmapped_proxy_row_count": unmapped_count[view],
                "intersection_unique_id_count": len(cmp["intersection"]),
                "exact_only_unique_id_count": len(cmp["exact_only"]),
                "repaired_proxy_only_unique_id_count": len(cmp["repaired_only"]),
                "union_unique_id_count": len(cmp["union"]),
                "view_jaccard": cmp["jaccard"],
                "view_exact_recall_by_repaired_proxy": cmp["recall"],
                "view_repaired_proxy_precision_against_exact": cmp["precision"],
                "valid_pixel_count": valid,
                "incomplete_pixel_count": incomplete,
                "zero_overlap_pixel_count": status_counts[view]["complete_repaired_zero_overlap"],
                "nonzero_overlap_pixel_count": status_counts[view]["complete_repaired_nonzero_overlap"],
                "comparison_status": "complete_repaired_comparison_valid" if incomplete == 0 else "complete_repaired_comparison_incomplete",
                "interpretation": "complete_repaired_nonzero_overlap" if cmp["intersection"] else "complete_repaired_zero_overlap" if incomplete == 0 else "complete_repaired_comparison_incomplete",
            }
        )
    return out


def _aggregate_by_group(
    *,
    scene: str,
    condition: str,
    subset_name: str,
    exact_by_pixel: dict[PixelKey, list[dict[str, str]]],
    repaired_by_pixel: dict[PixelKey, list[dict[str, Any]]],
    pixel_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    exact_by_group: dict[str, set[str]] = defaultdict(set)
    repaired_by_group: dict[str, set[str]] = defaultdict(set)
    exact_views: dict[str, set[str]] = defaultdict(set)
    exact_pixels: Counter[str] = Counter()
    exact_rows_count: Counter[str] = Counter()
    repaired_rows_count: Counter[str] = Counter()
    unmapped_count: Counter[str] = Counter()
    status_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for key, rows in exact_by_pixel.items():
        group = _normalize_group(rows[0].get("view_group", ""), key[0])
        exact_views[group].add(key[0])
        exact_pixels[group] += 1
        exact_rows_count[group] += len(rows)
        exact_by_group[group].update(_id_set(rows, "gaussian_id"))
    for key, rows in repaired_by_pixel.items():
        group = _normalize_group(rows[0].get("view_group", "") if rows else "", key[0])
        repaired_rows_count[group] += len(rows)
        repaired_by_group[group].update(_id_set([row for row in rows if not row.get("repair_warning")], "verified_final_gaussian_index"))
    for row in pixel_rows:
        group = str(row.get("view_group", "other_clean"))
        unmapped_count[group] += _safe_int(row.get("unmapped_proxy_row_count")) or 0
        status_counts[group][str(row.get("comparison_status", ""))] += 1
        status_counts[group][str(row.get("interpretation", ""))] += 1
    out = []
    for group in GROUP_ORDER:
        exact = exact_by_group.get(group, set())
        repaired = repaired_by_group.get(group, set())
        cmp = _compare(exact, repaired)
        valid = status_counts[group]["complete_repaired_comparison_valid"]
        incomplete = exact_pixels[group] - valid
        has_exact = exact_pixels[group] > 0
        out.append(
            {
                "scene": scene,
                "condition": condition,
                "subset_name": subset_name,
                "group_name": group,
                "exact_view_count": len(exact_views[group]),
                "exact_pixel_count": exact_pixels[group],
                "exact_row_count": exact_rows_count[group],
                "exact_unique_id_count": len(exact),
                "repaired_proxy_row_count": repaired_rows_count[group],
                "repaired_proxy_unique_id_count": len(repaired),
                "unmapped_proxy_row_count": unmapped_count[group],
                "intersection_unique_id_count": len(cmp["intersection"]),
                "exact_only_unique_id_count": len(cmp["exact_only"]),
                "repaired_proxy_only_unique_id_count": len(cmp["repaired_only"]),
                "union_unique_id_count": len(cmp["union"]),
                "group_jaccard": cmp["jaccard"],
                "group_exact_recall_by_repaired_proxy": cmp["recall"],
                "group_repaired_proxy_precision_against_exact": cmp["precision"],
                "comparison_status": "complete_repaired_comparison_valid" if has_exact and incomplete == 0 else "missing_exact_ids" if not has_exact else "complete_repaired_comparison_incomplete",
                "interpretation": "complete_repaired_nonzero_overlap" if cmp["intersection"] else "complete_repaired_zero_overlap" if has_exact and incomplete == 0 else "complete_repaired_comparison_incomplete",
                "caveat": "" if has_exact else "No PR21.1e exact rows exist for this group; do not claim group overlap.",
            }
        )
    return out


def _delta_rows(pr212c_summary: dict[str, Any], pixel_rows: list[dict[str, Any]], summary_values: dict[str, Any]) -> list[dict[str, Any]]:
    zero_count = len([row for row in pixel_rows if row.get("interpretation") == "complete_repaired_zero_overlap"])
    nonzero_count = len([row for row in pixel_rows if row.get("interpretation") == "complete_repaired_nonzero_overlap"])
    metrics = [
        ("exact_pixel_count", pr212c_summary.get("exact_pixel_count"), summary_values["exact_pixel_count"]),
        ("valid_pixel_count", pr212c_summary.get("complete_valid_pixel_count", pr212c_summary.get("repaired_proxy_row_count_on_exact_pixels")), summary_values["complete_valid_pixel_count"]),
        ("missing_pixel_count", pr212c_summary.get("complete_missing_pixel_count", 7 if not _truth(pr212c_summary.get("pr212c_ready_for_pr214")) else 0), summary_values["complete_missing_pixel_count"]),
        ("mean_pixel_jaccard", pr212c_summary.get("repaired_mean_pixel_jaccard"), summary_values["complete_mean_pixel_jaccard"]),
        ("median_pixel_jaccard", pr212c_summary.get("repaired_median_pixel_jaccard"), summary_values["complete_median_pixel_jaccard"]),
        ("mean_exact_recall", pr212c_summary.get("repaired_mean_exact_recall_by_proxy"), summary_values["complete_mean_exact_recall_by_proxy"]),
        ("mean_proxy_precision", pr212c_summary.get("repaired_mean_proxy_precision_against_exact"), summary_values["complete_mean_proxy_precision_against_exact"]),
        ("repaired_proxy_unique_id_count", pr212c_summary.get("repaired_proxy_unique_id_count_on_exact_pixels"), summary_values["repaired_proxy_unique_id_count_on_exact_pixels"]),
        ("zero_overlap_pixel_count", "", zero_count),
        ("nonzero_overlap_pixel_count", "", nonzero_count),
    ]
    out = []
    complete = summary_values["complete_missing_pixel_count"] == 0
    nonzero = summary_values["complete_nonzero_overlap_found"]
    for metric, old, new in metrics:
        old_float = _safe_float(old)
        new_float = _safe_float(new)
        if complete and not nonzero:
            interp = "coverage_completed_zero_overlap_unchanged"
        elif complete and nonzero:
            interp = "coverage_completed_nonzero_overlap_found"
        elif not complete:
            interp = "still_partial"
        else:
            interp = "not_comparable"
        out.append(
            {
                "metric": metric,
                "pr212c_value": old,
                "pr212d_value": new,
                "delta": new_float - old_float if old_float is not None and new_float is not None else "",
                "interpretation": interp,
            }
        )
    return out


def _claim_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    complete_zero = _truth(summary.get("complete_zero_overlap_claim_safe_within_exact_available_scope"))
    complete_valid = _truth(summary.get("pr212d_ready_for_pr214"))
    return [
        {
            "claim_id": "C1",
            "claim": "After exact-pixel-anchored repair, PR20 repaired proxy IDs and PR21 exact contributor IDs are disjoint on all chair exact-available pixels.",
            "status": "supported" if complete_zero else "unsupported_or_partial",
            "evidence": f"valid_pixels={summary.get('complete_valid_pixel_count')}; missing_pixels={summary.get('complete_missing_pixel_count')}; nonzero_overlap={summary.get('complete_nonzero_overlap_found')}",
            "scope": "chair exact-available pixels",
            "paper_safe_wording": "After anchoring the proxy repair to the PR21 exact-pixel set and remapping PR20 proxy IDs into verified final checkpoint indices, repaired proxy indices remain disjoint from exact contributor IDs on all chair exact-available pixels.",
            "unsafe_wording_to_avoid": "The proxy method is wrong in all scenes.",
            "recommended_next_step": summary.get("recommended_next_step", ""),
        },
        {
            "claim_id": "C2",
            "claim": "The original PR21.2 zero-overlap was not solely an ID-namespace artifact.",
            "status": "supported" if complete_zero else "unsupported_or_changed",
            "evidence": f"original_zero_overlap_was_namespace_artifact_only={summary.get('original_zero_overlap_was_namespace_artifact_only')}",
            "scope": "evaluated chair scope",
            "paper_safe_wording": "The completed repaired comparison preserves zero overlap, indicating that the original mismatch was not solely due to ID-namespace mismatch within the evaluated chair scope.",
            "unsafe_wording_to_avoid": "The original raw-ID comparison was fully valid.",
            "recommended_next_step": summary.get("recommended_next_step", ""),
        },
        {
            "claim_id": "C3",
            "claim": "The proxy evidence is safe for intervention.",
            "status": "unsupported",
            "evidence": "Exact contribution magnitude and intervention experiments are unavailable.",
            "scope": "all current outputs",
            "paper_safe_wording": "Even with complete repaired ID comparison, proxy IDs remain diagnostic-only because exact contribution magnitudes and intervention experiments are unavailable.",
            "unsafe_wording_to_avoid": "We can reject or downweight proxy Gaussians.",
            "recommended_next_step": "Do not intervene before exact magnitude and intervention validation.",
        },
        {
            "claim_id": "C4",
            "claim": "ViewTrust-GS is a defense.",
            "status": "unsupported",
            "evidence": "No defense, rejection, reweighting, or densification gating is enabled.",
            "scope": "current repository state",
            "paper_safe_wording": "The current pipeline remains an observation-only exact attribution audit.",
            "unsafe_wording_to_avoid": "ViewTrust-GS defends against poisoning.",
            "recommended_next_step": "Keep observation and intervention claims separate.",
        },
        {
            "claim_id": "C5",
            "claim": "PR21.4 can proceed.",
            "status": "supported" if complete_valid else "unsupported",
            "evidence": f"pr212d_ready_for_pr214={summary.get('pr212d_ready_for_pr214')}",
            "scope": "namespace-safe contributor-ID comparison",
            "paper_safe_wording": "With complete namespace-safe contributor-ID comparison established, the next technical gap is exact contribution magnitude / alpha-transmittance-aware replay.",
            "unsafe_wording_to_avoid": "PR21.4 is intervention-ready.",
            "recommended_next_step": summary.get("recommended_next_step", ""),
        },
    ]


def _write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# PR21.2d Exact-Pixel-Anchored PR20 Proxy Repair",
        "",
        "## Purpose",
        "PR21.2d bypasses the incomplete PR21.2b repaired preview by anchoring directly on PR21 exact pixels and extracting matching PR20 original proxy rows.",
        "",
        "## Input Readiness",
        f"Mapping source: `{summary.get('mapping_source')}` with confidence `{summary.get('mapping_confidence')}`.",
        f"PR21.2c-fix diagnosed missing preview coverage; PR21.2d uses PR20 original rows instead.",
        "",
        "## Exact Pixel Anchor",
        f"Exact pixels: `{summary.get('exact_pixel_count')}`; exact rows: `{summary.get('exact_row_count')}`; views: `{summary.get('exact_view_count_with_rows')}`.",
        "",
        "## PR20 Proxy Coverage on Exact Anchor",
        f"Exact pixels with PR20 proxy rows: `{summary.get('exact_pixels_with_pr20_proxy_rows')}`.",
        f"Exact pixels missing PR20 proxy rows: `{summary.get('exact_pixels_missing_pr20_proxy_rows')}`.",
        "",
        "## Verified Mapping Audit",
        f"Mapping columns: `{summary.get('mapping_source_id_column')}` -> `{summary.get('mapping_target_index_column')}`.",
        f"Unmapped proxy rows: `{summary.get('unmapped_proxy_row_count_on_exact_pixels')}`.",
        "",
        "## Complete Repaired Pixel Comparison",
        f"Mean / median Jaccard: `{summary.get('complete_mean_pixel_jaccard')}` / `{summary.get('complete_median_pixel_jaccard')}`.",
        f"Zero-overlap preserved: `{summary.get('complete_zero_overlap_preserved')}`.",
        f"Nonzero overlap found: `{summary.get('complete_nonzero_overlap_found')}`.",
        "",
        "## View and Group Results",
        "See `pr212d_complete_repaired_view_exact_vs_proxy.csv` and `pr212d_complete_repaired_group_exact_vs_proxy.csv`. Groups without exact rows remain caveated.",
        "",
        "## PR21.2c Partial vs PR21.2d Complete",
        f"Complete valid pixels: `{summary.get('complete_valid_pixel_count')}`; missing pixels: `{summary.get('complete_missing_pixel_count')}`.",
        "",
        "## Claim Status",
        "See `pr212d_claim_status_table.csv`.",
        "",
        "## Safety Boundary",
        "- Observation-only.",
        "- Not defense.",
        "- Not intervention-ready.",
        "- No exact magnitude.",
        "- Drums excluded.",
        "",
        "## Next Step",
        str(summary.get("recommended_next_step", "")),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_wording(path: Path) -> None:
    lines = [
        "# PR21.2d Paper Wording",
        "",
        "## A. Complete Zero-Overlap Preserved",
        "",
        "After anchoring the proxy repair to the PR21 exact-pixel set and remapping PR20 proxy IDs into verified final checkpoint Gaussian indices, repaired proxy indices remain disjoint from PR21 exact contributor IDs on all 43 chair exact-available pixels. This supports the conclusion that the original proxy/exact mismatch was not solely an ID-namespace artifact within the evaluated chair scope.",
        "",
        "## B. Nonzero Overlap Found",
        "",
        "After exact-pixel-anchored repair, repaired PR20 proxy indices partially overlap PR21 exact contributor IDs. Thus, the earlier zero-overlap result should be treated as namespace- or coverage-sensitive, and the proxy evidence should be reinterpreted.",
        "",
        "## C. Conservative Limitation",
        "",
        "This remains contributor-ID-only evidence. It does not measure exact alpha, transmittance, splat weight, or render contribution magnitude, and it does not make proxy IDs safe for intervention.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_next_step(path: Path, summary: dict[str, Any]) -> None:
    if not _truth(summary.get("pr212d_ready_for_pr214")):
        text = "Recommend archive with covered-scope wording or fix proxy export before PR21.4."
    elif _truth(summary.get("complete_nonzero_overlap_found")):
        text = "Recommend PR21.3a positioning update before PR21.4."
    else:
        text = "Recommend PR21.4 exact contribution magnitude / alpha-transmittance-aware replay."
    path.write_text("# PR21.2d Next-Step Decision Memo\n\n" + text + "\n", encoding="utf-8")


def _manifest_rows(output_dir: Path, inputs: list[tuple[str, Path, bool]]) -> list[dict[str, Any]]:
    items = [(name, path, required, "input") for name, path, required in inputs]
    items.extend((name, output_dir / name, True, "output_pr212d") for name in OUTPUT_FILES)
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


def build_pr212d_exact_pixel_anchored_proxy_repair(
    *,
    pr200_chair_dir: Path,
    pr211_chair_dir: Path,
    pr212_chair_dir: Path,
    pr212a_chair_dir: Path,
    pr212b_chair_dir: Path,
    pr212c_chair_dir: Path,
    pr212cfix_chair_dir: Path,
    pr213_chair_dir: Path,
    run_dir: Path,
    output_dir: Path,
    scene: str = "chair",
    condition: str = "corrupt_occluder",
    subset_name: str = "seed_20260710",
    write_markdown: bool = False,
) -> tuple[dict[str, Any], int]:
    del write_markdown
    output_dir.mkdir(parents=True, exist_ok=True)
    pr211_summary = load_json(pr211_chair_dir / "pr211_exact_sparse_attribution_summary.json")
    pr212_summary = load_json(pr212_chair_dir / "pr212_chair_exact_vs_proxy_summary.json")
    pr212a_summary = load_json(pr212a_chair_dir / "pr212a_chair_id_namespace_audit_summary.json")
    pr212b_summary = load_json(pr212b_chair_dir / "pr212b_pr20_proxy_id_source_audit_summary.json")
    pr212c_summary = load_json(pr212c_chair_dir / "pr212c_repaired_exact_vs_proxy_summary.json")
    pr212cfix_summary = load_json(pr212cfix_chair_dir / "pr212cfix_corrected_summary.json")
    exact_rows = load_csv_rows(pr211_chair_dir / "pr211_exact_pixel_gaussian_contributions.csv")
    pr20_rows = load_csv_rows(pr200_chair_dir / "pr200_pixel_gaussian_contributions.csv")

    exact_by_pixel = _by_pixel(exact_rows)
    pr20_by_pixel = _by_pixel(pr20_rows)
    anchor_rows = _anchor_rows(scene, condition, subset_name, exact_by_pixel)
    proxy_rows_on_anchor, coverage_rows = _extract_pr20_anchor_rows(scene, condition, subset_name, exact_by_pixel, pr20_by_pixel)
    checkpoint_count = _safe_int(pr212a_summary.get("checkpoint_gaussian_count"))
    mapping, mapping_audit = _load_mapping(
        pr212b_summary=pr212b_summary,
        pr212b_dir=pr212b_chair_dir,
        run_dir=run_dir,
        checkpoint_gaussian_count=checkpoint_count,
        proxy_rows_on_anchor=proxy_rows_on_anchor,
    )
    repaired_rows = _repaired_rows(proxy_rows_on_anchor=proxy_rows_on_anchor, mapping=mapping, audit=mapping_audit)
    repaired_by_pixel = _by_pixel(repaired_rows)
    pixel_rows = _pixel_comparison_rows(
        scene=scene,
        condition=condition,
        subset_name=subset_name,
        exact_by_pixel=exact_by_pixel,
        repaired_by_pixel=repaired_by_pixel,
    )
    view_rows = _aggregate_by_view(
        scene=scene,
        condition=condition,
        subset_name=subset_name,
        exact_by_pixel=exact_by_pixel,
        repaired_by_pixel=repaired_by_pixel,
        pixel_rows=pixel_rows,
    )
    group_rows = _aggregate_by_group(
        scene=scene,
        condition=condition,
        subset_name=subset_name,
        exact_by_pixel=exact_by_pixel,
        repaired_by_pixel=repaired_by_pixel,
        pixel_rows=pixel_rows,
    )

    input_warnings = _validate_readiness(pr212b_summary, pr212cfix_summary, pr211_summary, exact_rows)
    valid_pixels = [row for row in pixel_rows if row["comparison_status"] == "complete_repaired_comparison_valid"]
    missing_pixels = [row for row in pixel_rows if row["comparison_status"] != "complete_repaired_comparison_valid"]
    nonzero = any((_safe_int(row.get("intersection_count")) or 0) > 0 for row in pixel_rows)
    complete = bool(pixel_rows and not missing_pixels and not input_warnings and mapping_audit.get("mapping_status") == "verified_complete_for_exact_anchor")
    zero_preserved = bool(complete and not nonzero)
    jaccards = [float(row["jaccard"]) for row in valid_pixels if row.get("jaccard") not in ("", None)]
    recalls = [float(row["exact_recall_by_repaired_proxy"]) for row in valid_pixels if row.get("exact_recall_by_repaired_proxy") not in ("", None)]
    precisions = [float(row["repaired_proxy_precision_against_exact"]) for row in valid_pixels if row.get("repaired_proxy_precision_against_exact") not in ("", None)]
    view_jaccards = [float(row["view_jaccard"]) for row in view_rows if row.get("view_jaccard") not in ("", None)]
    group_jaccards = [float(row["group_jaccard"]) for row in group_rows if row.get("group_jaccard") not in ("", None)]
    exact_pixels_with_proxy = len([row for row in coverage_rows if row["coverage_status"] == "pr20_proxy_present"])
    repaired_ids = _id_set([row for row in repaired_rows if not row.get("repair_warning")], "verified_final_gaussian_index")
    unmapped_rows = len([row for row in repaired_rows if row.get("repair_warning")])
    exact_views = {key[0] for key in exact_by_pixel}
    if complete and not nonzero:
        recommended = "Run PR21.4 exact contribution magnitude / alpha-transmittance-aware replay."
    elif complete and nonzero:
        recommended = "Update PR21.3 positioning to reflect repaired nonzero overlap before PR21.4."
    else:
        recommended = "Archive covered-scope result or fix PR20 proxy export."

    summary = {
        "schema_name": "viewtrust.pr212d.exact_pixel_anchored_proxy_repair.summary",
        "schema_version": 1,
        "created_at_utc": _utc_now(),
        "scene": scene,
        "condition": condition,
        "subset_name": subset_name,
        "observation_only": True,
        "training_intervention": False,
        "defense_enabled": False,
        "view_rejection_enabled": False,
        "densification_gating_enabled": False,
        "third_party_modified": False,
        "pr200_chair_input_dir": str(pr200_chair_dir),
        "pr211_chair_input_dir": str(pr211_chair_dir),
        "pr212_chair_input_dir": str(pr212_chair_dir),
        "pr212a_chair_input_dir": str(pr212a_chair_dir),
        "pr212b_chair_input_dir": str(pr212b_chair_dir),
        "pr212c_chair_input_dir": str(pr212c_chair_dir),
        "pr212cfix_chair_input_dir": str(pr212cfix_chair_dir),
        "pr213_chair_input_dir": str(pr213_chair_dir),
        "run_dir": str(run_dir),
        "exact_evidence_quality": pr211_summary.get("evidence_quality", "exact_sparse_contributor_id_only"),
        "exact_contributor_id_only_available": bool(exact_rows),
        "exact_render_contribution_available": False,
        "exact_contribution_magnitude_available": False,
        "mapping_source": mapping_audit.get("mapping_source", ""),
        "mapping_source_id_column": mapping_audit.get("source_id_column", ""),
        "mapping_target_index_column": mapping_audit.get("target_index_column", ""),
        "mapping_confidence": mapping_audit.get("mapping_confidence", ""),
        "exact_pixel_count": len(exact_by_pixel),
        "exact_row_count": len(exact_rows),
        "exact_view_count_with_rows": len(exact_views),
        "pr20_proxy_pixels_on_exact_anchor_count": exact_pixels_with_proxy,
        "exact_pixels_with_pr20_proxy_rows": exact_pixels_with_proxy,
        "exact_pixels_missing_pr20_proxy_rows": len(exact_by_pixel) - exact_pixels_with_proxy,
        "repaired_proxy_row_count_on_exact_pixels": len(repaired_rows),
        "repaired_proxy_unique_id_count_on_exact_pixels": len(repaired_ids),
        "unmapped_proxy_row_count_on_exact_pixels": unmapped_rows,
        "complete_valid_pixel_count": len(valid_pixels),
        "complete_missing_pixel_count": len(missing_pixels),
        "complete_mean_pixel_jaccard": _mean(jaccards),
        "complete_median_pixel_jaccard": _median(jaccards),
        "complete_mean_exact_recall_by_proxy": _mean(recalls),
        "complete_mean_proxy_precision_against_exact": _mean(precisions),
        "complete_view_mean_jaccard": _mean(view_jaccards),
        "complete_group_mean_jaccard": _mean(group_jaccards),
        "complete_zero_overlap_preserved": zero_preserved,
        "complete_nonzero_overlap_found": nonzero,
        "complete_zero_overlap_claim_safe_within_exact_available_scope": zero_preserved,
        "original_zero_overlap_was_namespace_artifact_only": False if complete else "",
        "proxy_degeneracy_supported_by_complete_repaired_exact": False,
        "direct_collateral_exact_overlap_established": False,
        "train013_exact_control_separation_established": False,
        "proxy_safe_for_intervention": False,
        "drums_used_as_exact_evidence": False,
        "pr212d_ready_for_pr214": bool(complete),
        "input_warnings": input_warnings,
        "pr212_summary_schema": pr212_summary.get("schema_name", ""),
        "recommended_next_step": recommended,
    }
    delta_rows = _delta_rows(pr212c_summary, pixel_rows, summary)
    claims = _claim_rows(summary)

    write_json(output_dir / "pr212d_exact_pixel_anchored_proxy_repair_summary.json", summary)
    write_csv_rows(output_dir / "pr212d_exact_pixel_anchor.csv", anchor_rows, ANCHOR_FIELDS)
    proxy_fields = sorted({key for row in proxy_rows_on_anchor for key in row})
    write_csv_rows(output_dir / "pr212d_pr20_proxy_rows_on_exact_pixels.csv", proxy_rows_on_anchor, proxy_fields)
    write_csv_rows(output_dir / "pr212d_anchor_proxy_coverage.csv", coverage_rows, COVERAGE_FIELDS)
    write_csv_rows(output_dir / "pr212d_verified_mapping_audit.csv", [mapping_audit], MAPPING_AUDIT_FIELDS)
    write_csv_rows(output_dir / "pr212d_exact_pixel_anchored_repaired_proxy_rows.csv", repaired_rows, REPAIRED_FIELDS)
    write_csv_rows(output_dir / "pr212d_complete_repaired_pixel_exact_vs_proxy.csv", pixel_rows, PIXEL_FIELDS)
    write_csv_rows(output_dir / "pr212d_complete_repaired_view_exact_vs_proxy.csv", view_rows, VIEW_FIELDS)
    write_csv_rows(output_dir / "pr212d_complete_repaired_group_exact_vs_proxy.csv", group_rows, GROUP_FIELDS)
    write_csv_rows(output_dir / "pr212d_pr212c_partial_vs_complete_delta.csv", delta_rows, DELTA_FIELDS)
    write_csv_rows(output_dir / "pr212d_claim_status_table.csv", claims, CLAIM_FIELDS)
    _write_report(output_dir / "pr212d_exact_pixel_anchored_proxy_repair_report.md", summary)
    _write_wording(output_dir / "pr212d_paper_wording.md")
    _write_next_step(output_dir / "pr212d_next_step_decision_memo.md", summary)
    inputs = [
        ("pr200_chair_dir", pr200_chair_dir, True),
        ("pr211_chair_dir", pr211_chair_dir, True),
        ("pr212_chair_dir", pr212_chair_dir, True),
        ("pr212a_chair_dir", pr212a_chair_dir, True),
        ("pr212b_chair_dir", pr212b_chair_dir, True),
        ("pr212c_chair_dir", pr212c_chair_dir, True),
        ("pr212cfix_chair_dir", pr212cfix_chair_dir, True),
        ("pr213_chair_dir", pr213_chair_dir, True),
        ("run_dir", run_dir, True),
    ]
    manifest = output_dir / "artifact_manifest.csv"
    write_csv_rows(manifest, _manifest_rows(output_dir, inputs), MANIFEST_FIELDS)
    write_csv_rows(manifest, _manifest_rows(output_dir, inputs), MANIFEST_FIELDS)
    return summary, 0
