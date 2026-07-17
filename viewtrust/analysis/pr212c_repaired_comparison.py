"""PR21.2c repaired chair exact-vs-proxy comparison."""

from __future__ import annotations

import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from viewtrust.analysis.clean_prior_normalization import load_csv_rows, load_json, write_csv_rows, write_json


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
    "mapping_confidence_min",
    "mapping_confidence_mode",
    "repair_warning_count",
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
    "partial_pixel_count",
    "repaired_zero_overlap_pixel_count",
    "repaired_nonzero_overlap_pixel_count",
    "repair_warning_count",
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
]

DELTA_FIELDS = ["metric", "original_value", "repaired_value", "delta", "interpretation"]

DEGENERACY_FIELDS = [
    "comparison_name",
    "set_a_name",
    "set_b_name",
    "set_a_unique_count",
    "set_b_unique_count",
    "intersection_count",
    "union_count",
    "jaccard",
    "conclusion",
    "caveat",
]

CLAIM_FIELDS = [
    "claim_id",
    "claim",
    "status",
    "evidence",
    "scope",
    "paper_safe_wording",
    "unsafe_wording_to_avoid",
    "recommended_next_step",
]

MANIFEST_FIELDS = ["relative_path", "path", "exists", "file_type", "size_bytes", "required", "artifact_group"]

OUTPUT_FILES = [
    "pr212c_repaired_exact_vs_proxy_summary.json",
    "pr212c_repaired_pixel_exact_vs_proxy.csv",
    "pr212c_repaired_view_exact_vs_proxy.csv",
    "pr212c_repaired_group_exact_vs_proxy.csv",
    "pr212c_original_vs_repaired_delta.csv",
    "pr212c_repaired_proxy_degeneracy_reassessment.csv",
    "pr212c_claim_status_table.csv",
    "pr212c_repaired_exact_vs_proxy_report.md",
    "pr212c_paper_wording.md",
    "pr212c_next_step_decision_memo.md",
    "artifact_manifest.csv",
]

GROUP_ORDER = ["direct_corrupted", "collateral", "clean_control", "other_clean"]


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
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _safe_float(value: Any) -> float | None:
    try:
        if value in ("", None):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def _ratio(num: int, den: int) -> float | None:
    return num / den if den else None


def _jaccard(a: set[str], b: set[str]) -> float | None:
    union = a | b
    return len(a & b) / len(union) if union else None


def _norm_id(value: Any) -> str:
    intval = _safe_int(value)
    return "" if intval is None else str(intval)


def _pixel_key(row: dict[str, Any]) -> tuple[str, int, int]:
    return str(row.get("view_name", "")), _safe_int(row.get("pixel_x")) or 0, _safe_int(row.get("pixel_y")) or 0


def _normalize_group(value: Any, view_name: str = "") -> str:
    text = str(value or "").strip()
    if text in GROUP_ORDER:
        return text
    if text in {"co_visible_collateral", "collateral_view"}:
        return "collateral"
    if text in {"clean_prior_demoted", "clean_control_view"}:
        return "clean_control"
    if text == "direct_corrupted":
        return "direct_corrupted"
    if view_name in {"train_004", "train_009", "train_012", "train_017"}:
        return "direct_corrupted"
    if view_name == "train_014":
        return "collateral"
    if view_name == "train_013":
        return "clean_control"
    return "other_clean"


def _compare_sets(exact: set[str], repaired: set[str]) -> dict[str, Any]:
    inter = exact & repaired
    exact_only = exact - repaired
    repaired_only = repaired - exact
    union = exact | repaired
    return {
        "intersection": inter,
        "exact_only": exact_only,
        "repaired_only": repaired_only,
        "union": union,
        "jaccard": _jaccard(exact, repaired),
        "recall": _ratio(len(inter), len(exact)),
        "precision": _ratio(len(inter), len(repaired)),
    }


def _confidence_rank(value: str) -> int:
    return {"none": 0, "low": 1, "medium": 2, "high": 3}.get(value, 0)


def _confidence_min(values: list[str]) -> str:
    clean = [value for value in values if value]
    if not clean:
        return ""
    return min(clean, key=_confidence_rank)


def _mode(values: list[str]) -> str:
    clean = [value for value in values if value]
    return Counter(clean).most_common(1)[0][0] if clean else ""


def _valid_mapping_row(row: dict[str, Any]) -> bool:
    status = str(row.get("mapping_status", "")).strip()
    confidence = str(row.get("mapping_confidence", "")).strip()
    return bool(_norm_id(row.get("verified_final_gaussian_index"))) and (
        status == "verified_mapping_candidate" or confidence == "high"
    )


def _build_exact(rows: list[dict[str, str]]) -> tuple[dict[tuple[str, int, int], set[str]], dict[tuple[str, int, int], list[dict[str, str]]], dict[tuple[str, int, int], str]]:
    by_pixel: dict[tuple[str, int, int], set[str]] = defaultdict(set)
    row_lists: dict[tuple[str, int, int], list[dict[str, str]]] = defaultdict(list)
    groups: dict[tuple[str, int, int], str] = {}
    for row in rows:
        gid = _norm_id(row.get("gaussian_id"))
        if not gid:
            continue
        key = _pixel_key(row)
        by_pixel[key].add(gid)
        row_lists[key].append(row)
        groups.setdefault(key, _normalize_group(row.get("view_group"), key[0]))
    return dict(by_pixel), dict(row_lists), groups


def _build_repaired(rows: list[dict[str, str]]) -> tuple[
    dict[tuple[str, int, int], set[str]],
    dict[tuple[str, int, int], list[dict[str, str]]],
    dict[tuple[str, int, int], int],
    dict[tuple[str, int, int], int],
    dict[tuple[str, int, int], str],
]:
    by_pixel: dict[tuple[str, int, int], set[str]] = defaultdict(set)
    valid_rows: dict[tuple[str, int, int], list[dict[str, str]]] = defaultdict(list)
    total_counts: dict[tuple[str, int, int], int] = defaultdict(int)
    warning_counts: dict[tuple[str, int, int], int] = defaultdict(int)
    groups: dict[tuple[str, int, int], str] = {}
    for row in rows:
        key = _pixel_key(row)
        total_counts[key] += 1
        groups.setdefault(key, _normalize_group(row.get("view_group"), key[0]))
        if row.get("repair_warning"):
            warning_counts[key] += 1
        if not _valid_mapping_row(row):
            continue
        gid = _norm_id(row.get("verified_final_gaussian_index"))
        by_pixel[key].add(gid)
        valid_rows[key].append(row)
    return dict(by_pixel), dict(valid_rows), dict(total_counts), dict(warning_counts), groups


def _input_readiness(pr212b: dict[str, Any]) -> tuple[bool, list[str]]:
    expected = {
        "explicit_mapping_available": True,
        "mapping_confidence": "high",
        "exact_available_proxy_rows_repair_feasible": True,
        "exact_available_mapping_coverage_rate": 1.0,
        "repaired_zero_overlap_claim_safe": True,
        "pr212b_ready_for_pr212c": True,
        "proxy_safe_for_intervention": False,
        "pr212b_ready_for_pr214": False,
    }
    failures = []
    for key, expected_value in expected.items():
        value = pr212b.get(key)
        if isinstance(expected_value, bool):
            ok = _truth(value) is expected_value
        elif isinstance(expected_value, float):
            ok = _safe_float(value) == expected_value
        else:
            ok = value == expected_value
        if not ok:
            failures.append(f"{key}={value!r} expected {expected_value!r}")
    return not failures, failures


def _validate_pr211(summary: dict[str, Any], exact_rows: list[dict[str, str]]) -> list[str]:
    warnings = []
    if summary.get("evidence_quality") != "exact_sparse_contributor_id_only":
        warnings.append("PR21.1e evidence_quality is not exact_sparse_contributor_id_only")
    if not exact_rows:
        warnings.append("PR21.1e exact contributor rows are missing")
    if _truth(summary.get("exact_render_contribution_succeeded")):
        warnings.append("PR21.1e unexpectedly reports exact render contribution")
    return warnings


def _pixel_rows(
    *,
    scene: str,
    condition: str,
    subset_name: str,
    exact_by_pixel: dict[tuple[str, int, int], set[str]],
    exact_groups: dict[tuple[str, int, int], str],
    repaired_by_pixel: dict[tuple[str, int, int], set[str]],
    repaired_valid_rows: dict[tuple[str, int, int], list[dict[str, str]]],
    repaired_total_counts: dict[tuple[str, int, int], int],
    repaired_warning_counts: dict[tuple[str, int, int], int],
) -> list[dict[str, Any]]:
    rows = []
    for key in sorted(exact_by_pixel):
        view, x, y = key
        exact = exact_by_pixel[key]
        repaired = repaired_by_pixel.get(key, set())
        cmp = _compare_sets(exact, repaired)
        valid = repaired_valid_rows.get(key, [])
        total_proxy_rows = repaired_total_counts.get(key, 0)
        unmapped = max(total_proxy_rows - len(valid), 0)
        confidences = [str(row.get("mapping_confidence", "")) for row in valid]
        if not exact:
            status = "missing_exact_ids"
            interpretation = "repaired_comparison_incomplete"
        elif not repaired and total_proxy_rows == 0:
            status = "missing_repaired_proxy_ids"
            interpretation = "repaired_comparison_incomplete"
        elif unmapped:
            status = "repaired_comparison_partial"
            interpretation = "repaired_comparison_incomplete"
        else:
            status = "repaired_comparison_valid"
            interpretation = "repaired_nonzero_overlap" if cmp["intersection"] else "repaired_zero_overlap"
        rows.append(
            {
                "scene": scene,
                "condition": condition,
                "subset_name": subset_name,
                "view_name": view,
                "view_group": exact_groups.get(key, _normalize_group("", view)),
                "pixel_x": x,
                "pixel_y": y,
                "exact_id_count": len(exact),
                "repaired_proxy_id_count": len(repaired),
                "unmapped_proxy_row_count": unmapped,
                "intersection_count": len(cmp["intersection"]),
                "exact_only_count": len(cmp["exact_only"]),
                "repaired_proxy_only_count": len(cmp["repaired_only"]),
                "union_count": len(cmp["union"]),
                "jaccard": cmp["jaccard"],
                "exact_recall_by_repaired_proxy": cmp["recall"],
                "repaired_proxy_precision_against_exact": cmp["precision"],
                "mapping_confidence_min": _confidence_min(confidences),
                "mapping_confidence_mode": _mode(confidences),
                "repair_warning_count": repaired_warning_counts.get(key, 0),
                "comparison_status": status,
                "interpretation": interpretation,
            }
        )
    return rows


def _aggregate_by_view(
    *,
    scene: str,
    condition: str,
    subset_name: str,
    exact_rows: list[dict[str, str]],
    pixel_rows: list[dict[str, Any]],
    repaired_valid_rows: dict[tuple[str, int, int], list[dict[str, str]]],
) -> list[dict[str, Any]]:
    exact_by_view: dict[str, set[str]] = defaultdict(set)
    exact_row_count: Counter[str] = Counter()
    exact_pixel_count: Counter[str] = Counter()
    repaired_by_view: dict[str, set[str]] = defaultdict(set)
    repaired_row_count: Counter[str] = Counter()
    unmapped: Counter[str] = Counter()
    warnings: Counter[str] = Counter()
    group_for_view: dict[str, str] = {}
    status_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in exact_rows:
        view = str(row.get("view_name", ""))
        gid = _norm_id(row.get("gaussian_id"))
        if gid:
            exact_by_view[view].add(gid)
            exact_row_count[view] += 1
            group_for_view.setdefault(view, _normalize_group(row.get("view_group"), view))
    for row in pixel_rows:
        view = str(row["view_name"])
        exact_pixel_count[view] += 1
        unmapped[view] += _safe_int(row.get("unmapped_proxy_row_count")) or 0
        warnings[view] += _safe_int(row.get("repair_warning_count")) or 0
        status_counts[view][str(row.get("comparison_status", ""))] += 1
        status_counts[view][str(row.get("interpretation", ""))] += 1
        group_for_view.setdefault(view, str(row.get("view_group") or _normalize_group("", view)))
    for key, rows in repaired_valid_rows.items():
        view = key[0]
        for row in rows:
            gid = _norm_id(row.get("verified_final_gaussian_index"))
            if gid:
                repaired_by_view[view].add(gid)
                repaired_row_count[view] += 1
                group_for_view.setdefault(view, _normalize_group(row.get("view_group"), view))
    out = []
    for view in sorted(set(exact_by_view) | set(repaired_by_view) | set(exact_pixel_count)):
        exact = exact_by_view.get(view, set())
        repaired = repaired_by_view.get(view, set())
        cmp = _compare_sets(exact, repaired)
        valid_count = status_counts[view]["repaired_comparison_valid"]
        partial_count = (
            status_counts[view]["repaired_comparison_partial"]
            + status_counts[view]["missing_repaired_proxy_ids"]
            + status_counts[view]["missing_exact_ids"]
        )
        status = "repaired_comparison_valid" if valid_count and not partial_count else "repaired_comparison_partial"
        interpretation = (
            "repaired_comparison_incomplete"
            if partial_count
            else "repaired_nonzero_overlap"
            if cmp["intersection"]
            else "repaired_zero_overlap"
        )
        out.append(
            {
                "scene": scene,
                "condition": condition,
                "subset_name": subset_name,
                "view_name": view,
                "view_group": group_for_view.get(view, _normalize_group("", view)),
                "exact_pixel_count": exact_pixel_count[view],
                "exact_row_count": exact_row_count[view],
                "exact_unique_id_count": len(exact),
                "repaired_proxy_row_count": repaired_row_count[view],
                "repaired_proxy_unique_id_count": len(repaired),
                "unmapped_proxy_row_count": unmapped[view],
                "intersection_unique_id_count": len(cmp["intersection"]),
                "exact_only_unique_id_count": len(cmp["exact_only"]),
                "repaired_proxy_only_unique_id_count": len(cmp["repaired_only"]),
                "union_unique_id_count": len(cmp["union"]),
                "view_jaccard": cmp["jaccard"],
                "view_exact_recall_by_repaired_proxy": cmp["recall"],
                "view_repaired_proxy_precision_against_exact": cmp["precision"],
                "valid_pixel_count": valid_count,
                "partial_pixel_count": partial_count,
                "repaired_zero_overlap_pixel_count": status_counts[view]["repaired_zero_overlap"],
                "repaired_nonzero_overlap_pixel_count": status_counts[view]["repaired_nonzero_overlap"],
                "repair_warning_count": warnings[view],
                "comparison_status": status,
                "interpretation": interpretation,
            }
        )
    return out


def _aggregate_by_group(
    *,
    scene: str,
    condition: str,
    subset_name: str,
    exact_rows: list[dict[str, str]],
    pixel_rows: list[dict[str, Any]],
    repaired_valid_rows: dict[tuple[str, int, int], list[dict[str, str]]],
) -> list[dict[str, Any]]:
    exact_by_group: dict[str, set[str]] = defaultdict(set)
    exact_views: dict[str, set[str]] = defaultdict(set)
    exact_rows_count: Counter[str] = Counter()
    exact_pixels: dict[str, set[tuple[str, int, int]]] = defaultdict(set)
    repaired_by_group: dict[str, set[str]] = defaultdict(set)
    repaired_rows_count: Counter[str] = Counter()
    unmapped: Counter[str] = Counter()
    status_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in exact_rows:
        view = str(row.get("view_name", ""))
        group = _normalize_group(row.get("view_group"), view)
        gid = _norm_id(row.get("gaussian_id"))
        if gid:
            exact_by_group[group].add(gid)
            exact_views[group].add(view)
            exact_rows_count[group] += 1
            exact_pixels[group].add(_pixel_key(row))
    for row in pixel_rows:
        group = str(row.get("view_group") or "other_clean")
        unmapped[group] += _safe_int(row.get("unmapped_proxy_row_count")) or 0
        status_counts[group][str(row.get("comparison_status", ""))] += 1
        status_counts[group][str(row.get("interpretation", ""))] += 1
    for key, rows in repaired_valid_rows.items():
        for row in rows:
            group = _normalize_group(row.get("view_group"), key[0])
            gid = _norm_id(row.get("verified_final_gaussian_index"))
            if gid:
                repaired_by_group[group].add(gid)
                repaired_rows_count[group] += 1
    out = []
    for group in GROUP_ORDER:
        exact = exact_by_group.get(group, set())
        repaired = repaired_by_group.get(group, set())
        cmp = _compare_sets(exact, repaired)
        valid = status_counts[group]["repaired_comparison_valid"]
        partial = (
            status_counts[group]["repaired_comparison_partial"]
            + status_counts[group]["missing_repaired_proxy_ids"]
            + status_counts[group]["missing_exact_ids"]
        )
        if not exact and repaired:
            status = "missing_exact_ids"
            interpretation = "repaired_comparison_incomplete"
        else:
            status = "repaired_comparison_valid" if valid and not partial else "repaired_comparison_partial"
            interpretation = (
                "repaired_comparison_incomplete"
                if partial
                else "repaired_nonzero_overlap"
                if cmp["intersection"]
                else "repaired_zero_overlap"
            )
        out.append(
            {
                "scene": scene,
                "condition": condition,
                "subset_name": subset_name,
                "group_name": group,
                "exact_view_count": len(exact_views.get(group, set())),
                "exact_pixel_count": len(exact_pixels.get(group, set())),
                "exact_row_count": exact_rows_count[group],
                "exact_unique_id_count": len(exact),
                "repaired_proxy_row_count": repaired_rows_count[group],
                "repaired_proxy_unique_id_count": len(repaired),
                "unmapped_proxy_row_count": unmapped[group],
                "intersection_unique_id_count": len(cmp["intersection"]),
                "exact_only_unique_id_count": len(cmp["exact_only"]),
                "repaired_proxy_only_unique_id_count": len(cmp["repaired_only"]),
                "union_unique_id_count": len(cmp["union"]),
                "group_jaccard": cmp["jaccard"],
                "group_exact_recall_by_repaired_proxy": cmp["recall"],
                "group_repaired_proxy_precision_against_exact": cmp["precision"],
                "comparison_status": status,
                "interpretation": interpretation,
            }
        )
    return out


def _metric_interpretation(original: float | None, repaired: float | None) -> str:
    if original is None or repaired is None:
        return "not_comparable"
    if original == 0 and repaired == 0:
        return "unchanged_zero_overlap_after_repair"
    if repaired > original:
        return "overlap_increased_after_repair"
    if repaired < original:
        return "overlap_decreased_after_repair"
    return "not_comparable"


def _original_proxy_unique_count(pixel_rows: list[dict[str, str]]) -> int:
    ids: set[str] = set()
    for row in pixel_rows:
        for value in str(row.get("proxy_gaussian_ids_semicolon", "")).split(";"):
            gid = _norm_id(value)
            if gid:
                ids.add(gid)
    return len(ids)


def _delta_rows(
    original_summary: dict[str, Any],
    original_pixels: list[dict[str, str]],
    repaired_summary_values: dict[str, Any],
) -> list[dict[str, Any]]:
    metrics = [
        ("pixel_mean_jaccard", original_summary.get("mean_pixel_jaccard"), repaired_summary_values.get("repaired_mean_pixel_jaccard")),
        ("pixel_median_jaccard", original_summary.get("median_pixel_jaccard"), repaired_summary_values.get("repaired_median_pixel_jaccard")),
        ("pixel_mean_exact_recall", original_summary.get("mean_exact_recall_by_proxy"), repaired_summary_values.get("repaired_mean_exact_recall_by_proxy")),
        ("pixel_mean_proxy_precision", original_summary.get("mean_proxy_precision_against_exact"), repaired_summary_values.get("repaired_mean_proxy_precision_against_exact")),
        ("view_mean_jaccard", original_summary.get("view_mean_jaccard_for_exact_available_views"), repaired_summary_values.get("repaired_view_mean_jaccard")),
        ("exact_pixel_count", original_summary.get("exact_pixel_count"), repaired_summary_values.get("exact_pixel_count")),
        ("exact_row_count", original_summary.get("exact_row_count"), repaired_summary_values.get("exact_row_count")),
        ("exact_view_count_with_rows", original_summary.get("exact_view_count_with_rows"), repaired_summary_values.get("exact_view_count_with_rows")),
        ("proxy_unique_id_count", _original_proxy_unique_count(original_pixels), repaired_summary_values.get("repaired_proxy_unique_id_count_on_exact_pixels")),
        ("repaired_proxy_unique_id_count", "", repaired_summary_values.get("repaired_proxy_unique_id_count_on_exact_pixels")),
    ]
    out = []
    for metric, original, repaired in metrics:
        o = _safe_float(original)
        r = _safe_float(repaired)
        out.append(
            {
                "metric": metric,
                "original_value": "" if original is None else original,
                "repaired_value": "" if repaired is None else repaired,
                "delta": r - o if o is not None and r is not None else "",
                "interpretation": _metric_interpretation(o, r),
            }
        )
    return out


def _set_by_view_from_repaired(valid_rows: dict[tuple[str, int, int], list[dict[str, str]]]) -> dict[str, set[str]]:
    by_view: dict[str, set[str]] = defaultdict(set)
    for key, rows in valid_rows.items():
        for row in rows:
            gid = _norm_id(row.get("verified_final_gaussian_index"))
            if gid:
                by_view[key[0]].add(gid)
    return dict(by_view)


def _degeneracy_rows(
    exact_by_pixel: dict[tuple[str, int, int], set[str]],
    repaired_valid_rows: dict[tuple[str, int, int], list[dict[str, str]]],
) -> list[dict[str, Any]]:
    repaired_by_view = _set_by_view_from_repaired(repaired_valid_rows)
    repaired_by_group: dict[str, set[str]] = defaultdict(set)
    exact_all: set[str] = set()
    exact_by_group: dict[str, set[str]] = defaultdict(set)
    for key, ids in exact_by_pixel.items():
        group = _normalize_group("", key[0])
        exact_by_group[group].update(ids)
        exact_all.update(ids)
    for view, ids in repaired_by_view.items():
        repaired_by_group[_normalize_group("", view)].update(ids)

    comparisons = [
        ("direct_repaired_proxy_pool_vs_collateral", "direct_corrupted", "collateral"),
        ("direct_repaired_proxy_pool_vs_clean_control", "direct_corrupted", "clean_control"),
        ("collateral_repaired_proxy_pool_vs_clean_control", "collateral", "clean_control"),
    ]
    out = []
    for name, a, b in comparisons:
        set_a = repaired_by_group.get(a, set())
        set_b = repaired_by_group.get(b, set())
        j = _jaccard(set_a, set_b)
        if j is None:
            conclusion = "insufficient_exact_rows"
        elif j >= 0.95 and set_a and set_b:
            conclusion = "repaired_proxy_degeneracy_persists"
        elif j > 0 and set_a and set_b:
            conclusion = "repaired_proxy_degeneracy_reduced"
        else:
            conclusion = "repaired_proxy_degeneracy_invalidated"
        out.append(_degeneracy_row(name, a, b, set_a, set_b, conclusion, "Proxy pool overlap is diagnostic-only and not exact contribution magnitude."))

    repaired_exact_scope = set().union(*repaired_by_group.values()) if repaired_by_group else set()
    inter = exact_all & repaired_exact_scope
    out.append(
        _degeneracy_row(
            "exact_available_repaired_proxy_pool_vs_exact_contributors",
            "repaired_proxy_on_exact_pixels",
            "exact_contributors",
            repaired_exact_scope,
            exact_all,
            "exact_overlap_present" if inter else "exact_overlap_absent",
            "Exact evidence is contributor-ID-only and available only on PR21.1e chair pixels.",
        )
    )
    if not exact_by_group.get("collateral") or not exact_by_group.get("clean_control"):
        out.append(
            _degeneracy_row(
                "group_level_exact_scope_caveat",
                "collateral_exact_or_clean_control_exact",
                "direct_exact",
                exact_by_group.get("collateral", set()) | exact_by_group.get("clean_control", set()),
                exact_by_group.get("direct_corrupted", set()),
                "insufficient_exact_rows",
                "Do not claim direct-collateral exact overlap unless exact rows exist for both groups.",
            )
        )
    return out


def _degeneracy_row(name: str, a_name: str, b_name: str, a: set[str], b: set[str], conclusion: str, caveat: str) -> dict[str, Any]:
    inter = a & b
    union = a | b
    return {
        "comparison_name": name,
        "set_a_name": a_name,
        "set_b_name": b_name,
        "set_a_unique_count": len(a),
        "set_b_unique_count": len(b),
        "intersection_count": len(inter),
        "union_count": len(union),
        "jaccard": len(inter) / len(union) if union else "",
        "conclusion": conclusion,
        "caveat": caveat,
    }


def _claim_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    zero_supported = _truth(summary.get("repaired_zero_overlap_preserved")) and _truth(summary.get("repaired_zero_overlap_claim_safe_within_exact_available_scope"))
    valid = _truth(summary.get("pr212c_ready_for_pr214"))
    return [
        {
            "claim_id": "C1",
            "claim": "After verified namespace repair, PR20 repaired proxy IDs and PR21 exact contributor IDs show zero overlap on chair exact-available pixels/views.",
            "status": "supported" if zero_supported else "unsupported_or_partial",
            "evidence": f"mean_jaccard={summary.get('repaired_mean_pixel_jaccard')}; nonzero_overlap={summary.get('repaired_nonzero_overlap_found')}",
            "scope": "chair PR21.1e exact-available pixels/views",
            "paper_safe_wording": "After remapping PR20 proxy IDs into the final checkpoint namespace, repaired proxy indices remain disjoint from exact contributor IDs on chair exact-available pixels/views.",
            "unsafe_wording_to_avoid": "The proxy method is always wrong.",
            "recommended_next_step": summary.get("recommended_next_step", ""),
        },
        {
            "claim_id": "C2",
            "claim": "Original PR21.2 zero-overlap was not solely caused by ID namespace mismatch.",
            "status": "supported" if zero_supported else "unsupported_or_changed",
            "evidence": f"original_namespace_artifact_only={summary.get('original_zero_overlap_was_namespace_artifact_only')}",
            "scope": "repaired chair comparison only",
            "paper_safe_wording": "The repaired comparison preserves the zero-overlap result, indicating that the original mismatch was not only an ID-namespace artifact within the evaluated scope.",
            "unsafe_wording_to_avoid": "The original PR21.2 comparison was fully valid.",
            "recommended_next_step": summary.get("recommended_next_step", ""),
        },
        {
            "claim_id": "C3",
            "claim": "PR20 proxy IDs are now safe for intervention.",
            "status": "unsupported",
            "evidence": "No exact contribution magnitude or intervention experiment is available.",
            "scope": "all current PR21.2c outputs",
            "paper_safe_wording": "Even after namespace repair, proxy IDs remain diagnostic-only because exact contribution magnitudes and intervention experiments are unavailable.",
            "unsafe_wording_to_avoid": "We can reject or downweight proxy Gaussians.",
            "recommended_next_step": "Run exact contribution magnitude / alpha-transmittance-aware replay before considering interventions.",
        },
        {
            "claim_id": "C4",
            "claim": "ViewTrust-GS is a defense.",
            "status": "unsupported",
            "evidence": "No defense, rejection, reweighting, or gating is enabled.",
            "scope": "current repository state",
            "paper_safe_wording": "The current pipeline remains an observation-only exact attribution audit.",
            "unsafe_wording_to_avoid": "ViewTrust-GS defends against poisoning.",
            "recommended_next_step": "Keep observation and intervention claims separate.",
        },
        {
            "claim_id": "C5",
            "claim": "PR21.4 is now unblocked.",
            "status": "supported" if valid else "unsupported",
            "evidence": f"pr212c_ready_for_pr214={summary.get('pr212c_ready_for_pr214')}",
            "scope": "namespace-safe chair repaired comparison",
            "paper_safe_wording": "With repaired namespace comparison complete, the next technical gap is exact contribution magnitude.",
            "unsafe_wording_to_avoid": "PR21.4 can be used for intervention.",
            "recommended_next_step": summary.get("recommended_next_step", ""),
        },
    ]


def _write_report(path: Path, summary: dict[str, Any], view_rows: list[dict[str, Any]], group_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# PR21.2c Repaired Chair Exact-vs-Proxy Comparison",
        "",
        "## Purpose",
        "PR21.2c recomputes exact-vs-proxy comparison after PR21.2b maps PR20 proxy IDs into verified final checkpoint indices.",
        "",
        "## Input Readiness",
        f"Mapping source: `{summary.get('repaired_mapping_source')}`.",
        f"Mapping confidence: `{summary.get('repaired_mapping_confidence')}`.",
        f"Comparison partial: `{summary.get('comparison_partial')}`.",
        "",
        "## Pixel-Level Repaired Comparison",
        f"Exact pixels: `{summary.get('exact_pixel_count')}`.",
        f"Exact rows: `{summary.get('exact_row_count')}`.",
        f"Repaired proxy rows on exact pixels: `{summary.get('repaired_proxy_row_count_on_exact_pixels')}`.",
        f"Mean / median Jaccard: `{summary.get('repaired_mean_pixel_jaccard')}` / `{summary.get('repaired_median_pixel_jaccard')}`.",
        f"Mean recall / precision: `{summary.get('repaired_mean_exact_recall_by_proxy')}` / `{summary.get('repaired_mean_proxy_precision_against_exact')}`.",
        "",
        "## View-Level Repaired Comparison",
    ]
    for row in view_rows:
        lines.append(
            f"- `{row['view_name']}` ({row['view_group']}): jaccard `{row['view_jaccard']}`, intersection `{row['intersection_unique_id_count']}`, status `{row['comparison_status']}`."
        )
    lines.extend(["", "## Group-Level Repaired Comparison"])
    for row in group_rows:
        lines.append(
            f"- `{row['group_name']}`: exact views `{row['exact_view_count']}`, jaccard `{row['group_jaccard']}`, status `{row['comparison_status']}`."
        )
    lines.extend(
        [
            "",
            "Group-level evidence is limited to groups with PR21.1e exact rows. Do not claim direct-collateral exact overlap unless exact rows exist for both groups.",
            "",
            "## Original vs Repaired Difference",
            f"Repaired zero-overlap preserved: `{summary.get('repaired_zero_overlap_preserved')}`.",
            f"Repaired nonzero overlap found: `{summary.get('repaired_nonzero_overlap_found')}`.",
            "",
            "## Repaired Proxy Degeneracy",
            f"Proxy degeneracy supported by repaired exact evidence: `{summary.get('proxy_degeneracy_supported_by_repaired_exact')}`.",
            "",
            "## Claim Status",
            "See `pr212c_claim_status_table.csv`.",
            "",
            "## Safety Boundary",
            "- Observation-only.",
            "- Not defense.",
            "- Not intervention-ready.",
            "- Proxy IDs remain unsafe for intervention.",
            "- No exact magnitude evidence.",
            "- Drums not used as exact evidence.",
            "",
            "## Next Step",
            str(summary.get("recommended_next_step", "")),
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_wording(path: Path) -> None:
    lines = [
        "# PR21.2c Paper Wording",
        "",
        "## If Repaired Zero-Overlap Is Preserved",
        "",
        "After remapping PR20 proxy identifiers into verified final checkpoint Gaussian indices, the repaired proxy sets remain disjoint from PR21 exact contributor-ID sets on chair exact-available pixels and views. This indicates that the original zero-overlap observation was not solely an ID-namespace artifact within the evaluated scope.",
        "",
        "## If Repaired Nonzero Overlap Is Found",
        "",
        "After namespace repair, repaired PR20 proxy sets partially overlap PR21 exact contributor-ID sets. The original raw-ID zero-overlap result should therefore be interpreted as partly namespace-driven, and proxy evidence should be reinterpreted rather than discarded.",
        "",
        "## Conservative Limitation",
        "",
        "This comparison is contributor-ID-only. It does not estimate exact alpha, transmittance, splat weight, or render-contribution magnitude, and it does not make proxy IDs safe for intervention.",
        "",
        "## Next Step",
        "",
        "The next technical gap is exact contribution magnitude / alpha-transmittance-aware replay, not a defense or intervention mechanism.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_next_step(path: Path, summary: dict[str, Any]) -> None:
    if not _truth(summary.get("pr212c_ready_for_pr214")):
        recommendation = "Recommend PR21.2c-fix before anything else because repaired comparison coverage is partial."
    elif _truth(summary.get("repaired_nonzero_overlap_found")):
        recommendation = "Recommend PR21.3a interpretation update before PR21.4 because repaired proxy/exact agreement changes the research story."
    else:
        recommendation = "Recommend PR21.4 exact contribution magnitude / alpha-transmittance-aware replay. Namespace-safe contributor-ID comparison is now complete; the next gap is contribution magnitude."
    path.write_text("# PR21.2c Next-Step Decision Memo\n\n" + recommendation + "\n", encoding="utf-8")


def _manifest_rows(output_dir: Path, inputs: list[tuple[str, Path, bool]]) -> list[dict[str, Any]]:
    items = [(name, path, required, "input") for name, path, required in inputs]
    items.extend((name, output_dir / name, True, "output_pr212c") for name in OUTPUT_FILES)
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


def build_pr212c_repaired_exact_vs_proxy_comparison(
    *,
    pr200_chair_dir: Path,
    pr211_chair_dir: Path,
    pr212_chair_dir: Path,
    pr212a_chair_dir: Path,
    pr212b_chair_dir: Path,
    pr213_chair_dir: Path,
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
    pr212b_summary = load_json(pr212b_chair_dir / "pr212b_pr20_proxy_id_source_audit_summary.json")
    pr212a_summary = load_json(pr212a_chair_dir / "pr212a_chair_id_namespace_audit_summary.json")
    pr213_summary = load_json(pr213_chair_dir / "pr213_chair_exact_evidence_positioning_summary.json")
    exact_rows = load_csv_rows(pr211_chair_dir / "pr211_exact_pixel_gaussian_contributions.csv")
    repaired_rows = load_csv_rows(pr212b_chair_dir / "pr212b_pr20_proxy_repaired_preview.csv")
    original_pixel_rows = load_csv_rows(pr212_chair_dir / "pr212_chair_pixel_exact_vs_proxy.csv")

    readiness_ok, readiness_failures = _input_readiness(pr212b_summary)
    input_warnings = readiness_failures + _validate_pr211(pr211_summary, exact_rows)
    if not pr212_summary:
        input_warnings.append("PR21.2 original comparison summary is missing")
    if not repaired_rows:
        input_warnings.append("PR21.2b repaired proxy preview is missing")

    exact_by_pixel, exact_row_lists, exact_groups = _build_exact(exact_rows)
    repaired_by_pixel, repaired_valid_rows, repaired_total_counts, repaired_warning_counts, repaired_groups = _build_repaired(repaired_rows)
    for key, group in repaired_groups.items():
        exact_groups.setdefault(key, group)

    pixels = _pixel_rows(
        scene=scene,
        condition=condition,
        subset_name=subset_name,
        exact_by_pixel=exact_by_pixel,
        exact_groups=exact_groups,
        repaired_by_pixel=repaired_by_pixel,
        repaired_valid_rows=repaired_valid_rows,
        repaired_total_counts=repaired_total_counts,
        repaired_warning_counts=repaired_warning_counts,
    )
    views = _aggregate_by_view(
        scene=scene,
        condition=condition,
        subset_name=subset_name,
        exact_rows=exact_rows,
        pixel_rows=pixels,
        repaired_valid_rows=repaired_valid_rows,
    )
    groups = _aggregate_by_group(
        scene=scene,
        condition=condition,
        subset_name=subset_name,
        exact_rows=exact_rows,
        pixel_rows=pixels,
        repaired_valid_rows=repaired_valid_rows,
    )

    pixel_jaccards = [float(row["jaccard"]) for row in pixels if row.get("jaccard") not in ("", None)]
    pixel_recalls = [float(row["exact_recall_by_repaired_proxy"]) for row in pixels if row.get("exact_recall_by_repaired_proxy") not in ("", None)]
    pixel_precisions = [float(row["repaired_proxy_precision_against_exact"]) for row in pixels if row.get("repaired_proxy_precision_against_exact") not in ("", None)]
    view_jaccards = [float(row["view_jaccard"]) for row in views if row.get("view_jaccard") not in ("", None)]
    group_jaccards = [float(row["group_jaccard"]) for row in groups if row.get("group_jaccard") not in ("", None)]

    exact_all = set().union(*exact_by_pixel.values()) if exact_by_pixel else set()
    repaired_exact_scope: set[str] = set()
    repaired_row_count_on_exact = 0
    for key in exact_by_pixel:
        repaired_exact_scope.update(repaired_by_pixel.get(key, set()))
        repaired_row_count_on_exact += len(repaired_valid_rows.get(key, []))
    unmapped_on_exact = sum(max(repaired_total_counts.get(key, 0) - len(repaired_valid_rows.get(key, [])), 0) for key in exact_by_pixel)
    partial = bool(input_warnings or any(row["comparison_status"] != "repaired_comparison_valid" for row in pixels))
    nonzero = any((_safe_int(row.get("intersection_count")) or 0) > 0 for row in pixels)
    zero_preserved = bool(pixels and not partial and not nonzero)

    repaired_summary_values = {
        "exact_pixel_count": len(exact_by_pixel),
        "exact_row_count": len(exact_rows),
        "exact_view_count_with_rows": len({key[0] for key in exact_by_pixel}),
        "repaired_proxy_unique_id_count_on_exact_pixels": len(repaired_exact_scope),
        "repaired_mean_pixel_jaccard": _mean(pixel_jaccards),
        "repaired_median_pixel_jaccard": _median(pixel_jaccards),
        "repaired_mean_exact_recall_by_proxy": _mean(pixel_recalls),
        "repaired_mean_proxy_precision_against_exact": _mean(pixel_precisions),
        "repaired_view_mean_jaccard": _mean(view_jaccards),
        "repaired_group_mean_jaccard": _mean(group_jaccards),
    }
    deltas = _delta_rows(pr212_summary, original_pixel_rows, repaired_summary_values)
    degeneracy = _degeneracy_rows(exact_by_pixel, repaired_valid_rows)
    direct_collateral_exact_overlap = any(
        row["group_name"] == "collateral" and (_safe_int(row.get("exact_view_count")) or 0) > 0 for row in groups
    ) and any(row["group_name"] == "direct_corrupted" and (_safe_int(row.get("exact_view_count")) or 0) > 0 for row in groups)
    train013_exact_control = any(
        row["group_name"] == "clean_control" and (_safe_int(row.get("exact_view_count")) or 0) > 0 for row in groups
    )
    comparison_valid = bool(readiness_ok and pixels and not partial)
    ready_for_pr214 = bool(comparison_valid)
    if zero_preserved:
        recommended = "Run PR21.4 exact contribution magnitude / alpha-transmittance-aware replay."
    elif nonzero and comparison_valid:
        recommended = "Interpret repaired overlap and update PR21.3 positioning before PR21.4."
    else:
        recommended = "Fix repaired proxy coverage before PR21.4."

    summary = {
        "schema_name": "viewtrust.pr212c.repaired_exact_vs_proxy.summary",
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
        "pr213_chair_input_dir": str(pr213_chair_dir),
        "exact_evidence_quality": pr211_summary.get("evidence_quality", "exact_sparse_contributor_id_only"),
        "exact_contributor_id_only_available": bool(exact_rows),
        "exact_render_contribution_available": False,
        "exact_contribution_magnitude_available": False,
        "repaired_mapping_source": pr212b_summary.get("mapping_source", ""),
        "repaired_mapping_confidence": pr212b_summary.get("mapping_confidence", ""),
        "exact_pixel_count": len(exact_by_pixel),
        "exact_row_count": len(exact_rows),
        "exact_view_count_with_rows": len({key[0] for key in exact_by_pixel}),
        "repaired_proxy_row_count_on_exact_pixels": repaired_row_count_on_exact,
        "repaired_proxy_unique_id_count_on_exact_pixels": len(repaired_exact_scope),
        "unmapped_proxy_row_count_on_exact_pixels": unmapped_on_exact,
        "repaired_mean_pixel_jaccard": repaired_summary_values["repaired_mean_pixel_jaccard"],
        "repaired_median_pixel_jaccard": repaired_summary_values["repaired_median_pixel_jaccard"],
        "repaired_mean_exact_recall_by_proxy": repaired_summary_values["repaired_mean_exact_recall_by_proxy"],
        "repaired_mean_proxy_precision_against_exact": repaired_summary_values["repaired_mean_proxy_precision_against_exact"],
        "repaired_view_mean_jaccard": repaired_summary_values["repaired_view_mean_jaccard"],
        "repaired_group_mean_jaccard": repaired_summary_values["repaired_group_mean_jaccard"],
        "repaired_zero_overlap_preserved": zero_preserved,
        "repaired_nonzero_overlap_found": nonzero,
        "original_zero_overlap_was_namespace_artifact_only": False if comparison_valid else "",
        "repaired_zero_overlap_claim_safe_within_exact_available_scope": zero_preserved,
        "proxy_degeneracy_supported_by_repaired_exact": False,
        "direct_collateral_exact_overlap_established": bool(direct_collateral_exact_overlap),
        "train013_exact_control_separation_established": bool(train013_exact_control),
        "proxy_safe_for_intervention": False,
        "drums_used_as_exact_evidence": False,
        "pr212c_ready_for_pr214": ready_for_pr214,
        "comparison_partial": partial,
        "input_readiness_ok": readiness_ok,
        "input_warnings": input_warnings,
        "pr212a_same_namespace_supported": pr212a_summary.get("same_global_gaussian_id_namespace_supported", ""),
        "pr213_positioning_input_schema": pr213_summary.get("schema_name", ""),
        "recommended_next_step": recommended,
    }
    claims = _claim_rows(summary)

    write_json(output_dir / "pr212c_repaired_exact_vs_proxy_summary.json", summary)
    write_csv_rows(output_dir / "pr212c_repaired_pixel_exact_vs_proxy.csv", pixels, PIXEL_FIELDS)
    write_csv_rows(output_dir / "pr212c_repaired_view_exact_vs_proxy.csv", views, VIEW_FIELDS)
    write_csv_rows(output_dir / "pr212c_repaired_group_exact_vs_proxy.csv", groups, GROUP_FIELDS)
    write_csv_rows(output_dir / "pr212c_original_vs_repaired_delta.csv", deltas, DELTA_FIELDS)
    write_csv_rows(output_dir / "pr212c_repaired_proxy_degeneracy_reassessment.csv", degeneracy, DEGENERACY_FIELDS)
    write_csv_rows(output_dir / "pr212c_claim_status_table.csv", claims, CLAIM_FIELDS)
    _write_report(output_dir / "pr212c_repaired_exact_vs_proxy_report.md", summary, views, groups)
    _write_wording(output_dir / "pr212c_paper_wording.md")
    _write_next_step(output_dir / "pr212c_next_step_decision_memo.md", summary)
    inputs = [
        ("pr200_chair_dir", pr200_chair_dir, True),
        ("pr211_chair_dir", pr211_chair_dir, True),
        ("pr212_chair_dir", pr212_chair_dir, True),
        ("pr212a_chair_dir", pr212a_chair_dir, True),
        ("pr212b_chair_dir", pr212b_chair_dir, True),
        ("pr213_chair_dir", pr213_chair_dir, True),
    ]
    manifest = output_dir / "artifact_manifest.csv"
    write_csv_rows(manifest, _manifest_rows(output_dir, inputs), MANIFEST_FIELDS)
    write_csv_rows(manifest, _manifest_rows(output_dir, inputs), MANIFEST_FIELDS)
    return summary, 0
