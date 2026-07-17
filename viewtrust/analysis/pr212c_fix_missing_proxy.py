"""PR21.2c-fix diagnostics for missing repaired proxy pixels."""

from __future__ import annotations

import hashlib
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from viewtrust.analysis.clean_prior_normalization import load_csv_rows, load_json, write_csv_rows, write_json


MISSING_FIELDS = [
    "scene",
    "condition",
    "subset_name",
    "view_name",
    "view_group",
    "pixel_x",
    "pixel_y",
    "exact_id_count",
    "exact_ids_semicolon",
    "pr212c_comparison_status",
    "pr212c_interpretation",
    "pr212c_jaccard",
    "notes",
]

TRACE_FIELDS = [
    "view_name",
    "pixel_x",
    "pixel_y",
    "exact_row_count",
    "exact_unique_id_count",
    "pr20_original_proxy_row_count",
    "pr20_original_unique_raw_id_count",
    "pr212b_repaired_preview_row_count",
    "pr212b_repaired_preview_unique_verified_index_count",
    "pr212b_empty_verified_index_count",
    "pr212b_mapping_status_counts",
    "pr212b_mapping_confidence_counts",
    "pr212b_repair_warning_count",
    "pr212b_preview_comparison_row_count",
    "pr212c_comparison_status",
    "inferred_failure_mode",
    "fix_possible",
    "recommended_fix",
    "notes",
]

COVERAGE_FIELDS = ["pixel_set_name", "pixel_count", "view_count", "train_009_count", "train_012_count", "hash", "notes"]

COORD_FIELDS = [
    "view_name",
    "pixel_x",
    "pixel_y",
    "normal_match_pr20",
    "string_normalized_match_pr20",
    "int_cast_match_pr20",
    "xy_swap_match_pr20",
    "y_flip_match_pr20",
    "x_flip_match_pr20",
    "xy_swap_y_flip_match_pr20",
    "likely_coordinate_issue",
    "notes",
]

CORRECTED_FIELDS = [
    "scene",
    "condition",
    "subset_name",
    "view_name",
    "view_group",
    "pixel_x",
    "pixel_y",
    "exact_id_count",
    "corrected_repaired_proxy_id_count",
    "corrected_unmapped_proxy_row_count",
    "intersection_count",
    "exact_only_count",
    "repaired_proxy_only_count",
    "union_count",
    "jaccard",
    "exact_recall_by_repaired_proxy",
    "repaired_proxy_precision_against_exact",
    "comparison_status",
    "interpretation",
    "correction_source",
    "correction_warning",
]

CLAIM_FIELDS = [
    "scope_name",
    "support_status",
    "evidence",
    "paper_safe_wording",
    "unsafe_wording_to_avoid",
    "recommended_next_step",
]

MANIFEST_FIELDS = ["relative_path", "path", "exists", "file_type", "size_bytes", "required", "artifact_group"]

OUTPUT_FILES = [
    "pr212cfix_corrected_summary.json",
    "pr212cfix_missing_pixel_list.csv",
    "pr212cfix_missing_pixel_trace.csv",
    "pr212cfix_pixel_set_coverage.csv",
    "pr212cfix_coordinate_format_audit.csv",
    "pr212cfix_corrected_repaired_pixel_comparison_preview.csv",
    "pr212cfix_claim_scope_recommendation.csv",
    "pr212cfix_missing_repaired_proxy_pixels_report.md",
    "pr212cfix_paper_wording.md",
    "pr212cfix_next_step_decision_memo.md",
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
    intval = _safe_int(value)
    return "" if intval is None else str(intval)


def _pixel_key(row: dict[str, Any]) -> PixelKey:
    return str(row.get("view_name", "")).strip(), _safe_int(row.get("pixel_x")) or 0, _safe_int(row.get("pixel_y")) or 0


def _raw_pixel_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return str(row.get("view_name", "")), str(row.get("pixel_x", "")), str(row.get("pixel_y", ""))


def _fmt_ids(values: set[str]) -> str:
    return ";".join(sorted(values, key=lambda value: (0, int(value)) if value.isdigit() else (1, value)))


def _counter_text(values: list[str]) -> str:
    return ";".join(f"{key}:{count}" for key, count in sorted(Counter(values).items()))


def _valid_repaired_row(row: dict[str, Any]) -> bool:
    gid = _norm_id(row.get("verified_final_gaussian_index"))
    if not gid:
        return False
    status = str(row.get("mapping_status", "")).strip()
    confidence = str(row.get("mapping_confidence", "")).strip()
    return status == "verified_mapping_candidate" or confidence == "high"


def _compare(exact: set[str], repaired: set[str]) -> dict[str, Any]:
    inter = exact & repaired
    exact_only = exact - repaired
    repaired_only = repaired - exact
    union = exact | repaired
    return {
        "intersection_count": len(inter),
        "exact_only_count": len(exact_only),
        "repaired_proxy_only_count": len(repaired_only),
        "union_count": len(union),
        "jaccard": len(inter) / len(union) if union else None,
        "recall": _ratio(len(inter), len(exact)),
        "precision": _ratio(len(inter), len(repaired)),
    }


def _by_pixel(rows: list[dict[str, str]]) -> dict[PixelKey, list[dict[str, str]]]:
    out: dict[PixelKey, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        out[_pixel_key(row)].append(row)
    return dict(out)


def _id_set(rows: list[dict[str, str]], column: str) -> set[str]:
    return {gid for gid in (_norm_id(row.get(column)) for row in rows) if gid}


def _pixel_hash(keys: set[PixelKey]) -> str:
    text = "\n".join(f"{view},{x},{y}" for view, x, y in sorted(keys))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _coverage_row(name: str, keys: set[PixelKey], notes: str = "") -> dict[str, Any]:
    return {
        "pixel_set_name": name,
        "pixel_count": len(keys),
        "view_count": len({key[0] for key in keys}),
        "train_009_count": len([key for key in keys if key[0] == "train_009"]),
        "train_012_count": len([key for key in keys if key[0] == "train_012"]),
        "hash": _pixel_hash(keys),
        "notes": notes,
    }


def _missing_pr212c_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    missing = []
    for row in rows:
        status = str(row.get("comparison_status", ""))
        interp = str(row.get("interpretation", ""))
        repaired_count = _safe_int(row.get("repaired_proxy_id_count")) or 0
        corrected_count = _safe_int(row.get("corrected_repaired_proxy_id_count")) or 0
        if status == "missing_repaired_proxy_ids" or interp == "repaired_comparison_incomplete" or (repaired_count == 0 and corrected_count == 0):
            missing.append(row)
    return missing


def _validate_inputs(pr212c: dict[str, Any], pr212b: dict[str, Any]) -> list[str]:
    warnings = []
    expected_pr212c = {
        "exact_pixel_count": 43,
        "repaired_mean_pixel_jaccard": 0.0,
        "repaired_mean_exact_recall_by_proxy": 0.0,
        "repaired_mean_proxy_precision_against_exact": 0.0,
        "repaired_zero_overlap_claim_safe_within_exact_available_scope": False,
        "pr212c_ready_for_pr214": False,
    }
    for key, expected in expected_pr212c.items():
        value = pr212c.get(key)
        if value in ("", None):
            warnings.append(f"PR21.2c summary missing {key}")
        elif isinstance(expected, bool) and _truth(value) is not expected:
            warnings.append(f"PR21.2c {key}={value!r}, expected {expected!r}")
        elif isinstance(expected, (int, float)) and _safe_float(value) != float(expected):
            warnings.append(f"PR21.2c {key}={value!r}, expected {expected!r}")
    if "fix" not in str(pr212c.get("recommended_next_step", "")).lower():
        warnings.append("PR21.2c recommended_next_step does not clearly indicate fixing repaired proxy coverage")
    expected_pr212b = {
        "explicit_mapping_available": True,
        "mapping_confidence": "high",
        "all_pr20_proxy_rows_repair_feasible": True,
        "exact_available_proxy_rows_repair_feasible": True,
        "exact_available_mapping_coverage_rate": 1.0,
        "proxy_safe_for_intervention": False,
    }
    for key, expected in expected_pr212b.items():
        value = pr212b.get(key)
        if value in ("", None):
            warnings.append(f"PR21.2b summary missing {key}")
        elif isinstance(expected, bool) and _truth(value) is not expected:
            warnings.append(f"PR21.2b {key}={value!r}, expected {expected!r}")
        elif isinstance(expected, float) and _safe_float(value) != expected:
            warnings.append(f"PR21.2b {key}={value!r}, expected {expected!r}")
        elif isinstance(expected, str) and value != expected:
            warnings.append(f"PR21.2b {key}={value!r}, expected {expected!r}")
    return warnings


def _missing_pixel_list(
    *,
    scene: str,
    condition: str,
    subset_name: str,
    missing_rows: list[dict[str, str]],
    exact_by_pixel: dict[PixelKey, list[dict[str, str]]],
) -> list[dict[str, Any]]:
    rows = []
    for row in missing_rows:
        key = _pixel_key(row)
        exact_ids = _id_set(exact_by_pixel.get(key, []), "gaussian_id")
        rows.append(
            {
                "scene": scene,
                "condition": condition,
                "subset_name": subset_name,
                "view_name": key[0],
                "view_group": row.get("view_group", ""),
                "pixel_x": key[1],
                "pixel_y": key[2],
                "exact_id_count": len(exact_ids),
                "exact_ids_semicolon": _fmt_ids(exact_ids),
                "pr212c_comparison_status": row.get("comparison_status", ""),
                "pr212c_interpretation": row.get("interpretation", ""),
                "pr212c_jaccard": row.get("jaccard", ""),
                "notes": "missing repaired proxy rows in PR21.2c comparison",
            }
        )
    return rows


def _failure_mode(
    *,
    pr20_rows: list[dict[str, str]],
    repaired_rows: list[dict[str, str]],
    preview_rows: list[dict[str, str]],
    coord_issue: bool,
) -> tuple[str, str, str, str]:
    if not pr20_rows:
        return (
            "absent_from_pr20_original_proxy",
            "false",
            "Scope claim to repaired-proxy-covered exact pixels or rerun PR20 proxy export with the same selected pixel anchor.",
            "No original proxy rows exist for this exact pixel.",
        )
    if not repaired_rows:
        return (
            "present_in_pr20_but_missing_from_pr212b_repaired_preview",
            "false",
            "Regenerate PR21.2b repaired preview or re-export proxy rows with final checkpoint indices.",
            "Original proxy rows exist but no repaired preview rows match the same pixel key.",
        )
    valid_count = len([row for row in repaired_rows if _valid_repaired_row(row)])
    if valid_count:
        return (
            "present_in_pr212b_but_filtered_by_pr212c",
            "true",
            "Rerun PR21.2c with corrected coverage logic after confirming verified mapping rows are accepted.",
            "Repaired preview rows contain valid final indices for this pixel.",
        )
    if any(_norm_id(row.get("verified_final_gaussian_index")) for row in repaired_rows):
        statuses = {str(row.get("mapping_status", "")) for row in repaired_rows}
        warnings = {str(row.get("repair_warning", "")) for row in repaired_rows if row.get("repair_warning")}
        if statuses and "verified_mapping_candidate" not in statuses:
            return (
                "mapping_status_too_strict",
                "true",
                "Inspect PR21.2b mapping_status semantics before relaxing PR21.2c filtering.",
                "Rows have final indices but are not tagged as verified mapping candidates.",
            )
        if warnings:
            return (
                "repair_warning_filter_too_strict",
                "true",
                "Inspect repair_warning semantics before accepting warning-bearing rows.",
                "Rows have final indices but also repair warnings.",
            )
    if coord_issue:
        return (
            "coordinate_or_type_mismatch",
            "false",
            "Fix coordinate alignment before rerunning PR21.2c.",
            "Coordinate transforms or formatting may explain missing rows.",
        )
    if preview_rows:
        return (
            "source_scope_mismatch",
            "false",
            "Compare PR21.2b preview scope against PR21.2c exact-pixel anchor.",
            "Repaired comparison preview has rows but repaired proxy preview does not provide valid rows.",
        )
    return ("unknown", "false", "Inspect source artifacts manually.", "No single source-stage explanation matched.")


def _coordinate_audit_rows(missing_keys: list[PixelKey], pr20_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    raw_keys = {_raw_pixel_key(row) for row in pr20_rows}
    norm_keys = {_pixel_key(row) for row in pr20_rows}
    stripped_keys = {(str(row.get("view_name", "")).strip(), str(row.get("pixel_x", "")).strip(), str(row.get("pixel_y", "")).strip()) for row in pr20_rows}
    max_x = max([key[1] for key in norm_keys], default=0)
    max_y = max([key[2] for key in norm_keys], default=0)
    out = []
    for view, x, y in missing_keys:
        raw = (view, str(x), str(y))
        normal = raw in raw_keys
        string_norm = (view.strip(), str(x).strip(), str(y).strip()) in stripped_keys
        int_cast = (view, x, y) in norm_keys
        xy_swap = (view, y, x) in norm_keys
        y_flip = (view, x, max_y - y) in norm_keys if max_y else False
        x_flip = (view, max_x - x, y) in norm_keys if max_x else False
        xy_swap_y_flip = (view, y, max_y - x) in norm_keys if max_y else False
        issue = bool(not int_cast and (string_norm or xy_swap or y_flip or x_flip or xy_swap_y_flip))
        notes = []
        if string_norm and not normal:
            notes.append("string formatting differs")
        if xy_swap:
            notes.append("xy swap candidate")
        if y_flip:
            notes.append("y flip candidate")
        if x_flip:
            notes.append("x flip candidate")
        out.append(
            {
                "view_name": view,
                "pixel_x": x,
                "pixel_y": y,
                "normal_match_pr20": _bool_text(normal),
                "string_normalized_match_pr20": _bool_text(string_norm),
                "int_cast_match_pr20": _bool_text(int_cast),
                "xy_swap_match_pr20": _bool_text(xy_swap),
                "y_flip_match_pr20": _bool_text(y_flip),
                "x_flip_match_pr20": _bool_text(x_flip),
                "xy_swap_y_flip_match_pr20": _bool_text(xy_swap_y_flip),
                "likely_coordinate_issue": _bool_text(issue),
                "notes": "; ".join(notes),
            }
        )
    return out


def _trace_rows(
    *,
    missing_keys: list[PixelKey],
    exact_by_pixel: dict[PixelKey, list[dict[str, str]]],
    pr20_by_pixel: dict[PixelKey, list[dict[str, str]]],
    repaired_by_pixel: dict[PixelKey, list[dict[str, str]]],
    pr212b_compare_by_pixel: dict[PixelKey, list[dict[str, str]]],
    pr212c_by_pixel: dict[PixelKey, list[dict[str, str]]],
    coord_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    coord_issue_by_key = {_pixel_key(row): _truth(row.get("likely_coordinate_issue")) for row in coord_rows}
    out = []
    for key in missing_keys:
        exact_rows = exact_by_pixel.get(key, [])
        pr20_rows = pr20_by_pixel.get(key, [])
        repaired_rows = repaired_by_pixel.get(key, [])
        preview_rows = pr212b_compare_by_pixel.get(key, [])
        pr212c_rows = pr212c_by_pixel.get(key, [])
        valid_verified = _id_set([row for row in repaired_rows if _valid_repaired_row(row)], "verified_final_gaussian_index")
        empty_verified = len([row for row in repaired_rows if not _norm_id(row.get("verified_final_gaussian_index"))])
        failure, fix_possible, recommended, notes = _failure_mode(
            pr20_rows=pr20_rows,
            repaired_rows=repaired_rows,
            preview_rows=preview_rows,
            coord_issue=coord_issue_by_key.get(key, False),
        )
        out.append(
            {
                "view_name": key[0],
                "pixel_x": key[1],
                "pixel_y": key[2],
                "exact_row_count": len(exact_rows),
                "exact_unique_id_count": len(_id_set(exact_rows, "gaussian_id")),
                "pr20_original_proxy_row_count": len(pr20_rows),
                "pr20_original_unique_raw_id_count": len(_id_set(pr20_rows, "gaussian_id")),
                "pr212b_repaired_preview_row_count": len(repaired_rows),
                "pr212b_repaired_preview_unique_verified_index_count": len(valid_verified),
                "pr212b_empty_verified_index_count": empty_verified,
                "pr212b_mapping_status_counts": _counter_text([str(row.get("mapping_status", "")) for row in repaired_rows]),
                "pr212b_mapping_confidence_counts": _counter_text([str(row.get("mapping_confidence", "")) for row in repaired_rows]),
                "pr212b_repair_warning_count": len([row for row in repaired_rows if row.get("repair_warning")]),
                "pr212b_preview_comparison_row_count": len(preview_rows),
                "pr212c_comparison_status": pr212c_rows[0].get("comparison_status", "") if pr212c_rows else "",
                "inferred_failure_mode": failure,
                "fix_possible": fix_possible,
                "recommended_fix": recommended,
                "notes": notes,
            }
        )
    return out


def _corrected_preview_rows(
    *,
    scene: str,
    condition: str,
    subset_name: str,
    exact_by_pixel: dict[PixelKey, list[dict[str, str]]],
    repaired_by_pixel: dict[PixelKey, list[dict[str, str]]],
) -> list[dict[str, Any]]:
    rows = []
    for key in sorted(exact_by_pixel):
        exact_rows = exact_by_pixel[key]
        exact_ids = _id_set(exact_rows, "gaussian_id")
        proxy_rows = repaired_by_pixel.get(key, [])
        valid_rows = [row for row in proxy_rows if _valid_repaired_row(row)]
        repaired_ids = _id_set(valid_rows, "verified_final_gaussian_index")
        invalid_count = len(proxy_rows) - len(valid_rows)
        cmp = _compare(exact_ids, repaired_ids)
        if not proxy_rows:
            status = "missing_repaired_proxy_ids"
            interpretation = "repaired_comparison_incomplete"
            source = "no_repaired_preview_rows"
            warning = "no proxy rows fabricated"
        elif invalid_count:
            status = "repaired_comparison_partial"
            interpretation = "repaired_comparison_incomplete"
            source = "pr212b_repaired_preview"
            warning = "some repaired preview rows were not clearly valid"
        else:
            status = "repaired_comparison_valid"
            interpretation = "repaired_nonzero_overlap" if cmp["intersection_count"] else "repaired_zero_overlap"
            source = "pr212b_repaired_preview_verified_final_gaussian_index"
            warning = ""
        first = exact_rows[0] if exact_rows else {}
        rows.append(
            {
                "scene": scene,
                "condition": condition,
                "subset_name": subset_name,
                "view_name": key[0],
                "view_group": first.get("view_group", ""),
                "pixel_x": key[1],
                "pixel_y": key[2],
                "exact_id_count": len(exact_ids),
                "corrected_repaired_proxy_id_count": len(repaired_ids),
                "corrected_unmapped_proxy_row_count": invalid_count,
                "intersection_count": cmp["intersection_count"],
                "exact_only_count": cmp["exact_only_count"],
                "repaired_proxy_only_count": cmp["repaired_proxy_only_count"],
                "union_count": cmp["union_count"],
                "jaccard": cmp["jaccard"],
                "exact_recall_by_repaired_proxy": cmp["recall"],
                "repaired_proxy_precision_against_exact": cmp["precision"],
                "comparison_status": status,
                "interpretation": interpretation,
                "correction_source": source,
                "correction_warning": warning,
            }
        )
    return rows


def _claim_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    full_safe = _truth(summary.get("corrected_zero_overlap_claim_safe_within_exact_available_scope"))
    covered_safe = _truth(summary.get("corrected_zero_overlap_preserved_on_covered_pixels"))
    partial = not _truth(summary.get("coverage_problem_resolved"))
    return [
        {
            "scope_name": "full_exact_available_scope",
            "support_status": "supported" if full_safe else "unsupported",
            "evidence": f"corrected_missing_pixel_count={summary.get('corrected_missing_pixel_count')}; nonzero_overlap={summary.get('repaired_nonzero_overlap_found')}",
            "paper_safe_wording": "After repairing the proxy namespace and correcting coverage, repaired proxy indices remain disjoint from exact contributor IDs across all chair exact-available pixels.",
            "unsafe_wording_to_avoid": "Repaired zero-overlap holds for all chair exact pixels when coverage is partial.",
            "recommended_next_step": summary.get("recommended_next_step", ""),
        },
        {
            "scope_name": "repaired_proxy_covered_exact_scope",
            "support_status": "supported" if covered_safe else "unsupported",
            "evidence": f"covered_valid_pixels={summary.get('corrected_valid_pixel_count')}; missing_pixels={summary.get('corrected_missing_pixel_count')}",
            "paper_safe_wording": "On repaired-proxy-covered chair exact pixels, repaired proxy indices remain disjoint from exact contributor IDs.",
            "unsafe_wording_to_avoid": "Covered-pixel evidence proves full exact-available-scope behavior.",
            "recommended_next_step": "Use covered-scope wording only if full coverage remains unresolved." if partial else summary.get("recommended_next_step", ""),
        },
        {
            "scope_name": "train_009_scope",
            "support_status": "diagnostic",
            "evidence": f"missing_pixel_views={summary.get('missing_pixel_views')}",
            "paper_safe_wording": "Per-view coverage should be reported separately for train_009 and train_012.",
            "unsafe_wording_to_avoid": "A complete train_009 result automatically generalizes to train_012.",
            "recommended_next_step": "Inspect per-view rows in the corrected preview.",
        },
        {
            "scope_name": "train_012_scope",
            "support_status": "partial" if "train_012" in str(summary.get("missing_pixel_views", "")) else "diagnostic",
            "evidence": f"missing_pixel_views={summary.get('missing_pixel_views')}",
            "paper_safe_wording": "Train_012 contains the unresolved missing repaired-proxy pixels if coverage remains partial.",
            "unsafe_wording_to_avoid": "Train_012 is fully covered without confirming the missing rows.",
            "recommended_next_step": summary.get("recommended_next_step", ""),
        },
        {
            "scope_name": "direct_corrupted_group_scope",
            "support_status": "partial" if partial else "supported",
            "evidence": f"corrected_missing_pixel_count={summary.get('corrected_missing_pixel_count')}",
            "paper_safe_wording": "Direct-corrupted group claims inherit the repaired-proxy coverage boundary.",
            "unsafe_wording_to_avoid": "Group-level repaired zero-overlap is complete when some exact pixels lack proxy rows.",
            "recommended_next_step": summary.get("recommended_next_step", ""),
        },
        {
            "scope_name": "intervention_scope",
            "support_status": "unsupported",
            "evidence": "proxy_safe_for_intervention=false; exact contribution magnitude unavailable",
            "paper_safe_wording": "This diagnostic remains observation-only and does not make proxy IDs safe for intervention.",
            "unsafe_wording_to_avoid": "Use repaired proxy IDs to reject, downweight, or gate updates.",
            "recommended_next_step": "Do not introduce intervention until exact magnitude evidence and intervention experiments exist.",
        },
    ]


def _write_report(path: Path, summary: dict[str, Any], trace_rows: list[dict[str, Any]]) -> None:
    failure_counts = summary.get("missing_pixel_failure_mode_counts", {})
    lines = [
        "# PR21.2c-fix Missing Repaired Proxy Pixel Diagnosis",
        "",
        "## Purpose",
        "This report diagnoses why PR21.2c produced repaired comparison rows with 36 valid pixels and 7 missing repaired-proxy pixels.",
        "",
        "## Missing Pixel List",
        f"Missing pixels: `{summary.get('missing_pixel_count')}`.",
        f"Missing pixel views: `{summary.get('missing_pixel_views')}`.",
        "",
        "## Source Trace",
        f"Failure-mode counts: `{failure_counts}`.",
    ]
    for row in trace_rows[:20]:
        lines.append(
            f"- `{row['view_name']}:{row['pixel_x']},{row['pixel_y']}` -> `{row['inferred_failure_mode']}`; PR20 rows `{row['pr20_original_proxy_row_count']}`, repaired preview rows `{row['pr212b_repaired_preview_row_count']}`."
        )
    lines.extend(
        [
            "",
            "## Pixel Set Coverage",
            f"Exact pixels present in PR20 proxy rows: `{summary.get('exact_pixels_present_in_pr20_proxy_count')}`.",
            f"Exact pixels present in PR21.2b repaired preview: `{summary.get('exact_pixels_present_in_pr212b_repaired_preview_count')}`.",
            "",
            "## Coordinate and Formatting Audit",
            "Coordinate checks are diagnostic only. PR21.2c-fix does not silently apply coordinate transforms.",
            "",
            "## Corrected Preview",
            f"Corrected valid pixels: `{summary.get('corrected_valid_pixel_count')}`.",
            f"Corrected missing pixels: `{summary.get('corrected_missing_pixel_count')}`.",
            f"Corrected mean pixel Jaccard: `{summary.get('corrected_mean_pixel_jaccard')}`.",
            "",
            "## Claim Scope",
            f"Full exact-available claim safe: `{summary.get('corrected_zero_overlap_claim_safe_within_exact_available_scope')}`.",
            f"Covered-pixel zero-overlap preserved: `{summary.get('corrected_zero_overlap_preserved_on_covered_pixels')}`.",
            "",
            "## Safety Boundary",
            "- Observation-only.",
            "- Not defense.",
            "- Not intervention-ready.",
            "- No exact contribution magnitude.",
            "- Drums excluded as exact evidence.",
            "",
            "## Next Step",
            str(summary.get("recommended_next_step", "")),
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_wording(path: Path) -> None:
    lines = [
        "# PR21.2c-fix Paper Wording",
        "",
        "## A. If Full Coverage Is Resolved",
        "",
        "After repairing the proxy namespace and correcting coverage, repaired proxy indices remain disjoint from exact contributor IDs across all chair exact-available pixels.",
        "",
        "## B. If Partial Coverage Remains",
        "",
        "After repairing the proxy namespace, repaired proxy indices remain disjoint from exact contributor IDs on repaired-proxy-covered chair exact pixels. However, a subset of exact pixels lacks corresponding proxy rows, so full exact-available-scope claims are deferred.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_next_step(path: Path, summary: dict[str, Any]) -> None:
    if _truth(summary.get("coverage_problem_resolved")):
        text = "Recommend rerunning PR21.2c cleanly with corrected coverage logic and then proceeding to PR21.4 if zero-overlap remains."
    else:
        text = "Recommend PR21.2d proxy re-export with exact-pixel anchor, or conservative covered-scope wording and archive if re-export is not planned."
    path.write_text("# PR21.2c-fix Next-Step Decision Memo\n\n" + text + "\n", encoding="utf-8")


def _manifest_rows(output_dir: Path, inputs: list[tuple[str, Path, bool]]) -> list[dict[str, Any]]:
    items = [(name, path, required, "input") for name, path, required in inputs]
    items.extend((name, output_dir / name, True, "output_pr212cfix") for name in OUTPUT_FILES)
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


def build_pr212c_fix_missing_repaired_proxy_pixels(
    *,
    pr200_chair_dir: Path,
    pr211_chair_dir: Path,
    pr212_chair_dir: Path,
    pr212a_chair_dir: Path,
    pr212b_chair_dir: Path,
    pr212c_chair_dir: Path,
    pr213_chair_dir: Path,
    output_dir: Path,
    scene: str = "chair",
    condition: str = "corrupt_occluder",
    subset_name: str = "seed_20260710",
    write_markdown: bool = False,
) -> tuple[dict[str, Any], int]:
    del pr212_chair_dir, pr212a_chair_dir, pr213_chair_dir, write_markdown
    output_dir.mkdir(parents=True, exist_ok=True)
    pr212c_summary = load_json(pr212c_chair_dir / "pr212c_repaired_exact_vs_proxy_summary.json")
    pr212b_summary = load_json(pr212b_chair_dir / "pr212b_pr20_proxy_id_source_audit_summary.json")
    input_warnings = _validate_inputs(pr212c_summary, pr212b_summary)

    exact_rows = load_csv_rows(pr211_chair_dir / "pr211_exact_pixel_gaussian_contributions.csv")
    pr20_rows = load_csv_rows(pr200_chair_dir / "pr200_pixel_gaussian_contributions.csv")
    repaired_rows = load_csv_rows(pr212b_chair_dir / "pr212b_pr20_proxy_repaired_preview.csv")
    pr212b_compare_rows = load_csv_rows(pr212b_chair_dir / "pr212b_repaired_exact_vs_proxy_preview.csv")
    pr212c_pixel_rows = load_csv_rows(pr212c_chair_dir / "pr212c_repaired_pixel_exact_vs_proxy.csv")

    exact_by_pixel = _by_pixel(exact_rows)
    pr20_by_pixel = _by_pixel(pr20_rows)
    repaired_by_pixel = _by_pixel(repaired_rows)
    pr212b_compare_by_pixel = _by_pixel(pr212b_compare_rows)
    pr212c_by_pixel = _by_pixel(pr212c_pixel_rows)

    missing_pr212c = _missing_pr212c_rows(pr212c_pixel_rows)
    missing_keys = sorted({_pixel_key(row) for row in missing_pr212c})
    missing_list = _missing_pixel_list(
        scene=scene,
        condition=condition,
        subset_name=subset_name,
        missing_rows=missing_pr212c,
        exact_by_pixel=exact_by_pixel,
    )
    coord_rows = _coordinate_audit_rows(missing_keys, pr20_rows)
    trace_rows = _trace_rows(
        missing_keys=missing_keys,
        exact_by_pixel=exact_by_pixel,
        pr20_by_pixel=pr20_by_pixel,
        repaired_by_pixel=repaired_by_pixel,
        pr212b_compare_by_pixel=pr212b_compare_by_pixel,
        pr212c_by_pixel=pr212c_by_pixel,
        coord_rows=coord_rows,
    )

    exact_pixels = set(exact_by_pixel)
    pr20_pixels = set(pr20_by_pixel)
    repaired_pixels = set(repaired_by_pixel)
    pr212c_valid_pixels = {
        _pixel_key(row)
        for row in pr212c_pixel_rows
        if row.get("comparison_status") == "repaired_comparison_valid" and (_safe_int(row.get("repaired_proxy_id_count")) or 0) > 0
    }
    coverage_rows = [
        _coverage_row("exact_pixels", exact_pixels, "PR21.1e exact contributor pixel anchor"),
        _coverage_row("pr20_proxy_pixels", pr20_pixels, "Original PR20 proxy contribution pixels"),
        _coverage_row("pr212b_repaired_preview_pixels", repaired_pixels, "PR21.2b repaired preview pixels"),
        _coverage_row("pr212c_valid_repaired_proxy_pixels", pr212c_valid_pixels, "PR21.2c valid repaired proxy pixels"),
        _coverage_row("exact_pixels_missing_in_pr20_proxy", exact_pixels - pr20_pixels),
        _coverage_row("exact_pixels_missing_in_pr212b_repaired_preview", exact_pixels - repaired_pixels),
        _coverage_row("exact_pixels_missing_in_pr212c_valid_proxy", exact_pixels - pr212c_valid_pixels),
        _coverage_row("train_009_exact_pixels", {key for key in exact_pixels if key[0] == "train_009"}),
        _coverage_row("train_012_exact_pixels", {key for key in exact_pixels if key[0] == "train_012"}),
        _coverage_row("train_012_missing_pixels", {key for key in missing_keys if key[0] == "train_012"}),
    ]

    corrected_rows = _corrected_preview_rows(
        scene=scene,
        condition=condition,
        subset_name=subset_name,
        exact_by_pixel=exact_by_pixel,
        repaired_by_pixel=repaired_by_pixel,
    )
    corrected_valid = [row for row in corrected_rows if row["comparison_status"] == "repaired_comparison_valid"]
    corrected_missing = [row for row in corrected_rows if row["comparison_status"] != "repaired_comparison_valid"]
    covered_zero = bool(corrected_valid) and all((_safe_int(row.get("intersection_count")) or 0) == 0 for row in corrected_valid)
    nonzero = any((_safe_int(row.get("intersection_count")) or 0) > 0 for row in corrected_rows)
    coverage_resolved = not corrected_missing
    jaccards = [float(row["jaccard"]) for row in corrected_valid if row.get("jaccard") not in ("", None)]
    recalls = [float(row["exact_recall_by_repaired_proxy"]) for row in corrected_valid if row.get("exact_recall_by_repaired_proxy") not in ("", None)]
    precisions = [float(row["repaired_proxy_precision_against_exact"]) for row in corrected_valid if row.get("repaired_proxy_precision_against_exact") not in ("", None)]
    failure_counts = dict(Counter(str(row["inferred_failure_mode"]) for row in trace_rows))
    if coverage_resolved and covered_zero and not nonzero:
        recommended = "Rerun PR21.2c with corrected coverage logic, then proceed to PR21.4 if zero-overlap remains."
    elif any(row["inferred_failure_mode"] == "coordinate_or_type_mismatch" for row in trace_rows):
        recommended = "Fix coordinate alignment before rerunning PR21.2c."
    elif any(row["inferred_failure_mode"] == "absent_from_pr20_original_proxy" for row in trace_rows):
        recommended = "Scope claim to repaired-proxy-covered exact pixels or rerun PR20 proxy export with the same selected pixel anchor."
    else:
        recommended = "Rerun PR21.2b/PR21.2c diagnostics after resolving repaired preview coverage."

    summary = {
        "schema_name": "viewtrust.pr212cfix.missing_repaired_proxy_pixels.summary",
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
        "pr212c_input_dir": str(pr212c_chair_dir),
        "pr212b_input_dir": str(pr212b_chair_dir),
        "exact_pixel_count": len(exact_pixels),
        "original_pr212c_valid_pixel_count": len(pr212c_valid_pixels),
        "original_pr212c_missing_pixel_count": len(missing_keys),
        "missing_pixel_count": len(missing_keys),
        "missing_pixel_views": sorted({key[0] for key in missing_keys}),
        "missing_pixel_failure_mode_counts": failure_counts,
        "exact_pixels_present_in_pr20_proxy_count": len(exact_pixels & pr20_pixels),
        "exact_pixels_present_in_pr212b_repaired_preview_count": len(exact_pixels & repaired_pixels),
        "corrected_valid_pixel_count": len(corrected_valid),
        "corrected_missing_pixel_count": len(corrected_missing),
        "corrected_mean_pixel_jaccard": _mean(jaccards),
        "corrected_median_pixel_jaccard": _median(jaccards),
        "corrected_mean_exact_recall_by_proxy": _mean(recalls),
        "corrected_mean_proxy_precision_against_exact": _mean(precisions),
        "corrected_zero_overlap_preserved_on_covered_pixels": covered_zero,
        "corrected_zero_overlap_claim_safe_within_exact_available_scope": bool(coverage_resolved and covered_zero and not nonzero),
        "repaired_nonzero_overlap_found": nonzero,
        "coverage_problem_resolved": coverage_resolved,
        "proxy_safe_for_intervention": False,
        "exact_render_contribution_available": False,
        "exact_contribution_magnitude_available": False,
        "drums_used_as_exact_evidence": False,
        "pr212cfix_ready_for_pr212c_rerun": bool(coverage_resolved and not nonzero),
        "pr212cfix_ready_for_pr214": False,
        "input_warnings": input_warnings,
        "recommended_next_step": recommended,
    }
    claim_rows = _claim_rows(summary)

    write_json(output_dir / "pr212cfix_corrected_summary.json", summary)
    write_csv_rows(output_dir / "pr212cfix_missing_pixel_list.csv", missing_list, MISSING_FIELDS)
    write_csv_rows(output_dir / "pr212cfix_missing_pixel_trace.csv", trace_rows, TRACE_FIELDS)
    write_csv_rows(output_dir / "pr212cfix_pixel_set_coverage.csv", coverage_rows, COVERAGE_FIELDS)
    write_csv_rows(output_dir / "pr212cfix_coordinate_format_audit.csv", coord_rows, COORD_FIELDS)
    write_csv_rows(output_dir / "pr212cfix_corrected_repaired_pixel_comparison_preview.csv", corrected_rows, CORRECTED_FIELDS)
    write_csv_rows(output_dir / "pr212cfix_claim_scope_recommendation.csv", claim_rows, CLAIM_FIELDS)
    _write_report(output_dir / "pr212cfix_missing_repaired_proxy_pixels_report.md", summary, trace_rows)
    _write_wording(output_dir / "pr212cfix_paper_wording.md")
    _write_next_step(output_dir / "pr212cfix_next_step_decision_memo.md", summary)
    inputs = [
        ("pr200_chair_dir", pr200_chair_dir, True),
        ("pr211_chair_dir", pr211_chair_dir, True),
        ("pr212b_chair_dir", pr212b_chair_dir, True),
        ("pr212c_chair_dir", pr212c_chair_dir, True),
    ]
    manifest = output_dir / "artifact_manifest.csv"
    write_csv_rows(manifest, _manifest_rows(output_dir, inputs), MANIFEST_FIELDS)
    write_csv_rows(manifest, _manifest_rows(output_dir, inputs), MANIFEST_FIELDS)
    return summary, 0
