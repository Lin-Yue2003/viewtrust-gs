"""Post-hoc exact view-group binding helpers for PR19.3.

This module binds PR17/PR18 view-group semantics onto PR19.2 exact Gaussian
logs. It is offline analysis only: it does not change training, rendering,
optimization, PR17 normalization, PR18 diagnosis, or PR19 scoring.
"""

from __future__ import annotations

import csv
import json
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from viewtrust.analysis.clean_prior_normalization import load_csv_rows, load_json, normalize_bool
from viewtrust.analysis.offline_signals import safe_float


PR193_OUTPUT_FILES = [
    "pr193_view_group_map.csv",
    "pr193_view_group_binding_summary.json",
    "gaussian_identity_table_grouped.csv",
    "gaussian_lifecycle_events_grouped.csv",
    "view_gaussian_event_attribution_grouped.csv",
    "gaussian_support_summary_grouped.csv",
    "pr193_exact_group_overlap_summary.csv",
    "pr193_train013_exact_control.csv",
    "pr193_direct_collateral_exact_overlap.csv",
    "pr193_pr19_exact_input_bundle_manifest.csv",
    "pr193_missing_inputs.csv",
    "pr193_report.md",
    "artifact_manifest.csv",
]

EXACT_REQUIRED_FILES = [
    "gaussian_identity_table.csv",
    "gaussian_lifecycle_events.csv",
    "view_gaussian_event_attribution.csv",
    "gaussian_support_summary.csv",
    "exact_gaussian_logging_summary.json",
    "exact_gaussian_logging_validation.json",
    "artifact_manifest.csv",
]

PR19_READY_EXACT_FILES = [
    "gaussian_identity_table.csv",
    "gaussian_lifecycle_events.csv",
    "view_gaussian_event_attribution.csv",
    "gaussian_support_summary.csv",
    "exact_gaussian_logging_summary.json",
    "exact_gaussian_logging_validation.json",
    "artifact_manifest.csv",
]

VIEW_GROUP_MAP_FIELDS = [
    "scene",
    "condition",
    "subset_name",
    "view_name",
    "view_group",
    "was_corrupted",
    "pr17_raw_rank",
    "pr17_normalized_rank",
    "pr17_raw_false_positive",
    "pr17_normalized_false_positive",
    "pr18_spillover_class",
    "pr18_spillover_confidence",
    "camera_neighbor_evidence",
    "index_neighbor_evidence",
    "gaussian_overlap_evidence",
    "clean_prior_pattern",
    "collateral_lift_pattern",
    "source_priority_reason",
    "included_in_exact_group_binding",
]

GROUP_STAT_FIELDS = [
    "direct_corrupted_unique_view_count",
    "collateral_unique_view_count",
    "clean_prior_unique_view_count",
    "other_clean_unique_view_count",
    "direct_corrupted_event_count",
    "collateral_event_count",
    "clean_prior_event_count",
    "other_clean_event_count",
    "corrupted_plus_collateral_unique_view_ratio",
    "corrupted_plus_collateral_event_ratio",
    "clean_prior_unique_view_ratio",
    "clean_prior_event_ratio",
    "dominant_view_group",
    "dominant_view_group_ratio",
    "has_direct_corrupted_support",
    "has_collateral_support",
    "has_clean_prior_support",
    "exact_group_binding_source",
    "exact_group_binding_warnings",
]

OVERLAP_FIELDS = [
    "scene",
    "condition",
    "subset_name",
    "gaussian_id",
    "has_direct_corrupted_support",
    "has_collateral_support",
    "has_clean_prior_support",
    "direct_corrupted_unique_view_count",
    "collateral_unique_view_count",
    "clean_prior_unique_view_count",
    "direct_corrupted_event_count",
    "collateral_event_count",
    "clean_prior_event_count",
    "corrupted_plus_collateral_unique_view_ratio",
    "corrupted_plus_collateral_event_ratio",
    "clean_prior_unique_view_ratio",
    "clean_prior_event_ratio",
    "dominant_view_group",
]

DIRECT_COLLATERAL_FIELDS = [
    "scene",
    "condition",
    "subset_name",
    "direct_corrupted_view_names",
    "collateral_view_names",
    "direct_supported_gaussian_count",
    "collateral_supported_gaussian_count",
    "exact_overlap_gaussian_count",
    "exact_overlap_jaccard",
    "exact_overlap_ratio_over_direct",
    "exact_overlap_ratio_over_collateral",
    "direct_collateral_exact_overlap_supported",
    "evidence_quality",
    "notes",
]

TRAIN013_FIELDS = [
    "scene",
    "condition",
    "subset_name",
    "view_name",
    "view_group",
    "was_corrupted",
    "exact_gaussian_support_count",
    "high_event_support_gaussian_count",
    "direct_collateral_overlap_gaussian_count",
    "clean_prior_event_count",
    "clean_prior_unique_view_count",
    "corrupted_plus_collateral_event_ratio_mean",
    "clean_prior_event_ratio_mean",
    "train013_exact_control_supported",
    "reason",
    "evidence_quality",
    "notes",
]

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


def _number(value: Any, default: float = 0.0) -> float:
    parsed = safe_float(value)
    return default if parsed is None else parsed


def _bool_text(value: bool) -> str:
    return str(bool(value)).lower()


def _split_names(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    text = str(value or "")
    if not text:
        return []
    text = text.replace(",", ";").replace("|", ";")
    return [item.strip() for item in text.split(";") if item.strip()]


def _source_view(row: dict[str, Any]) -> str:
    return str(row.get("view_name") or row.get("source_view_name") or row.get("birth_view_name") or "")


def _gaussian_id(row: dict[str, Any]) -> str:
    return str(row.get("gaussian_id", "") or "")


def _field_union(rows: list[dict[str, Any]], extras: list[str]) -> list[str]:
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    for field in extras:
        if field not in fields:
            fields.append(field)
    return fields


def _copy_value(row: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        value = row.get(key)
        if value not in ("", None):
            return value
    return ""


def _condition_match(row: dict[str, Any], scene: str, condition: str, subset_name: str) -> bool:
    return (
        str(row.get("scene", "")) == scene
        and str(row.get("condition", "")) == condition
        and str(row.get("subset_name", "")) == subset_name
    )


def _lookup_by_view(rows: list[dict[str, Any]], scene: str, condition: str, subset_name: str) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for row in rows:
        if _condition_match(row, scene, condition, subset_name) and row.get("view_name"):
            output[str(row["view_name"])] = row
    return output


def _transition_marks_demoted(row: dict[str, Any]) -> bool:
    text = ";".join(str(value).lower() for value in row.values())
    return "clean_prior_demoted" in text or "prior_demoted" in text


def load_pr17_rows(pr17_dir: Path) -> list[dict[str, str]]:
    return load_csv_rows(pr17_dir / "clean_prior_normalized_rows.csv")


def load_pr18_classification(pr18_dir: Path) -> list[dict[str, str]]:
    return load_csv_rows(pr18_dir / "pr18_spillover_classification.csv")


def load_pr18_transition(pr18_dir: Path) -> list[dict[str, str]]:
    return load_csv_rows(pr18_dir / "pr18_view_identity_transition.csv")


def build_view_group_map(
    *,
    pr17_rows: list[dict[str, Any]],
    pr18_rows: list[dict[str, Any]],
    pr18_transition_rows: list[dict[str, Any]],
    scene: str,
    condition: str,
    subset_name: str,
    exact_view_names: list[str],
) -> tuple[list[dict[str, Any]], list[str]]:
    warnings: list[str] = []
    pr17_by_view = _lookup_by_view(pr17_rows, scene, condition, subset_name)
    pr18_by_view = _lookup_by_view(pr18_rows, scene, condition, subset_name)
    transition_by_view = _lookup_by_view(pr18_transition_rows, scene, condition, subset_name)
    view_names = sorted({*pr17_by_view, *pr18_by_view, *transition_by_view, *exact_view_names})
    output: list[dict[str, Any]] = []
    for view_name in view_names:
        pr17 = pr17_by_view.get(view_name, {})
        pr18 = pr18_by_view.get(view_name, {})
        transition = transition_by_view.get(view_name, {})
        spillover_class = str(pr18.get("spillover_class", ""))
        was_corrupted = _truth(pr17.get("was_corrupted")) or _truth(pr18.get("was_corrupted"))
        raw_fp = _truth(_copy_value(pr17, ["raw_false_positive", "raw_top_k_false_positive"]))
        normalized_fp = _truth(_copy_value(pr17, ["normalized_false_positive", "normalized_top_k_false_positive"]))
        clean_prior_pattern = (
            spillover_class in {"clean_prior_false_positive", "prior_demoted"}
            or _transition_marks_demoted(transition)
            or (view_name == "train_013" and raw_fp and not normalized_fp and not was_corrupted)
        )
        collateral_lift_pattern = spillover_class == "co_visible_collateral" or _truth(pr18.get("collateral_lift_pattern"))
        if was_corrupted:
            group = "direct_corrupted"
            reason = "pr17_was_corrupted_priority"
        elif spillover_class == "co_visible_collateral":
            group = "co_visible_collateral"
            reason = "pr18_co_visible_collateral"
        elif clean_prior_pattern:
            group = "clean_prior_demoted"
            reason = "clean_prior_demoted_pattern"
        else:
            group = "other_clean"
            reason = "default_non_corrupted"
        if view_name in exact_view_names and view_name not in pr17_by_view:
            warnings.append(f"exact view {view_name} missing PR17 row; defaulted with PR18/other_clean context")
        output.append(
            {
                "scene": scene,
                "condition": condition,
                "subset_name": subset_name,
                "view_name": view_name,
                "view_group": group,
                "was_corrupted": _bool_text(was_corrupted),
                "pr17_raw_rank": _copy_value(pr17, ["raw_rank"]),
                "pr17_normalized_rank": _copy_value(pr17, ["normalized_rank"]),
                "pr17_raw_false_positive": _bool_text(raw_fp),
                "pr17_normalized_false_positive": _bool_text(normalized_fp),
                "pr18_spillover_class": spillover_class,
                "pr18_spillover_confidence": pr18.get("spillover_confidence", ""),
                "camera_neighbor_evidence": pr18.get("camera_neighbor_evidence", ""),
                "index_neighbor_evidence": pr18.get("index_neighbor_evidence", ""),
                "gaussian_overlap_evidence": pr18.get("gaussian_overlap_evidence", ""),
                "clean_prior_pattern": _bool_text(clean_prior_pattern),
                "collateral_lift_pattern": _bool_text(collateral_lift_pattern),
                "source_priority_reason": reason,
                "included_in_exact_group_binding": _bool_text(view_name in exact_view_names or bool(pr17)),
            }
        )
    return output, warnings


def _view_group_lookup(view_group_rows: list[dict[str, Any]]) -> dict[str, str]:
    return {str(row.get("view_name", "")): str(row.get("view_group", "other_clean")) for row in view_group_rows}


def _exact_view_names(event_rows: list[dict[str, Any]], attribution_rows: list[dict[str, Any]]) -> list[str]:
    views = {_source_view(row) for row in event_rows + attribution_rows if _source_view(row)}
    return sorted(views)


def add_view_groups(rows: list[dict[str, Any]], lookup: dict[str, str]) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        view_name = _source_view(row)
        output.append({**row, "view_group": lookup.get(view_name, "other_clean")})
    return output


def _empty_group_stats() -> dict[str, Any]:
    return {
        "_views": defaultdict(set),
        "_events": Counter(),
        "_warnings": [],
    }


def compute_gaussian_group_stats(
    *,
    attribution_rows: list[dict[str, Any]],
    support_rows: list[dict[str, Any]],
    view_group_lookup: dict[str, str],
) -> dict[str, dict[str, Any]]:
    raw: dict[str, dict[str, Any]] = defaultdict(_empty_group_stats)
    for row in attribution_rows:
        gid = _gaussian_id(row)
        view = _source_view(row)
        if not gid or not view:
            continue
        group = view_group_lookup.get(view, "other_clean")
        weight = _number(row.get("contribution_value"), 1.0)
        raw[gid]["_views"][group].add(view)
        raw[gid]["_events"][group] += weight if weight > 0 else 1.0
    for row in support_rows:
        gid = _gaussian_id(row)
        if not gid or gid in raw:
            continue
        for view in _split_names(row.get("support_view_names")):
            group = view_group_lookup.get(view, "other_clean")
            raw[gid]["_views"][group].add(view)
            raw[gid]["_events"][group] += 1.0
        if gid in raw:
            raw[gid]["_warnings"].append("group stats fell back to support_view_names")

    finalized: dict[str, dict[str, Any]] = {}
    for gid, stats in raw.items():
        views_by_group = stats["_views"]
        events_by_group = stats["_events"]
        unique_total = sum(len(views_by_group[group]) for group in views_by_group)
        event_total = sum(events_by_group.values())
        unique_total = unique_total or 0
        event_total = event_total or 0.0
        group_event_items = {
            "direct_corrupted": events_by_group.get("direct_corrupted", 0.0),
            "co_visible_collateral": events_by_group.get("co_visible_collateral", 0.0),
            "clean_prior_demoted": events_by_group.get("clean_prior_demoted", 0.0),
            "other_clean": events_by_group.get("other_clean", 0.0),
        }
        dominant = max(group_event_items.items(), key=lambda item: (item[1], item[0]))[0] if event_total else ""
        direct_unique = len(views_by_group.get("direct_corrupted", set()))
        collateral_unique = len(views_by_group.get("co_visible_collateral", set()))
        clean_prior_unique = len(views_by_group.get("clean_prior_demoted", set()))
        other_unique = len(views_by_group.get("other_clean", set()))
        direct_events = group_event_items["direct_corrupted"]
        collateral_events = group_event_items["co_visible_collateral"]
        clean_prior_events = group_event_items["clean_prior_demoted"]
        other_events = group_event_items["other_clean"]
        finalized[gid] = {
            "direct_corrupted_unique_view_count": direct_unique,
            "collateral_unique_view_count": collateral_unique,
            "clean_prior_unique_view_count": clean_prior_unique,
            "other_clean_unique_view_count": other_unique,
            "direct_corrupted_event_count": direct_events,
            "collateral_event_count": collateral_events,
            "clean_prior_event_count": clean_prior_events,
            "other_clean_event_count": other_events,
            "corrupted_plus_collateral_unique_view_ratio": (direct_unique + collateral_unique) / unique_total if unique_total else 0.0,
            "corrupted_plus_collateral_event_ratio": (direct_events + collateral_events) / event_total if event_total else 0.0,
            "clean_prior_unique_view_ratio": clean_prior_unique / unique_total if unique_total else 0.0,
            "clean_prior_event_ratio": clean_prior_events / event_total if event_total else 0.0,
            "dominant_view_group": dominant,
            "dominant_view_group_ratio": group_event_items.get(dominant, 0.0) / event_total if event_total and dominant else 0.0,
            "has_direct_corrupted_support": _bool_text(direct_unique > 0 or direct_events > 0),
            "has_collateral_support": _bool_text(collateral_unique > 0 or collateral_events > 0),
            "has_clean_prior_support": _bool_text(clean_prior_unique > 0 or clean_prior_events > 0),
            "exact_group_binding_source": "pr193_pr17_pr18_view_group_binding",
            "exact_group_binding_warnings": ";".join(stats["_warnings"]),
        }
    return finalized


def _merge_group_stats(rows: list[dict[str, Any]], stats_by_gid: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        row_stats = stats_by_gid.get(_gaussian_id(row), {})
        output.append({**row, **{field: row_stats.get(field, 0 if field.endswith("_count") else "") for field in GROUP_STAT_FIELDS}})
    return output


def _overlap_rows(scene: str, condition: str, subset_name: str, stats_by_gid: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for gid, stats in sorted(stats_by_gid.items(), key=lambda item: item[0]):
        row = {
            "scene": scene,
            "condition": condition,
            "subset_name": subset_name,
            "gaussian_id": gid,
            **{field: stats.get(field, "") for field in GROUP_STAT_FIELDS},
        }
        rows.append(row)
    return rows


def _direct_collateral_overlap(
    *,
    scene: str,
    condition: str,
    subset_name: str,
    view_group_rows: list[dict[str, Any]],
    stats_by_gid: dict[str, dict[str, Any]],
    evidence_quality: str,
) -> dict[str, Any]:
    direct_views = sorted(row["view_name"] for row in view_group_rows if row.get("view_group") == "direct_corrupted")
    collateral_views = sorted(row["view_name"] for row in view_group_rows if row.get("view_group") == "co_visible_collateral")
    direct_ids = {gid for gid, stats in stats_by_gid.items() if _truth(stats.get("has_direct_corrupted_support"))}
    collateral_ids = {gid for gid, stats in stats_by_gid.items() if _truth(stats.get("has_collateral_support"))}
    overlap = direct_ids & collateral_ids
    union = direct_ids | collateral_ids
    supported = len(collateral_ids) > 0 and len(overlap) > 0 and (len(overlap) / len(union) if union else 0.0) > 0.0
    return {
        "scene": scene,
        "condition": condition,
        "subset_name": subset_name,
        "direct_corrupted_view_names": ";".join(direct_views),
        "collateral_view_names": ";".join(collateral_views),
        "direct_supported_gaussian_count": len(direct_ids),
        "collateral_supported_gaussian_count": len(collateral_ids),
        "exact_overlap_gaussian_count": len(overlap),
        "exact_overlap_jaccard": len(overlap) / len(union) if union else 0.0,
        "exact_overlap_ratio_over_direct": len(overlap) / len(direct_ids) if direct_ids else 0.0,
        "exact_overlap_ratio_over_collateral": len(overlap) / len(collateral_ids) if collateral_ids else 0.0,
        "direct_collateral_exact_overlap_supported": _bool_text(supported),
        "evidence_quality": evidence_quality,
        "notes": "" if supported else "no exact direct/collateral Gaussian overlap detected",
    }


def _train013_control(
    *,
    scene: str,
    condition: str,
    subset_name: str,
    view_group_rows: list[dict[str, Any]],
    attribution_rows: list[dict[str, Any]],
    stats_by_gid: dict[str, dict[str, Any]],
    top_k: int,
    evidence_quality: str,
) -> list[dict[str, Any]]:
    train_row = next((row for row in view_group_rows if row.get("view_name") == "train_013"), None)
    if not train_row:
        return []
    train_ids: Counter[str] = Counter()
    for row in attribution_rows:
        if _source_view(row) == "train_013" and _gaussian_id(row):
            train_ids[_gaussian_id(row)] += _number(row.get("contribution_value"), 1.0)
    direct_collateral_ids = {
        gid
        for gid, stats in stats_by_gid.items()
        if _truth(stats.get("has_direct_corrupted_support")) or _truth(stats.get("has_collateral_support"))
    }
    overlap = set(train_ids) & direct_collateral_ids
    sorted_events = sorted(train_ids.values(), reverse=True)
    threshold = sorted_events[min(top_k - 1, len(sorted_events) - 1)] if sorted_events else 0
    high_event = [gid for gid, count in train_ids.items() if count >= threshold and threshold > 0]
    ratios = [stats_by_gid.get(gid, {}) for gid in train_ids]
    corrupted_mean = _mean([row.get("corrupted_plus_collateral_event_ratio") for row in ratios])
    clean_prior_mean = _mean([row.get("clean_prior_event_ratio") for row in ratios])
    low_overlap = len(overlap) == 0
    does_not_dominate = clean_prior_mean >= corrupted_mean
    supported = (
        train_row.get("view_group") == "clean_prior_demoted"
        and not _truth(train_row.get("was_corrupted"))
        and bool(train_ids)
        and low_overlap
        and does_not_dominate
    )
    reason = "clean_prior_demoted_low_exact_overlap" if supported else "train013 overlaps direct/collateral IDs or is unavailable"
    return [
        {
            "scene": scene,
            "condition": condition,
            "subset_name": subset_name,
            "view_name": "train_013",
            "view_group": train_row.get("view_group", ""),
            "was_corrupted": train_row.get("was_corrupted", ""),
            "exact_gaussian_support_count": len(train_ids),
            "high_event_support_gaussian_count": len(high_event),
            "direct_collateral_overlap_gaussian_count": len(overlap),
            "clean_prior_event_count": sum(stats_by_gid.get(gid, {}).get("clean_prior_event_count", 0.0) for gid in train_ids),
            "clean_prior_unique_view_count": sum(stats_by_gid.get(gid, {}).get("clean_prior_unique_view_count", 0) for gid in train_ids),
            "corrupted_plus_collateral_event_ratio_mean": corrupted_mean,
            "clean_prior_event_ratio_mean": clean_prior_mean,
            "train013_exact_control_supported": _bool_text(supported),
            "reason": reason,
            "evidence_quality": evidence_quality,
            "notes": "",
        }
    ]


def _mean(values: list[Any]) -> float:
    numbers = [safe_float(value) for value in values]
    finite = [value for value in numbers if value is not None]
    return sum(finite) / len(finite) if finite else 0.0


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


def write_artifact_manifest(output_dir: Path, exact_log_dir: Path, pr17_dir: Path, pr18_dir: Path) -> None:
    items: list[tuple[str, Path, bool, str]] = [
        ("exact_log_dir", exact_log_dir, True, "input"),
        ("pr17_dir", pr17_dir, True, "input"),
        ("pr18_dir", pr18_dir, True, "input"),
    ]
    items.extend((name, output_dir / name, True, "output_pr193") for name in PR193_OUTPUT_FILES)
    manifest = output_dir / "artifact_manifest.csv"
    write_csv_rows(manifest, _artifact_rows(items), MANIFEST_FIELDS)
    write_csv_rows(manifest, _artifact_rows(items), MANIFEST_FIELDS)


def write_bundle_manifest(output_dir: Path, bundle_root: Path) -> None:
    items: list[tuple[str, Path, bool, str]] = []
    for name in [
        "pr193_view_group_map.csv",
        "pr193_view_group_binding_summary.json",
        "pr193_pr19_exact_input_bundle_manifest.csv",
    ]:
        items.append((name, bundle_root / name, True, "pr19_exact_input_bundle"))
    for name in PR19_READY_EXACT_FILES:
        items.append((f"exact_gaussian_logging/{name}", bundle_root / "exact_gaussian_logging" / name, True, "pr19_exact_input_bundle"))
    rows = _artifact_rows(items)
    write_csv_rows(output_dir / "pr193_pr19_exact_input_bundle_manifest.csv", rows, MANIFEST_FIELDS)
    write_csv_rows(bundle_root / "pr193_pr19_exact_input_bundle_manifest.csv", rows, MANIFEST_FIELDS)


def _copy_pr19_ready_bundle(
    *,
    output_dir: Path,
    exact_log_dir: Path,
    grouped_identity: list[dict[str, Any]],
    grouped_events: list[dict[str, Any]],
    grouped_attribution: list[dict[str, Any]],
    grouped_support: list[dict[str, Any]],
    identity_fields: list[str],
    event_fields: list[str],
    attribution_fields: list[str],
    support_fields: list[str],
    summary: dict[str, Any],
) -> None:
    bundle_root = output_dir / "pr19_exact_input_bundle"
    exact_bundle = bundle_root / "exact_gaussian_logging"
    exact_bundle.mkdir(parents=True, exist_ok=True)
    write_csv_rows(exact_bundle / "gaussian_identity_table.csv", grouped_identity, identity_fields)
    write_csv_rows(exact_bundle / "gaussian_lifecycle_events.csv", grouped_events, event_fields)
    write_csv_rows(exact_bundle / "view_gaussian_event_attribution.csv", grouped_attribution, attribution_fields)
    write_csv_rows(exact_bundle / "gaussian_support_summary.csv", grouped_support, support_fields)
    source_summary = load_json(exact_log_dir / "exact_gaussian_logging_summary.json")
    source_summary.update(
        {
            "view_group_binding_source": "pr193_pr17_pr18_view_group_binding",
            "pr193_view_group_binding_summary": str(output_dir / "pr193_view_group_binding_summary.json"),
        }
    )
    write_json(exact_bundle / "exact_gaussian_logging_summary.json", source_summary)
    validation = load_json(exact_log_dir / "exact_gaussian_logging_validation.json")
    validation.update({"view_group_binding_applied": True})
    write_json(exact_bundle / "exact_gaussian_logging_validation.json", validation)
    shutil.copy2(output_dir / "pr193_view_group_map.csv", bundle_root / "pr193_view_group_map.csv")
    write_json(bundle_root / "pr193_view_group_binding_summary.json", summary)
    exact_items = [(name, exact_bundle / name, True, "pr19_exact_input_bundle") for name in PR19_READY_EXACT_FILES]
    write_csv_rows(exact_bundle / "artifact_manifest.csv", _artifact_rows(exact_items), MANIFEST_FIELDS)
    write_csv_rows(exact_bundle / "artifact_manifest.csv", _artifact_rows(exact_items), MANIFEST_FIELDS)
    write_bundle_manifest(output_dir, bundle_root)


def _write_empty_bundle_manifest(output_dir: Path) -> None:
    write_csv_rows(output_dir / "pr193_pr19_exact_input_bundle_manifest.csv", [], MANIFEST_FIELDS)


def _write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# PR19.3 Exact View-Group Binding",
        "",
        "PR19.3 is offline observation only. It binds PR17/PR18 view-group semantics onto PR19.2 exact Gaussian logs.",
        "It does not change training, rendering, PR13 scoring, PR17 normalization, PR18 diagnosis, or PR19 risk scoring.",
        "Corruption labels are used only for post-hoc grouping and evaluation, not for scoring.",
        "",
        "## Summary",
        f"- Scene: `{summary.get('scene')}`",
        f"- Condition: `{summary.get('condition')}`",
        f"- Subset: `{summary.get('subset_name')}`",
        f"- Direct corrupted views: `{summary.get('direct_corrupted_view_count')}`",
        f"- Collateral views: `{summary.get('co_visible_collateral_view_count')}`",
        f"- Clean-prior-demoted views: `{summary.get('clean_prior_demoted_view_count')}`",
        f"- Gaussians with collateral support: `{summary.get('total_gaussians_with_collateral_support')}`",
        f"- Gaussians with clean-prior support: `{summary.get('total_gaussians_with_clean_prior_support')}`",
        f"- Direct/collateral exact overlap supported: `{summary.get('direct_collateral_exact_overlap_supported')}`",
        f"- Train013 exact control supported: `{summary.get('train013_exact_control_supported')}`",
        f"- PR19-ready bundle written: `{summary.get('pr19_ready_bundle_written')}`",
        "",
        "The PR19-ready bundle, when requested, is written under `pr19_exact_input_bundle/exact_gaussian_logging/`.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def bind_exact_view_groups(
    *,
    exact_log_dir: Path,
    pr17_dir: Path,
    pr18_dir: Path,
    scene: str,
    condition: str,
    subset_name: str,
    output_dir: Path,
    top_k: int = 20,
    write_markdown: bool = False,
    copy_pr19_ready_bundle: bool = False,
    strict: bool = False,
    allow_missing: bool = False,
) -> tuple[dict[str, Any], int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    missing_rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    for name in EXACT_REQUIRED_FILES:
        path = exact_log_dir / name
        if not path.exists():
            missing_rows.append({"input_name": name, "path": str(path), "exists": "false", "required": "true", "details": "missing exact log input"})
    for name in ["clean_prior_normalized_rows.csv"]:
        path = pr17_dir / name
        if not path.exists():
            missing_rows.append({"input_name": name, "path": str(path), "exists": "false", "required": "true", "details": "missing PR17 input"})
    for name in ["pr18_spillover_classification.csv"]:
        path = pr18_dir / name
        if not path.exists():
            missing_rows.append({"input_name": name, "path": str(path), "exists": "false", "required": "true", "details": "missing PR18 input"})

    identity_rows = load_csv_rows(exact_log_dir / "gaussian_identity_table.csv")
    event_rows = load_csv_rows(exact_log_dir / "gaussian_lifecycle_events.csv")
    attribution_rows = load_csv_rows(exact_log_dir / "view_gaussian_event_attribution.csv")
    support_rows = load_csv_rows(exact_log_dir / "gaussian_support_summary.csv")
    exact_summary = load_json(exact_log_dir / "exact_gaussian_logging_summary.json")
    exact_validation = load_json(exact_log_dir / "exact_gaussian_logging_validation.json")
    pr17_rows = load_pr17_rows(pr17_dir)
    pr18_rows = load_pr18_classification(pr18_dir)
    pr18_transition_rows = load_pr18_transition(pr18_dir)

    condition_pr17_rows = [row for row in pr17_rows if _condition_match(row, scene, condition, subset_name)]
    if not condition_pr17_rows:
        missing_rows.append(
            {
                "input_name": "clean_prior_normalized_rows.csv",
                "path": str(pr17_dir / "clean_prior_normalized_rows.csv"),
                "exists": _bool_text((pr17_dir / "clean_prior_normalized_rows.csv").exists()),
                "required": "true",
                "details": "no PR17 rows for requested scene/condition/subset",
            }
        )
    exact_views = _exact_view_names(event_rows, attribution_rows)
    view_group_rows, group_warnings = build_view_group_map(
        pr17_rows=pr17_rows,
        pr18_rows=pr18_rows,
        pr18_transition_rows=pr18_transition_rows,
        scene=scene,
        condition=condition,
        subset_name=subset_name,
        exact_view_names=exact_views,
    )
    warnings.extend(group_warnings)
    lookup = _view_group_lookup(view_group_rows)
    grouped_events = add_view_groups(event_rows, lookup)
    grouped_attribution = add_view_groups(attribution_rows, lookup)
    stats_by_gid = compute_gaussian_group_stats(
        attribution_rows=grouped_attribution,
        support_rows=support_rows,
        view_group_lookup=lookup,
    )
    grouped_identity = _merge_group_stats(identity_rows, stats_by_gid)
    grouped_support = _merge_group_stats(support_rows, stats_by_gid)
    overlap_rows = _overlap_rows(scene, condition, subset_name, stats_by_gid)
    direct_collateral = _direct_collateral_overlap(
        scene=scene,
        condition=condition,
        subset_name=subset_name,
        view_group_rows=view_group_rows,
        stats_by_gid=stats_by_gid,
        evidence_quality=str(exact_summary.get("evidence_quality") or "exact"),
    )
    train013_rows = _train013_control(
        scene=scene,
        condition=condition,
        subset_name=subset_name,
        view_group_rows=view_group_rows,
        attribution_rows=grouped_attribution,
        stats_by_gid=stats_by_gid,
        top_k=top_k,
        evidence_quality=str(exact_summary.get("evidence_quality") or "exact"),
    )

    group_counts = Counter(row.get("view_group", "other_clean") for row in view_group_rows)
    train013_row = train013_rows[0] if train013_rows else {}
    pr19_ready_written = bool(copy_pr19_ready_bundle)
    summary = {
        "schema_name": "viewtrust.pr193.exact_view_group_binding.summary",
        "schema_version": 1,
        "created_at_utc": _utc_now(),
        "scene": scene,
        "condition": condition,
        "subset_name": subset_name,
        "exact_log_dir": str(exact_log_dir),
        "pr17_dir": str(pr17_dir),
        "pr18_dir": str(pr18_dir),
        "output_dir": str(output_dir),
        "observation_only": True,
        "training_intervention": False,
        "defense_enabled": False,
        "uses_corruption_labels_for_scoring": False,
        "uses_corruption_labels_for_evaluation": True,
        "exact_gaussian_logging_enabled": True,
        "exact_gaussian_evidence_available": bool(identity_rows and attribution_rows),
        "input_exact_evidence_quality": exact_summary.get("evidence_quality", ""),
        "output_exact_evidence_quality": exact_summary.get("evidence_quality", ""),
        "integration_source": exact_summary.get("integration_source", ""),
        "parent_mapping_source": exact_summary.get("parent_mapping_source", ""),
        "total_views": len(view_group_rows),
        "direct_corrupted_view_count": group_counts["direct_corrupted"],
        "co_visible_collateral_view_count": group_counts["co_visible_collateral"],
        "clean_prior_demoted_view_count": group_counts["clean_prior_demoted"],
        "other_clean_view_count": group_counts["other_clean"],
        "total_gaussians": len(identity_rows) or len(stats_by_gid),
        "total_lifecycle_event_rows": len(event_rows),
        "total_attribution_rows": len(attribution_rows),
        "total_gaussians_with_direct_corrupted_support": sum(1 for row in stats_by_gid.values() if _truth(row.get("has_direct_corrupted_support"))),
        "total_gaussians_with_collateral_support": sum(1 for row in stats_by_gid.values() if _truth(row.get("has_collateral_support"))),
        "total_gaussians_with_clean_prior_support": sum(1 for row in stats_by_gid.values() if _truth(row.get("has_clean_prior_support"))),
        "total_gaussians_with_corrupted_plus_collateral_support": sum(
            1
            for row in stats_by_gid.values()
            if _truth(row.get("has_direct_corrupted_support")) or _truth(row.get("has_collateral_support"))
        ),
        "train013_present": bool(train013_rows),
        "train013_view_group": train013_row.get("view_group", ""),
        "train013_exact_control_supported": _truth(train013_row.get("train013_exact_control_supported")),
        "direct_collateral_exact_overlap_supported": _truth(direct_collateral.get("direct_collateral_exact_overlap_supported")),
        "pr19_ready_bundle_written": pr19_ready_written,
        "exact_validation_identity_consistency_passed": exact_validation.get("identity_consistency_passed", ""),
        "warnings": warnings,
    }

    write_csv_rows(output_dir / "pr193_view_group_map.csv", view_group_rows, VIEW_GROUP_MAP_FIELDS)
    write_json(output_dir / "pr193_view_group_binding_summary.json", summary)
    identity_fields = _field_union(grouped_identity, GROUP_STAT_FIELDS)
    event_fields = _field_union(grouped_events, ["view_group"])
    attribution_fields = _field_union(grouped_attribution, ["view_group"])
    support_fields = _field_union(grouped_support, GROUP_STAT_FIELDS)
    write_csv_rows(output_dir / "gaussian_identity_table_grouped.csv", grouped_identity, identity_fields)
    write_csv_rows(output_dir / "gaussian_lifecycle_events_grouped.csv", grouped_events, event_fields)
    write_csv_rows(output_dir / "view_gaussian_event_attribution_grouped.csv", grouped_attribution, attribution_fields)
    write_csv_rows(output_dir / "gaussian_support_summary_grouped.csv", grouped_support, support_fields)
    write_csv_rows(output_dir / "pr193_exact_group_overlap_summary.csv", overlap_rows, OVERLAP_FIELDS)
    write_csv_rows(output_dir / "pr193_train013_exact_control.csv", train013_rows, TRAIN013_FIELDS)
    write_csv_rows(output_dir / "pr193_direct_collateral_exact_overlap.csv", [direct_collateral], DIRECT_COLLATERAL_FIELDS)
    write_csv_rows(output_dir / "pr193_missing_inputs.csv", missing_rows, MISSING_FIELDS)
    if write_markdown:
        _write_report(output_dir / "pr193_report.md", summary)
    else:
        _write_report(output_dir / "pr193_report.md", summary)
    _write_empty_bundle_manifest(output_dir)
    if copy_pr19_ready_bundle:
        _copy_pr19_ready_bundle(
            output_dir=output_dir,
            exact_log_dir=exact_log_dir,
            grouped_identity=grouped_identity,
            grouped_events=grouped_events,
            grouped_attribution=grouped_attribution,
            grouped_support=grouped_support,
            identity_fields=identity_fields,
            event_fields=event_fields,
            attribution_fields=attribution_fields,
            support_fields=support_fields,
            summary=summary,
        )
    write_artifact_manifest(output_dir, exact_log_dir, pr17_dir, pr18_dir)

    missing_required = [name for name in PR193_OUTPUT_FILES if not (output_dir / name).is_file()]
    if missing_required:
        raise RuntimeError(f"missing PR19.3 outputs: {missing_required}")
    if strict and missing_rows and not allow_missing:
        return summary, 1
    if missing_rows and not allow_missing:
        return summary, 1
    return summary, 0
