"""Offline attribution semantics audit helpers for PR19.5."""

from __future__ import annotations

import csv
import json
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from viewtrust.analysis.clean_prior_normalization import load_csv_rows, load_json, normalize_bool
from viewtrust.analysis.offline_signals import safe_float


PR195_OUTPUT_FILES = [
    "pr195_attribution_semantics_summary.json",
    "pr195_support_mode_failure_analysis.csv",
    "pr195_event_type_group_distribution.csv",
    "pr195_view_group_event_distribution.csv",
    "pr195_high_event_semantics_audit.csv",
    "pr195_suspicious_alive_degeneracy_audit.csv",
    "pr195_birth_prune_semantics_audit.csv",
    "pr195_train013_semantics_audit.csv",
    "pr195_required_attribution_field_gap.csv",
    "pr195_pr20_readiness_assessment.csv",
    "pr195_next_step_recommendation.md",
    "pr195_missing_inputs.csv",
    "pr195_report.md",
    "artifact_manifest.csv",
]

PR193_REQUIRED_FILES = [
    "pr193_view_group_map.csv",
    "pr193_view_group_binding_summary.json",
    "gaussian_lifecycle_events_grouped.csv",
    "view_gaussian_event_attribution_grouped.csv",
    "gaussian_support_summary_grouped.csv",
    "pr193_direct_collateral_exact_overlap.csv",
    "pr193_train013_exact_control.csv",
]

PR194_REQUIRED_FILES = [
    "pr194_exact_support_filter_summary.json",
    "pr194_support_mode_comparison.csv",
    "pr194_direct_collateral_overlap_by_mode.csv",
    "pr194_train013_control_by_mode.csv",
    "pr194_filtered_gaussian_support_by_mode.csv",
    "pr194_view_group_event_concentration.csv",
    "pr194_nontrivial_overlap_candidates.csv",
]

MISSING_FIELDS = ["input_name", "path", "exists", "required", "details"]
MANIFEST_FIELDS = ["relative_path", "path", "exists", "file_type", "size_bytes", "required", "artifact_group"]

FAILURE_FIELDS = [
    "support_mode",
    "filtered_gaussian_count",
    "direct_supported_gaussian_count",
    "collateral_supported_gaussian_count",
    "overlap_count",
    "jaccard",
    "train013_supported_gaussian_count",
    "train013_control_supported",
    "failure_type",
    "interpretation",
    "usable_for_pr19_exact_mode",
    "usable_for_intervention",
]

EVENT_TYPE_GROUP_FIELDS = [
    "event_type",
    "view_group",
    "row_count",
    "unique_gaussian_count",
    "alive_final_count",
    "mean_contribution_value",
    "max_contribution_value",
]

VIEW_GROUP_EVENT_FIELDS = [
    "view_group",
    "row_count",
    "unique_gaussian_count",
    "unique_view_count",
    "event_type_distribution",
    "mean_events_per_gaussian",
    "mean_events_per_view",
    "notes",
]

HIGH_EVENT_FIELDS = [
    "support_mode",
    "threshold",
    "direct_selected_count",
    "collateral_selected_count",
    "clean_prior_selected_count",
    "other_clean_selected_count",
    "top_event_view_groups",
    "top_event_view_names",
    "interpretation",
]

SUSPICIOUS_ALIVE_FIELDS = [
    "support_mode",
    "broad_overlap_count",
    "suspicious_alive_overlap_count",
    "overlap_equivalence_ratio",
    "broad_jaccard",
    "suspicious_alive_jaccard",
    "degeneracy_confirmed",
    "interpretation",
]

BIRTH_PRUNE_FIELDS = [
    "support_kind",
    "selected_gaussian_count",
    "direct_supported_gaussian_count",
    "collateral_supported_gaussian_count",
    "clean_prior_supported_gaussian_count",
    "other_clean_supported_gaussian_count",
    "interpretation",
]

TRAIN013_FIELDS = [
    "train013_view_group",
    "support_mode",
    "train013_supported_gaussian_count",
    "train013_direct_collateral_overlap_count",
    "train013_control_supported",
    "meaningful_or_diagnostic",
    "interpretation",
]

FIELD_GAP_FIELDS = ["field_name", "currently_available", "source_file", "required_for", "reason", "priority"]
READINESS_FIELDS = ["criterion", "passed", "evidence", "blocker", "recommended_action"]

REQUIRED_ATTRIBUTION_FIELDS = [
    ("gaussian_id", "stable Gaussian identity", "match view events to Gaussian IDs", "high"),
    ("view_name", "view attribution", "bind evidence to source views", "high"),
    ("view_group", "PR17/PR18 grouping", "compare direct/collateral/control views", "high"),
    ("event_type", "lifecycle context", "separate birth/prune/update/visibility semantics", "high"),
    ("contribution_value", "event weighting", "weight view-Gaussian support", "medium"),
    ("residual_value", "residual attribution", "residual-weighted suspicious support", "high"),
    ("pixel_x", "render contribution", "sparse pixel attribution", "high"),
    ("pixel_y", "render contribution", "sparse pixel attribution", "high"),
    ("alpha_contribution", "render contribution", "splat alpha contribution attribution", "high"),
    ("splat_weight", "render contribution", "per-Gaussian splat contribution attribution", "high"),
    ("rendered_rgb", "residual attribution", "render residual calculation", "medium"),
    ("gt_rgb", "residual attribution", "render residual calculation", "medium"),
    ("residual_l1", "residual attribution", "per-pixel residual weighting", "high"),
    ("residual_ssim", "residual attribution", "per-view structural residual weighting", "medium"),
    ("gradient_norm", "gradient attribution", "per-view update influence", "medium"),
    ("delta_position", "gradient/update attribution", "parameter update attribution", "medium"),
    ("delta_opacity", "gradient/update attribution", "parameter update attribution", "medium"),
    ("delta_scale", "gradient/update attribution", "parameter update attribution", "medium"),
    ("delta_sh", "gradient/update attribution", "parameter update attribution", "medium"),
    ("target_region_flag", "artifact localization", "target-region support filtering", "high"),
    ("artifact_region_flag", "artifact localization", "artifact-region support filtering", "high"),
]


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


def _mean(values: list[Any]) -> float | str:
    numbers = [safe_float(value) for value in values]
    finite = [value for value in numbers if value is not None]
    return statistics.fmean(finite) if finite else ""


def _max(values: list[Any]) -> float | str:
    numbers = [safe_float(value) for value in values]
    finite = [value for value in numbers if value is not None]
    return max(finite) if finite else ""


def _columns(rows: list[dict[str, Any]]) -> set[str]:
    return {field for row in rows for field in row}


def _mode_rows(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {str(row.get("support_mode", "")): row for row in rows if row.get("support_mode")}


def _classify_mode(row: dict[str, str]) -> tuple[str, str, bool, bool]:
    mode = str(row.get("support_mode", ""))
    direct = int(_number(row.get("direct_supported_gaussian_count")))
    collateral = int(_number(row.get("collateral_supported_gaussian_count")))
    other = int(_number(row.get("other_clean_supported_gaussian_count")))
    overlap = int(_number(row.get("direct_collateral_overlap_count") or row.get("exact_overlap_gaussian_count")))
    jaccard = _number(row.get("direct_collateral_jaccard") or row.get("exact_overlap_jaccard"))
    train_control = _truth(row.get("train013_control_supported"))
    nontrivial = _truth(row.get("nontrivial_overlap_supported"))
    degenerate = _truth(row.get("broad_overlap_degeneracy_flag"))
    usable = nontrivial and not degenerate and direct > 0 and collateral > 0 and overlap > 0 and jaccard < 0.95
    if mode == "broad" and jaccard >= 0.95:
        return "broad_degenerate", "Broad support reproduces final-alive visibility/update overlap and is not causal evidence.", False, False
    if mode == "suspicious_alive" and jaccard >= 0.95:
        return "suspicious_alive_degenerate", "Suspicious-alive support remains equivalent to broad overlap.", False, False
    if direct <= 0 and collateral <= 0 and other > 0 and mode == "dominant_source":
        return "other_clean_dominant", "Dominant-source evidence is carried by other_clean, not direct/collateral views.", False, False
    if direct > 0 and collateral <= 0:
        return "no_collateral_support", "This mode selects direct support but no collateral support.", False, False
    if direct <= 0 and collateral > 0:
        return "no_direct_support", "This mode selects collateral support but no direct support.", False, False
    if mode in {"birth", "prune", "low_entropy"} and train_control and direct <= 0 and collateral <= 0:
        return "train013_only_diagnostic", "This mode supports train013 control only; it is diagnostic but not a PR19 exact mode.", False, False
    if direct > 0 and collateral > 0 and overlap <= 0:
        return "no_overlap", "Direct and collateral supports are present but do not overlap by exact Gaussian ID.", False, False
    if usable:
        return "potentially_useful", "This mode has non-degenerate direct/collateral exact overlap.", True, True
    return "insufficient_attribution_fields", "Mode is not usable without render/residual/gradient attribution semantics.", False, False


def build_support_mode_failure_rows(comparison_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    output = []
    for row in comparison_rows:
        failure, interpretation, usable_pr19, usable_intervention = _classify_mode(row)
        output.append(
            {
                "support_mode": row.get("support_mode", ""),
                "filtered_gaussian_count": row.get("filtered_gaussian_count", ""),
                "direct_supported_gaussian_count": row.get("direct_supported_gaussian_count", ""),
                "collateral_supported_gaussian_count": row.get("collateral_supported_gaussian_count", ""),
                "overlap_count": row.get("direct_collateral_overlap_count", ""),
                "jaccard": row.get("direct_collateral_jaccard", ""),
                "train013_supported_gaussian_count": row.get("train013_supported_gaussian_count", ""),
                "train013_control_supported": row.get("train013_control_supported", ""),
                "failure_type": failure,
                "interpretation": interpretation,
                "usable_for_pr19_exact_mode": _bool_text(usable_pr19),
                "usable_for_intervention": _bool_text(usable_intervention),
            }
        )
    return output


def build_event_type_group_distribution(
    lifecycle_rows: list[dict[str, str]],
    attribution_rows: list[dict[str, str]],
    support_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    alive = {row.get("gaussian_id", "") for row in support_rows if _truth(row.get("is_alive_final"))}
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in [*lifecycle_rows, *attribution_rows]:
        event_type = str(row.get("event_type") or "unknown")
        view_group = str(row.get("view_group") or "other_clean")
        grouped[(event_type, view_group)].append(row)
    rows = []
    for (event_type, view_group), group_rows in sorted(grouped.items()):
        ids = {row.get("gaussian_id", "") for row in group_rows if row.get("gaussian_id")}
        rows.append(
            {
                "event_type": event_type,
                "view_group": view_group,
                "row_count": len(group_rows),
                "unique_gaussian_count": len(ids),
                "alive_final_count": len(ids & alive),
                "mean_contribution_value": _mean([row.get("contribution_value") for row in group_rows]),
                "max_contribution_value": _max([row.get("contribution_value") for row in group_rows]),
            }
        )
    return rows


def build_view_group_event_distribution(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("view_group") or "other_clean")].append(row)
    output = []
    for view_group, group_rows in sorted(grouped.items()):
        ids = {row.get("gaussian_id", "") for row in group_rows if row.get("gaussian_id")}
        views = {row.get("view_name", "") for row in group_rows if row.get("view_name")}
        events = Counter(str(row.get("event_type") or "unknown") for row in group_rows)
        output.append(
            {
                "view_group": view_group,
                "row_count": len(group_rows),
                "unique_gaussian_count": len(ids),
                "unique_view_count": len(views),
                "event_type_distribution": ";".join(f"{key}:{value}" for key, value in sorted(events.items())),
                "mean_events_per_gaussian": len(group_rows) / len(ids) if ids else 0.0,
                "mean_events_per_view": len(group_rows) / len(views) if views else 0.0,
                "notes": "",
            }
        )
    return output


def build_high_event_audit(summary: dict[str, Any], filtered_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    rows = [row for row in filtered_rows if row.get("support_mode") == "high_event" and _truth(row.get("included_by_filter"))]
    counts = {
        "direct_corrupted": sum(1 for row in rows if _truth(row.get("direct_corrupted_support"))),
        "co_visible_collateral": sum(1 for row in rows if _truth(row.get("collateral_support"))),
        "clean_prior_demoted": sum(1 for row in rows if _truth(row.get("clean_prior_support"))),
        "other_clean": sum(1 for row in rows if _truth(row.get("other_clean_support"))),
    }
    top_groups = Counter(str(row.get("dominant_view_group", "")) for row in rows if row.get("dominant_view_group"))
    top_views = Counter(str(row.get("dominant_view_name", "")) for row in rows if row.get("dominant_view_name"))
    interpretation = "high_event selects direct support without collateral support" if counts["direct_corrupted"] > 0 and counts["co_visible_collateral"] == 0 else "high_event support is mixed or unavailable"
    return [
        {
            "support_mode": "high_event",
            "threshold": summary.get("high_event_threshold", ""),
            "direct_selected_count": counts["direct_corrupted"],
            "collateral_selected_count": counts["co_visible_collateral"],
            "clean_prior_selected_count": counts["clean_prior_demoted"],
            "other_clean_selected_count": counts["other_clean"],
            "top_event_view_groups": ";".join(f"{key}:{value}" for key, value in top_groups.most_common(5)),
            "top_event_view_names": ";".join(f"{key}:{value}" for key, value in top_views.most_common(5)),
            "interpretation": interpretation,
        }
    ]


def build_suspicious_alive_audit(comparison_by_mode: dict[str, dict[str, str]]) -> list[dict[str, Any]]:
    broad = comparison_by_mode.get("broad", {})
    suspicious = comparison_by_mode.get("suspicious_alive", {})
    broad_overlap = _number(broad.get("direct_collateral_overlap_count"))
    suspicious_overlap = _number(suspicious.get("direct_collateral_overlap_count"))
    suspicious_jaccard = _number(suspicious.get("direct_collateral_jaccard"))
    broad_jaccard = _number(broad.get("direct_collateral_jaccard"))
    degeneracy = suspicious_jaccard >= 0.95 and suspicious_overlap > 0
    return [
        {
            "support_mode": "suspicious_alive",
            "broad_overlap_count": broad_overlap,
            "suspicious_alive_overlap_count": suspicious_overlap,
            "overlap_equivalence_ratio": suspicious_overlap / broad_overlap if broad_overlap else 0.0,
            "broad_jaccard": broad_jaccard,
            "suspicious_alive_jaccard": suspicious_jaccard,
            "degeneracy_confirmed": _bool_text(degeneracy),
            "interpretation": "suspicious_alive remains broad-degenerate" if degeneracy else "suspicious_alive is not broad-equivalent",
        }
    ]


def build_birth_prune_audit(comparison_by_mode: dict[str, dict[str, str]], failure_by_mode: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for mode, kind in [("birth", "birth_newborn_or_parent_support"), ("prune", "prune_death_support")]:
        row = comparison_by_mode.get(mode, {})
        failure = failure_by_mode.get(mode, {})
        rows.append(
            {
                "support_kind": kind,
                "selected_gaussian_count": row.get("filtered_gaussian_count", ""),
                "direct_supported_gaussian_count": row.get("direct_supported_gaussian_count", ""),
                "collateral_supported_gaussian_count": row.get("collateral_supported_gaussian_count", ""),
                "clean_prior_supported_gaussian_count": row.get("clean_prior_supported_gaussian_count", ""),
                "other_clean_supported_gaussian_count": row.get("other_clean_supported_gaussian_count", ""),
                "interpretation": failure.get("interpretation", ""),
            }
        )
    return rows


def build_train013_audit(train_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    output = []
    for row in train_rows:
        control = _truth(row.get("train013_control_supported"))
        mode = row.get("support_mode", "")
        diagnostic = control and mode in {"birth", "prune", "low_entropy"}
        output.append(
            {
                "train013_view_group": row.get("train013_view_group", ""),
                "support_mode": mode,
                "train013_supported_gaussian_count": row.get("train013_supported_gaussian_count", ""),
                "train013_direct_collateral_overlap_count": row.get("train013_direct_collateral_overlap_count", ""),
                "train013_control_supported": row.get("train013_control_supported", ""),
                "meaningful_or_diagnostic": "diagnostic_only" if diagnostic else ("meaningful" if control else "not_supported"),
                "interpretation": "train013 control is diagnostic only without nontrivial direct/collateral overlap" if diagnostic else row.get("reason", ""),
            }
        )
    return output


def build_field_gap_rows(all_columns: set[str]) -> list[dict[str, Any]]:
    source = "pr193/pr194 exact grouped logs"
    rows = []
    for field_name, required_for, reason, priority in REQUIRED_ATTRIBUTION_FIELDS:
        rows.append(
            {
                "field_name": field_name,
                "currently_available": _bool_text(field_name in all_columns),
                "source_file": source if field_name in all_columns else "",
                "required_for": required_for,
                "reason": reason,
                "priority": priority,
            }
        )
    return rows


def attribution_sufficiency_level(columns: set[str]) -> str:
    if {"gradient_norm", "delta_position", "delta_opacity", "delta_scale", "delta_sh"} & columns:
        return "gradient_weighted_contribution_available"
    if {"residual_value", "residual_l1", "residual_ssim"} & columns:
        return "residual_weighted_contribution_available"
    if {"pixel_x", "pixel_y", "alpha_contribution", "splat_weight"} & columns:
        return "render_contribution_available"
    if "contribution_value" in columns:
        return "event_weighted_context"
    if {"event_type", "gaussian_id", "view_name"} <= columns:
        return "lifecycle_context_only"
    return "stable_identity_only"


def build_readiness_rows(
    *,
    stable_ids: bool,
    view_binding: bool,
    broad_degeneracy: bool,
    nontrivial: bool,
    train013_meaningful: bool,
    columns: set[str],
) -> list[dict[str, Any]]:
    residual = bool({"residual_value", "residual_l1", "residual_ssim"} & columns)
    render = bool({"pixel_x", "pixel_y", "alpha_contribution", "splat_weight"} & columns)
    gradient = bool({"gradient_norm", "delta_position", "delta_opacity", "delta_scale", "delta_sh"} & columns)
    intervention = nontrivial and (residual or render or gradient)
    criteria = [
        ("stable Gaussian IDs available", stable_ids, "gaussian_id present", "", "keep exact ID logging"),
        ("view group binding available", view_binding, "view_group present", "", "keep PR19.3 binding"),
        ("broad degeneracy detected", broad_degeneracy, "PR19.4 broad jaccard >= 0.95", "", "do not use broad mode causally"),
        ("nontrivial exact overlap available", nontrivial, "non-broad direct/collateral overlap", "no nontrivial overlap" if not nontrivial else "", "add stronger attribution"),
        ("train013 control meaningful", train013_meaningful, "train013 control in usable mode", "train013 control is diagnostic only" if not train013_meaningful else "", "separate diagnostic controls from usable exact modes"),
        ("residual-weighted attribution available", residual, "residual columns present", "missing residual columns" if not residual else "", "log residual-weighted per-Gaussian attribution"),
        ("render contribution attribution available", render, "render contribution columns present", "missing pixel/splat contribution columns" if not render else "", "log sparse render contribution"),
        ("gradient-weighted update attribution available", gradient, "gradient/update delta columns present", "missing gradient/update columns" if not gradient else "", "optionally log gradient-weighted updates"),
        ("usable for training intervention", intervention, "nontrivial overlap plus stronger attribution", "not enough attribution semantics" if not intervention else "", "do not intervene yet"),
        ("safe to proceed to densification gating", False, "not a defense PR", "intervention evidence is insufficient", "do not gate densification"),
    ]
    return [
        {
            "criterion": criterion,
            "passed": _bool_text(passed),
            "evidence": evidence,
            "blocker": blocker,
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


def write_artifact_manifest(output_dir: Path, pr193_dir: Path, pr194_dir: Path) -> None:
    items: list[tuple[str, Path, bool, str]] = [
        ("pr193_dir", pr193_dir, True, "input"),
        ("pr194_dir", pr194_dir, True, "input"),
    ]
    items.extend((name, output_dir / name, True, "output_pr195") for name in PR195_OUTPUT_FILES)
    manifest = output_dir / "artifact_manifest.csv"
    write_csv_rows(manifest, _artifact_rows(items), MANIFEST_FIELDS)
    write_csv_rows(manifest, _artifact_rows(items), MANIFEST_FIELDS)


def _write_recommendation(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# PR19.5 Next Step Recommendation",
        "",
        "Current exact lifecycle logs are not sufficient for intervention.",
        "",
        f"- Corrected PR19 exact mode: `{summary.get('corrected_recommended_pr19_exact_mode')}`",
        f"- Attribution sufficiency: `{summary.get('attribution_sufficiency_level')}`",
        f"- PR20 ready for intervention: `{summary.get('pr20_ready_for_intervention')}`",
        "",
        "Recommended next PR: implement sparse per-view per-Gaussian render contribution and residual-weighted attribution before any intervention.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_report(path: Path, summary: dict[str, Any], failure_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# PR19.5 Exact Attribution Semantics Audit",
        "",
        "PR19.5 is offline observation only. It does not modify training, rendering, `third_party`, scoring, or defense behavior.",
        "Corruption labels are used only for grouping and evaluation, not for scoring.",
        "",
        "## Verdict",
        f"- Broad degeneracy confirmed: `{summary.get('broad_degeneracy_confirmed')}`",
        f"- Suspicious-alive degeneracy confirmed: `{summary.get('suspicious_alive_degeneracy_confirmed')}`",
        f"- Nontrivial direct/collateral overlap found: `{summary.get('nontrivial_direct_collateral_overlap_found')}`",
        f"- Corrected PR19 exact mode: `{summary.get('corrected_recommended_pr19_exact_mode')}`",
        f"- Attribution sufficiency level: `{summary.get('attribution_sufficiency_level')}`",
        f"- PR20 ready for intervention: `{summary.get('pr20_ready_for_intervention')}`",
        "",
        "## Support Mode Failures",
        "",
        "| mode | failure | usable PR19 | usable intervention |",
        "| --- | --- | --- | --- |",
    ]
    for row in failure_rows:
        lines.append(f"| {row['support_mode']} | {row['failure_type']} | {row['usable_for_pr19_exact_mode']} | {row['usable_for_intervention']} |")
    lines.extend(
        [
            "",
            "## Recommendation",
            "",
            "Implement sparse per-view per-Gaussian render contribution and residual-weighted attribution before intervention.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def audit_exact_attribution_semantics(
    *,
    pr193_dir: Path,
    pr194_dir: Path,
    output_dir: Path,
    scene: str,
    condition: str,
    subset_name: str,
    write_markdown: bool = False,
    strict: bool = False,
    allow_missing: bool = False,
) -> tuple[dict[str, Any], int]:
    del write_markdown
    output_dir.mkdir(parents=True, exist_ok=True)
    missing_rows = []
    for name in PR193_REQUIRED_FILES:
        path = pr193_dir / name
        if not path.exists():
            missing_rows.append({"input_name": name, "path": str(path), "exists": "false", "required": "true", "details": "missing PR19.3 input"})
    for name in PR194_REQUIRED_FILES:
        path = pr194_dir / name
        if not path.exists():
            missing_rows.append({"input_name": name, "path": str(path), "exists": "false", "required": "true", "details": "missing PR19.4 input"})

    view_group_rows = load_csv_rows(pr193_dir / "pr193_view_group_map.csv")
    lifecycle_rows = load_csv_rows(pr193_dir / "gaussian_lifecycle_events_grouped.csv")
    attribution_rows = load_csv_rows(pr193_dir / "view_gaussian_event_attribution_grouped.csv")
    support_rows = load_csv_rows(pr193_dir / "gaussian_support_summary_grouped.csv")
    pr193_summary = load_json(pr193_dir / "pr193_view_group_binding_summary.json")
    pr194_summary = load_json(pr194_dir / "pr194_exact_support_filter_summary.json")
    comparison_rows = load_csv_rows(pr194_dir / "pr194_support_mode_comparison.csv")
    filtered_rows = load_csv_rows(pr194_dir / "pr194_filtered_gaussian_support_by_mode.csv")
    train_rows = load_csv_rows(pr194_dir / "pr194_train013_control_by_mode.csv")
    concentration_rows = load_csv_rows(pr194_dir / "pr194_view_group_event_concentration.csv")

    comparison_by_mode = _mode_rows(comparison_rows)
    failure_rows = build_support_mode_failure_rows(comparison_rows)
    failure_by_mode = {row["support_mode"]: row for row in failure_rows}
    event_distribution = build_event_type_group_distribution(lifecycle_rows, attribution_rows, support_rows)
    view_distribution = build_view_group_event_distribution([*lifecycle_rows, *attribution_rows])
    high_event_audit = build_high_event_audit(pr194_summary, filtered_rows)
    suspicious_alive_audit = build_suspicious_alive_audit(comparison_by_mode)
    birth_prune_audit = build_birth_prune_audit(comparison_by_mode, failure_by_mode)
    train013_audit = build_train013_audit(train_rows)
    all_columns = _columns([*view_group_rows, *lifecycle_rows, *attribution_rows, *support_rows, *filtered_rows, *concentration_rows])
    field_gap_rows = build_field_gap_rows(all_columns)

    nontrivial = any(_truth(row.get("usable_for_pr19_exact_mode")) for row in failure_rows)
    broad_degeneracy = _truth(pr194_summary.get("broad_overlap_degeneracy_detected")) or _truth(comparison_by_mode.get("broad", {}).get("broad_overlap_degeneracy_flag"))
    suspicious_degeneracy = _truth(suspicious_alive_audit[0].get("degeneracy_confirmed"))
    diagnostic_modes = [
        row["support_mode"]
        for row in train_rows
        if _truth(row.get("train013_control_supported")) and row.get("support_mode") in {"birth", "prune", "low_entropy"}
    ]
    train013_only = bool(diagnostic_modes) and not nontrivial
    valid_modes = [row["support_mode"] for row in failure_rows if _truth(row.get("usable_for_pr19_exact_mode"))]
    pr194_recommendation = str(pr194_summary.get("recommended_pr19_exact_mode") or "none")
    pr194_recommendation_is_valid = pr194_recommendation in valid_modes
    corrected_mode = pr194_recommendation if pr194_recommendation_is_valid else "none"
    sufficiency = attribution_sufficiency_level(all_columns)
    readiness_rows = build_readiness_rows(
        stable_ids="gaussian_id" in all_columns,
        view_binding="view_group" in all_columns,
        broad_degeneracy=broad_degeneracy,
        nontrivial=nontrivial,
        train013_meaningful=any(
            _truth(row.get("train013_control_supported")) and row.get("support_mode") in valid_modes
            for row in train_rows
        ),
        columns=all_columns,
    )
    pr20_ready = _truth(next(row["passed"] for row in readiness_rows if row["criterion"] == "usable for training intervention"))
    field_available = {row["field_name"]: _truth(row["currently_available"]) for row in field_gap_rows}
    warnings = []
    if not nontrivial:
        warnings.append("no usable nontrivial direct/collateral exact overlap was found")
    if not pr194_recommendation_is_valid and pr194_recommendation != "none":
        warnings.append(f"PR19.4 recommendation {pr194_recommendation} is diagnostic but not a valid exact mode")

    summary = {
        "schema_name": "viewtrust.pr195.exact_attribution_semantics.summary",
        "schema_version": 1,
        "created_at_utc": _utc_now(),
        "scene": scene,
        "condition": condition,
        "subset_name": subset_name,
        "pr193_dir": str(pr193_dir),
        "pr194_dir": str(pr194_dir),
        "output_dir": str(output_dir),
        "observation_only": True,
        "training_intervention": False,
        "defense_enabled": False,
        "uses_corruption_labels_for_scoring": False,
        "uses_corruption_labels_for_evaluation": True,
        "broad_degeneracy_confirmed": broad_degeneracy,
        "suspicious_alive_degeneracy_confirmed": suspicious_degeneracy,
        "nontrivial_direct_collateral_overlap_found": nontrivial,
        "train013_control_found_only_in_non_direct_modes": train013_only,
        "pr194_recommendation_is_valid": pr194_recommendation_is_valid,
        "corrected_recommended_pr19_exact_mode": corrected_mode,
        "diagnostic_modes_with_train013_control": diagnostic_modes,
        "attribution_sufficiency_level": sufficiency,
        "pr20_ready_for_intervention": pr20_ready,
        "pr20_requires_sparse_render_contribution": not (field_available.get("pixel_x") and field_available.get("pixel_y") and field_available.get("splat_weight")),
        "pr20_requires_residual_weighted_attribution": not (field_available.get("residual_value") or field_available.get("residual_l1")),
        "pr20_requires_gradient_weighted_update_attribution": not any(field_available.get(field) for field in ["gradient_norm", "delta_position", "delta_opacity", "delta_scale", "delta_sh"]),
        "pr193_input_exact_evidence_quality": pr193_summary.get("input_exact_evidence_quality", ""),
        "pr194_recommended_pr19_exact_mode": pr194_recommendation,
        "warnings": warnings,
    }

    write_json(output_dir / "pr195_attribution_semantics_summary.json", summary)
    write_csv_rows(output_dir / "pr195_support_mode_failure_analysis.csv", failure_rows, FAILURE_FIELDS)
    write_csv_rows(output_dir / "pr195_event_type_group_distribution.csv", event_distribution, EVENT_TYPE_GROUP_FIELDS)
    write_csv_rows(output_dir / "pr195_view_group_event_distribution.csv", view_distribution, VIEW_GROUP_EVENT_FIELDS)
    write_csv_rows(output_dir / "pr195_high_event_semantics_audit.csv", high_event_audit, HIGH_EVENT_FIELDS)
    write_csv_rows(output_dir / "pr195_suspicious_alive_degeneracy_audit.csv", suspicious_alive_audit, SUSPICIOUS_ALIVE_FIELDS)
    write_csv_rows(output_dir / "pr195_birth_prune_semantics_audit.csv", birth_prune_audit, BIRTH_PRUNE_FIELDS)
    write_csv_rows(output_dir / "pr195_train013_semantics_audit.csv", train013_audit, TRAIN013_FIELDS)
    write_csv_rows(output_dir / "pr195_required_attribution_field_gap.csv", field_gap_rows, FIELD_GAP_FIELDS)
    write_csv_rows(output_dir / "pr195_pr20_readiness_assessment.csv", readiness_rows, READINESS_FIELDS)
    _write_recommendation(output_dir / "pr195_next_step_recommendation.md", summary)
    write_csv_rows(output_dir / "pr195_missing_inputs.csv", missing_rows, MISSING_FIELDS)
    _write_report(output_dir / "pr195_report.md", summary, failure_rows)
    write_artifact_manifest(output_dir, pr193_dir, pr194_dir)

    missing_required = [name for name in PR195_OUTPUT_FILES if not (output_dir / name).is_file()]
    if missing_required:
        raise RuntimeError(f"missing PR19.5 outputs: {missing_required}")
    if missing_rows and strict and not allow_missing:
        return summary, 1
    if missing_rows and not allow_missing:
        return summary, 1
    return summary, 0
