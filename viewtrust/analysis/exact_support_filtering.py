"""Offline exact support filtering helpers for PR19.4."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from viewtrust.analysis.clean_prior_normalization import load_csv_rows, load_json, normalize_bool
from viewtrust.analysis.offline_signals import safe_float


DEFAULT_SUPPORT_MODES = [
    "broad",
    "birth",
    "prune",
    "high_event",
    "dominant_source",
    "low_entropy",
    "suspicious_alive",
]

PR194_OUTPUT_FILES = [
    "pr194_exact_support_filter_summary.json",
    "pr194_support_mode_comparison.csv",
    "pr194_filtered_gaussian_support_by_mode.csv",
    "pr194_direct_collateral_overlap_by_mode.csv",
    "pr194_train013_control_by_mode.csv",
    "pr194_gaussian_mode_membership.csv",
    "pr194_view_group_event_concentration.csv",
    "pr194_nontrivial_overlap_candidates.csv",
    "pr194_missing_inputs.csv",
    "pr194_report.md",
    "artifact_manifest.csv",
]

PR193_REQUIRED_FILES = [
    "pr193_view_group_map.csv",
    "pr193_view_group_binding_summary.json",
    "gaussian_identity_table_grouped.csv",
    "gaussian_lifecycle_events_grouped.csv",
    "view_gaussian_event_attribution_grouped.csv",
    "gaussian_support_summary_grouped.csv",
]

GROUPS = ["direct_corrupted", "co_visible_collateral", "clean_prior_demoted", "other_clean"]
GROUP_FIELD_PREFIX = {
    "direct_corrupted": "direct_corrupted",
    "co_visible_collateral": "collateral",
    "clean_prior_demoted": "clean_prior",
    "other_clean": "other_clean",
}

MANIFEST_FIELDS = ["relative_path", "path", "exists", "file_type", "size_bytes", "required", "artifact_group"]
MISSING_FIELDS = ["input_name", "path", "exists", "required", "details"]

MODE_COMPARISON_FIELDS = [
    "scene",
    "condition",
    "subset_name",
    "support_mode",
    "total_gaussians_considered",
    "filtered_gaussian_count",
    "direct_supported_gaussian_count",
    "collateral_supported_gaussian_count",
    "clean_prior_supported_gaussian_count",
    "other_clean_supported_gaussian_count",
    "direct_collateral_overlap_count",
    "direct_collateral_jaccard",
    "direct_collateral_ratio_over_direct",
    "direct_collateral_ratio_over_collateral",
    "train013_supported_gaussian_count",
    "train013_direct_collateral_overlap_count",
    "train013_overlap_ratio_with_direct_collateral",
    "train013_control_supported",
    "nontrivial_overlap_supported",
    "broad_overlap_degeneracy_flag",
    "notes",
]

DIRECT_COLLATERAL_FIELDS = [
    "scene",
    "condition",
    "subset_name",
    "support_mode",
    "direct_corrupted_view_names",
    "collateral_view_names",
    "direct_supported_gaussian_count",
    "collateral_supported_gaussian_count",
    "exact_overlap_gaussian_count",
    "exact_overlap_jaccard",
    "exact_overlap_ratio_over_direct",
    "exact_overlap_ratio_over_collateral",
    "nontrivial_overlap_supported",
    "evidence_quality",
    "notes",
]

TRAIN013_FIELDS = [
    "scene",
    "condition",
    "subset_name",
    "support_mode",
    "train013_present",
    "train013_view_group",
    "train013_supported_gaussian_count",
    "direct_collateral_supported_gaussian_count",
    "train013_direct_collateral_overlap_count",
    "train013_overlap_ratio_with_direct_collateral",
    "train013_clean_prior_event_ratio_mean",
    "train013_corrupted_plus_collateral_event_ratio_mean",
    "train013_control_supported",
    "evidence_quality",
    "reason",
    "notes",
]

FILTERED_FIELDS = [
    "scene",
    "condition",
    "subset_name",
    "support_mode",
    "gaussian_id",
    "root_gaussian_id",
    "parent_gaussian_id",
    "birth_event_type",
    "death_event_type",
    "is_alive_final",
    "direct_corrupted_support",
    "collateral_support",
    "clean_prior_support",
    "other_clean_support",
    "direct_corrupted_event_count",
    "collateral_event_count",
    "clean_prior_event_count",
    "other_clean_event_count",
    "source_entropy",
    "source_concentration",
    "dominant_view_group",
    "dominant_view_name",
    "corrupted_plus_collateral_event_ratio",
    "clean_prior_event_ratio",
    "included_by_filter",
    "filter_reason",
]

NONTRIVIAL_FIELDS = [
    "scene",
    "condition",
    "subset_name",
    "support_mode",
    "gaussian_id",
    "root_gaussian_id",
    "parent_gaussian_id",
    "birth_event_type",
    "death_event_type",
    "is_alive_final",
    "direct_corrupted_event_count",
    "collateral_event_count",
    "clean_prior_event_count",
    "other_clean_event_count",
    "source_entropy",
    "source_concentration",
    "dominant_view_group",
    "corrupted_plus_collateral_event_ratio",
    "clean_prior_event_ratio",
    "filter_reason",
]

CONCENTRATION_FIELDS = [
    "scene",
    "condition",
    "subset_name",
    "gaussian_id",
    "root_gaussian_id",
    "parent_gaussian_id",
    "is_alive_final",
    "total_event_count",
    "direct_corrupted_event_count",
    "collateral_event_count",
    "clean_prior_event_count",
    "other_clean_event_count",
    "source_entropy",
    "source_concentration",
    "view_entropy",
    "view_concentration",
    "dominant_view_group",
    "dominant_view_name",
    "corrupted_plus_collateral_event_ratio",
    "clean_prior_event_ratio",
    "birth_from_direct_or_collateral",
    "birth_event_count",
    "prune_event_count",
    "max_view_event_count",
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


def _source_view(row: dict[str, Any]) -> str:
    return str(row.get("view_name") or row.get("source_view_name") or "")


def _gaussian_id(row: dict[str, Any]) -> str:
    return str(row.get("gaussian_id", "") or "")


def _group(row: dict[str, Any], view_groups: dict[str, str]) -> str:
    value = str(row.get("view_group", "") or "")
    if value:
        return value
    return view_groups.get(_source_view(row), "other_clean")


def _event_weight(row: dict[str, Any]) -> float:
    for key in ("contribution_value", "event_count", "birth_event_count", "clone_birth_count", "split_birth_count", "prune_death_count"):
        value = _number(row.get(key), 0.0)
        if value > 0:
            return value
    return 1.0


def _event_type(row: dict[str, Any]) -> str:
    return str(row.get("event_type") or row.get("lifecycle_action") or "")


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = math.ceil((max(0.0, min(100.0, percentile)) / 100.0) * len(ordered)) - 1
    return ordered[max(0, min(len(ordered) - 1, index))]


def _entropy(weights: list[float]) -> tuple[float, float, float]:
    total = sum(weight for weight in weights if weight > 0)
    if total <= 0:
        return 0.0, 0.0, 0.0
    probabilities = [weight / total for weight in weights if weight > 0]
    entropy = -sum(p * math.log(p) for p in probabilities)
    max_entropy = math.log(len(probabilities)) if len(probabilities) > 1 else 1.0
    normalized = entropy / max_entropy if max_entropy else 0.0
    concentration = 1.0 - normalized
    return entropy, max(0.0, min(1.0, normalized)), max(0.0, min(1.0, concentration))


def _meta_by_id(identity_rows: list[dict[str, str]], support_rows: list[dict[str, str]]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for row in [*identity_rows, *support_rows]:
        gid = _gaussian_id(row)
        if not gid:
            continue
        current = output.setdefault(gid, {"gaussian_id": gid})
        for key in (
            "scene",
            "condition",
            "subset_name",
            "root_gaussian_id",
            "parent_gaussian_id",
            "birth_event_type",
            "death_event_type",
            "is_alive_final",
        ):
            if current.get(key) in ("", None) and row.get(key) not in ("", None):
                current[key] = row.get(key)
            elif key not in current:
                current[key] = row.get(key, "")
    return output


def _empty_support_state() -> dict[str, Any]:
    return {
        "views_by_group": defaultdict(set),
        "events_by_group": Counter(),
        "events_by_view": Counter(),
        "reasons": [],
    }


def _add_support(state: dict[str, Any], view_name: str, view_group: str, weight: float, reason: str) -> None:
    if not view_name:
        return
    group = view_group if view_group in GROUPS else "other_clean"
    state["views_by_group"][group].add(view_name)
    state["events_by_group"][group] += max(weight, 1.0)
    state["events_by_view"][view_name] += max(weight, 1.0)
    if reason not in state["reasons"]:
        state["reasons"].append(reason)


def _aggregate_pairs(
    attribution_rows: list[dict[str, Any]],
    view_groups: dict[str, str],
) -> tuple[dict[str, dict[str, dict[str, Any]]], list[float]]:
    pairs: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    weights: list[float] = []
    for row in attribution_rows:
        gid = _gaussian_id(row)
        view = _source_view(row)
        if not gid or not view:
            continue
        weight = _event_weight(row)
        group = _group(row, view_groups)
        pair = pairs[gid].setdefault(view, {"view_name": view, "view_group": group, "weight": 0.0})
        pair["weight"] += weight
    for by_view in pairs.values():
        for pair in by_view.values():
            weights.append(float(pair["weight"]))
    return pairs, weights


def _base_concentration(
    *,
    meta: dict[str, dict[str, Any]],
    pairs: dict[str, dict[str, dict[str, Any]]],
    lifecycle_rows: list[dict[str, Any]],
    view_groups: dict[str, str],
) -> dict[str, dict[str, Any]]:
    lifecycle_counts: dict[str, Counter[str]] = defaultdict(Counter)
    birth_direct_collateral: set[str] = set()
    for row in lifecycle_rows:
        gid = _gaussian_id(row)
        if not gid:
            continue
        event_type = _event_type(row)
        lifecycle_counts[gid][event_type] += 1
        if event_type in {"clone_birth", "split_birth", "densify_birth_unknown"} and _group(row, view_groups) in {"direct_corrupted", "co_visible_collateral"}:
            birth_direct_collateral.add(gid)
    output: dict[str, dict[str, Any]] = {}
    for gid in sorted(set(meta) | set(pairs)):
        group_weights = Counter()
        view_weights = Counter()
        for view, pair in pairs.get(gid, {}).items():
            group_weights[pair["view_group"]] += pair["weight"]
            view_weights[view] += pair["weight"]
        _, group_entropy, group_concentration = _entropy(list(group_weights.values()))
        _, view_entropy, view_concentration = _entropy(list(view_weights.values()))
        total = sum(group_weights.values())
        dominant_group = max(group_weights.items(), key=lambda item: (item[1], item[0]))[0] if group_weights else ""
        dominant_view = max(view_weights.items(), key=lambda item: (item[1], item[0]))[0] if view_weights else ""
        direct = group_weights.get("direct_corrupted", 0.0)
        collateral = group_weights.get("co_visible_collateral", 0.0)
        clean_prior = group_weights.get("clean_prior_demoted", 0.0)
        output[gid] = {
            **meta.get(gid, {"gaussian_id": gid}),
            "total_event_count": total,
            "direct_corrupted_event_count": direct,
            "collateral_event_count": collateral,
            "clean_prior_event_count": clean_prior,
            "other_clean_event_count": group_weights.get("other_clean", 0.0),
            "source_entropy": group_entropy,
            "source_concentration": group_concentration,
            "view_entropy": view_entropy,
            "view_concentration": view_concentration,
            "dominant_view_group": dominant_group,
            "dominant_view_name": dominant_view,
            "corrupted_plus_collateral_event_ratio": (direct + collateral) / total if total else 0.0,
            "clean_prior_event_ratio": clean_prior / total if total else 0.0,
            "birth_from_direct_or_collateral": _bool_text(gid in birth_direct_collateral),
            "birth_event_count": sum(
                lifecycle_counts[gid].get(event_type, 0)
                for event_type in ("clone_birth", "split_birth", "densify_birth_unknown")
            ),
            "prune_event_count": lifecycle_counts[gid].get("prune_death", 0),
            "max_view_event_count": max(view_weights.values()) if view_weights else 0.0,
        }
    return output


def _mode_supports(
    *,
    mode: str,
    meta: dict[str, dict[str, Any]],
    pairs: dict[str, dict[str, dict[str, Any]]],
    lifecycle_rows: list[dict[str, Any]],
    concentration: dict[str, dict[str, Any]],
    view_groups: dict[str, str],
    high_event_threshold: float,
    dominant_source_threshold: float,
    low_entropy_threshold: float,
    min_event_count: int,
    alive_only: bool,
) -> dict[str, dict[str, Any]]:
    supports: dict[str, dict[str, Any]] = defaultdict(_empty_support_state)

    def alive_ok(gid: str) -> bool:
        return (not alive_only) or _truth(meta.get(gid, {}).get("is_alive_final"))

    if mode == "broad":
        for gid, by_view in pairs.items():
            if not alive_ok(gid):
                continue
            for pair in by_view.values():
                _add_support(supports[gid], pair["view_name"], pair["view_group"], pair["weight"], "broad_visibility_update_support")
    elif mode == "birth":
        for row in lifecycle_rows:
            event_type = _event_type(row)
            if event_type not in {"clone_birth", "split_birth", "densify_birth_unknown"}:
                continue
            view = _source_view(row)
            group = _group(row, view_groups)
            gid = _gaussian_id(row)
            if gid and alive_ok(gid):
                _add_support(supports[gid], view, group, _event_weight(row), "birth_newborn_support")
            parent = str(row.get("parent_gaussian_id") or "")
            if parent and parent in meta and alive_ok(parent):
                _add_support(supports[parent], view, group, _event_weight(row), "birth_parent_support")
    elif mode == "prune":
        for row in lifecycle_rows:
            if _event_type(row) != "prune_death":
                continue
            gid = _gaussian_id(row)
            if gid and alive_ok(gid):
                _add_support(supports[gid], _source_view(row), _group(row, view_groups), _event_weight(row), "prune_support")
    elif mode == "high_event":
        for gid, by_view in pairs.items():
            if not alive_ok(gid):
                continue
            for pair in by_view.values():
                if pair["weight"] >= high_event_threshold and pair["weight"] >= min_event_count:
                    _add_support(supports[gid], pair["view_name"], pair["view_group"], pair["weight"], "high_event_pair_support")
    elif mode == "dominant_source":
        for gid, by_view in pairs.items():
            if not alive_ok(gid):
                continue
            total = concentration.get(gid, {}).get("total_event_count", 0.0)
            if total <= 0:
                continue
            group_totals = {
                "direct_corrupted": concentration[gid].get("direct_corrupted_event_count", 0.0),
                "co_visible_collateral": concentration[gid].get("collateral_event_count", 0.0),
                "clean_prior_demoted": concentration[gid].get("clean_prior_event_count", 0.0),
                "other_clean": concentration[gid].get("other_clean_event_count", 0.0),
            }
            dominant_groups = [group for group, value in group_totals.items() if value >= min_event_count and value / total >= dominant_source_threshold]
            for pair in by_view.values():
                if pair["view_group"] in dominant_groups:
                    _add_support(supports[gid], pair["view_name"], pair["view_group"], pair["weight"], "dominant_source_group_support")
    elif mode == "low_entropy":
        for gid, by_view in pairs.items():
            if not alive_ok(gid) or concentration.get(gid, {}).get("source_entropy", 1.0) > low_entropy_threshold:
                continue
            dominant = concentration.get(gid, {}).get("dominant_view_group", "")
            for pair in by_view.values():
                if pair["view_group"] == dominant:
                    _add_support(supports[gid], pair["view_name"], pair["view_group"], pair["weight"], "low_entropy_dominant_support")
    elif mode == "suspicious_alive":
        for gid, by_view in pairs.items():
            if not _truth(meta.get(gid, {}).get("is_alive_final")):
                continue
            row = concentration.get(gid, {})
            criteria = []
            if row.get("source_entropy", 1.0) <= low_entropy_threshold:
                criteria.append("low_source_entropy")
            if row.get("source_concentration", 0.0) >= (1.0 - low_entropy_threshold):
                criteria.append("high_source_concentration")
            if row.get("corrupted_plus_collateral_event_ratio", 0.0) >= dominant_source_threshold:
                criteria.append("high_corrupted_plus_collateral_event_ratio")
            if row.get("clean_prior_event_ratio", 1.0) <= 0.10:
                criteria.append("low_clean_prior_event_ratio")
            if _truth(row.get("birth_from_direct_or_collateral")):
                criteria.append("newborn_from_direct_or_collateral")
            if row.get("max_view_event_count", 0.0) >= high_event_threshold and row.get("max_view_event_count", 0.0) >= min_event_count:
                criteria.append("high_event_percentile")
            if row.get("birth_event_count", 0) + row.get("prune_event_count", 0) >= min_event_count:
                criteria.append("abnormal_lifecycle_instability")
            if len(criteria) < 2:
                continue
            for pair in by_view.values():
                if pair["view_group"] in {"direct_corrupted", "co_visible_collateral"} or pair["weight"] >= high_event_threshold:
                    _add_support(supports[gid], pair["view_name"], pair["view_group"], pair["weight"], "suspicious_alive:" + ";".join(criteria))
    else:
        raise ValueError(f"unknown support mode: {mode}")

    return supports


def _finalize_support_row(
    *,
    scene: str,
    condition: str,
    subset_name: str,
    mode: str,
    gid: str,
    meta: dict[str, Any],
    state: dict[str, Any] | None,
    concentration: dict[str, Any],
) -> dict[str, Any]:
    state = state or _empty_support_state()
    group_events = state["events_by_group"]
    group_views = state["views_by_group"]
    direct = group_events.get("direct_corrupted", 0.0)
    collateral = group_events.get("co_visible_collateral", 0.0)
    clean_prior = group_events.get("clean_prior_demoted", 0.0)
    other = group_events.get("other_clean", 0.0)
    included = bool(sum(group_events.values()) > 0)
    return {
        "scene": scene,
        "condition": condition,
        "subset_name": subset_name,
        "support_mode": mode,
        "gaussian_id": gid,
        "root_gaussian_id": meta.get("root_gaussian_id", ""),
        "parent_gaussian_id": meta.get("parent_gaussian_id", ""),
        "birth_event_type": meta.get("birth_event_type", ""),
        "death_event_type": meta.get("death_event_type", ""),
        "is_alive_final": meta.get("is_alive_final", ""),
        "direct_corrupted_support": _bool_text(bool(group_views.get("direct_corrupted"))),
        "collateral_support": _bool_text(bool(group_views.get("co_visible_collateral"))),
        "clean_prior_support": _bool_text(bool(group_views.get("clean_prior_demoted"))),
        "other_clean_support": _bool_text(bool(group_views.get("other_clean"))),
        "direct_corrupted_event_count": direct,
        "collateral_event_count": collateral,
        "clean_prior_event_count": clean_prior,
        "other_clean_event_count": other,
        "source_entropy": concentration.get("source_entropy", 0.0),
        "source_concentration": concentration.get("source_concentration", 0.0),
        "dominant_view_group": concentration.get("dominant_view_group", ""),
        "dominant_view_name": concentration.get("dominant_view_name", ""),
        "corrupted_plus_collateral_event_ratio": concentration.get("corrupted_plus_collateral_event_ratio", 0.0),
        "clean_prior_event_ratio": concentration.get("clean_prior_event_ratio", 0.0),
        "included_by_filter": _bool_text(included),
        "filter_reason": ";".join(state["reasons"]) if included else "",
    }


def _mode_summary(
    *,
    scene: str,
    condition: str,
    subset_name: str,
    mode: str,
    rows: list[dict[str, Any]],
    view_group_rows: list[dict[str, Any]],
    train013_group: str,
    evidence_quality: str,
    final_alive_count: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    included = [row for row in rows if _truth(row.get("included_by_filter"))]
    direct_ids = {row["gaussian_id"] for row in included if _truth(row.get("direct_corrupted_support"))}
    collateral_ids = {row["gaussian_id"] for row in included if _truth(row.get("collateral_support"))}
    clean_prior_ids = {row["gaussian_id"] for row in included if _truth(row.get("clean_prior_support"))}
    other_ids = {row["gaussian_id"] for row in included if _truth(row.get("other_clean_support"))}
    overlap = direct_ids & collateral_ids
    union = direct_ids | collateral_ids
    train013_ids = {
        row["gaussian_id"]
        for row in included
        if _truth(row.get("clean_prior_support"))
    }
    direct_collateral_ids = direct_ids | collateral_ids
    train_overlap = train013_ids & direct_collateral_ids
    jaccard = len(overlap) / len(union) if union else 0.0
    broad_degenerate = (
        mode == "broad"
        and jaccard >= 0.95
        and len(direct_ids) == len(collateral_ids)
        and len(overlap) == len(direct_ids)
    )
    if mode != "broad" and final_alive_count:
        broad_degenerate = (
            jaccard >= 0.95
            and len(included) >= 0.90 * final_alive_count
            and len(train013_ids) >= 0.90 * final_alive_count
        )
    nontrivial = mode != "broad" and len(direct_ids) > 0 and len(collateral_ids) > 0 and len(overlap) > 0 and jaccard < 0.95 and not broad_degenerate
    train_ratio = len(train_overlap) / len(train013_ids) if train013_ids else 0.0
    train_cp_mean = _mean([row.get("clean_prior_event_ratio") for row in rows if row["gaussian_id"] in train013_ids])
    train_cc_mean = _mean([row.get("corrupted_plus_collateral_event_ratio") for row in rows if row["gaussian_id"] in train013_ids])
    train_reason = "ok"
    train_control = True
    if not train013_group:
        train_reason = "train013_missing"
        train_control = False
    elif train013_group != "clean_prior_demoted":
        train_reason = "train013_not_clean_prior_demoted"
        train_control = False
    elif not train013_ids:
        train_reason = "train013_no_support"
        train_control = False
    elif mode == "broad":
        train_reason = "support_mode_too_broad"
        train_control = False
    elif train_ratio >= 0.10:
        train_reason = "train013_overlaps_direct_collateral"
        train_control = False
    elif train_cc_mean > train_cp_mean:
        train_reason = "train013_dominates_suspicious_ids"
        train_control = False

    direct_views = sorted(row["view_name"] for row in view_group_rows if row.get("view_group") == "direct_corrupted")
    collateral_views = sorted(row["view_name"] for row in view_group_rows if row.get("view_group") == "co_visible_collateral")
    comparison = {
        "scene": scene,
        "condition": condition,
        "subset_name": subset_name,
        "support_mode": mode,
        "total_gaussians_considered": len(rows),
        "filtered_gaussian_count": len(included),
        "direct_supported_gaussian_count": len(direct_ids),
        "collateral_supported_gaussian_count": len(collateral_ids),
        "clean_prior_supported_gaussian_count": len(clean_prior_ids),
        "other_clean_supported_gaussian_count": len(other_ids),
        "direct_collateral_overlap_count": len(overlap),
        "direct_collateral_jaccard": jaccard,
        "direct_collateral_ratio_over_direct": len(overlap) / len(direct_ids) if direct_ids else 0.0,
        "direct_collateral_ratio_over_collateral": len(overlap) / len(collateral_ids) if collateral_ids else 0.0,
        "train013_supported_gaussian_count": len(train013_ids),
        "train013_direct_collateral_overlap_count": len(train_overlap),
        "train013_overlap_ratio_with_direct_collateral": train_ratio,
        "train013_control_supported": _bool_text(train_control),
        "nontrivial_overlap_supported": _bool_text(nontrivial),
        "broad_overlap_degeneracy_flag": _bool_text(broad_degenerate),
        "notes": "" if not broad_degenerate else "overlap is degenerate broad support",
    }
    direct_collateral = {
        "scene": scene,
        "condition": condition,
        "subset_name": subset_name,
        "support_mode": mode,
        "direct_corrupted_view_names": ";".join(direct_views),
        "collateral_view_names": ";".join(collateral_views),
        "direct_supported_gaussian_count": len(direct_ids),
        "collateral_supported_gaussian_count": len(collateral_ids),
        "exact_overlap_gaussian_count": len(overlap),
        "exact_overlap_jaccard": jaccard,
        "exact_overlap_ratio_over_direct": len(overlap) / len(direct_ids) if direct_ids else 0.0,
        "exact_overlap_ratio_over_collateral": len(overlap) / len(collateral_ids) if collateral_ids else 0.0,
        "nontrivial_overlap_supported": _bool_text(nontrivial),
        "evidence_quality": evidence_quality,
        "notes": "" if nontrivial else "no nontrivial direct/collateral exact overlap under this mode",
    }
    train013 = {
        "scene": scene,
        "condition": condition,
        "subset_name": subset_name,
        "support_mode": mode,
        "train013_present": _bool_text(bool(train013_group)),
        "train013_view_group": train013_group,
        "train013_supported_gaussian_count": len(train013_ids),
        "direct_collateral_supported_gaussian_count": len(direct_collateral_ids),
        "train013_direct_collateral_overlap_count": len(train_overlap),
        "train013_overlap_ratio_with_direct_collateral": train_ratio,
        "train013_clean_prior_event_ratio_mean": train_cp_mean,
        "train013_corrupted_plus_collateral_event_ratio_mean": train_cc_mean,
        "train013_control_supported": _bool_text(train_control),
        "evidence_quality": evidence_quality,
        "reason": train_reason,
        "notes": "",
    }
    return comparison, direct_collateral, train013


def _mean(values: list[Any]) -> float:
    numbers = [safe_float(value) for value in values]
    finite = [number for number in numbers if number is not None]
    return sum(finite) / len(finite) if finite else 0.0


def _recommend_mode(comparison_rows: list[dict[str, Any]], train_rows: list[dict[str, Any]]) -> tuple[str, list[str]]:
    train_by_mode = {row["support_mode"]: row for row in train_rows}
    nontrivial = {row["support_mode"]: row for row in comparison_rows if _truth(row.get("nontrivial_overlap_supported"))}
    reasons: list[str] = []
    if not nontrivial:
        reasons.append("no nontrivial direct/collateral exact overlap; diagnostic train013 modes are not valid PR19 exact modes")
        return "none", reasons
    for mode in ["suspicious_alive", "dominant_source"]:
        if mode in nontrivial and _truth(train_by_mode.get(mode, {}).get("train013_control_supported")):
            return mode, reasons
    if "high_event" in nontrivial:
        return "high_event", reasons
    reasons.append("no non-degenerate support mode satisfied direct/collateral overlap and train013 control criteria")
    return "none", reasons


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


def write_artifact_manifest(output_dir: Path, pr193_dir: Path) -> None:
    items: list[tuple[str, Path, bool, str]] = [("pr193_dir", pr193_dir, True, "input")]
    items.extend((name, output_dir / name, True, "output_pr194") for name in PR194_OUTPUT_FILES)
    manifest = output_dir / "artifact_manifest.csv"
    write_csv_rows(manifest, _artifact_rows(items), MANIFEST_FIELDS)
    write_csv_rows(manifest, _artifact_rows(items), MANIFEST_FIELDS)


def _write_report(path: Path, summary: dict[str, Any], comparison_rows: list[dict[str, Any]], train_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# PR19.4 Exact Support Filtering",
        "",
        "PR19.4 is offline observation only. It does not modify training, rendering, `third_party`, PR17/PR18 scoring, PR19.3 binding, or any defense behavior.",
        "Corruption labels are used only for grouping and evaluation, not for scoring.",
        "",
        "## Why This Exists",
        "",
        "PR19.3 broad support can be degenerate when many views observe or update the same final alive Gaussians. PR19.4 compares stricter support modes against that broad baseline.",
        "",
        "## Summary",
        f"- Broad direct/collateral Jaccard: `{summary.get('broad_direct_collateral_jaccard')}`",
        f"- Broad degeneracy detected: `{summary.get('broad_overlap_degeneracy_detected')}`",
        f"- Nontrivial overlap modes: `{';'.join(summary.get('nontrivial_modes_with_direct_collateral_overlap', []))}`",
        f"- Train013 control modes: `{';'.join(summary.get('modes_with_train013_control_supported', []))}`",
        f"- Recommended PR19 exact mode: `{summary.get('recommended_pr19_exact_mode')}`",
        "",
        "## Support Mode Comparison",
        "",
        "| mode | filtered | overlap | jaccard | train013 control | nontrivial |",
        "| --- | ---: | ---: | ---: | --- | --- |",
    ]
    for row in comparison_rows:
        lines.append(
            f"| {row['support_mode']} | {row['filtered_gaussian_count']} | {row['direct_collateral_overlap_count']} | {row['direct_collateral_jaccard']} | {row['train013_control_supported']} | {row['nontrivial_overlap_supported']} |"
        )
    lines.extend(
        [
            "",
            "## Train013 Control",
            "",
            "| mode | supported | overlap | reason |",
            "| --- | --- | ---: | --- |",
        ]
    )
    for row in train_rows:
        lines.append(
            f"| {row['support_mode']} | {row['train013_control_supported']} | {row['train013_direct_collateral_overlap_count']} | {row['reason']} |"
        )
    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "Filtered event support is still observational. If every non-broad mode is unsupported or degenerate, the correct conclusion is that more precise attribution is needed before intervention.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def analyze_exact_support_filters(
    *,
    pr193_dir: Path,
    output_dir: Path,
    scene: str,
    condition: str,
    subset_name: str,
    support_modes: list[str] | None = None,
    top_k: int = 20,
    event_percentile: float = 95.0,
    dominant_source_threshold: float = 0.5,
    low_entropy_threshold: float = 0.35,
    min_event_count: int = 3,
    alive_only: bool = False,
    write_markdown: bool = False,
    strict: bool = False,
    allow_missing: bool = False,
) -> tuple[dict[str, Any], int]:
    del top_k
    output_dir.mkdir(parents=True, exist_ok=True)
    support_modes = support_modes or list(DEFAULT_SUPPORT_MODES)
    missing_rows = []
    for name in PR193_REQUIRED_FILES:
        path = pr193_dir / name
        if not path.exists():
            missing_rows.append({"input_name": name, "path": str(path), "exists": "false", "required": "true", "details": "missing PR19.3 input"})

    view_group_rows = load_csv_rows(pr193_dir / "pr193_view_group_map.csv")
    identity_rows = load_csv_rows(pr193_dir / "gaussian_identity_table_grouped.csv")
    lifecycle_rows = load_csv_rows(pr193_dir / "gaussian_lifecycle_events_grouped.csv")
    attribution_rows = load_csv_rows(pr193_dir / "view_gaussian_event_attribution_grouped.csv")
    support_rows = load_csv_rows(pr193_dir / "gaussian_support_summary_grouped.csv")
    pr193_summary = load_json(pr193_dir / "pr193_view_group_binding_summary.json")
    view_groups = {row.get("view_name", ""): row.get("view_group", "other_clean") for row in view_group_rows if row.get("view_name")}
    train013_group = view_groups.get("train_013", "")
    meta = _meta_by_id(identity_rows, support_rows)
    pairs, pair_weights = _aggregate_pairs(attribution_rows, view_groups)
    high_event_threshold = max(_percentile(pair_weights, event_percentile), float(min_event_count))
    concentration = _base_concentration(meta=meta, pairs=pairs, lifecycle_rows=lifecycle_rows, view_groups=view_groups)
    final_alive_count = sum(1 for row in meta.values() if _truth(row.get("is_alive_final")))
    all_ids = sorted(set(meta) | set(pairs) | set(concentration))
    evidence_quality = str(pr193_summary.get("output_exact_evidence_quality") or pr193_summary.get("input_exact_evidence_quality") or "exact")

    filtered_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    direct_collateral_rows: list[dict[str, Any]] = []
    train013_rows: list[dict[str, Any]] = []
    membership: dict[str, dict[str, Any]] = {}
    nontrivial_rows: list[dict[str, Any]] = []

    for mode in support_modes:
        supports = _mode_supports(
            mode=mode,
            meta=meta,
            pairs=pairs,
            lifecycle_rows=lifecycle_rows,
            concentration=concentration,
            view_groups=view_groups,
            high_event_threshold=high_event_threshold,
            dominant_source_threshold=dominant_source_threshold,
            low_entropy_threshold=low_entropy_threshold,
            min_event_count=min_event_count,
            alive_only=alive_only,
        )
        mode_rows = [
            _finalize_support_row(
                scene=scene,
                condition=condition,
                subset_name=subset_name,
                mode=mode,
                gid=gid,
                meta=meta.get(gid, {"gaussian_id": gid}),
                state=supports.get(gid),
                concentration=concentration.get(gid, {}),
            )
            for gid in all_ids
        ]
        comparison, direct_collateral, train013 = _mode_summary(
            scene=scene,
            condition=condition,
            subset_name=subset_name,
            mode=mode,
            rows=mode_rows,
            view_group_rows=view_group_rows,
            train013_group=train013_group,
            evidence_quality=evidence_quality,
            final_alive_count=final_alive_count,
        )
        filtered_rows.extend(mode_rows)
        comparison_rows.append(comparison)
        direct_collateral_rows.append(direct_collateral)
        train013_rows.append(train013)
        for row in mode_rows:
            gid = row["gaussian_id"]
            member = membership.setdefault(
                gid,
                {
                    "scene": scene,
                    "condition": condition,
                    "subset_name": subset_name,
                    "gaussian_id": gid,
                    "root_gaussian_id": row.get("root_gaussian_id", ""),
                    "parent_gaussian_id": row.get("parent_gaussian_id", ""),
                    "is_alive_final": row.get("is_alive_final", ""),
                    "included_mode_count": 0,
                },
            )
            included = _truth(row.get("included_by_filter"))
            member[f"{mode}_included"] = _bool_text(included)
            if included:
                member["included_mode_count"] += 1
        if _truth(direct_collateral.get("nontrivial_overlap_supported")):
            for row in mode_rows:
                if _truth(row.get("direct_corrupted_support")) and _truth(row.get("collateral_support")):
                    nontrivial_rows.append({key: row.get(key, "") for key in NONTRIVIAL_FIELDS})

    concentration_rows = [
        {
            "scene": scene,
            "condition": condition,
            "subset_name": subset_name,
            "gaussian_id": gid,
            **{field: concentration.get(gid, {}).get(field, "") for field in CONCENTRATION_FIELDS if field not in {"scene", "condition", "subset_name", "gaussian_id"}},
        }
        for gid in all_ids
    ]
    broad_row = next((row for row in comparison_rows if row["support_mode"] == "broad"), {})
    nontrivial_modes = [row["support_mode"] for row in comparison_rows if _truth(row.get("nontrivial_overlap_supported"))]
    train_control_modes = [row["support_mode"] for row in train013_rows if _truth(row.get("train013_control_supported"))]
    recommended, recommendation_warnings = _recommend_mode(comparison_rows, train013_rows)
    warnings = list(recommendation_warnings)
    if _truth(broad_row.get("broad_overlap_degeneracy_flag")):
        warnings.append("broad support degeneracy detected; do not interpret broad overlap as causal")

    summary = {
        "schema_name": "viewtrust.pr194.exact_support_filter.summary",
        "schema_version": 1,
        "created_at_utc": _utc_now(),
        "scene": scene,
        "condition": condition,
        "subset_name": subset_name,
        "pr193_dir": str(pr193_dir),
        "output_dir": str(output_dir),
        "observation_only": True,
        "training_intervention": False,
        "defense_enabled": False,
        "uses_corruption_labels_for_scoring": False,
        "uses_corruption_labels_for_evaluation": True,
        "input_exact_evidence_quality": evidence_quality,
        "support_modes": support_modes,
        "event_percentile": event_percentile,
        "dominant_source_threshold": dominant_source_threshold,
        "low_entropy_threshold": low_entropy_threshold,
        "min_event_count": min_event_count,
        "alive_only": alive_only,
        "high_event_threshold": high_event_threshold,
        "broad_direct_collateral_jaccard": broad_row.get("direct_collateral_jaccard", 0.0),
        "broad_overlap_degeneracy_detected": _truth(broad_row.get("broad_overlap_degeneracy_flag")),
        "nontrivial_modes_with_direct_collateral_overlap": nontrivial_modes,
        "modes_with_train013_control_supported": train_control_modes,
        "diagnostic_modes_with_train013_control": train_control_modes,
        "recommended_pr19_exact_mode": recommended,
        "warnings": warnings,
    }

    membership_fields = ["scene", "condition", "subset_name", "gaussian_id", "root_gaussian_id", "parent_gaussian_id", "is_alive_final", "included_mode_count"]
    membership_fields.extend(f"{mode}_included" for mode in support_modes)
    write_json(output_dir / "pr194_exact_support_filter_summary.json", summary)
    write_csv_rows(output_dir / "pr194_support_mode_comparison.csv", comparison_rows, MODE_COMPARISON_FIELDS)
    write_csv_rows(output_dir / "pr194_filtered_gaussian_support_by_mode.csv", filtered_rows, FILTERED_FIELDS)
    write_csv_rows(output_dir / "pr194_direct_collateral_overlap_by_mode.csv", direct_collateral_rows, DIRECT_COLLATERAL_FIELDS)
    write_csv_rows(output_dir / "pr194_train013_control_by_mode.csv", train013_rows, TRAIN013_FIELDS)
    write_csv_rows(output_dir / "pr194_gaussian_mode_membership.csv", list(membership.values()), membership_fields)
    write_csv_rows(output_dir / "pr194_view_group_event_concentration.csv", concentration_rows, CONCENTRATION_FIELDS)
    write_csv_rows(output_dir / "pr194_nontrivial_overlap_candidates.csv", nontrivial_rows, NONTRIVIAL_FIELDS)
    write_csv_rows(output_dir / "pr194_missing_inputs.csv", missing_rows, MISSING_FIELDS)
    _write_report(output_dir / "pr194_report.md", summary, comparison_rows, train013_rows)
    write_artifact_manifest(output_dir, pr193_dir)

    missing_required = [name for name in PR194_OUTPUT_FILES if not (output_dir / name).is_file()]
    if missing_required:
        raise RuntimeError(f"missing PR19.4 outputs: {missing_required}")
    if missing_rows and strict and not allow_missing:
        return summary, 1
    if missing_rows and not allow_missing:
        return summary, 1
    return summary, 0
