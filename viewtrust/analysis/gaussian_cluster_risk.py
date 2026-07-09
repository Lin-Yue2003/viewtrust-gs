"""Pure helpers for PR19 Gaussian cluster and event-cluster risk analysis."""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from viewtrust.analysis.clean_prior_normalization import load_csv_rows, load_json, normalize_bool
from viewtrust.analysis.offline_signals import positive_part, robust_z_scores, safe_float


PR19_OUTPUT_FILES = [
    "pr19_gaussian_cluster_risk_summary.json",
    "pr19_evidence_availability.csv",
    "pr19_view_group_map.csv",
    "pr19_cluster_risk_rows.csv",
    "pr19_cluster_risk_rankings.csv",
    "pr19_group_concentration_summary.csv",
    "pr19_direct_collateral_overlap.csv",
    "pr19_train013_control_summary.csv",
    "pr19_intervention_candidate_preview.csv",
    "pr19_missing_outputs.csv",
    "pr19_report.md",
    "artifact_manifest.csv",
]

GAUSSIAN_ID_KEYS = [
    "gaussian_id",
    "parent_id",
    "child_gaussian_id",
    "source_gaussian_id",
]
GAUSSIAN_ID_LIST_KEYS = ["gaussian_ids", "affected_gaussian_ids"]
INFLUENCE_FILES = [
    "view_lifecycle_attribution.csv",
    "view_iteration_events.csv",
    "view_influence.csv",
]
EXACT_LOG_FILES = [
    "gaussian_lifecycle_events.csv",
    "view_gaussian_event_attribution.csv",
    "gaussian_identity_table.csv",
    "gaussian_support_summary.csv",
]


def write_csv_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: "" if row.get(field) is None else row.get(field) for field in fieldnames})


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_pr17_rows(pr17_dir: Path) -> list[dict[str, str]]:
    return load_csv_rows(pr17_dir / "clean_prior_normalized_rows.csv")


def load_pr18_classification(pr18_dir: Path) -> list[dict[str, str]]:
    return load_csv_rows(pr18_dir / "pr18_spillover_classification.csv")


def load_pr18_condition_summary(pr18_dir: Path) -> list[dict[str, str]]:
    return load_csv_rows(pr18_dir / "pr18_condition_summary.csv")


def resolve_offline_artifact_inputs(signal_dir: Path) -> dict[str, Path]:
    output: dict[str, Path] = {}
    for row in load_csv_rows(signal_dir / "offline_viewtrust_artifact_manifest.csv"):
        raw_path = str(row.get("path", ""))
        if not raw_path:
            continue
        path = Path(raw_path)
        relative = str(row.get("relative_path", ""))
        group = str(row.get("artifact_group", ""))
        if group == "input_clean" or relative.startswith("input_clean/"):
            output.setdefault("input_clean_dir", path.parent)
        elif group == "input_corrupt" or relative.startswith("input_corrupt/"):
            output.setdefault("input_corrupt_dir", path.parent)
        elif group == "input_comparison" or relative.startswith("input_comparison/"):
            output.setdefault("input_comparison_dir", path.parent)
    return output


def _truth(value: Any) -> bool:
    return normalize_bool(value) is True


def _number(value: Any, default: float = 0.0) -> float:
    parsed = safe_float(value)
    return default if parsed is None else parsed


def _split_names(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    text = str(value or "")
    if not text:
        return []
    for delimiter in [",", "|"]:
        text = text.replace(delimiter, ";")
    return [item.strip() for item in text.split(";") if item.strip()]


def _source_view(row: dict[str, Any]) -> str:
    return str(row.get("source_view_name") or row.get("view_name") or "")


def _event_action(row: dict[str, Any]) -> str:
    return str(row.get("lifecycle_action") or row.get("event_type") or row.get("action") or "view_event")


def _event_weight(row: dict[str, Any]) -> float:
    for key in ("event_count", "birth_event_count", "clone_birth_count", "split_birth_count", "prune_death_count"):
        value = safe_float(row.get(key))
        if value is not None and value > 0:
            return value
    return 1.0


def _gaussian_ids_from_row(row: dict[str, Any]) -> list[str]:
    ids: list[str] = []
    for key in GAUSSIAN_ID_KEYS:
        value = str(row.get(key, "") or "")
        if value:
            ids.append(value)
    for key in GAUSSIAN_ID_LIST_KEYS:
        ids.extend(_split_names(row.get(key, "")))
    return sorted(dict.fromkeys(ids))


def inspect_gaussian_id_availability(paths: dict[str, Path]) -> dict[str, Any]:
    files_used: list[str] = []
    missing_files: list[str] = []
    exact = False
    aggregate = False
    warnings: list[str] = []
    for group_key in ("input_clean_dir", "input_corrupt_dir"):
        root = paths.get(group_key)
        if not root:
            missing_files.append(group_key)
            continue
        for path in _candidate_influence_paths(root):
            if not path.exists():
                continue
            rows = load_csv_rows(path)
            if rows:
                aggregate = True
                files_used.append(str(path))
            for row in rows:
                if _gaussian_ids_from_row(row):
                    exact = True
                    break
    comparison = paths.get("input_comparison_dir")
    if comparison and (comparison / "view_influence_comparison.csv").exists():
        aggregate = True
        files_used.append(str(comparison / "view_influence_comparison.csv"))
    elif not comparison:
        missing_files.append("input_comparison_dir")
    if not files_used:
        warnings.append("no influence files found")
    return {
        "exact_gaussian_ids_available": exact,
        "aggregate_event_proxy_available": aggregate,
        "files_used": sorted(dict.fromkeys(files_used)),
        "missing_files": sorted(dict.fromkeys(missing_files)),
        "warnings": warnings,
    }


def _candidate_influence_paths(root: Path) -> list[Path]:
    paths = [root / name for name in INFLUENCE_FILES]
    paths.extend(root / name for name in EXACT_LOG_FILES)
    paths.extend(root / "exact_gaussian_logging" / name for name in EXACT_LOG_FILES)
    return paths


def build_view_group_map(
    *,
    pr17_rows: list[dict[str, Any]],
    pr18_rows: list[dict[str, Any]],
    scene: str,
    subset_name: str,
    condition: str,
    candidate_only: bool = False,
) -> list[dict[str, Any]]:
    pr18_by_view = {
        str(row.get("view_name", "")): row
        for row in pr18_rows
        if row.get("scene") == scene and row.get("subset_name") == subset_name and row.get("condition") == condition
    }
    output: list[dict[str, Any]] = []
    for row in pr17_rows:
        if row.get("scene") != scene or row.get("subset_name") != subset_name or row.get("condition") != condition:
            continue
        view_name = str(row.get("view_name", ""))
        spillover = pr18_by_view.get(view_name, {})
        spillover_class = str(spillover.get("spillover_class", ""))
        if _truth(row.get("was_corrupted")):
            group = "direct_corrupted"
        elif spillover_class == "co_visible_collateral":
            group = "co_visible_collateral"
        elif spillover_class in {"clean_prior_false_positive", "prior_demoted"} or (
            view_name == "train_013" and _truth(row.get("raw_false_positive")) and not _truth(row.get("normalized_false_positive"))
        ):
            group = "clean_prior_demoted"
        else:
            group = "other_clean"
        included = (not candidate_only) or group != "other_clean"
        output.append(
            {
                "scene": scene,
                "subset_name": subset_name,
                "condition": condition,
                "view_name": view_name,
                "view_group": group,
                "was_corrupted": row.get("was_corrupted", ""),
                "pr17_raw_rank": row.get("raw_rank", ""),
                "pr17_normalized_rank": row.get("normalized_rank", ""),
                "pr18_spillover_class": spillover_class,
                "pr18_spillover_confidence": spillover.get("spillover_confidence", ""),
                "camera_neighbor_evidence": spillover.get("camera_neighbor_evidence", ""),
                "index_neighbor_evidence": spillover.get("index_neighbor_evidence", ""),
                "gaussian_overlap_evidence": spillover.get("gaussian_overlap_evidence", ""),
                "included_in_candidate_analysis": included,
            }
        )
    return output


def _view_group_lookup(rows: list[dict[str, Any]]) -> dict[str, str]:
    return {str(row.get("view_name", "")): str(row.get("view_group", "other_clean")) for row in rows}


def _read_influence_rows(paths: dict[str, Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group_key in ("input_clean_dir", "input_corrupt_dir"):
        root = paths.get(group_key)
        if not root:
            continue
        for path in _candidate_influence_paths(root):
            for row in load_csv_rows(path):
                rows.append({"_source_file": path.name, "_source_group": group_key, **row})
    comparison = paths.get("input_comparison_dir")
    if comparison:
        for row in load_csv_rows(comparison / "view_influence_comparison.csv"):
            rows.append({"_source_file": "view_influence_comparison.csv", "_source_group": "input_comparison", **row})
    return rows


def build_exact_gaussian_support_sets(
    *,
    scene: str,
    subset_name: str,
    condition: str,
    paths: dict[str, Path],
    view_group_rows: list[dict[str, Any]],
    min_cluster_support: int,
) -> list[dict[str, Any]]:
    groups = _view_group_lookup(view_group_rows)
    clusters: dict[str, dict[str, Any]] = {}
    for row in _read_influence_rows(paths):
        ids = _gaussian_ids_from_row(row)
        if not ids:
            continue
        source_view = _source_view(row)
        if source_view and source_view not in groups:
            continue
        for gaussian_id in ids:
            cluster = clusters.setdefault(
                f"gaussian:{gaussian_id}",
                _empty_cluster(scene, subset_name, condition, "exact_gaussian_id", f"gaussian:{gaussian_id}", "gaussian_id"),
            )
            _add_event_to_cluster(cluster, row, source_view, groups.get(source_view, "other_clean"), unique_gaussian_id=gaussian_id)
    return [row for row in clusters.values() if row["_event_weight_total"] >= min_cluster_support]


def build_aggregate_event_clusters(
    *,
    scene: str,
    subset_name: str,
    condition: str,
    paths: dict[str, Path],
    view_group_rows: list[dict[str, Any]],
    iteration_bucket_size: int,
    min_cluster_support: int,
) -> list[dict[str, Any]]:
    groups = _view_group_lookup(view_group_rows)
    clusters: dict[str, dict[str, Any]] = {}
    for row in _read_influence_rows(paths):
        source_view = _source_view(row)
        if source_view and source_view not in groups:
            continue
        action = _event_action(row)
        iteration_bucket = _iteration_bucket(row, iteration_bucket_size)
        delta_bucket = _delta_bucket(row)
        cluster_id = f"event:{action}:{iteration_bucket}:{delta_bucket}"
        cluster = clusters.setdefault(
            cluster_id,
            _empty_cluster(scene, subset_name, condition, "aggregate_event_proxy", cluster_id, "event_cluster"),
        )
        cluster["lifecycle_action"] = action
        cluster["iteration_bucket"] = iteration_bucket
        _add_event_to_cluster(cluster, row, source_view, groups.get(source_view, "other_clean"), unique_gaussian_id="")
    return [row for row in clusters.values() if row["_event_weight_total"] >= min_cluster_support]


def _empty_cluster(
    scene: str,
    subset_name: str,
    condition: str,
    evidence_level: str,
    cluster_id: str,
    cluster_kind: str,
) -> dict[str, Any]:
    return {
        "scene": scene,
        "subset_name": subset_name,
        "condition": condition,
        "evidence_level": evidence_level,
        "cluster_id": cluster_id,
        "cluster_kind": cluster_kind,
        "lifecycle_action": "",
        "iteration_bucket": "",
        "source_view_group": "",
        "_source_view_names": set(),
        "_source_groups": Counter(),
        "_source_weight_by_view": Counter(),
        "_gaussian_ids": set(),
        "_event_weight_total": 0.0,
        "event_count_total": 0.0,
        "unique_gaussian_count_total": 0.0,
        "final_alive_count_total": 0.0,
        "final_dead_count_total": 0.0,
        "clone_birth_count_total": 0.0,
        "split_birth_count_total": 0.0,
        "prune_death_count_total": 0.0,
        "_visibility_delta_values": [],
        "_clean_delta_values": [],
        "warnings": "",
    }


def _add_event_to_cluster(
    cluster: dict[str, Any],
    row: dict[str, Any],
    source_view: str,
    view_group: str,
    *,
    unique_gaussian_id: str,
) -> None:
    weight = _event_weight(row)
    if source_view:
        cluster["_source_view_names"].add(source_view)
        cluster["_source_weight_by_view"][source_view] += weight
    cluster["_source_groups"][view_group] += weight
    if unique_gaussian_id:
        cluster["_gaussian_ids"].add(unique_gaussian_id)
    cluster["_event_weight_total"] += weight
    cluster["event_count_total"] += weight
    cluster["unique_gaussian_count_total"] += _number(row.get("unique_gaussian_count"), 1.0 if unique_gaussian_id else 0.0)
    cluster["final_alive_count_total"] += _number(row.get("final_alive_count"))
    cluster["final_dead_count_total"] += _number(row.get("final_dead_count"))
    cluster["clone_birth_count_total"] += _number(row.get("clone_birth_count"))
    cluster["split_birth_count_total"] += _number(row.get("split_birth_count"))
    cluster["prune_death_count_total"] += _number(row.get("prune_death_count"))
    action = _event_action(row)
    if action and not cluster.get("lifecycle_action"):
        cluster["lifecycle_action"] = action
    if not cluster.get("iteration_bucket"):
        cluster["iteration_bucket"] = _iteration_bucket(row, 100)
    for key in ("visibility_ratio_delta", "visibility_delta", "mean_visibility_ratio"):
        if safe_float(row.get(key)) is not None:
            cluster["_visibility_delta_values"].append(abs(_number(row.get(key))))
    for key in ("birth_event_count_delta", "prune_death_count_delta", "birth_survival_ratio_delta"):
        if safe_float(row.get(key)) is not None:
            cluster["_clean_delta_values"].append(abs(_number(row.get(key))))


def _iteration_bucket(row: dict[str, Any], bucket_size: int) -> str:
    value = safe_float(row.get("source_iteration") or row.get("iteration"))
    if value is None:
        return "all"
    start = int(value // max(bucket_size, 1)) * max(bucket_size, 1)
    return f"{start}-{start + max(bucket_size, 1) - 1}"


def _delta_bucket(row: dict[str, Any]) -> str:
    values = [
        abs(_number(row.get(key)))
        for key in ("birth_event_count_delta", "prune_death_count_delta", "visibility_ratio_delta")
        if safe_float(row.get(key)) is not None
    ]
    if not values:
        return "no_delta"
    value = max(values)
    if value >= 10:
        return "large_delta"
    if value >= 1:
        return "medium_delta"
    return "small_delta"


def compute_source_concentration(group_counts: Counter[str]) -> tuple[float, float]:
    total = sum(group_counts.values())
    if total <= 0:
        return 0.0, 0.0
    probabilities = [count / total for count in group_counts.values() if count > 0]
    entropy = -sum(p * math.log(p) for p in probabilities)
    max_entropy = math.log(len(probabilities)) if len(probabilities) > 1 else 1.0
    concentration = 1.0 - (entropy / max_entropy if max_entropy else 0.0)
    return entropy, max(0.0, min(1.0, concentration))


def compute_cluster_risk_components(rows: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    normalized_rows = [_finalize_cluster(row) for row in rows]
    by_condition: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in normalized_rows:
        by_condition[(row["scene"], row["subset_name"], row["condition"])].append(row)
    weights = config.get("risk_weights", {})
    for group_rows in by_condition.values():
        lifecycle_scores = _positive_robust_scores([row["_lifecycle_instability_raw"] for row in group_rows], config)
        weak_scores = _positive_robust_scores([row["_weak_support_raw"] for row in group_rows], config)
        visibility_scores = _positive_robust_scores([row["_visibility_delta_raw"] for row in group_rows], config)
        clean_delta_scores = _positive_robust_scores([row["_clean_vs_corrupt_delta_raw"] for row in group_rows], config)
        for index, row in enumerate(group_rows):
            row["lifecycle_instability_score"] = lifecycle_scores[index]
            row["weak_support_score"] = weak_scores[index]
            row["visibility_delta_score"] = visibility_scores[index]
            row["clean_vs_corrupt_delta_score"] = clean_delta_scores[index]
            proxy_delta_or_weak = (
                row["clean_vs_corrupt_delta_score"]
                if row["evidence_level"] == "aggregate_event_proxy"
                else row["weak_support_score"]
            )
            row["gaussian_cluster_risk"] = (
                _number(weights.get("source_concentration"), 0.30) * row["source_concentration_score"]
                + _number(weights.get("corrupted_plus_collateral_ratio"), 0.25) * row["corrupted_plus_collateral_ratio"]
                + _number(weights.get("lifecycle_instability"), 0.20) * row["lifecycle_instability_score"]
                + _number(weights.get("weak_support"), 0.15) * proxy_delta_or_weak
                + _number(weights.get("visibility_delta"), 0.10) * row["visibility_delta_score"]
            )
    return normalized_rows


def _positive_robust_scores(values: list[Any], config: dict[str, Any]) -> list[float]:
    normalization = config.get("normalization", {})
    eps = _number(normalization.get("eps"), 1e-8)
    mad_scale = _number(normalization.get("mad_scale"), 1.4826)
    return positive_part(robust_z_scores(values, eps=eps, mad_scale=mad_scale))


def _mean(values: list[Any]) -> float | None:
    numbers = [value for value in (safe_float(item) for item in values) if value is not None]
    return statistics.fmean(numbers) if numbers else None


def _finalize_cluster(row: dict[str, Any]) -> dict[str, Any]:
    group_counts: Counter[str] = row.pop("_source_groups")
    source_views = sorted(row.pop("_source_view_names"))
    source_weights = row.pop("_source_weight_by_view")
    gaussian_ids = row.pop("_gaussian_ids")
    visibility = row.pop("_visibility_delta_values")
    clean_delta = row.pop("_clean_delta_values")
    entropy, concentration = compute_source_concentration(group_counts)
    total = sum(group_counts.values()) or 1.0
    event_total = _number(row.get("event_count_total"), 0.0)
    birth_total = _number(row.get("clone_birth_count_total")) + _number(row.get("split_birth_count_total"))
    prune_total = _number(row.get("prune_death_count_total"))
    dead_total = _number(row.get("final_dead_count_total"))
    row.update(
        {
            "source_view_names": ";".join(source_views),
            "source_view_count": len(source_views),
            "direct_corrupted_source_count": group_counts.get("direct_corrupted", 0.0),
            "collateral_source_count": group_counts.get("co_visible_collateral", 0.0),
            "clean_prior_source_count": group_counts.get("clean_prior_demoted", 0.0),
            "other_clean_source_count": group_counts.get("other_clean", 0.0),
            "source_entropy": entropy,
            "source_concentration_score": concentration,
            "corrupted_plus_collateral_ratio": (
                group_counts.get("direct_corrupted", 0.0) + group_counts.get("co_visible_collateral", 0.0)
            )
            / total,
            "clean_prior_ratio": group_counts.get("clean_prior_demoted", 0.0) / total,
            "birth_ratio": birth_total / event_total if event_total else 0.0,
            "prune_ratio": prune_total / event_total if event_total else 0.0,
            "death_ratio": dead_total / event_total if event_total else 0.0,
            "source_view_group": _mixed_group_label(group_counts),
            "_lifecycle_instability_raw": (birth_total + prune_total + dead_total) / event_total if event_total else 0.0,
            "_weak_support_raw": 1.0 / max(len(source_views), 1),
            "_visibility_delta_raw": _mean(visibility) or 0.0,
            "_clean_vs_corrupt_delta_raw": _mean(clean_delta) or 0.0,
        }
    )
    if gaussian_ids:
        row["unique_gaussian_count_total"] = len(gaussian_ids)
    return row


def _mixed_group_label(group_counts: Counter[str]) -> str:
    present = [group for group, count in sorted(group_counts.items()) if count > 0]
    return present[0] if len(present) == 1 else "mixed"


def rank_cluster_risks(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked: list[dict[str, Any]] = []
    by_condition: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_condition[(row["scene"], row["subset_name"], row["condition"])].append(row)
    for group_rows in by_condition.values():
        ordered = sorted(
            group_rows,
            key=lambda row: (_number(row.get("gaussian_cluster_risk")), str(row.get("cluster_id", ""))),
            reverse=True,
        )
        for rank, row in enumerate(ordered, start=1):
            row["rank"] = rank
            ranked.append(row)
    return ranked


def compute_group_overlap_summary(rows: list[dict[str, Any]], view_group_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    clusters_by_view: dict[str, set[str]] = defaultdict(set)
    risk_by_cluster = {str(row.get("cluster_id")): _number(row.get("gaussian_cluster_risk")) for row in rows}
    for row in rows:
        for view in _split_names(row.get("source_view_names", "")):
            clusters_by_view[view].add(str(row.get("cluster_id", "")))
    direct = [row["view_name"] for row in view_group_rows if row.get("view_group") == "direct_corrupted"]
    collateral = [row["view_name"] for row in view_group_rows if row.get("view_group") == "co_visible_collateral"]
    output: list[dict[str, Any]] = []
    for direct_view in direct:
        for collateral_view in collateral:
            direct_clusters = clusters_by_view.get(direct_view, set())
            collateral_clusters = clusters_by_view.get(collateral_view, set())
            shared = direct_clusters & collateral_clusters
            union = direct_clusters | collateral_clusters
            output.append(
                {
                    "scene": view_group_rows[0].get("scene", "") if view_group_rows else "",
                    "subset_name": view_group_rows[0].get("subset_name", "") if view_group_rows else "",
                    "condition": view_group_rows[0].get("condition", "") if view_group_rows else "",
                    "direct_view": direct_view,
                    "collateral_view": collateral_view,
                    "overlap_type": "shared_cluster",
                    "shared_cluster_count": len(shared),
                    "direct_cluster_count": len(direct_clusters),
                    "collateral_cluster_count": len(collateral_clusters),
                    "overlap_ratio": len(shared) / len(union) if union else 0.0,
                    "mean_shared_cluster_risk": _mean([risk_by_cluster[item] for item in shared]) or 0.0,
                    "evidence_level": rows[0].get("evidence_level", "") if rows else "unavailable",
                    "supported": bool(shared),
                    "warnings": "" if shared else "no shared clusters",
                }
            )
    return output


def compute_train013_control_summary(rows: list[dict[str, Any]], view_group_rows: list[dict[str, Any]], top_k: int) -> dict[str, Any]:
    train_row = next((row for row in view_group_rows if row.get("view_name") == "train_013"), {})
    train_clusters = [row for row in rows if "train_013" in _split_names(row.get("source_view_names", ""))]
    direct_collateral_clusters = [
        row
        for row in rows
        if row.get("corrupted_plus_collateral_ratio", 0.0) and _number(row.get("corrupted_plus_collateral_ratio")) >= 0.5
    ]
    train_ids = {row["cluster_id"] for row in train_clusters}
    direct_ids = {row["cluster_id"] for row in direct_collateral_clusters}
    overlap = train_ids & direct_ids
    high_train = [row for row in train_clusters if int(row.get("rank", 10**9) or 10**9) <= top_k]
    supported = bool(train_row) and len(overlap) == 0 and len(high_train) <= max(1, len(train_clusters) // 3)
    return {
        "scene": view_group_rows[0].get("scene", "") if view_group_rows else "",
        "subset_name": view_group_rows[0].get("subset_name", "") if view_group_rows else "",
        "condition": view_group_rows[0].get("condition", "") if view_group_rows else "",
        "train013_present": bool(train_row),
        "train013_view_group": train_row.get("view_group", ""),
        "train013_cluster_count": len(train_clusters),
        "train013_high_risk_cluster_count": len(high_train),
        "train013_mean_cluster_risk": _mean([row.get("gaussian_cluster_risk") for row in train_clusters]) or 0.0,
        "train013_clean_prior_demoted": train_row.get("view_group") == "clean_prior_demoted",
        "train013_low_overlap_with_direct_collateral": len(overlap) == 0,
        "control_supported": supported,
        "interpretation": "train_013 remains separated from direct/collateral high-risk clusters"
        if supported
        else "train_013 overlaps with risky clusters or is unavailable",
    }


def compute_condition_summary_rows(rows: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    by_condition: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_condition[(row["scene"], row["subset_name"], row["condition"])].append(row)
    output = []
    for (scene, subset_name, condition), condition_rows in sorted(by_condition.items()):
        top = sorted(condition_rows, key=lambda row: int(row.get("rank", 10**9)))[:top_k]
        sources = Counter()
        for row in top:
            sources["direct_corrupted"] += _number(row.get("direct_corrupted_source_count"))
            sources["co_visible_collateral"] += _number(row.get("collateral_source_count"))
            sources["clean_prior_demoted"] += _number(row.get("clean_prior_source_count"))
            sources["other_clean"] += _number(row.get("other_clean_source_count"))
        mean_ratio = _mean([row.get("corrupted_plus_collateral_ratio") for row in top]) or 0.0
        output.append(
            {
                "scene": scene,
                "subset_name": subset_name,
                "condition": condition,
                "evidence_level": top[0].get("evidence_level", "") if top else "unavailable",
                "top_k": top_k,
                "topk_mean_corrupted_plus_collateral_ratio": mean_ratio,
                "topk_mean_clean_prior_ratio": _mean([row.get("clean_prior_ratio") for row in top]) or 0.0,
                "topk_mean_source_concentration": _mean([row.get("source_concentration_score") for row in top]) or 0.0,
                "topk_direct_corrupted_view_count": sources["direct_corrupted"],
                "topk_collateral_view_count": sources["co_visible_collateral"],
                "topk_clean_prior_view_count": sources["clean_prior_demoted"],
                "topk_other_clean_view_count": sources["other_clean"],
                "support_concentration_status": "direct_collateral_concentrated" if mean_ratio >= 0.5 else "mixed_or_weak",
                "interpretation": "top clusters are dominated by direct corrupted plus collateral sources"
                if mean_ratio >= 0.5
                else "top clusters are not dominated by direct/collateral sources",
            }
        )
    return output


def build_intervention_preview(rows: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    output = []
    for row in sorted(rows, key=lambda item: int(item.get("rank", 10**9)))[:top_k]:
        exact = row.get("evidence_level") == "exact_gaussian_id"
        action = "not_actionable_proxy_only"
        if exact and _number(row.get("prune_ratio")) > 0:
            action = "candidate_for_future_densification_delay"
        elif exact and _number(row.get("visibility_delta_score")) > 0:
            action = "candidate_for_future_opacity_update_suppression"
        elif exact:
            action = "candidate_for_further_exact_gaussian_logging"
        output.append(
            {
                "scene": row.get("scene", ""),
                "subset_name": row.get("subset_name", ""),
                "condition": row.get("condition", ""),
                "cluster_id": row.get("cluster_id", ""),
                "evidence_level": row.get("evidence_level", ""),
                "gaussian_cluster_risk": row.get("gaussian_cluster_risk", ""),
                "candidate_reason": _main_reason(row),
                "source_view_names": row.get("source_view_names", ""),
                "source_view_groups": row.get("source_view_group", ""),
                "corrupted_plus_collateral_ratio": row.get("corrupted_plus_collateral_ratio", ""),
                "clean_prior_ratio": row.get("clean_prior_ratio", ""),
                "weak_support_score": row.get("weak_support_score", ""),
                "lifecycle_instability_score": row.get("lifecycle_instability_score", ""),
                "suggested_future_action": action,
                "do_not_apply_intervention": True,
            }
        )
    return output


def _main_reason(row: dict[str, Any]) -> str:
    reasons = []
    if _number(row.get("corrupted_plus_collateral_ratio")) >= 0.5:
        reasons.append("direct_collateral_concentration")
    if _number(row.get("lifecycle_instability_score")) > 0:
        reasons.append("lifecycle_instability")
    if _number(row.get("weak_support_score")) > 0:
        reasons.append("weak_support")
    if _number(row.get("visibility_delta_score")) > 0:
        reasons.append("visibility_delta")
    return ";".join(reasons) or "ranked_cluster"


def compute_summary(
    *,
    scenes: list[str],
    conditions: list[str],
    subset_names: list[str],
    top_k: int,
    evidence_rows: list[dict[str, Any]],
    cluster_rows: list[dict[str, Any]],
    group_rows: list[dict[str, Any]],
    train013_rows: list[dict[str, Any]],
    intervention_rows: list[dict[str, Any]],
    missing_rows: list[dict[str, Any]],
    warnings: list[str],
) -> dict[str, Any]:
    top_rows = [row for row in cluster_rows if int(row.get("rank", 10**9) or 10**9) <= top_k]
    evidence_counts = Counter(str(row.get("evidence_level", "unavailable")) for row in evidence_rows)
    return {
        "schema_name": "viewtrust.pr19.gaussian_cluster_risk.summary",
        "schema_version": 1,
        "scenes": scenes,
        "conditions": conditions,
        "subset_names": subset_names,
        "top_k": top_k,
        "valid_condition_count": sum(1 for row in evidence_rows if row.get("evidence_level") != "unavailable"),
        "missing_condition_count": len(missing_rows),
        "exact_gaussian_condition_count": evidence_counts["exact_gaussian_id"],
        "aggregate_proxy_condition_count": evidence_counts["aggregate_event_proxy"],
        "unavailable_condition_count": evidence_counts["unavailable"],
        "high_risk_cluster_count": len(top_rows),
        "mean_corrupted_plus_collateral_ratio_topk": _mean([row.get("corrupted_plus_collateral_ratio") for row in top_rows]),
        "mean_clean_prior_ratio_topk": _mean([row.get("clean_prior_ratio") for row in top_rows]),
        "mean_source_concentration_topk": _mean([row.get("source_concentration_score") for row in top_rows]),
        "train013_control_supported": any(_truth(row.get("control_supported")) for row in train013_rows),
        "direct_collateral_overlap_supported": any(_truth(row.get("support_concentration_status") == "direct_collateral_concentrated") for row in group_rows),
        "intervention_candidate_count": len(intervention_rows),
        "evidence_level_counts": dict(evidence_counts),
        "uses_corruption_labels_for_scoring": False,
        "uses_corruption_labels_for_evaluation": True,
        "observation_only": True,
        "training_intervention": False,
        "defense_enabled": False,
        "warnings": warnings,
    }


def write_artifact_manifest(output_dir: Path, input_root: Path, plan_dir: Path, pr17_dir: Path, pr18_dir: Path) -> None:
    fields = ["relative_path", "path", "exists", "file_type", "size_bytes", "required", "artifact_group"]

    def rows() -> list[dict[str, Any]]:
        items: list[tuple[str, Path, bool, str]] = [
            ("input_root", input_root, False, "input"),
            ("plan_dir", plan_dir, True, "input"),
            ("pr17_dir", pr17_dir, True, "input"),
            ("pr18_dir", pr18_dir, True, "input"),
        ]
        items.extend((name, output_dir / name, True, "output_pr19") for name in PR19_OUTPUT_FILES)
        output = []
        for relative, path, required, group in items:
            output.append(
                {
                    "relative_path": relative,
                    "path": str(path),
                    "exists": str(path.exists()).lower(),
                    "file_type": "directory" if path.is_dir() else path.suffix.lstrip("."),
                    "size_bytes": path.stat().st_size if path.is_file() else "",
                    "required": str(required).lower(),
                    "artifact_group": group,
                }
            )
        return output

    manifest = output_dir / "artifact_manifest.csv"
    write_csv_rows(manifest, rows(), fields)
    write_csv_rows(manifest, rows(), fields)
