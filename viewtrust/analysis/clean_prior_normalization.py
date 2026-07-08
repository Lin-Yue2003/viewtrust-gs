"""Offline clean-prior normalization helpers for PR17."""

from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from viewtrust.analysis.offline_signals import positive_part, robust_z_scores, safe_float


REQUIRED_OFFLINE_FILES = [
    "offline_viewtrust_summary.json",
    "offline_viewtrust_rankings.csv",
    "offline_viewtrust_signals.csv",
    "offline_viewtrust_signal_ablation.csv",
    "offline_viewtrust_group_metrics.csv",
    "offline_viewtrust_config.json",
    "offline_viewtrust_report.md",
    "offline_viewtrust_artifact_manifest.csv",
]

PR17_OUTPUT_FILES = [
    "clean_prior_normalized_summary.json",
    "clean_prior_normalized_rows.csv",
    "clean_prior_normalized_rankings.csv",
    "clean_prior_normalized_group_metrics.csv",
    "clean_prior_normalized_ablation.csv",
    "clean_prior_false_positive_reduction.csv",
    "clean_prior_view_identity_diagnosis.csv",
    "clean_prior_component_comparison.csv",
    "clean_prior_missing_outputs.csv",
    "clean_prior_report.md",
    "artifact_manifest.csv",
]

SCORE_NAMES = [
    "raw_risk",
    "clean_prior_risk",
    "positive_delta_risk",
    "prior_suppressed_risk",
    "rank_lift_score",
    "normalized_viewtrust_risk",
]

PRIOR_WEIGHTS = {
    "loss_component": 0.2,
    "visibility_component": 0.15,
    "birth_component": 0.2,
    "prune_component": 0.25,
    "survival_component": 0.1,
}


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: "" if row.get(field) is None else row.get(field) for field in fieldnames})


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def normalize_bool(value: Any) -> bool | None:
    if value in ("", None):
        return None
    text = str(value).strip().lower()
    if value is True or text in {"true", "1", "yes"}:
        return True
    if value is False or text in {"false", "0", "no"}:
        return False
    return None


def _mean(values: list[Any]) -> float | None:
    numbers = [number for number in (safe_float(value) for value in values) if number is not None]
    return statistics.fmean(numbers) if numbers else None


def _median(values: list[Any]) -> float | None:
    numbers = [number for number in (safe_float(value) for value in values) if number is not None]
    return statistics.median(numbers) if numbers else None


def _view_name(row: dict[str, Any]) -> str:
    return str(row.get("view_name") or row.get("source_view_name") or "")


def _rank(rows: list[dict[str, Any]], score_key: str, rank_key: str) -> list[dict[str, Any]]:
    ranked = sorted(rows, key=lambda row: (safe_float(row.get(score_key)) or 0.0, _view_name(row)), reverse=True)
    for index, row in enumerate(ranked, start=1):
        row[rank_key] = index
    return ranked


def _is_corrupted(row: dict[str, Any]) -> bool:
    return normalize_bool(row.get("was_corrupted")) is True


def resolve_artifact_manifest_paths(signal_dir: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for row in load_csv_rows(signal_dir / "offline_viewtrust_artifact_manifest.csv"):
        relative = str(row.get("relative_path", ""))
        group = str(row.get("artifact_group", ""))
        raw_path = str(row.get("path", ""))
        if not raw_path:
            continue
        path = Path(raw_path)
        if group == "input_clean" or relative.startswith("input_clean/"):
            paths.setdefault("input_clean_dir", path.parent)
        elif group == "input_corrupt" or relative.startswith("input_corrupt/"):
            paths.setdefault("input_corrupt_dir", path.parent)
        elif group == "input_comparison" or relative.startswith("input_comparison/"):
            paths.setdefault("input_comparison_dir", path.parent)
    return paths


def load_offline_signal_dir(signal_dir: Path) -> dict[str, Any]:
    missing = [name for name in REQUIRED_OFFLINE_FILES if not (signal_dir / name).is_file()]
    summary = load_json(signal_dir / "offline_viewtrust_summary.json")
    return {
        "signal_dir": signal_dir,
        "summary": summary,
        "signals": load_csv_rows(signal_dir / "offline_viewtrust_signals.csv"),
        "rankings": load_csv_rows(signal_dir / "offline_viewtrust_rankings.csv"),
        "artifact_paths": resolve_artifact_manifest_paths(signal_dir),
        "missing_files": missing,
    }


def _positive_z(values: list[Any], config: dict[str, Any]) -> list[float]:
    normalization = config.get("normalization", {})
    eps = safe_float(normalization.get("eps")) or 1e-8
    mad_scale = safe_float(normalization.get("mad_scale")) or 1.4826
    return positive_part(robust_z_scores(values, eps=eps, mad_scale=mad_scale))


def _clean_rows_from_manifest(offline: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], str]:
    clean_dir = offline.get("artifact_paths", {}).get("input_clean_dir")
    if clean_dir:
        rows = load_csv_rows(clean_dir / "view_influence.csv")
        if rows:
            return {row.get("view_name", ""): row for row in rows if row.get("view_name")}, "manifest_inputs"
    return {}, ""


def _clean_feature(row: dict[str, Any], clean_row: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        if clean_row.get(key) not in ("", None):
            return clean_row.get(key)
        if row.get(key) not in ("", None):
            return row.get(key)
    return ""


def compute_clean_prior_rows(offline: dict[str, Any], config: dict[str, Any]) -> list[dict[str, Any]]:
    signal_rows = offline.get("signals", [])
    clean_by_view, manifest_source = _clean_rows_from_manifest(offline)
    prepared: list[dict[str, Any]] = []
    for row in signal_rows:
        view_name = _view_name(row)
        clean = clean_by_view.get(view_name, {})
        source = manifest_source if clean else "clean_columns"
        loss = _clean_feature(row, clean, ["mean_total_loss", "mean_loss", "clean_mean_loss", "clean_mean_total_loss"])
        visibility = _clean_feature(row, clean, ["mean_visibility_ratio", "clean_mean_visibility_ratio"])
        birth = _clean_feature(row, clean, ["birth_event_count_after_view", "clean_birth_event_count_after_view"])
        prune = _clean_feature(row, clean, ["prune_death_count_after_view", "clean_prune_death_count_after_view"])
        survival = _clean_feature(row, clean, ["birth_survival_ratio_after_view", "clean_birth_survival_ratio_after_view"])
        prepared.append(
            {
                "view_name": view_name,
                "loss_value": loss,
                "visibility_value": visibility,
                "birth_value": birth,
                "prune_value": prune,
                "survival_anomaly_value": 1.0 - (safe_float(survival) if safe_float(survival) is not None else 1.0),
                "prior_source": source,
            }
        )

    loss_scores = _positive_z([row["loss_value"] for row in prepared], config)
    visibility_scores = _positive_z([row["visibility_value"] for row in prepared], config)
    birth_scores = _positive_z([row["birth_value"] for row in prepared], config)
    prune_scores = _positive_z([row["prune_value"] for row in prepared], config)
    survival_scores = _positive_z([row["survival_anomaly_value"] for row in prepared], config)
    for index, row in enumerate(prepared):
        components = {
            "loss_component": loss_scores[index],
            "visibility_component": visibility_scores[index],
            "birth_component": birth_scores[index],
            "prune_component": prune_scores[index],
            "survival_component": survival_scores[index],
        }
        row["clean_prior_risk"] = sum(PRIOR_WEIGHTS[name] * value for name, value in components.items())
        row.update({f"clean_prior_{name}": value for name, value in components.items()})
        warnings = []
        if row["prior_source"] == "clean_columns" and not any(row.get(key) not in ("", None) for key in ("loss_value", "birth_value", "prune_value")):
            warnings.append("limited_clean_prior_features")
        row["component_warnings"] = ";".join(warnings)
    return prepared


def compute_normalized_rows(
    *,
    offline: dict[str, Any],
    scene: str,
    subset_name: str,
    condition: str,
    subset_seed: str,
    config: dict[str, Any],
    top_k: int,
    raw_score_key: str,
) -> list[dict[str, Any]]:
    prior_by_view = {row["view_name"]: row for row in compute_clean_prior_rows(offline, config)}
    rows: list[dict[str, Any]] = []
    for source in offline.get("signals", []):
        view_name = _view_name(source)
        prior = prior_by_view.get(view_name, {})
        raw_risk = safe_float(source.get(raw_score_key)) or 0.0
        clean_prior = safe_float(prior.get("clean_prior_risk")) or 0.0
        delta = raw_risk - clean_prior
        weights = config.get("normalized_score", {})
        row = {
            "scene": scene,
            "subset_name": subset_name,
            "subset_seed": subset_seed,
            "condition": condition,
            "view_name": view_name,
            "view_split": source.get("view_split", ""),
            "was_corrupted": source.get("was_corrupted", ""),
            "raw_risk": raw_risk,
            "clean_prior_risk": clean_prior,
            "delta_risk": delta,
            "positive_delta_risk": max(0.0, delta),
            "prior_suppressed_risk": raw_risk / (1.0 + clean_prior),
            "prior_source": prior.get("prior_source", "clean_columns"),
            "component_warnings": prior.get("component_warnings", ""),
        }
        rows.append(row)

    _rank(rows, "raw_risk", "raw_rank")
    _rank(rows, "clean_prior_risk", "clean_prior_rank")
    view_count = len(rows)
    for row in rows:
        rank_lift = (safe_float(row.get("clean_prior_rank")) or 0.0) - (safe_float(row.get("raw_rank")) or 0.0)
        row["rank_lift_score"] = rank_lift
        row["normalized_viewtrust_risk"] = (
            (safe_float(weights.get("positive_delta_weight")) or 1.0) * row["positive_delta_risk"]
            + (safe_float(weights.get("prior_suppressed_weight")) or 0.25) * row["prior_suppressed_risk"]
            + (safe_float(weights.get("rank_lift_weight")) or 0.10) * max(0.0, rank_lift) / max(view_count - 1, 1)
        )
        row["normalized_consistency"] = 1.0 / (1.0 + row["normalized_viewtrust_risk"])
    _rank(rows, "normalized_viewtrust_risk", "normalized_rank")
    for row in rows:
        row["raw_top_k"] = int(row["raw_rank"]) <= top_k
        row["normalized_top_k"] = int(row["normalized_rank"]) <= top_k
        row["raw_false_positive"] = row["raw_top_k"] and not _is_corrupted(row)
        row["normalized_false_positive"] = row["normalized_top_k"] and not _is_corrupted(row)
    return sorted(rows, key=lambda row: (str(row["scene"]), str(row["subset_name"]), str(row["condition"]), int(row["normalized_rank"])))


def rank_normalized_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranking_rows: list[dict[str, Any]] = []
    keys = {
        "raw_risk": "raw_risk",
        "clean_prior_risk": "clean_prior_risk",
        "positive_delta_risk": "positive_delta_risk",
        "prior_suppressed_risk": "prior_suppressed_risk",
        "rank_lift_score": "rank_lift_score",
        "normalized_viewtrust_risk": "normalized_viewtrust_risk",
    }
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["scene"], row["subset_name"], row["condition"])].append(row)
    for (scene, subset_name, condition), group_rows in grouped.items():
        for score_name, score_key in keys.items():
            for rank, row in enumerate(_rank([dict(item) for item in group_rows], score_key, "_tmp_rank"), start=1):
                ranking_rows.append(
                    {
                        "scene": scene,
                        "subset_name": subset_name,
                        "subset_seed": row.get("subset_seed", ""),
                        "condition": condition,
                        "score_name": score_name,
                        "rank": rank,
                        "view_name": row.get("view_name", ""),
                        "was_corrupted": row.get("was_corrupted", ""),
                        "score": row.get(score_key, ""),
                        "raw_risk": row.get("raw_risk", ""),
                        "clean_prior_risk": row.get("clean_prior_risk", ""),
                        "delta_risk": row.get("delta_risk", ""),
                        "rank_lift_score": row.get("rank_lift_score", ""),
                        "prior_source": row.get("prior_source", ""),
                    }
                )
    return ranking_rows


def _metric_for_score(rows: list[dict[str, Any]], score_key: str, top_k: int) -> dict[str, Any]:
    ranked = sorted(rows, key=lambda row: (safe_float(row.get(score_key)) or 0.0, row.get("view_name", "")), reverse=True)
    top = ranked[: min(top_k, len(ranked))]
    corrupted = [row for row in rows if _is_corrupted(row)]
    uncorrupted = [row for row in rows if not _is_corrupted(row)]
    corrupted_top = [row for row in top if _is_corrupted(row)]
    return {
        "top_k": top_k,
        "corrupted_in_top_k": len(corrupted_top),
        "precision_at_k": len(corrupted_top) / len(top) if top else 0.0,
        "recall_at_k": len(corrupted_top) / len(corrupted) if corrupted else 0.0,
        "mean_corrupted_score": _mean([row.get(score_key) for row in corrupted]),
        "mean_uncorrupted_score": _mean([row.get(score_key) for row in uncorrupted]),
        "score_gap": (_mean([row.get(score_key) for row in corrupted]) or 0.0) - (_mean([row.get(score_key) for row in uncorrupted]) or 0.0),
        "top1_view_name": ranked[0].get("view_name", "") if ranked else "",
        "top1_was_corrupted": ranked[0].get("was_corrupted", "") if ranked else "",
    }


def compute_normalized_ablation_metrics(rows: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["scene"], row["subset_name"], row["condition"])].append(row)
    for (scene, subset_name, condition), group_rows in grouped.items():
        for score_name in SCORE_NAMES:
            metrics = _metric_for_score(group_rows, score_name, top_k)
            output.append(
                {
                    "scene": scene,
                    "subset_name": subset_name,
                    "condition": condition,
                    "score_name": score_name,
                    **metrics,
                    "status": "ok",
                }
            )
    return output


def compute_normalized_group_metrics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["scene"], row["subset_name"], row["condition"])].append(row)
    for (scene, subset_name, condition), group_rows in grouped.items():
        groups = {
            "all": group_rows,
            "corrupted": [row for row in group_rows if _is_corrupted(row)],
            "uncorrupted": [row for row in group_rows if not _is_corrupted(row)],
        }
        for group, items in groups.items():
            output.append(
                {
                    "scene": scene,
                    "subset_name": subset_name,
                    "condition": condition,
                    "group": group,
                    "view_count": len(items),
                    "mean_raw_risk": _mean([row.get("raw_risk") for row in items]),
                    "mean_clean_prior_risk": _mean([row.get("clean_prior_risk") for row in items]),
                    "mean_normalized_viewtrust_risk": _mean([row.get("normalized_viewtrust_risk") for row in items]),
                    "median_normalized_viewtrust_risk": _median([row.get("normalized_viewtrust_risk") for row in items]),
                }
            )
    return output


def compute_false_positive_reduction(rows: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["scene"], row["subset_name"], row["condition"])].append(row)
    for (scene, subset_name, condition), group_rows in grouped.items():
        raw_top = {row["view_name"]: row for row in group_rows if row["raw_top_k"]}
        norm_top = {row["view_name"]: row for row in group_rows if row["normalized_top_k"]}
        raw_fp = {name for name, row in raw_top.items() if not _is_corrupted(row)}
        norm_fp = {name for name, row in norm_top.items() if not _is_corrupted(row)}
        raw_corrupt = {name for name, row in raw_top.items() if _is_corrupted(row)}
        norm_corrupt = {name for name, row in norm_top.items() if _is_corrupted(row)}
        raw_metrics = _metric_for_score(group_rows, "raw_risk", top_k)
        norm_metrics = _metric_for_score(group_rows, "normalized_viewtrust_risk", top_k)
        output.append(
            {
                "scene": scene,
                "subset_name": subset_name,
                "condition": condition,
                "raw_false_positive_count_at_k": len(raw_fp),
                "normalized_false_positive_count_at_k": len(norm_fp),
                "false_positive_reduction": len(raw_fp) - len(norm_fp),
                "raw_recall_at_k": raw_metrics["recall_at_k"],
                "normalized_recall_at_k": norm_metrics["recall_at_k"],
                "recall_delta": norm_metrics["recall_at_k"] - raw_metrics["recall_at_k"],
                "raw_precision_at_k": raw_metrics["precision_at_k"],
                "normalized_precision_at_k": norm_metrics["precision_at_k"],
                "precision_delta": norm_metrics["precision_at_k"] - raw_metrics["precision_at_k"],
                "removed_false_positive_views": ";".join(sorted(raw_fp - norm_fp)),
                "newly_added_false_positive_views": ";".join(sorted(norm_fp - raw_fp)),
                "recovered_corrupted_views": ";".join(sorted(norm_corrupt - raw_corrupt)),
                "lost_corrupted_views": ";".join(sorted(raw_corrupt - norm_corrupt)),
            }
        )
    return output


def compute_view_identity_diagnosis(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_view: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_view[(row["scene"], row["view_name"])].append(row)
    output: list[dict[str, Any]] = []
    for (scene, view_name), items in sorted(by_view.items()):
        uncorrupted = [row for row in items if not _is_corrupted(row)]
        corrupted = [row for row in items if _is_corrupted(row)]
        raw_uncorrupt_top = [row for row in uncorrupted if row["raw_top_k"]]
        norm_uncorrupt_top = [row for row in uncorrupted if row["normalized_top_k"]]
        output.append(
            {
                "scene": scene,
                "view_name": view_name,
                "observed_condition_count": len(items),
                "corrupted_count": len(corrupted),
                "uncorrupted_count": len(uncorrupted),
                "raw_top5_when_uncorrupted_count": len(raw_uncorrupt_top),
                "normalized_top5_when_uncorrupted_count": len(norm_uncorrupt_top),
                "raw_repeated_false_positive": len(raw_uncorrupt_top) >= 2,
                "normalized_repeated_false_positive": len(norm_uncorrupt_top) >= 2,
                "raw_mean_rank_when_uncorrupted": _mean([row.get("raw_rank") for row in uncorrupted]),
                "normalized_mean_rank_when_uncorrupted": _mean([row.get("normalized_rank") for row in uncorrupted]),
                "raw_mean_risk_when_uncorrupted": _mean([row.get("raw_risk") for row in uncorrupted]),
                "normalized_mean_risk_when_uncorrupted": _mean([row.get("normalized_viewtrust_risk") for row in uncorrupted]),
                "clean_prior_risk_mean": _mean([row.get("clean_prior_risk") for row in items]),
                "normalized_bias_reduced": len(norm_uncorrupt_top) < len(raw_uncorrupt_top),
                "conditions_removed_from_top5_when_uncorrupted": ";".join(
                    sorted(f"{row['subset_name']}:{row['condition']}" for row in raw_uncorrupt_top if not row["normalized_top_k"])
                ),
                "conditions_remaining_top5_when_uncorrupted": ";".join(
                    sorted(f"{row['subset_name']}:{row['condition']}" for row in norm_uncorrupt_top)
                ),
            }
        )
    return output


def compute_component_comparison(ablation_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "scene": row.get("scene", ""),
            "subset_name": row.get("subset_name", ""),
            "condition": row.get("condition", ""),
            "score_name": row.get("score_name", ""),
            "precision_at_k": row.get("precision_at_k", ""),
            "recall_at_k": row.get("recall_at_k", ""),
            "score_gap": row.get("score_gap", ""),
            "top1_view_name": row.get("top1_view_name", ""),
            "top1_was_corrupted": row.get("top1_was_corrupted", ""),
            "status": row.get("status", ""),
        }
        for row in ablation_rows
    ]


def build_summary(
    *,
    scenes: list[str],
    conditions: list[str],
    subset_names: list[str],
    top_k: int,
    valid_result_count: int,
    missing_rows: list[dict[str, Any]],
    fp_rows: list[dict[str, Any]],
    diagnosis_rows: list[dict[str, Any]],
    warnings: list[str],
) -> dict[str, Any]:
    raw_precision = [row.get("raw_precision_at_k") for row in fp_rows]
    norm_precision = [row.get("normalized_precision_at_k") for row in fp_rows]
    raw_recall = [row.get("raw_recall_at_k") for row in fp_rows]
    norm_recall = [row.get("normalized_recall_at_k") for row in fp_rows]
    raw_fp = [row.get("raw_false_positive_count_at_k") for row in fp_rows]
    norm_fp = [row.get("normalized_false_positive_count_at_k") for row in fp_rows]
    raw_repeated = [f"{row['scene']}:{row['view_name']}" for row in diagnosis_rows if row.get("raw_repeated_false_positive") is True]
    norm_repeated = [f"{row['scene']}:{row['view_name']}" for row in diagnosis_rows if row.get("normalized_repeated_false_positive") is True]
    lookup = {(row["scene"], row["view_name"]): row for row in diagnosis_rows}
    return {
        "schema_name": "viewtrust.pr17.clean_prior_normalized.summary",
        "schema_version": 1,
        "scenes": scenes,
        "conditions": conditions,
        "subset_names": subset_names,
        "top_k": top_k,
        "valid_result_count": valid_result_count,
        "missing_result_count": len(missing_rows),
        "raw_mean_precision_at_k": _mean(raw_precision),
        "raw_mean_recall_at_k": _mean(raw_recall),
        "normalized_mean_precision_at_k": _mean(norm_precision),
        "normalized_mean_recall_at_k": _mean(norm_recall),
        "raw_mean_false_positive_count_at_k": _mean(raw_fp),
        "normalized_mean_false_positive_count_at_k": _mean(norm_fp),
        "mean_false_positive_reduction": _mean([row.get("false_positive_reduction") for row in fp_rows]),
        "mean_recall_delta": _mean([row.get("recall_delta") for row in fp_rows]),
        "mean_precision_delta": _mean([row.get("precision_delta") for row in fp_rows]),
        "train_013_raw_false_positive_count": sum(row.get("raw_top5_when_uncorrupted_count") or 0 for key, row in lookup.items() if key[1] == "train_013"),
        "train_013_normalized_false_positive_count": sum(row.get("normalized_top5_when_uncorrupted_count") or 0 for key, row in lookup.items() if key[1] == "train_013"),
        "train_014_raw_false_positive_count": sum(row.get("raw_top5_when_uncorrupted_count") or 0 for key, row in lookup.items() if key[1] == "train_014"),
        "train_014_normalized_false_positive_count": sum(row.get("normalized_top5_when_uncorrupted_count") or 0 for key, row in lookup.items() if key[1] == "train_014"),
        "normalized_repeated_false_positive_views": sorted(norm_repeated),
        "raw_repeated_false_positive_views": sorted(raw_repeated),
        "uses_corruption_labels_for_scoring": False,
        "uses_corruption_labels_for_evaluation": True,
        "observation_only": True,
        "training_intervention": False,
        "defense_enabled": False,
        "warnings": warnings,
    }


def write_report(path: Path, summary: dict[str, Any]) -> None:
    report = "\n".join(
        [
            "# PR17 Clean-Prior Normalized Offline ViewTrust",
            "",
            "PR17 is offline analysis only.",
            "It is not a trust score used during training.",
            "It is not a defense.",
            "It is not a poison classifier.",
            "It does not reject views, suppress updates, reweight loss, or gate densification.",
            "Corruption labels are used only for evaluation summaries, not for scoring or ranking.",
            "",
            "## Formula",
            "`normalized_viewtrust_risk = positive_delta_risk + 0.25 * prior_suppressed_risk + 0.10 * max(0, rank_lift_score) / max(view_count - 1, 1)`",
            "",
            "## Summary",
            f"- Raw mean precision@k: `{summary.get('raw_mean_precision_at_k')}`",
            f"- Normalized mean precision@k: `{summary.get('normalized_mean_precision_at_k')}`",
            f"- Raw mean recall@k: `{summary.get('raw_mean_recall_at_k')}`",
            f"- Normalized mean recall@k: `{summary.get('normalized_mean_recall_at_k')}`",
            f"- Raw repeated false positives: `{summary.get('raw_repeated_false_positive_views')}`",
            f"- Normalized repeated false positives: `{summary.get('normalized_repeated_false_positive_views')}`",
            "",
            "## Interpretation",
            "High normalized risk is post-hoc evidence of corruption-induced lift, not proof of maliciousness.",
            "Clean-prior normalization separates stable view identity prior from additional corruption-associated anomaly.",
            "",
            "## Limitations",
            "- Clean priors are estimated from available clean-side features and may be incomplete.",
            "- This is a prerequisite analysis before any training-time intervention, not an intervention.",
            "- Missing PR16 outputs are reported rather than hidden.",
            "",
        ]
    )
    path.write_text(report, encoding="utf-8")


def write_artifact_manifest(output_dir: Path, input_root: Path, plan_dir: Path) -> None:
    fields = ["relative_path", "path", "exists", "file_type", "size_bytes", "required", "artifact_group"]
    def rows() -> list[dict[str, Any]]:
        items = [
            ("input_root", input_root, "false", "input"),
            ("plan_dir", plan_dir, "false", "input"),
        ]
        output = []
        for relative, path, required, group in items:
            output.append(
                {
                    "relative_path": relative,
                    "path": str(path),
                    "exists": str(path.exists()).lower(),
                    "file_type": "directory" if path.is_dir() else "",
                    "size_bytes": "",
                    "required": required,
                    "artifact_group": group,
                }
            )
        for name in PR17_OUTPUT_FILES:
            path = output_dir / name
            output.append(
                {
                    "relative_path": name,
                    "path": str(path),
                    "exists": str(path.exists()).lower(),
                    "file_type": path.suffix.lstrip("."),
                    "size_bytes": path.stat().st_size if path.is_file() else "",
                    "required": "true",
                    "artifact_group": "output_pr17",
                }
            )
        return output
    manifest = output_dir / "artifact_manifest.csv"
    write_csv_rows(manifest, rows(), fields)
    write_csv_rows(manifest, rows(), fields)
