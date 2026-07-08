"""Pure helpers for PR14 multi-condition offline signal validation."""

from __future__ import annotations

import csv
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


REQUIRED_OFFLINE_SIGNAL_FILES = [
    "offline_viewtrust_summary.json",
    "offline_viewtrust_signals.csv",
    "offline_viewtrust_rankings.csv",
    "offline_viewtrust_group_metrics.csv",
    "offline_viewtrust_signal_ablation.csv",
    "offline_viewtrust_config.json",
    "offline_viewtrust_report.md",
    "offline_viewtrust_artifact_manifest.csv",
]

ABLATION_SIGNAL_NAMES = [
    "loss_only",
    "visibility_only",
    "birth_only",
    "prune_only",
    "survival_only",
    "delta_only",
    "lifecycle_only",
    "full_signal",
]


def safe_float(value: Any) -> float | None:
    if value in ("", None):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number


def truthy(value: Any) -> bool:
    return value is True or str(value).lower() == "true"


def csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def json_file(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _timestamp_sort_key(path: Path) -> tuple[str, str]:
    return path.name.rsplit("_", 1)[-1], path.name


def discover_offline_signal_dirs(input_root: Path, conditions: list[str]) -> dict[str, Path | None]:
    result: dict[str, Path | None] = {}
    for condition in conditions:
        prefix = f"offline_viewtrust_{condition}_pr13"
        candidates = [
            path
            for path in input_root.iterdir()
            if path.is_dir() and path.name.startswith(prefix)
        ] if input_root.exists() else []
        sorted_candidates = sorted(candidates, key=_timestamp_sort_key, reverse=True)
        valid = [
            path
            for path in sorted_candidates
            if validate_offline_signal_dir(path)[0] == "ok"
        ]
        result[condition] = valid[0] if valid else sorted_candidates[0] if sorted_candidates else None
    return result


def load_offline_signal_summary(signal_dir: Path) -> dict[str, Any]:
    return json_file(signal_dir / "offline_viewtrust_summary.json")


def load_offline_rankings(signal_dir: Path) -> list[dict[str, str]]:
    return csv_rows(signal_dir / "offline_viewtrust_rankings.csv")


def load_offline_ablation(signal_dir: Path) -> list[dict[str, str]]:
    return csv_rows(signal_dir / "offline_viewtrust_signal_ablation.csv")


def _required_file_errors(signal_dir: Path) -> list[str]:
    return [
        f"missing required file: {name}"
        for name in REQUIRED_OFFLINE_SIGNAL_FILES
        if not (signal_dir / name).is_file()
    ]


def _summary_validation_errors(summary: dict[str, Any]) -> list[str]:
    checks = [
        ("observation_only", summary.get("observation_only") is True),
        ("uses_corruption_labels_for_scoring", summary.get("uses_corruption_labels_for_scoring") is False),
        ("uses_corruption_labels_for_evaluation", summary.get("uses_corruption_labels_for_evaluation") is True),
        ("training_intervention", summary.get("training_intervention") is False),
        ("defense_enabled", summary.get("defense_enabled") is False),
        ("view_count", (safe_float(summary.get("view_count")) or 0.0) > 0),
        ("corrupted_view_count", (safe_float(summary.get("corrupted_view_count")) or 0.0) > 0),
    ]
    return [f"invalid summary field: {name}" for name, ok in checks if not ok]


def _ranking_validation_errors(rankings: list[dict[str, str]]) -> list[str]:
    if not rankings:
        return ["offline_viewtrust_rankings.csv has no rows"]
    errors: list[str] = []
    previous_score: float | None = None
    for expected_rank, row in enumerate(rankings, start=1):
        rank = int(safe_float(row.get("rank")) or -1)
        if rank != expected_rank:
            errors.append("ranking ranks are not consecutive from 1")
            break
        score = safe_float(row.get("offline_viewtrust_risk"))
        if score is None:
            errors.append("ranking row missing offline_viewtrust_risk")
            break
        if previous_score is not None and score > previous_score:
            errors.append("rankings are not sorted by offline_viewtrust_risk descending")
            break
        previous_score = score
    return errors


def validate_offline_signal_dir(signal_dir: Path) -> tuple[str, list[str]]:
    file_errors = _required_file_errors(signal_dir)
    if file_errors:
        return "invalid", file_errors

    summary = load_offline_signal_summary(signal_dir)
    rankings = load_offline_rankings(signal_dir)
    ablation = load_offline_ablation(signal_dir)
    group_metrics = csv_rows(signal_dir / "offline_viewtrust_group_metrics.csv")
    signals = csv_rows(signal_dir / "offline_viewtrust_signals.csv")

    errors = _summary_validation_errors(summary)
    if not signals:
        errors.append("offline_viewtrust_signals.csv has no rows")
    errors.extend(_ranking_validation_errors(rankings))
    if not ablation:
        errors.append("offline_viewtrust_signal_ablation.csv has no rows")
    if not group_metrics:
        errors.append("offline_viewtrust_group_metrics.csv has no rows")
    return ("failed_validation", errors) if errors else ("ok", [])


def _count_corrupted_in_top(rankings: list[dict[str, str]], k: int) -> int:
    return sum(1 for row in rankings[: min(k, len(rankings))] if truthy(row.get("was_corrupted")))


def condition_result_from_signal_dir(
    *,
    scene: str,
    condition: str,
    clean_condition: str,
    signal_dir: Path | None,
    top_k: int,
) -> dict[str, Any]:
    if signal_dir is None:
        return {
            "scene": scene,
            "condition": condition,
            "clean_condition": clean_condition,
            "offline_signal_dir": "",
            "status": "missing",
            "warnings": "missing PR13 offline signal output",
        }

    status, warnings = validate_offline_signal_dir(signal_dir)
    if status != "ok":
        return {
            "scene": scene,
            "condition": condition,
            "clean_condition": clean_condition,
            "offline_signal_dir": str(signal_dir),
            "status": status,
            "warnings": ";".join(warnings),
        }

    summary = load_offline_signal_summary(signal_dir)
    rankings = load_offline_rankings(signal_dir)
    top1 = rankings[0] if rankings else {}
    return {
        "scene": summary.get("scene") or scene,
        "condition": summary.get("corrupt_condition") or condition,
        "clean_condition": summary.get("clean_condition") or clean_condition,
        "clean_run_id": summary.get("clean_run_id", ""),
        "corrupt_run_id": summary.get("corrupt_run_id", ""),
        "offline_signal_dir": str(signal_dir),
        "view_count": summary.get("view_count", ""),
        "corrupted_view_count": summary.get("corrupted_view_count", ""),
        "uncorrupted_view_count": summary.get("uncorrupted_view_count", ""),
        "top_k": min(top_k, int(safe_float(summary.get("view_count")) or top_k)),
        "corrupted_in_top_k": summary.get("corrupted_in_top_k", ""),
        "precision_at_k": summary.get("precision_at_k", ""),
        "recall_at_k": summary.get("recall_at_k", ""),
        "mean_corrupted_risk": summary.get("mean_corrupted_risk", ""),
        "mean_uncorrupted_risk": summary.get("mean_uncorrupted_risk", ""),
        "risk_gap_corrupted_minus_uncorrupted": summary.get("risk_gap_corrupted_minus_uncorrupted", ""),
        "best_ablation_signal": summary.get("best_ablation_signal", ""),
        "top1_view_name": top1.get("view_name", ""),
        "top1_was_corrupted": top1.get("was_corrupted", ""),
        "top1_risk": top1.get("offline_viewtrust_risk", ""),
        "top1_main_reason": top1.get("main_reason", ""),
        "top3_corrupted_count": _count_corrupted_in_top(rankings, 3),
        "top5_corrupted_count": _count_corrupted_in_top(rankings, 5),
        "observation_only": summary.get("observation_only"),
        "uses_corruption_labels_for_scoring": summary.get("uses_corruption_labels_for_scoring"),
        "uses_corruption_labels_for_evaluation": summary.get("uses_corruption_labels_for_evaluation"),
        "training_intervention": summary.get("training_intervention"),
        "defense_enabled": summary.get("defense_enabled"),
        "status": "ok",
        "warnings": ";".join(summary.get("warnings", [])),
    }


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def aggregate_condition_results(
    *,
    scene: str,
    clean_condition: str,
    conditions: list[str],
    condition_results: list[dict[str, Any]],
    top_k: int,
) -> dict[str, Any]:
    valid = [row for row in condition_results if row.get("status") == "ok"]
    missing = [row["condition"] for row in condition_results if row.get("status") == "missing"]
    invalid = [row["condition"] for row in condition_results if row.get("status") not in ("ok", "missing")]
    precision_values = [safe_float(row.get("precision_at_k")) for row in valid]
    recall_values = [safe_float(row.get("recall_at_k")) for row in valid]
    gap_values = [safe_float(row.get("risk_gap_corrupted_minus_uncorrupted")) for row in valid]
    precision_numbers = [value for value in precision_values if value is not None]
    recall_numbers = [value for value in recall_values if value is not None]
    gap_numbers = [value for value in gap_values if value is not None]
    ablation_counts = Counter(str(row.get("best_ablation_signal", "")) for row in valid if row.get("best_ablation_signal"))
    valid_conditions = [row["condition"] for row in valid]
    return {
        "schema_name": "viewtrust.offline_signal.multi_condition.summary",
        "schema_version": 1,
        "scene": scene,
        "clean_condition": clean_condition,
        "conditions_requested": conditions,
        "conditions_found": [row["condition"] for row in condition_results if row.get("status") != "missing"],
        "conditions_missing": missing,
        "conditions_valid": valid_conditions,
        "condition_count_requested": len(conditions),
        "condition_count_valid": len(valid),
        "top_k": top_k,
        "observation_only": all(row.get("observation_only") is True for row in valid) if valid else True,
        "uses_corruption_labels_for_scoring": any(row.get("uses_corruption_labels_for_scoring") is True for row in valid),
        "uses_corruption_labels_for_evaluation": all(row.get("uses_corruption_labels_for_evaluation") is True for row in valid) if valid else True,
        "training_intervention": any(row.get("training_intervention") is True for row in valid),
        "defense_enabled": any(row.get("defense_enabled") is True for row in valid),
        "mean_precision_at_k": _mean(precision_numbers),
        "mean_recall_at_k": _mean(recall_numbers),
        "mean_risk_gap": _mean(gap_numbers),
        "median_precision_at_k": _median(precision_numbers),
        "median_recall_at_k": _median(recall_numbers),
        "median_risk_gap": _median(gap_numbers),
        "conditions_with_full_recall": sum(1 for value in recall_numbers if value >= 1.0),
        "conditions_with_positive_risk_gap": sum(1 for value in gap_numbers if value > 0.0),
        "best_condition_by_recall": max(valid, key=lambda row: safe_float(row.get("recall_at_k")) or -1.0).get("condition", "") if valid else "",
        "worst_condition_by_recall": min(valid, key=lambda row: safe_float(row.get("recall_at_k")) or 2.0).get("condition", "") if valid else "",
        "best_ablation_signal_counts": dict(sorted(ablation_counts.items())),
        "warnings": [
            *(f"missing condition output: {condition}" for condition in missing),
            *(f"invalid condition output: {condition}" for condition in invalid),
        ],
    }


def compute_condition_ranking(condition_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    valid = [row for row in condition_results if row.get("status") == "ok"]
    ranked = sorted(
        valid,
        key=lambda row: (
            -(safe_float(row.get("recall_at_k")) or 0.0),
            -(safe_float(row.get("precision_at_k")) or 0.0),
            -(safe_float(row.get("risk_gap_corrupted_minus_uncorrupted")) or 0.0),
            str(row.get("condition", "")),
        ),
    )
    rows = []
    for rank, row in enumerate(ranked, start=1):
        rows.append(
            {
                "rank": rank,
                "scene": row.get("scene", ""),
                "condition": row.get("condition", ""),
                "precision_at_k": row.get("precision_at_k", ""),
                "recall_at_k": row.get("recall_at_k", ""),
                "corrupted_in_top_k": row.get("corrupted_in_top_k", ""),
                "risk_gap_corrupted_minus_uncorrupted": row.get("risk_gap_corrupted_minus_uncorrupted", ""),
                "mean_corrupted_risk": row.get("mean_corrupted_risk", ""),
                "mean_uncorrupted_risk": row.get("mean_uncorrupted_risk", ""),
                "best_ablation_signal": row.get("best_ablation_signal", ""),
                "status": row.get("status", ""),
            }
        )
    for row in condition_results:
        if row.get("status") != "ok":
            rows.append(
                {
                    "rank": "",
                    "scene": row.get("scene", ""),
                    "condition": row.get("condition", ""),
                    "precision_at_k": "",
                    "recall_at_k": "",
                    "corrupted_in_top_k": "",
                    "risk_gap_corrupted_minus_uncorrupted": "",
                    "mean_corrupted_risk": "",
                    "mean_uncorrupted_risk": "",
                    "best_ablation_signal": "",
                    "status": row.get("status", ""),
                }
            )
    return rows


def compute_failure_cases(condition_results: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for row in condition_results:
        scene = row.get("scene", "")
        condition = row.get("condition", "")
        status = row.get("status")
        if status == "missing":
            failures.append(
                {
                    "scene": scene,
                    "condition": condition,
                    "failure_type": "missing_condition_output",
                    "view_name": "",
                    "was_corrupted": "",
                    "rank": "",
                    "offline_viewtrust_risk": "",
                    "main_reason": "",
                    "details": row.get("warnings", ""),
                }
            )
            continue
        if status != "ok":
            failures.append(
                {
                    "scene": scene,
                    "condition": condition,
                    "failure_type": "invalid_condition_output",
                    "view_name": "",
                    "was_corrupted": "",
                    "rank": "",
                    "offline_viewtrust_risk": "",
                    "main_reason": "",
                    "details": row.get("warnings", ""),
                }
            )
            continue

        signal_dir = Path(str(row.get("offline_signal_dir")))
        rankings = load_offline_rankings(signal_dir)
        if rankings and not truthy(rankings[0].get("was_corrupted")):
            failures.append(
                {
                    "scene": scene,
                    "condition": condition,
                    "failure_type": "top_ranked_uncorrupted_view",
                    "view_name": rankings[0].get("view_name", ""),
                    "was_corrupted": rankings[0].get("was_corrupted", ""),
                    "rank": rankings[0].get("rank", ""),
                    "offline_viewtrust_risk": rankings[0].get("offline_viewtrust_risk", ""),
                    "main_reason": rankings[0].get("main_reason", ""),
                    "details": "top ranked view is uncorrupted",
                }
            )
        for ranking in rankings:
            rank_value = int(safe_float(ranking.get("rank")) or 0)
            if truthy(ranking.get("was_corrupted")) and rank_value > top_k:
                failures.append(
                    {
                        "scene": scene,
                        "condition": condition,
                        "failure_type": "corrupted_view_not_in_top_k",
                        "view_name": ranking.get("view_name", ""),
                        "was_corrupted": ranking.get("was_corrupted", ""),
                        "rank": ranking.get("rank", ""),
                        "offline_viewtrust_risk": ranking.get("offline_viewtrust_risk", ""),
                        "main_reason": ranking.get("main_reason", ""),
                        "details": f"corrupted view rank exceeds top_k={top_k}",
                    }
                )
        if (safe_float(row.get("risk_gap_corrupted_minus_uncorrupted")) or 0.0) <= 0.0:
            failures.append(
                {
                    "scene": scene,
                    "condition": condition,
                    "failure_type": "low_risk_gap",
                    "view_name": "",
                    "was_corrupted": "",
                    "rank": "",
                    "offline_viewtrust_risk": "",
                    "main_reason": "",
                    "details": "risk gap corrupted minus uncorrupted is <= 0",
                }
            )
        if int(safe_float(row.get("corrupted_in_top_k")) or 0) == 0:
            failures.append(
                {
                    "scene": scene,
                    "condition": condition,
                    "failure_type": "zero_corrupted_in_top_k",
                    "view_name": "",
                    "was_corrupted": "",
                    "rank": "",
                    "offline_viewtrust_risk": "",
                    "main_reason": "",
                    "details": "no corrupted views in top_k",
                }
            )
    return failures
