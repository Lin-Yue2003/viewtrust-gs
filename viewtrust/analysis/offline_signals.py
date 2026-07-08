"""Pure helpers for PR13 offline ViewTrust candidate signals.

These functions are intentionally dependency-light and offline-only. They do
not use corruption labels for scoring; label-aware helpers are for post-hoc
evaluation summaries only.
"""

from __future__ import annotations

import statistics
from collections.abc import Iterable
from typing import Any


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


def robust_median(values: Iterable[Any]) -> float | None:
    numbers = [number for number in (safe_float(value) for value in values) if number is not None]
    return statistics.median(numbers) if numbers else None


def mad(values: Iterable[Any]) -> float | None:
    numbers = [number for number in (safe_float(value) for value in values) if number is not None]
    if not numbers:
        return None
    median = statistics.median(numbers)
    return statistics.median(abs(number - median) for number in numbers)


def robust_z_scores(values: Iterable[Any], eps: float = 1e-8, mad_scale: float = 1.4826) -> list[float]:
    raw_values = list(values)
    numbers = [safe_float(value) for value in raw_values]
    available = [number for number in numbers if number is not None]
    if not available:
        return [0.0 for _ in raw_values]

    median = statistics.median(available)
    mad_value = statistics.median(abs(number - median) for number in available)
    denominator = mad_scale * mad_value
    center = median
    if denominator <= eps:
        mean = statistics.fmean(available)
        variance = statistics.fmean((number - mean) ** 2 for number in available)
        std = variance ** 0.5
        if std <= eps:
            return [0.0 for _ in raw_values]
        center = mean
        denominator = std

    return [
        ((number - center) / (denominator + eps)) if number is not None else 0.0
        for number in numbers
    ]


def positive_part(values: Iterable[Any]) -> list[float]:
    return [max(0.0, safe_float(value) or 0.0) for value in values]


def safe_divide(numerator: Any, denominator: Any, default: float = 0.0) -> float:
    numerator_value = safe_float(numerator)
    denominator_value = safe_float(denominator)
    if numerator_value is None or denominator_value in (None, 0.0):
        return default
    return numerator_value / denominator_value


def _mean(values: Iterable[Any]) -> float | None:
    numbers = [number for number in (safe_float(value) for value in values) if number is not None]
    return statistics.fmean(numbers) if numbers else None


def _median(values: Iterable[Any]) -> float | None:
    numbers = [number for number in (safe_float(value) for value in values) if number is not None]
    return statistics.median(numbers) if numbers else None


def rank_descending(rows: list[dict[str, Any]], score_key: str) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (safe_float(row.get(score_key)) or 0.0, str(row.get("view_name", ""))),
        reverse=True,
    )


def precision_at_k(rows: list[dict[str, Any]], label_key: str, score_key: str, k: int) -> float:
    ranked = rank_descending(rows, score_key)
    top_k = min(max(k, 0), len(ranked))
    if top_k == 0:
        return 0.0
    positives = sum(1 for row in ranked[:top_k] if str(row.get(label_key, "")).lower() == "true")
    return positives / top_k


def recall_at_k(rows: list[dict[str, Any]], label_key: str, score_key: str, k: int) -> float:
    total_positive = sum(1 for row in rows if str(row.get(label_key, "")).lower() == "true")
    if total_positive == 0:
        return 0.0
    ranked = rank_descending(rows, score_key)
    top_k = min(max(k, 0), len(ranked))
    positives = sum(1 for row in ranked[:top_k] if str(row.get(label_key, "")).lower() == "true")
    return positives / total_positive


def _positive_robust_z(values: list[Any], eps: float, mad_scale: float) -> list[float]:
    return positive_part(robust_z_scores(values, eps=eps, mad_scale=mad_scale))


def compute_signal_components(rows: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    normalization = config.get("normalization", {})
    eps = safe_float(normalization.get("eps")) or 1e-8
    mad_scale = safe_float(normalization.get("mad_scale")) or 1.4826
    weights = config.get("weights", {})
    output_rows = [dict(row) for row in rows]

    loss_values = [
        row.get("mean_total_loss")
        if safe_float(row.get("mean_total_loss")) is not None
        else row.get("mean_loss")
        for row in output_rows
    ]
    visibility_drop_values = [row.get("visibility_drop") for row in output_rows]
    birth_rate_values = [row.get("birth_rate") for row in output_rows]
    prune_rate_values = [row.get("prune_death_rate") for row in output_rows]
    survival_anomaly_values = [
        1.0 - (safe_float(row.get("birth_survival_ratio_after_view")) or 1.0)
        for row in output_rows
    ]
    birth_delta_values = [row.get("birth_event_count_delta") for row in output_rows]
    prune_delta_values = [row.get("prune_death_count_delta") for row in output_rows]

    loss_scores = _positive_robust_z(loss_values, eps, mad_scale)
    visibility_scores = _positive_robust_z(visibility_drop_values, eps, mad_scale)
    birth_scores = _positive_robust_z(birth_rate_values, eps, mad_scale)
    prune_scores = _positive_robust_z(prune_rate_values, eps, mad_scale)
    survival_scores = _positive_robust_z(survival_anomaly_values, eps, mad_scale)
    birth_delta_scores = _positive_robust_z(birth_delta_values, eps, mad_scale)
    prune_delta_scores = _positive_robust_z(prune_delta_values, eps, mad_scale)

    for index, row in enumerate(output_rows):
        delta_component = statistics.fmean(
            [
                birth_delta_scores[index],
                prune_delta_scores[index],
                visibility_scores[index],
            ]
        )
        lifecycle_component = statistics.fmean(
            [
                birth_scores[index],
                prune_scores[index],
                survival_scores[index],
            ]
        )
        components = {
            "loss_component": loss_scores[index],
            "visibility_component": visibility_scores[index],
            "birth_component": birth_scores[index],
            "prune_component": prune_scores[index],
            "survival_component": survival_scores[index],
            "delta_component": delta_component,
            "lifecycle_component": lifecycle_component,
        }
        risk = sum(
            (safe_float(weights.get(name)) or 0.0) * value
            for name, value in components.items()
            if name != "lifecycle_component"
        )
        row.update(components)
        row["offline_viewtrust_risk"] = max(0.0, risk)
        row["offline_viewtrust_consistency"] = 1.0 / (1.0 + row["offline_viewtrust_risk"])
    return output_rows


def compute_group_metrics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups = {
        "all": rows,
        "corrupted": [row for row in rows if str(row.get("was_corrupted", "")).lower() == "true"],
        "uncorrupted": [row for row in rows if str(row.get("was_corrupted", "")).lower() != "true"],
    }
    fields = [
        "loss_component",
        "visibility_component",
        "birth_component",
        "prune_component",
        "survival_component",
        "delta_component",
        "lifecycle_component",
    ]
    result: list[dict[str, Any]] = []
    for name, group_rows in groups.items():
        row: dict[str, Any] = {
            "group": name,
            "view_count": len(group_rows),
            "mean_offline_viewtrust_risk": _mean(item.get("offline_viewtrust_risk") for item in group_rows),
            "median_offline_viewtrust_risk": _median(item.get("offline_viewtrust_risk") for item in group_rows),
            "max_offline_viewtrust_risk": max(
                [safe_float(item.get("offline_viewtrust_risk")) or 0.0 for item in group_rows],
                default=0.0,
            ),
            "mean_offline_viewtrust_consistency": _mean(
                item.get("offline_viewtrust_consistency") for item in group_rows
            ),
        }
        for field in fields:
            row[f"mean_{field}"] = _mean(item.get(field) for item in group_rows)
        result.append(row)
    return result


def compute_ablation_metrics(
    rows: list[dict[str, Any]],
    *,
    label_key: str = "was_corrupted",
    top_k: int = 5,
) -> list[dict[str, Any]]:
    variants = [
        ("loss_only", "loss_component"),
        ("visibility_only", "visibility_component"),
        ("birth_only", "birth_component"),
        ("prune_only", "prune_component"),
        ("survival_only", "survival_component"),
        ("delta_only", "delta_component"),
        ("lifecycle_only", "lifecycle_component"),
        ("full_signal", "offline_viewtrust_risk"),
    ]
    total_corrupted = sum(1 for row in rows if str(row.get(label_key, "")).lower() == "true")
    result: list[dict[str, Any]] = []
    for signal_name, score_key in variants:
        ranked = rank_descending(rows, score_key)
        top_count = min(max(top_k, 0), len(ranked))
        top_rows = ranked[:top_count]
        corrupted_rows = [
            (rank + 1, row)
            for rank, row in enumerate(ranked)
            if str(row.get(label_key, "")).lower() == "true"
        ]
        corrupted_scores = [row.get(score_key) for _, row in corrupted_rows]
        uncorrupted_scores = [
            row.get(score_key)
            for row in rows
            if str(row.get(label_key, "")).lower() != "true"
        ]
        mean_corrupted = _mean(corrupted_scores)
        mean_uncorrupted = _mean(uncorrupted_scores)
        corrupted_in_topk = sum(1 for row in top_rows if str(row.get(label_key, "")).lower() == "true")
        result.append(
            {
                "signal_name": signal_name,
                "top1_view_name": ranked[0].get("view_name", "") if ranked else "",
                "top1_was_corrupted": ranked[0].get(label_key, "") if ranked else "",
                "topk": top_count,
                "corrupted_in_topk": corrupted_in_topk,
                "precision_at_k": safe_divide(corrupted_in_topk, top_count),
                "recall_at_k": safe_divide(corrupted_in_topk, total_corrupted),
                "mean_corrupted_rank": _mean(rank for rank, _ in corrupted_rows),
                "median_corrupted_rank": _median(rank for rank, _ in corrupted_rows),
                "mean_corrupted_score": mean_corrupted,
                "mean_uncorrupted_score": mean_uncorrupted,
                "score_gap": (
                    mean_corrupted - mean_uncorrupted
                    if mean_corrupted is not None and mean_uncorrupted is not None
                    else None
                ),
            }
        )
    return result
