"""Pure helpers for PR15 offline rank consistency diagnosis."""

from __future__ import annotations

import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_CONDITIONS = [
    "corrupt_occluder",
    "corrupt_blur",
    "corrupt_exposure",
    "corrupt_color_shift",
    "corrupt_noise",
    "corrupt_mixed",
]

REQUIRED_CONDITION_FILES = [
    "offline_viewtrust_summary.json",
    "offline_viewtrust_signals.csv",
    "offline_viewtrust_rankings.csv",
    "offline_viewtrust_group_metrics.csv",
    "offline_viewtrust_signal_ablation.csv",
    "offline_viewtrust_config.json",
    "offline_viewtrust_report.md",
    "offline_viewtrust_artifact_manifest.csv",
]

PR15_OUTPUT_FILES = [
    "cross_condition_view_rank_table.csv",
    "cross_condition_view_rank_summary.json",
    "repeated_top_views.csv",
    "false_positive_topk_views.csv",
    "corrupted_view_rank_distribution.csv",
    "component_win_table.csv",
    "component_condition_summary.csv",
    "component_gap_table.csv",
    "rank_consistency_report.md",
    "artifact_manifest.csv",
]

COMPONENT_KEYWORDS = [
    "loss",
    "visibility",
    "birth",
    "prune",
    "survival",
    "delta",
    "lifecycle",
    "consistency",
]

REPORT_DISCLAIMER = """This PR15 report is offline observation only.
It is not a trust score used during training.
It is not a defense.
It is not a poison classifier.
It does not reject views, suppress updates, reweight loss, or gate densification.
Corruption labels are used only for evaluation summaries, not for scoring or ranking."""


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


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


def normalize_bool(value: Any) -> bool | None:
    if value in ("", None):
        return None
    text = str(value).strip().lower()
    if value is True or text in {"true", "1", "yes"}:
        return True
    if value is False or text in {"false", "0", "no"}:
        return False
    return None


def normalize_float(value: Any) -> float | None:
    if value in ("", None):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number


def normalize_int(value: Any) -> int | None:
    number = normalize_float(value)
    return int(number) if number is not None else None


def _first(row: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        if row.get(key) not in ("", None):
            return row.get(key)
    return ""


def _risk(row: dict[str, Any]) -> float | None:
    return normalize_float(_first(row, ["offline_viewtrust_risk", "risk", "score", "full_signal"]))


def _view_name(row: dict[str, Any]) -> str:
    return str(_first(row, ["view_name", "source_view_name"]))


def _rank(row: dict[str, Any]) -> int | None:
    return normalize_int(_first(row, ["rank", "offline_viewtrust_rank"]))


def _was_corrupted(row: dict[str, Any]) -> bool | None:
    return normalize_bool(_first(row, ["was_corrupted", "is_corrupted", "corrupted"]))


def _main_reason(row: dict[str, Any]) -> str:
    return str(_first(row, ["main_reason", "top_reason", "reason"]))


def _component_values(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in row.items()
        if any(keyword in key for keyword in COMPONENT_KEYWORDS)
    }


def _candidate_sort_key(path: Path, condition: str) -> tuple[int, str]:
    exact = f"offline_viewtrust_{condition}_pr14_input"
    if path.name == exact:
        return (3, path.name)
    if path.name.startswith(f"offline_viewtrust_{condition}_pr13"):
        return (2, path.name)
    return (1, path.name)


def _condition_candidate_dirs(input_root: Path, condition: str) -> list[Path]:
    if not input_root.exists():
        return []
    exact_names = {
        f"offline_viewtrust_{condition}_pr14_input",
        f"offline_viewtrust_{condition}_pr13_pr14_input",
    }
    candidates = [
        path
        for path in input_root.iterdir()
        if path.is_dir()
        and (
            path.name in exact_names
            or path.name.startswith(f"offline_viewtrust_{condition}_pr13")
            or path.name.startswith(f"offline_viewtrust_{condition}_pr131")
            or path.name.startswith(f"offline_viewtrust_{condition}_pr132")
        )
    ]
    return sorted(candidates, key=lambda path: _candidate_sort_key(path, condition), reverse=True)


def find_condition_signal_dir(input_root: Path, condition: str) -> Path | None:
    for path in _condition_candidate_dirs(input_root, condition):
        if validate_condition_signal_dir(path)[0] == "ok":
            return path
    candidates = _condition_candidate_dirs(input_root, condition)
    return candidates[0] if candidates else None


def validate_multi_condition_dir(multi_condition_dir: Path) -> tuple[dict[str, Any], list[str]]:
    required = [
        "offline_viewtrust_multi_condition_summary.json",
        "offline_viewtrust_multi_condition_results.csv",
        "offline_viewtrust_multi_condition_ablation.csv",
        "offline_viewtrust_condition_ranking.csv",
        "offline_viewtrust_failure_cases.csv",
    ]
    warnings = [
        f"missing PR14 artifact: {name}"
        for name in required
        if not (multi_condition_dir / name).is_file()
    ]
    summary = load_json(multi_condition_dir / "offline_viewtrust_multi_condition_summary.json")
    checks = [
        ("observation_only", summary.get("observation_only") is True),
        ("training_intervention", summary.get("training_intervention") is False),
        ("defense_enabled", summary.get("defense_enabled") is False),
        ("uses_corruption_labels_for_scoring", summary.get("uses_corruption_labels_for_scoring") is False),
        ("uses_corruption_labels_for_evaluation", summary.get("uses_corruption_labels_for_evaluation") is True),
    ]
    for name, ok in checks:
        if summary and not ok:
            warnings.append(f"unexpected PR14 summary field: {name}")
    if summary and normalize_int(summary.get("condition_count_valid")) != 6:
        warnings.append("PR14 condition_count_valid is not 6")
    if summary and summary.get("conditions_missing"):
        warnings.append("PR14 conditions_missing is not empty")
    return summary, warnings


def _ranking_schema_warnings(rankings: list[dict[str, Any]]) -> list[str]:
    if not rankings:
        return ["offline_viewtrust_rankings.csv has no rows"]
    warnings: list[str] = []
    first = rankings[0]
    if not _view_name(first):
        warnings.append("ranking missing required compatible column: view_name")
    if _risk(first) is None:
        warnings.append("ranking missing required compatible column: offline_viewtrust_risk")
    if _was_corrupted(first) is None:
        warnings.append("ranking missing evaluation column: was_corrupted")
    if not _main_reason(first):
        warnings.append("ranking missing optional compatible column: main_reason")

    ranks = [_rank(row) for row in rankings]
    risks = [_risk(row) for row in rankings]
    if all(rank is not None for rank in ranks):
        expected = list(range(1, len(rankings) + 1))
        if ranks != expected:
            warnings.append("ranking ranks are not consecutive from 1")
        previous: float | None = None
        for risk in risks:
            if risk is None:
                continue
            if previous is not None and risk > previous:
                warnings.append("offline_viewtrust_risk is not descending by rank")
                break
            previous = risk
    return warnings


def validate_condition_signal_dir(signal_dir: Path) -> tuple[str, list[str]]:
    missing = [
        f"missing condition artifact: {name}"
        for name in REQUIRED_CONDITION_FILES
        if not (signal_dir / name).is_file()
    ]
    if missing:
        return "invalid", missing

    summary = load_json(signal_dir / "offline_viewtrust_summary.json")
    critical: list[str] = []
    warnings: list[str] = []
    checks = [
        ("observation_only", summary.get("observation_only") is True),
        ("training_intervention", summary.get("training_intervention") is False),
        ("defense_enabled", summary.get("defense_enabled") is False),
        ("uses_corruption_labels_for_scoring", summary.get("uses_corruption_labels_for_scoring") is False),
        ("uses_corruption_labels_for_evaluation", summary.get("uses_corruption_labels_for_evaluation") is True),
        ("view_count", (normalize_int(summary.get("view_count")) or 0) > 0),
        ("corrupted_view_count", (normalize_int(summary.get("corrupted_view_count")) or 0) > 0),
    ]
    for name, ok in checks:
        if not ok:
            critical.append(f"invalid condition summary field: {name}")
    schema_warnings = _ranking_schema_warnings(load_csv_rows(signal_dir / "offline_viewtrust_rankings.csv"))
    required_schema_warnings = [
        warning
        for warning in schema_warnings
        if "view_name" in warning or "offline_viewtrust_risk" in warning or "has no rows" in warning
    ]
    critical.extend(required_schema_warnings)
    warnings.extend(warning for warning in schema_warnings if warning not in required_schema_warnings)
    return ("failed_validation", critical + warnings) if critical else ("ok", warnings)


def load_condition_rankings(signal_dir: Path) -> list[dict[str, Any]]:
    rankings = load_csv_rows(signal_dir / "offline_viewtrust_rankings.csv")
    signals_by_view = {
        _view_name(row): row
        for row in load_csv_rows(signal_dir / "offline_viewtrust_signals.csv")
        if _view_name(row)
    }
    merged: list[dict[str, Any]] = []
    for row in rankings:
        view_name = _view_name(row)
        combined = dict(signals_by_view.get(view_name, {}))
        combined.update(row)
        merged.append(combined)
    return infer_rankings_if_needed(merged)


def load_condition_signals(signal_dir: Path) -> list[dict[str, str]]:
    return load_csv_rows(signal_dir / "offline_viewtrust_signals.csv")


def load_condition_ablation(signal_dir: Path) -> list[dict[str, str]]:
    return load_csv_rows(signal_dir / "offline_viewtrust_signal_ablation.csv")


def infer_rankings_if_needed(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    if any(_rank(row) is None for row in rows):
        ranked = sorted(rows, key=lambda row: (_risk(row) or 0.0, _view_name(row)), reverse=True)
        for index, row in enumerate(ranked, start=1):
            row["rank"] = index
        return ranked
    return sorted(rows, key=lambda row: _rank(row) or 10**9)


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def _std(values: list[float]) -> float | None:
    if not values:
        return None
    mean = statistics.fmean(values)
    return (statistics.fmean((value - mean) ** 2 for value in values)) ** 0.5


def build_cross_condition_view_rank_table(
    condition_rankings: dict[str, list[dict[str, Any]]],
    top_k: int,
) -> list[dict[str, Any]]:
    by_view: dict[str, dict[str, Any]] = defaultdict(dict)
    conditions = list(condition_rankings)
    for condition, rankings in condition_rankings.items():
        for row in rankings:
            view_name = _view_name(row)
            if view_name:
                by_view[view_name][condition] = row

    output: list[dict[str, Any]] = []
    for view_name, per_condition in by_view.items():
        ranks = [float(rank) for row in per_condition.values() if (rank := _rank(row)) is not None]
        risks = [float(risk) for row in per_condition.values() if (risk := _risk(row)) is not None]
        reasons = [_main_reason(row) for row in per_condition.values() if _main_reason(row)]
        corrupted_flags = [_was_corrupted(row) for row in per_condition.values() if _was_corrupted(row) is not None]
        row: dict[str, Any] = {
            "view_name": view_name,
            "condition_count_observed": len(per_condition),
            "was_corrupted_in_any_condition": any(corrupted_flags),
            "corrupted_condition_count": sum(1 for value in corrupted_flags if value),
            "top1_count": sum(1 for rank in ranks if rank <= 1),
            "top3_count": sum(1 for rank in ranks if rank <= 3),
            "top5_count": sum(1 for rank in ranks if rank <= 5),
            "mean_rank": _mean(ranks),
            "median_rank": _median(ranks),
            "best_rank": min(ranks) if ranks else "",
            "worst_rank": max(ranks) if ranks else "",
            "rank_std": _std(ranks),
            "mean_risk": _mean(risks),
            "median_risk": _median(risks),
            "risk_std": _std(risks),
            "most_common_main_reason": Counter(reasons).most_common(1)[0][0] if reasons else "",
        }
        for condition in conditions:
            condition_row = per_condition.get(condition, {})
            rank = _rank(condition_row) if condition_row else None
            risk = _risk(condition_row) if condition_row else None
            was_corrupted = _was_corrupted(condition_row) if condition_row else None
            row[f"rank_{condition}"] = rank
            row[f"risk_{condition}"] = risk
            row[f"was_corrupted_{condition}"] = was_corrupted
            row[f"main_reason_{condition}"] = _main_reason(condition_row) if condition_row else ""
            row[f"in_top1_{condition}"] = rank is not None and rank <= 1
            row[f"in_top3_{condition}"] = rank is not None and rank <= 3
            row[f"in_top5_{condition}"] = rank is not None and rank <= 5
        output.append(row)
    return sorted(
        output,
        key=lambda row: (
            normalize_float(row.get("mean_rank")) or 10**9,
            -(normalize_int(row.get("top5_count")) or 0),
            str(row.get("view_name", "")),
        ),
    )


def build_repeated_top_views(cross_condition_rows: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in cross_condition_rows:
        top1_count = normalize_int(row.get("top1_count")) or 0
        top3_count = normalize_int(row.get("top3_count")) or 0
        top5_count = normalize_int(row.get("top5_count")) or 0
        if not (top1_count > 0 or top3_count >= 2 or top5_count >= 2):
            continue
        if top1_count >= 2:
            repeat_type = "repeated_top1"
        elif top3_count >= 2:
            repeat_type = "repeated_top3"
        elif top5_count >= 2:
            repeat_type = "repeated_top5"
        else:
            repeat_type = "single_top1_only"
        conditions_top1 = []
        conditions_top3 = []
        conditions_top5 = []
        for key, value in row.items():
            if key.startswith("in_top1_") and value is True:
                conditions_top1.append(key.removeprefix("in_top1_"))
            if key.startswith("in_top3_") and value is True:
                conditions_top3.append(key.removeprefix("in_top3_"))
            if key.startswith("in_top5_") and value is True:
                conditions_top5.append(key.removeprefix("in_top5_"))
        rows.append(
            {
                "view_name": row.get("view_name", ""),
                "top1_count": top1_count,
                "top3_count": top3_count,
                "top5_count": top5_count,
                "conditions_top1": ";".join(conditions_top1),
                "conditions_top3": ";".join(conditions_top3),
                "conditions_top5": ";".join(conditions_top5),
                "was_corrupted_in_any_condition": row.get("was_corrupted_in_any_condition"),
                "corrupted_condition_count": row.get("corrupted_condition_count"),
                "mean_rank": row.get("mean_rank"),
                "median_rank": row.get("median_rank"),
                "mean_risk": row.get("mean_risk"),
                "median_risk": row.get("median_risk"),
                "most_common_main_reason": row.get("most_common_main_reason"),
                "repeat_type": repeat_type,
            }
        )
    return rows


def build_false_positive_topk_views(
    condition_rankings: dict[str, list[dict[str, Any]]],
    cross_condition_rows: list[dict[str, Any]],
    top_k: int,
) -> list[dict[str, Any]]:
    cross_by_view = {str(row.get("view_name")): row for row in cross_condition_rows}
    counts = Counter(
        _view_name(row)
        for rankings in condition_rankings.values()
        for row in rankings
        if (_rank(row) or 10**9) <= top_k and _was_corrupted(row) is False
    )
    rows: list[dict[str, Any]] = []
    for condition, rankings in condition_rankings.items():
        for row in rankings:
            rank = _rank(row)
            if rank is None or rank > top_k or _was_corrupted(row) is not False:
                continue
            view_name = _view_name(row)
            cross = cross_by_view.get(view_name, {})
            output = {
                "condition": condition,
                "view_name": view_name,
                "rank": rank,
                "offline_viewtrust_risk": _risk(row),
                "main_reason": _main_reason(row),
                "is_repeated_false_positive": counts[view_name] > 1,
                "repeated_false_positive_count": counts[view_name],
                "mean_rank_across_conditions": cross.get("mean_rank", ""),
                "mean_risk_across_conditions": cross.get("mean_risk", ""),
                "top5_count_across_conditions": cross.get("top5_count", ""),
            }
            output.update(_component_values(row))
            rows.append(output)
    return rows


def build_corrupted_view_rank_distribution(condition_rankings: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for condition, rankings in condition_rankings.items():
        for row in rankings:
            if _was_corrupted(row) is not True:
                continue
            output = {
                "condition": condition,
                "view_name": _view_name(row),
                "rank": _rank(row),
                "offline_viewtrust_risk": _risk(row),
                "in_top1": (_rank(row) or 10**9) <= 1,
                "in_top3": (_rank(row) or 10**9) <= 3,
                "in_top5": (_rank(row) or 10**9) <= 5,
                "main_reason": _main_reason(row),
            }
            output.update(_component_values(row))
            rows.append(output)
    return sorted(rows, key=lambda row: (str(row.get("condition", "")), normalize_int(row.get("rank")) or 10**9))


def build_component_win_table(condition_ablations: dict[str, list[dict[str, str]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for condition, ablations in condition_ablations.items():
        enriched = []
        for row in ablations:
            enriched.append(
                {
                    "condition": condition,
                    "signal_name": row.get("signal_name", ""),
                    "corrupted_in_top_k": _first(row, ["corrupted_in_top_k", "corrupted_in_topk"]),
                    "precision_at_k": row.get("precision_at_k", ""),
                    "recall_at_k": row.get("recall_at_k", ""),
                    "risk_gap_corrupted_minus_uncorrupted": _first(
                        row,
                        ["risk_gap_corrupted_minus_uncorrupted", "score_gap"],
                    ),
                    "mean_corrupted_score": _first(row, ["mean_corrupted_score", "mean_corrupted_risk"]),
                    "mean_uncorrupted_score": _first(row, ["mean_uncorrupted_score", "mean_uncorrupted_risk"]),
                    "top1_view_name": row.get("top1_view_name", ""),
                    "top1_was_corrupted": row.get("top1_was_corrupted", ""),
                    "status": row.get("status", "ok"),
                }
            )
        ranked = sorted(
            enriched,
            key=lambda row: (
                -(normalize_float(row.get("recall_at_k")) or 0.0),
                -(normalize_float(row.get("precision_at_k")) or 0.0),
                -(normalize_float(row.get("risk_gap_corrupted_minus_uncorrupted")) or 0.0),
                str(row.get("signal_name", "")),
            ),
        )
        for rank, row in enumerate(ranked, start=1):
            row["rank_within_condition"] = rank
            rows.append(row)
    return sorted(
        rows,
        key=lambda row: (
            str(row.get("condition", "")),
            normalize_int(row.get("rank_within_condition")) or 10**9,
            str(row.get("signal_name", "")),
        ),
    )


def _signal_lookup(rows: list[dict[str, Any]], signal_name: str) -> dict[str, Any]:
    return next((row for row in rows if row.get("signal_name") == signal_name), {})


def _beats(left: dict[str, Any], right: dict[str, Any]) -> bool:
    if not left or not right:
        return False
    for key in ("recall_at_k", "precision_at_k", "risk_gap_corrupted_minus_uncorrupted"):
        left_value = normalize_float(left.get(key)) or 0.0
        right_value = normalize_float(right.get(key)) or 0.0
        if left_value > right_value:
            return True
        if left_value < right_value:
            return False
    return False


def _delta(left: dict[str, Any], right: dict[str, Any], key: str) -> float | None:
    if not left or not right:
        return None
    left_value = normalize_float(left.get(key))
    right_value = normalize_float(right.get(key))
    return left_value - right_value if left_value is not None and right_value is not None else None


def build_component_condition_summary(component_win_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in component_win_rows:
        by_condition[str(row.get("condition", ""))].append(row)
    rows: list[dict[str, Any]] = []
    for condition, items in sorted(by_condition.items()):
        best = sorted(items, key=lambda row: normalize_int(row.get("rank_within_condition")) or 10**9)[0]
        full = _signal_lookup(items, "full_signal")
        loss = _signal_lookup(items, "loss_only")
        lifecycle = _signal_lookup(items, "lifecycle_only")
        rows.append(
            {
                "condition": condition,
                "best_signal_name": best.get("signal_name", ""),
                "best_signal_recall": best.get("recall_at_k", ""),
                "best_signal_precision": best.get("precision_at_k", ""),
                "best_signal_risk_gap": best.get("risk_gap_corrupted_minus_uncorrupted", ""),
                "full_signal_precision": full.get("precision_at_k", ""),
                "full_signal_recall": full.get("recall_at_k", ""),
                "full_signal_risk_gap": full.get("risk_gap_corrupted_minus_uncorrupted", ""),
                "loss_only_precision": loss.get("precision_at_k", ""),
                "loss_only_recall": loss.get("recall_at_k", ""),
                "loss_only_risk_gap": loss.get("risk_gap_corrupted_minus_uncorrupted", ""),
                "lifecycle_only_precision": lifecycle.get("precision_at_k", ""),
                "lifecycle_only_recall": lifecycle.get("recall_at_k", ""),
                "lifecycle_only_risk_gap": lifecycle.get("risk_gap_corrupted_minus_uncorrupted", ""),
                "full_minus_loss_recall": _delta(full, loss, "recall_at_k"),
                "full_minus_loss_precision": _delta(full, loss, "precision_at_k"),
                "full_minus_loss_risk_gap": _delta(full, loss, "risk_gap_corrupted_minus_uncorrupted"),
                "full_minus_lifecycle_recall": _delta(full, lifecycle, "recall_at_k"),
                "full_minus_lifecycle_precision": _delta(full, lifecycle, "precision_at_k"),
                "full_minus_lifecycle_risk_gap": _delta(full, lifecycle, "risk_gap_corrupted_minus_uncorrupted"),
                "does_full_beat_loss": _beats(full, loss),
                "does_full_beat_lifecycle": _beats(full, lifecycle),
            }
        )
    return rows


def build_component_gap_table(component_win_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fields = [
        "condition",
        "signal_name",
        "mean_corrupted_score",
        "mean_uncorrupted_score",
        "risk_gap_corrupted_minus_uncorrupted",
        "precision_at_k",
        "recall_at_k",
        "corrupted_in_top_k",
        "top1_view_name",
        "top1_was_corrupted",
        "status",
    ]
    return [{field: row.get(field, "") for field in fields} for row in component_win_rows]


def build_summary_json(
    *,
    scene: str,
    conditions: list[str],
    top_k: int,
    pr14_summary: dict[str, Any],
    condition_rankings: dict[str, list[dict[str, Any]]],
    cross_condition_rows: list[dict[str, Any]],
    repeated_rows: list[dict[str, Any]],
    false_positive_rows: list[dict[str, Any]],
    component_summary_rows: list[dict[str, Any]],
    warnings: list[str],
) -> dict[str, Any]:
    repeated_top1 = [row.get("view_name") for row in repeated_rows if row.get("repeat_type") == "repeated_top1"]
    repeated_top3 = [
        row.get("view_name")
        for row in repeated_rows
        if (normalize_int(row.get("top3_count")) or 0) >= 2
    ]
    repeated_top5 = [
        row.get("view_name")
        for row in repeated_rows
        if (normalize_int(row.get("top5_count")) or 0) >= 2
    ]
    false_positive_counts = Counter(row.get("view_name", "") for row in false_positive_rows)
    corrupted_topk = {}
    for condition, rankings in condition_rankings.items():
        corrupted_rows = [row for row in rankings if _was_corrupted(row) is True]
        corrupted_topk[condition] = bool(corrupted_rows) and all((_rank(row) or 10**9) <= top_k for row in corrupted_rows)
    return {
        "schema_name": "viewtrust.offline_signal.rank_consistency.summary",
        "schema_version": 1,
        "scene": scene,
        "conditions": conditions,
        "top_k": top_k,
        "condition_count": len(conditions),
        "condition_count_valid": len(condition_rankings),
        "view_count": len(cross_condition_rows),
        "repeated_top1_views": repeated_top1,
        "repeated_top3_views": repeated_top3,
        "repeated_top5_views": repeated_top5,
        "false_positive_count": len(false_positive_rows),
        "repeated_false_positive_views": [
            view_name for view_name, count in sorted(false_positive_counts.items()) if view_name and count > 1
        ],
        "corrupted_views_all_in_topk_by_condition": corrupted_topk,
        "mean_precision_at_k_from_pr14": pr14_summary.get("mean_precision_at_k"),
        "mean_recall_at_k_from_pr14": pr14_summary.get("mean_recall_at_k"),
        "best_ablation_signal_counts": pr14_summary.get("best_ablation_signal_counts", {}),
        "full_signal_win_count_over_loss": sum(
            1 for row in component_summary_rows if row.get("does_full_beat_loss") is True
        ),
        "full_signal_win_count_over_lifecycle": sum(
            1 for row in component_summary_rows if row.get("does_full_beat_lifecycle") is True
        ),
        "conditions_where_loss_only_wins": [
            row.get("condition") for row in component_summary_rows if row.get("best_signal_name") == "loss_only"
        ],
        "conditions_where_lifecycle_only_wins": [
            row.get("condition") for row in component_summary_rows if row.get("best_signal_name") == "lifecycle_only"
        ],
        "observation_only": True,
        "training_intervention": False,
        "defense_enabled": False,
        "uses_corruption_labels_for_scoring": False,
        "uses_corruption_labels_for_evaluation": True,
        "warnings": warnings,
    }


def write_rank_consistency_report(
    path: Path,
    *,
    summary: dict[str, Any],
    multi_condition_dir: Path,
    input_root: Path,
    repeated_rows: list[dict[str, Any]],
    false_positive_rows: list[dict[str, Any]],
    corrupted_rows: list[dict[str, Any]],
    component_summary_rows: list[dict[str, Any]],
    warnings: list[str],
) -> None:
    repeated_lines = [
        f"- `{row.get('view_name')}` repeat=`{row.get('repeat_type')}` top5_count=`{row.get('top5_count')}`"
        for row in repeated_rows[:20]
    ]
    false_positive_lines = [
        f"- `{row.get('condition')}` `{row.get('view_name')}` rank=`{row.get('rank')}` reason=`{row.get('main_reason')}`"
        for row in false_positive_rows[:20]
    ]
    corrupted_lines = [
        f"- `{row.get('condition')}` `{row.get('view_name')}` rank=`{row.get('rank')}` reason=`{row.get('main_reason')}`"
        for row in corrupted_rows[:20]
    ]
    component_lines = [
        (
            f"- `{row.get('condition')}` best=`{row.get('best_signal_name')}` "
            f"full_minus_loss_recall=`{row.get('full_minus_loss_recall')}` "
            f"full_minus_lifecycle_recall=`{row.get('full_minus_lifecycle_recall')}`"
        )
        for row in component_summary_rows
    ]
    warning_lines = [f"- {warning}" for warning in warnings]
    report = "\n".join(
        [
            "# PR15 Cross-condition Rank Consistency and Component Diagnosis",
            "",
            "## Purpose",
            "PR15 diagnoses whether existing offline ViewTrust rankings are stable across natural corruption conditions and whether component ablations suggest value beyond loss-only ranking.",
            "",
            "## Inputs",
            f"- Multi-condition directory: `{multi_condition_dir}`",
            f"- Per-condition input root: `{input_root}`",
            f"- Scene: `{summary.get('scene')}`",
            f"- Conditions: `{', '.join(summary.get('conditions', []))}`",
            f"- Top-k: `{summary.get('top_k')}`",
            "",
            "## Offline-only guarantee",
            REPORT_DISCLAIMER,
            "",
            "## Cross-condition repeated top views",
            *(repeated_lines or ["- No repeated top views found in the analyzed outputs."]),
            "",
            "## False positive analysis",
            "False positive here means an uncorrupted view ranked within top-k during post-hoc evaluation. It may still be a naturally high-impact or hard-but-useful view.",
            *(false_positive_lines or ["- No uncorrupted top-k views found."]),
            "",
            "## Corrupted view rank distribution",
            *(corrupted_lines or ["- No corrupted rows were available for evaluation summaries."]),
            "",
            "## Component diagnosis",
            *(component_lines or ["- No component ablation rows were available."]),
            "",
            "## Interpretation guidance",
            "- Treat repeated high ranks as offline diagnostic evidence, not as causal proof.",
            "- Compare `full_signal`, `loss_only`, and `lifecycle_only` before attributing success to lifecycle information.",
            "- A repeated clean top-k view can indicate a candidate high-impact view that needs multi-seed validation.",
            "",
            "## Limitations",
            "- single scene",
            "- chair mini only",
            "- fixed corrupted-view subset",
            "- single seed",
            "- natural corruption only",
            "- no malicious attack validation yet",
            "- no multi-seed validation yet",
            "- no cross-object validation yet",
            "- no full-scene validation yet",
            "- no training-time intervention yet",
            "- no causal proof",
            "",
            "## Recommended next experiments",
            "1. multi-seed chair validation",
            "2. corrupted-subset variation",
            "3. cross-object synthetic scene validation",
            "4. longer training / full-scene observation validation",
            "5. synthetic target-poison benchmark",
            "6. suspicious Gaussian localization evaluation",
            "7. only after these, training-time intervention",
            "",
            "## Warnings",
            *(warning_lines or ["- None"]),
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")


def _file_type(path: Path) -> str:
    return path.suffix.lstrip(".") if path.suffix else ("directory" if path.is_dir() else "")


def _artifact_rows(
    *,
    output_dir: Path,
    multi_condition_dir: Path,
    input_root: Path,
    condition_dirs: dict[str, Path | None],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = [
        {
            "relative_path": "multi_condition_dir",
            "path": str(multi_condition_dir),
            "exists": str(multi_condition_dir.exists()).lower(),
            "file_type": "directory" if multi_condition_dir.is_dir() else "",
            "size_bytes": "",
            "required": "true",
            "artifact_group": "input_pr14",
            "description": "PR14.1 multi-condition aggregation directory",
        },
        {
            "relative_path": "input_root",
            "path": str(input_root),
            "exists": str(input_root.exists()).lower(),
            "file_type": "directory" if input_root.is_dir() else "",
            "size_bytes": "",
            "required": "true",
            "artifact_group": "input_conditions",
            "description": "Root containing per-condition PR13/PR14-input outputs",
        },
    ]
    for condition, signal_dir in condition_dirs.items():
        path = signal_dir or input_root / f"offline_viewtrust_{condition}_pr14_input"
        rows.append(
            {
                "relative_path": f"condition_dir/{condition}",
                "path": str(path),
                "exists": str(path.exists()).lower(),
                "file_type": "directory" if path.is_dir() else "",
                "size_bytes": "",
                "required": "false",
                "artifact_group": "input_condition",
                "description": f"Per-condition offline signal directory for {condition}",
            }
        )
    for name in PR15_OUTPUT_FILES:
        path = output_dir / name
        rows.append(
            {
                "relative_path": name,
                "path": str(path),
                "exists": str(path.exists()).lower(),
                "file_type": _file_type(path),
                "size_bytes": path.stat().st_size if path.is_file() else "",
                "required": "true",
                "artifact_group": "output_pr15",
                "description": "PR15 rank consistency output",
            }
        )
    return rows


def write_artifact_manifest(
    path: Path,
    *,
    output_dir: Path,
    multi_condition_dir: Path,
    input_root: Path,
    condition_dirs: dict[str, Path | None],
) -> None:
    fields = [
        "relative_path",
        "path",
        "exists",
        "file_type",
        "size_bytes",
        "required",
        "artifact_group",
        "description",
    ]
    rows = _artifact_rows(
        output_dir=output_dir,
        multi_condition_dir=multi_condition_dir,
        input_root=input_root,
        condition_dirs=condition_dirs,
    )
    write_csv_rows(path, rows, fields)
    rows = _artifact_rows(
        output_dir=output_dir,
        multi_condition_dir=multi_condition_dir,
        input_root=input_root,
        condition_dirs=condition_dirs,
    )
    write_csv_rows(path, rows, fields)
