#!/usr/bin/env python3
"""Build PR13 offline ViewTrust candidate signals from PR12.1 tables."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Any


SCHEMA_NAME = "viewtrust.offline_signal.summary"
SCHEMA_VERSION = 1
DEFAULT_CONFIG = Path("configs/offline_viewtrust_signal/default_pr13_signal.json")


def _bootstrap_project_imports() -> Path:
    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root))
    return project_root


def _json_file(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(fieldnames=fields, extrasaction="ignore", f=handle)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: "" if row.get(field) is None else row.get(field) for field in fields})


def _write_artifact_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    _write_csv(
        path,
        rows,
        [
            "relative_path",
            "path",
            "exists",
            "file_type",
            "size_bytes",
            "required",
            "artifact_group",
        ],
    )


def _float_or_none(value: Any) -> float | None:
    if value in ("", None):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _truthy(value: Any) -> bool:
    return str(value).lower() == "true" or value is True


def _delta(corrupt: Any, clean: Any) -> float | None:
    corrupt_value = _float_or_none(corrupt)
    clean_value = _float_or_none(clean)
    return corrupt_value - clean_value if corrupt_value is not None and clean_value is not None else None


def _safe_divide(numerator: Any, denominator: Any) -> float:
    numerator_value = _float_or_none(numerator)
    denominator_value = _float_or_none(denominator)
    if numerator_value is None or denominator_value in (None, 0.0):
        return 0.0
    return numerator_value / denominator_value


def _pick(*values: Any) -> Any:
    for value in values:
        if value not in ("", None):
            return value
    return ""


def _artifact_manifest_rows(
    *,
    clean_dir: Path,
    corrupt_dir: Path,
    comparison_dir: Path,
    output_dir: Path,
) -> list[dict[str, Any]]:
    inputs = [
        ("input_clean/view_influence_summary.json", clean_dir / "view_influence_summary.json", True, "input_clean"),
        ("input_clean/view_influence.csv", clean_dir / "view_influence.csv", True, "input_clean"),
        ("input_clean/view_lifecycle_attribution.csv", clean_dir / "view_lifecycle_attribution.csv", False, "input_clean"),
        ("input_clean/view_iteration_events.csv", clean_dir / "view_iteration_events.csv", False, "input_clean"),
        ("input_clean/view_influence_report.md", clean_dir / "view_influence_report.md", False, "input_clean"),
        ("input_corrupt/view_influence_summary.json", corrupt_dir / "view_influence_summary.json", True, "input_corrupt"),
        ("input_corrupt/view_influence.csv", corrupt_dir / "view_influence.csv", True, "input_corrupt"),
        ("input_corrupt/view_lifecycle_attribution.csv", corrupt_dir / "view_lifecycle_attribution.csv", False, "input_corrupt"),
        ("input_corrupt/view_iteration_events.csv", corrupt_dir / "view_iteration_events.csv", False, "input_corrupt"),
        ("input_corrupt/view_influence_report.md", corrupt_dir / "view_influence_report.md", False, "input_corrupt"),
        ("input_comparison/view_influence_comparison_summary.json", comparison_dir / "view_influence_comparison_summary.json", True, "input_comparison"),
        ("input_comparison/view_influence_comparison.csv", comparison_dir / "view_influence_comparison.csv", True, "input_comparison"),
        ("input_comparison/view_influence_comparison_report.md", comparison_dir / "view_influence_comparison_report.md", False, "input_comparison"),
    ]
    outputs = [
        "offline_viewtrust_summary.json",
        "offline_viewtrust_signals.csv",
        "offline_viewtrust_rankings.csv",
        "offline_viewtrust_group_metrics.csv",
        "offline_viewtrust_signal_ablation.csv",
        "offline_viewtrust_config.json",
        "offline_viewtrust_report.md",
        "offline_viewtrust_artifact_manifest.csv",
    ]
    rows = []
    for relative_path, path, required, group in inputs:
        rows.append(
            {
                "relative_path": relative_path,
                "path": str(path),
                "exists": str(path.exists()).lower(),
                "file_type": path.suffix.lstrip("."),
                "size_bytes": path.stat().st_size if path.is_file() else "",
                "required": str(required).lower(),
                "artifact_group": group,
            }
        )
    for relative_path in outputs:
        path = output_dir / relative_path
        rows.append(
            {
                "relative_path": relative_path,
                "path": str(path),
                "exists": str(path.exists()).lower(),
                "file_type": path.suffix.lstrip("."),
                "size_bytes": path.stat().st_size if path.is_file() else "",
                "required": "true",
                "artifact_group": "output_pr13",
            }
        )
    return rows


def _load_config(project_root: Path, config_path: Path | None, output_dir: Path) -> dict[str, Any]:
    source = config_path if config_path is not None else project_root / DEFAULT_CONFIG
    config = _json_file(source)
    if not config:
        raise FileNotFoundError(f"offline signal config not found or empty: {source}")
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, output_dir / "offline_viewtrust_config.json")
    return config


def _component_warning(row: dict[str, Any]) -> str:
    warnings: list[str] = []
    if _float_or_none(_pick(row.get("mean_total_loss"), row.get("mean_loss"))) is None:
        warnings.append("missing_loss_feature")
    if _float_or_none(row.get("visibility_drop")) is None:
        warnings.append("missing_visibility_delta")
    for field in (
        "birth_event_count_after_view",
        "prune_death_count_after_view",
        "birth_survival_ratio_after_view",
        "birth_event_count_delta",
        "prune_death_count_delta",
    ):
        if _float_or_none(row.get(field)) is None:
            warnings.append(f"missing_{field}")
    return ";".join(warnings)


def _prepare_rows(
    *,
    clean_dir: Path,
    corrupt_dir: Path,
    comparison_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], dict[str, Any]]:
    clean_summary = _json_file(clean_dir / "view_influence_summary.json")
    corrupt_summary = _json_file(corrupt_dir / "view_influence_summary.json")
    comparison_summary = _json_file(comparison_dir / "view_influence_comparison_summary.json")
    clean_rows = {
        row.get("view_name", ""): row
        for row in _csv_rows(clean_dir / "view_influence.csv")
        if row.get("view_name")
    }
    corrupt_rows = {
        row.get("view_name", ""): row
        for row in _csv_rows(corrupt_dir / "view_influence.csv")
        if row.get("view_name")
    }
    comparison_rows = {
        row.get("view_name", ""): row
        for row in _csv_rows(comparison_dir / "view_influence_comparison.csv")
        if row.get("view_name")
    }
    rows: list[dict[str, Any]] = []
    for view_name in sorted(set(clean_rows) | set(corrupt_rows) | set(comparison_rows)):
        clean = clean_rows.get(view_name, {})
        corrupt = corrupt_rows.get(view_name, {})
        comparison = comparison_rows.get(view_name, {})
        clean_visibility = _pick(comparison.get("clean_mean_visibility_ratio"), clean.get("mean_visibility_ratio"))
        corrupt_visibility = _pick(comparison.get("corrupt_mean_visibility_ratio"), corrupt.get("mean_visibility_ratio"))
        visibility_delta = _pick(
            comparison.get("visibility_ratio_delta"),
            _delta(corrupt_visibility, clean_visibility),
        )
        visibility_delta_float = _float_or_none(visibility_delta)
        birth_count = _pick(corrupt.get("birth_event_count_after_view"), comparison.get("corrupt_birth_event_count_after_view"))
        clone_birth_count = corrupt.get("clone_birth_count_after_view", "")
        split_birth_count = corrupt.get("split_birth_count_after_view", "")
        prune_count = _pick(corrupt.get("prune_death_count_after_view"), comparison.get("corrupt_prune_death_count_after_view"))
        survivor_count = corrupt.get("final_survivor_birth_count_after_view", "")
        dead_birth_count = corrupt.get("dead_birth_count_after_view", "")
        times_sampled = _pick(comparison.get("corrupt_times_sampled"), corrupt.get("times_sampled"), clean.get("times_sampled"), 0)
        row: dict[str, Any] = {
            "scene": _pick(corrupt_summary.get("scene"), clean_summary.get("scene"), comparison_summary.get("scene")),
            "clean_condition": _pick(clean_summary.get("condition"), comparison_summary.get("clean_condition")),
            "corrupt_condition": _pick(corrupt_summary.get("condition"), comparison_summary.get("corrupt_condition")),
            "view_name": view_name,
            "view_split": _pick(corrupt.get("view_split"), clean.get("view_split"), "train"),
            "was_corrupted": _pick(corrupt.get("was_corrupted"), comparison.get("was_corrupted"), "false"),
            "corruption_type": corrupt.get("corruption_type", ""),
            "clean_times_sampled": _pick(comparison.get("clean_times_sampled"), clean.get("times_sampled")),
            "corrupt_times_sampled": _pick(comparison.get("corrupt_times_sampled"), corrupt.get("times_sampled")),
            "times_sampled": times_sampled,
            "mean_loss": _pick(corrupt.get("mean_loss"), corrupt.get("mean_total_loss")),
            "mean_total_loss": _pick(corrupt.get("mean_total_loss"), corrupt.get("mean_loss")),
            "mean_l1_loss": corrupt.get("mean_l1_loss", ""),
            "mean_ssim_loss": corrupt.get("mean_ssim_loss", ""),
            "clean_mean_visibility_ratio": clean_visibility,
            "corrupt_mean_visibility_ratio": corrupt_visibility,
            "visibility_ratio_delta": visibility_delta,
            "visibility_drop": max(0.0, -(visibility_delta_float or 0.0)),
            "birth_event_count_after_view": birth_count,
            "clone_birth_count_after_view": clone_birth_count,
            "split_birth_count_after_view": split_birth_count,
            "prune_death_count_after_view": prune_count,
            "final_survivor_birth_count_after_view": survivor_count,
            "dead_birth_count_after_view": dead_birth_count,
            "birth_survival_ratio_after_view": _pick(
                corrupt.get("birth_survival_ratio_after_view"),
                comparison.get("corrupt_birth_survival_ratio_after_view"),
            ),
            "mean_view_psnr": corrupt.get("mean_view_psnr", ""),
            "mean_view_ssim": corrupt.get("mean_view_ssim", ""),
            "mean_view_l1": corrupt.get("mean_view_l1", ""),
            "birth_event_count_delta": _pick(
                comparison.get("birth_event_count_delta"),
                _delta(corrupt.get("birth_event_count_after_view"), clean.get("birth_event_count_after_view")),
            ),
            "prune_death_count_delta": _pick(
                comparison.get("prune_death_count_delta"),
                _delta(corrupt.get("prune_death_count_after_view"), clean.get("prune_death_count_after_view")),
            ),
            "birth_survival_ratio_delta": _pick(
                comparison.get("birth_survival_ratio_delta"),
                _delta(corrupt.get("birth_survival_ratio_after_view"), clean.get("birth_survival_ratio_after_view")),
            ),
        }
        row["birth_rate"] = _safe_divide(birth_count, times_sampled)
        row["clone_birth_rate"] = _safe_divide(clone_birth_count, times_sampled)
        row["split_birth_rate"] = _safe_divide(split_birth_count, times_sampled)
        row["prune_death_rate"] = _safe_divide(prune_count, times_sampled)
        row["dead_birth_rate"] = _safe_divide(dead_birth_count, max(_float_or_none(birth_count) or 0.0, 1.0))
        row["final_survivor_birth_rate"] = _safe_divide(
            survivor_count,
            max(_float_or_none(birth_count) or 0.0, 1.0),
        )
        row["component_warnings"] = _component_warning(row)
        rows.append(row)
    return rows, clean_summary, corrupt_summary, comparison_summary


def _rank_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    from viewtrust.analysis.offline_signals import rank_descending

    ranked_by_risk = rank_descending(rows, "offline_viewtrust_risk")
    risk_ranks = {id(row): rank for rank, row in enumerate(ranked_by_risk, start=1)}
    consistency_ranked = sorted(
        rows,
        key=lambda row: (_float_or_none(row.get("offline_viewtrust_consistency")) or 0.0, row.get("view_name", "")),
        reverse=True,
    )
    consistency_ranks = {id(row): rank for rank, row in enumerate(consistency_ranked, start=1)}
    output = []
    for row in ranked_by_risk:
        updated = dict(row)
        updated["risk_rank"] = risk_ranks[id(row)]
        updated["consistency_rank"] = consistency_ranks[id(row)]
        output.append(updated)
    return output


def _ranking_rows(signal_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    component_names = [
        "loss_component",
        "visibility_component",
        "birth_component",
        "prune_component",
        "survival_component",
        "delta_component",
        "lifecycle_component",
    ]
    rows = []
    for rank, row in enumerate(signal_rows, start=1):
        main_reason = max(
            component_names,
            key=lambda key: _float_or_none(row.get(key)) or 0.0,
        )
        rows.append(
            {
                "rank": rank,
                "view_name": row.get("view_name", ""),
                "was_corrupted": row.get("was_corrupted", ""),
                "corruption_type": row.get("corruption_type", ""),
                "offline_viewtrust_risk": row.get("offline_viewtrust_risk", ""),
                "offline_viewtrust_consistency": row.get("offline_viewtrust_consistency", ""),
                "loss_component": row.get("loss_component", ""),
                "visibility_component": row.get("visibility_component", ""),
                "birth_component": row.get("birth_component", ""),
                "prune_component": row.get("prune_component", ""),
                "survival_component": row.get("survival_component", ""),
                "delta_component": row.get("delta_component", ""),
                "lifecycle_component": row.get("lifecycle_component", ""),
                "main_reason": main_reason,
            }
        )
    return rows


def _summary(
    *,
    signal_rows: list[dict[str, Any]],
    group_rows: list[dict[str, Any]],
    ablation_rows: list[dict[str, Any]],
    clean_summary: dict[str, Any],
    corrupt_summary: dict[str, Any],
    comparison_summary: dict[str, Any],
    config: dict[str, Any],
    top_k: int,
    warnings: list[str],
) -> dict[str, Any]:
    from viewtrust.analysis.offline_signals import precision_at_k, recall_at_k

    top_count = min(top_k, len(signal_rows))
    corrupted_count = sum(1 for row in signal_rows if _truthy(row.get("was_corrupted")))
    uncorrupted_count = len(signal_rows) - corrupted_count
    corrupted_group = next((row for row in group_rows if row.get("group") == "corrupted"), {})
    uncorrupted_group = next((row for row in group_rows if row.get("group") == "uncorrupted"), {})
    best_ablation = max(
        ablation_rows,
        key=lambda row: (
            _float_or_none(row.get("recall_at_k")) or 0.0,
            _float_or_none(row.get("precision_at_k")) or 0.0,
            _float_or_none(row.get("score_gap")) or 0.0,
        ),
        default={},
    )
    labels = config.get("labels", {})
    return {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "scene": _pick(corrupt_summary.get("scene"), clean_summary.get("scene"), comparison_summary.get("scene")),
        "clean_condition": _pick(clean_summary.get("condition"), comparison_summary.get("clean_condition")),
        "corrupt_condition": _pick(corrupt_summary.get("condition"), comparison_summary.get("corrupt_condition")),
        "clean_run_id": _pick(clean_summary.get("run_id"), comparison_summary.get("clean_run_id")),
        "corrupt_run_id": _pick(corrupt_summary.get("run_id"), comparison_summary.get("corrupt_run_id")),
        "view_count": len(signal_rows),
        "corrupted_view_count": corrupted_count,
        "uncorrupted_view_count": uncorrupted_count,
        "observation_only": bool(clean_summary.get("observation_only") is True and corrupt_summary.get("observation_only") is True),
        "uses_corruption_labels_for_scoring": labels.get("use_corruption_labels_for_scoring") is True,
        "uses_corruption_labels_for_evaluation": labels.get("use_corruption_labels_for_evaluation") is True,
        "training_intervention": False,
        "defense_enabled": False,
        "top_k": top_count,
        "top_views_by_offline_viewtrust_risk": [
            {
                "rank": row.get("risk_rank"),
                "view_name": row.get("view_name"),
                "was_corrupted": row.get("was_corrupted"),
                "offline_viewtrust_risk": row.get("offline_viewtrust_risk"),
                "main_reason": ranking.get("main_reason"),
            }
            for row, ranking in zip(signal_rows[:top_count], _ranking_rows(signal_rows[:top_count]), strict=False)
        ],
        "corrupted_in_top_k": sum(1 for row in signal_rows[:top_count] if _truthy(row.get("was_corrupted"))),
        "precision_at_k": precision_at_k(signal_rows, "was_corrupted", "offline_viewtrust_risk", top_count),
        "recall_at_k": recall_at_k(signal_rows, "was_corrupted", "offline_viewtrust_risk", top_count),
        "mean_corrupted_risk": corrupted_group.get("mean_offline_viewtrust_risk"),
        "mean_uncorrupted_risk": uncorrupted_group.get("mean_offline_viewtrust_risk"),
        "risk_gap_corrupted_minus_uncorrupted": (
            (_float_or_none(corrupted_group.get("mean_offline_viewtrust_risk")) or 0.0)
            - (_float_or_none(uncorrupted_group.get("mean_offline_viewtrust_risk")) or 0.0)
        ),
        "best_ablation_signal": best_ablation.get("signal_name", ""),
        "warnings": warnings,
    }


def _markdown(summary: dict[str, Any], ranking_rows: list[dict[str, Any]], ablation_rows: list[dict[str, Any]]) -> str:
    top_lines = [
        f"- {row.get('rank')}. `{row.get('view_name')}` risk=`{row.get('offline_viewtrust_risk')}` reason=`{row.get('main_reason')}` corrupted=`{row.get('was_corrupted')}`"
        for row in ranking_rows[: summary.get("top_k", 5)]
    ]
    ablation_lines = [
        f"- `{row.get('signal_name')}` precision@k=`{row.get('precision_at_k')}` recall@k=`{row.get('recall_at_k')}` gap=`{row.get('score_gap')}`"
        for row in ablation_rows
    ]
    return "\n".join(
        [
            "# Offline ViewTrust Signal Report",
            "",
            "This report proposes offline candidate ViewTrust signals. The signal is not used during training.",
            "This offline candidate signal is not a trust score used during training.",
            "Corruption labels are used only after scoring for evaluation.",
            "A high offline risk score does not prove maliciousness. This is not a defense and not a poison classifier.",
            "",
            "## Inputs",
            f"- Scene: `{summary.get('scene')}`",
            f"- Conditions: `{summary.get('clean_condition')}` vs `{summary.get('corrupt_condition')}`",
            f"- Clean run: `{summary.get('clean_run_id')}`",
            f"- Corrupt run: `{summary.get('corrupt_run_id')}`",
            "",
            "## Observation-Only Guarantee",
            f"- Observation only: `{summary.get('observation_only')}`",
            f"- Uses corruption labels for scoring: `{summary.get('uses_corruption_labels_for_scoring')}`",
            f"- Uses corruption labels for evaluation: `{summary.get('uses_corruption_labels_for_evaluation')}`",
            f"- Training intervention: `{summary.get('training_intervention')}`",
            f"- Defense enabled: `{summary.get('defense_enabled')}`",
            "",
            "## Signal Components",
            "- `loss_component`: robust high-loss signal.",
            "- `visibility_component`: robust visibility-drop signal.",
            "- `birth_component`: robust birth-rate signal.",
            "- `prune_component`: robust prune-death-rate signal.",
            "- `survival_component`: robust low birth-survival signal.",
            "- `delta_component`: robust clean-vs-corrupt delta signal.",
            "- `offline_viewtrust_risk`: weighted offline candidate risk signal.",
            "- `offline_viewtrust_consistency`: `1 / (1 + risk)`.",
            "",
            "## Top-Ranked Views",
            *top_lines,
            "",
            "## Corrupted vs Uncorrupted Group Summary",
            f"- Corrupted views: `{summary.get('corrupted_view_count')}`",
            f"- Uncorrupted views: `{summary.get('uncorrupted_view_count')}`",
            f"- Mean corrupted risk: `{summary.get('mean_corrupted_risk')}`",
            f"- Mean uncorrupted risk: `{summary.get('mean_uncorrupted_risk')}`",
            f"- Risk gap: `{summary.get('risk_gap_corrupted_minus_uncorrupted')}`",
            "",
            "## Component Ablation",
            *ablation_lines,
            "",
            "## Interpretation Guidance",
            "- Use careful language: candidate offline signal, lifecycle anomaly ranking, and post-hoc evidence.",
            "- Do not describe this as detection, operational defense, or view rejection.",
            "",
            "## Known Limitations",
            "- PR13 does not convert temporal source-view attribution into causal attribution.",
            "- Lifecycle events are associated with the sampled source view context, not proven caused by that view alone.",
            "- Results should be evaluated across multiple corruptions and seeds before any method claim.",
            "",
            "## Recommended Next Experiments",
            "- Run the same offline signal analysis across all PR10 natural corruption conditions.",
            "- Repeat with multiple seeds and compare ablation stability.",
            "",
            "## Warnings",
            *[f"- {warning}" for warning in summary.get("warnings", [])],
            "" if summary.get("warnings") else "- None",
            "",
        ]
    )


def build_offline_viewtrust_signals(
    *,
    clean_view_influence_dir: Path,
    corrupt_view_influence_dir: Path,
    view_influence_comparison_dir: Path,
    output_dir: Path,
    signal_config: Path | None,
    top_k: int,
    write_markdown: bool,
    quiet: bool,
) -> dict[str, Any]:
    project_root = _bootstrap_project_imports()
    from viewtrust.analysis.offline_signals import (
        compute_ablation_metrics,
        compute_group_metrics,
        compute_signal_components,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    config = _load_config(project_root, signal_config, output_dir)
    prepared_rows, clean_summary, corrupt_summary, comparison_summary = _prepare_rows(
        clean_dir=clean_view_influence_dir,
        corrupt_dir=corrupt_view_influence_dir,
        comparison_dir=view_influence_comparison_dir,
    )
    signal_rows = _rank_rows(compute_signal_components(prepared_rows, config))
    ranking_rows = _ranking_rows(signal_rows)
    group_rows = compute_group_metrics(signal_rows)
    ablation_rows = compute_ablation_metrics(signal_rows, top_k=top_k)
    warnings = sorted(
        {
            warning
            for row in signal_rows
            for warning in str(row.get("component_warnings", "")).split(";")
            if warning
        }
    )
    if config.get("labels", {}).get("use_corruption_labels_for_scoring") is True:
        warnings.append("config unexpectedly enables corruption labels for scoring")
    summary = _summary(
        signal_rows=signal_rows,
        group_rows=group_rows,
        ablation_rows=ablation_rows,
        clean_summary=clean_summary,
        corrupt_summary=corrupt_summary,
        comparison_summary=comparison_summary,
        config=config,
        top_k=top_k,
        warnings=warnings,
    )

    signal_fields = [
        "scene", "clean_condition", "corrupt_condition", "view_name", "view_split",
        "was_corrupted", "corruption_type", "clean_times_sampled", "corrupt_times_sampled",
        "times_sampled", "mean_loss", "mean_total_loss", "mean_l1_loss", "mean_ssim_loss",
        "clean_mean_visibility_ratio", "corrupt_mean_visibility_ratio",
        "visibility_ratio_delta", "visibility_drop", "birth_event_count_after_view",
        "clone_birth_count_after_view", "split_birth_count_after_view",
        "prune_death_count_after_view", "final_survivor_birth_count_after_view",
        "dead_birth_count_after_view", "birth_survival_ratio_after_view",
        "birth_rate", "clone_birth_rate", "split_birth_rate", "prune_death_rate",
        "dead_birth_rate", "final_survivor_birth_rate", "birth_event_count_delta",
        "prune_death_count_delta", "birth_survival_ratio_delta", "loss_component",
        "visibility_component", "birth_component", "prune_component", "survival_component",
        "delta_component", "lifecycle_component", "offline_viewtrust_risk",
        "offline_viewtrust_consistency", "risk_rank", "consistency_rank",
        "component_warnings",
    ]
    ranking_fields = [
        "rank", "view_name", "was_corrupted", "corruption_type",
        "offline_viewtrust_risk", "offline_viewtrust_consistency",
        "loss_component", "visibility_component", "birth_component", "prune_component",
        "survival_component", "delta_component", "lifecycle_component", "main_reason",
    ]
    group_fields = [
        "group", "view_count", "mean_offline_viewtrust_risk",
        "median_offline_viewtrust_risk", "max_offline_viewtrust_risk",
        "mean_loss_component", "mean_visibility_component", "mean_birth_component",
        "mean_prune_component", "mean_survival_component", "mean_delta_component",
        "mean_lifecycle_component", "mean_offline_viewtrust_consistency",
    ]
    ablation_fields = [
        "signal_name", "top1_view_name", "top1_was_corrupted", "topk",
        "corrupted_in_topk", "precision_at_k", "recall_at_k",
        "mean_corrupted_rank", "median_corrupted_rank", "mean_corrupted_score",
        "mean_uncorrupted_score", "score_gap",
    ]
    _write_csv(output_dir / "offline_viewtrust_signals.csv", signal_rows, signal_fields)
    _write_csv(output_dir / "offline_viewtrust_rankings.csv", ranking_rows, ranking_fields)
    _write_csv(output_dir / "offline_viewtrust_group_metrics.csv", group_rows, group_fields)
    _write_csv(output_dir / "offline_viewtrust_signal_ablation.csv", ablation_rows, ablation_fields)
    if write_markdown:
        (output_dir / "offline_viewtrust_report.md").write_text(
            _markdown(summary, ranking_rows, ablation_rows),
            encoding="utf-8",
        )
    else:
        (output_dir / "offline_viewtrust_report.md").write_text(
            _markdown(summary, ranking_rows, ablation_rows),
            encoding="utf-8",
        )
    (output_dir / "offline_viewtrust_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest_rows = _artifact_manifest_rows(
        clean_dir=clean_view_influence_dir,
        corrupt_dir=corrupt_view_influence_dir,
        comparison_dir=view_influence_comparison_dir,
        output_dir=output_dir,
    )
    manifest_path = output_dir / "offline_viewtrust_artifact_manifest.csv"
    _write_artifact_manifest(manifest_path, manifest_rows)
    refreshed_manifest_rows = _artifact_manifest_rows(
        clean_dir=clean_view_influence_dir,
        corrupt_dir=corrupt_view_influence_dir,
        comparison_dir=view_influence_comparison_dir,
        output_dir=output_dir,
    )
    _write_artifact_manifest(manifest_path, refreshed_manifest_rows)
    if not quiet:
        print(f"offline ViewTrust signal rows: {len(signal_rows)}", file=sys.stderr)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-view-influence-dir", required=True, type=Path)
    parser.add_argument("--corrupt-view-influence-dir", required=True, type=Path)
    parser.add_argument("--view-influence-comparison-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--signal-config", type=Path)
    parser.add_argument("--write-markdown", action="store_true")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        summary = build_offline_viewtrust_signals(
            clean_view_influence_dir=args.clean_view_influence_dir,
            corrupt_view_influence_dir=args.corrupt_view_influence_dir,
            view_influence_comparison_dir=args.view_influence_comparison_dir,
            output_dir=args.output_dir,
            signal_config=args.signal_config,
            top_k=args.top_k,
            write_markdown=args.write_markdown,
            quiet=args.quiet,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
