#!/usr/bin/env python3
"""Analyze cross-condition consistency of existing offline ViewTrust rankings."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _bootstrap_project_imports() -> Path:
    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root))
    return project_root


def _dynamic_fields(base: list[str], rows: list[dict[str, Any]]) -> list[str]:
    seen = set(base)
    fields = list(base)
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    return fields


def analyze_rank_consistency(
    *,
    multi_condition_dir: Path,
    input_root: Path,
    scene: str,
    conditions: list[str],
    top_k: int,
    output_dir: Path,
    write_markdown: bool,
    quiet: bool,
) -> tuple[dict[str, Any], int]:
    _bootstrap_project_imports()
    from viewtrust.analysis.offline_rank_consistency import (
        build_component_condition_summary,
        build_component_gap_table,
        build_component_win_table,
        build_corrupted_view_rank_distribution,
        build_cross_condition_view_rank_table,
        build_false_positive_topk_views,
        build_repeated_top_views,
        build_summary_json,
        find_condition_signal_dir,
        load_condition_ablation,
        load_condition_rankings,
        validate_condition_signal_dir,
        validate_multi_condition_dir,
        write_artifact_manifest,
        write_csv_rows,
        write_rank_consistency_report,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    pr14_summary, warnings = validate_multi_condition_dir(multi_condition_dir)
    condition_dirs: dict[str, Path | None] = {}
    condition_rankings: dict[str, list[dict[str, Any]]] = {}
    condition_ablations: dict[str, list[dict[str, str]]] = {}

    for condition in conditions:
        signal_dir = find_condition_signal_dir(input_root, condition)
        condition_dirs[condition] = signal_dir
        if signal_dir is None:
            warnings.append(f"{condition}: missing per-condition offline signal directory")
            continue
        status, condition_warnings = validate_condition_signal_dir(signal_dir)
        warnings.extend(f"{condition}: {warning}" for warning in condition_warnings)
        if status != "ok":
            warnings.append(f"{condition}: skipped invalid condition directory {signal_dir}")
            continue
        condition_rankings[condition] = load_condition_rankings(signal_dir)
        condition_ablations[condition] = load_condition_ablation(signal_dir)

    cross_rows = build_cross_condition_view_rank_table(condition_rankings, top_k)
    repeated_rows = build_repeated_top_views(cross_rows, top_k)
    false_positive_rows = build_false_positive_topk_views(condition_rankings, cross_rows, top_k)
    corrupted_rows = build_corrupted_view_rank_distribution(condition_rankings)
    component_win_rows = build_component_win_table(condition_ablations)
    component_summary_rows = build_component_condition_summary(component_win_rows)
    component_gap_rows = build_component_gap_table(component_win_rows)
    summary = build_summary_json(
        scene=scene,
        conditions=conditions,
        top_k=top_k,
        pr14_summary=pr14_summary,
        condition_rankings=condition_rankings,
        cross_condition_rows=cross_rows,
        repeated_rows=repeated_rows,
        false_positive_rows=false_positive_rows,
        component_summary_rows=component_summary_rows,
        warnings=warnings,
    )

    cross_fields = [
        "view_name",
        "condition_count_observed",
        "was_corrupted_in_any_condition",
        "corrupted_condition_count",
        "top1_count",
        "top3_count",
        "top5_count",
        "mean_rank",
        "median_rank",
        "best_rank",
        "worst_rank",
        "rank_std",
        "mean_risk",
        "median_risk",
        "risk_std",
        "most_common_main_reason",
    ]
    for condition in conditions:
        cross_fields.extend(
            [
                f"rank_{condition}",
                f"risk_{condition}",
                f"was_corrupted_{condition}",
                f"main_reason_{condition}",
                f"in_top1_{condition}",
                f"in_top3_{condition}",
                f"in_top5_{condition}",
            ]
        )
    repeated_fields = [
        "view_name",
        "top1_count",
        "top3_count",
        "top5_count",
        "conditions_top1",
        "conditions_top3",
        "conditions_top5",
        "was_corrupted_in_any_condition",
        "corrupted_condition_count",
        "mean_rank",
        "median_rank",
        "mean_risk",
        "median_risk",
        "most_common_main_reason",
        "repeat_type",
    ]
    false_positive_fields = _dynamic_fields(
        [
            "condition",
            "view_name",
            "rank",
            "offline_viewtrust_risk",
            "main_reason",
            "is_repeated_false_positive",
            "repeated_false_positive_count",
            "mean_rank_across_conditions",
            "mean_risk_across_conditions",
            "top5_count_across_conditions",
        ],
        false_positive_rows,
    )
    corrupted_fields = _dynamic_fields(
        [
            "condition",
            "view_name",
            "rank",
            "offline_viewtrust_risk",
            "in_top1",
            "in_top3",
            "in_top5",
            "main_reason",
        ],
        corrupted_rows,
    )
    component_win_fields = [
        "condition",
        "signal_name",
        "corrupted_in_top_k",
        "precision_at_k",
        "recall_at_k",
        "risk_gap_corrupted_minus_uncorrupted",
        "mean_corrupted_score",
        "mean_uncorrupted_score",
        "rank_within_condition",
        "top1_view_name",
        "top1_was_corrupted",
        "status",
    ]
    component_summary_fields = [
        "condition",
        "best_signal_name",
        "best_signal_recall",
        "best_signal_precision",
        "best_signal_risk_gap",
        "full_signal_precision",
        "full_signal_recall",
        "full_signal_risk_gap",
        "loss_only_precision",
        "loss_only_recall",
        "loss_only_risk_gap",
        "lifecycle_only_precision",
        "lifecycle_only_recall",
        "lifecycle_only_risk_gap",
        "full_minus_loss_recall",
        "full_minus_loss_precision",
        "full_minus_loss_risk_gap",
        "full_minus_lifecycle_recall",
        "full_minus_lifecycle_precision",
        "full_minus_lifecycle_risk_gap",
        "does_full_beat_loss",
        "does_full_beat_lifecycle",
    ]
    component_gap_fields = [
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

    write_csv_rows(output_dir / "cross_condition_view_rank_table.csv", cross_rows, cross_fields)
    write_csv_rows(output_dir / "repeated_top_views.csv", repeated_rows, repeated_fields)
    write_csv_rows(output_dir / "false_positive_topk_views.csv", false_positive_rows, false_positive_fields)
    write_csv_rows(output_dir / "corrupted_view_rank_distribution.csv", corrupted_rows, corrupted_fields)
    write_csv_rows(output_dir / "component_win_table.csv", component_win_rows, component_win_fields)
    write_csv_rows(output_dir / "component_condition_summary.csv", component_summary_rows, component_summary_fields)
    write_csv_rows(output_dir / "component_gap_table.csv", component_gap_rows, component_gap_fields)
    (output_dir / "cross_condition_view_rank_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_rank_consistency_report(
        output_dir / "rank_consistency_report.md",
        summary=summary,
        multi_condition_dir=multi_condition_dir,
        input_root=input_root,
        repeated_rows=repeated_rows,
        false_positive_rows=false_positive_rows,
        corrupted_rows=corrupted_rows,
        component_summary_rows=component_summary_rows,
        warnings=warnings,
    )
    write_artifact_manifest(
        output_dir / "artifact_manifest.csv",
        output_dir=output_dir,
        multi_condition_dir=multi_condition_dir,
        input_root=input_root,
        condition_dirs=condition_dirs,
    )

    if not quiet:
        print(
            f"PR15 valid conditions: {summary['condition_count_valid']} / {summary['condition_count']}",
            file=sys.stderr,
        )
    return summary, 0 if condition_rankings else 1


def parse_args() -> argparse.Namespace:
    _bootstrap_project_imports()
    from viewtrust.analysis.offline_rank_consistency import DEFAULT_CONDITIONS

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--multi-condition-dir", required=True, type=Path)
    parser.add_argument("--input-root", default=Path("outputs/reports"), type=Path)
    parser.add_argument("--scene", default="chair")
    parser.add_argument("--conditions", nargs="+", default=DEFAULT_CONDITIONS)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--write-markdown", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        summary, exit_code = analyze_rank_consistency(
            multi_condition_dir=args.multi_condition_dir,
            input_root=args.input_root,
            scene=args.scene,
            conditions=args.conditions,
            top_k=args.top_k,
            output_dir=args.output_dir,
            write_markdown=args.write_markdown,
            quiet=args.quiet,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
