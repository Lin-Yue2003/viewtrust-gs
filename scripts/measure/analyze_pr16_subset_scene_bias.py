#!/usr/bin/env python3
"""Analyze PR16 subset and scene bias from existing offline outputs."""

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


def _resolve_path(project_root: Path, raw_path: str | Path) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else project_root / path


def _load_plan_subset_manifest(plan_dir: Path) -> list[dict[str, str]]:
    from viewtrust.analysis.subset_scene_bias import load_csv_rows

    return load_csv_rows(plan_dir / "pr16_subset_manifest.csv")


def _minimal_subset_manifest(
    *,
    plan_rows: list[dict[str, str]],
    scenes: list[str],
    subset_names: list[str],
) -> list[dict[str, Any]]:
    existing = {
        (row.get("scene", ""), row.get("subset_name", "")): row
        for row in plan_rows
    }
    rows: list[dict[str, Any]] = []
    for scene in scenes:
        for subset_name in subset_names:
            row = dict(existing.get((scene, subset_name), {}))
            row.setdefault("scene", scene)
            row.setdefault("subset_name", subset_name)
            row.setdefault("subset_seed", subset_name.removeprefix("seed_") if subset_name.startswith("seed_") else "")
            row.setdefault("corrupted_view_names", "")
            row.setdefault("status", "missing_plan_manifest" if not existing.get((scene, subset_name)) else "ok")
            row.setdefault("warnings", "")
            rows.append(row)
    return rows


def analyze_pr16_subset_scene_bias(
    *,
    input_root: Path,
    plan_dir: Path,
    output_dir: Path,
    scenes: list[str],
    conditions: list[str],
    subset_names: list[str],
    top_k: int,
    write_markdown: bool,
    quiet: bool,
) -> tuple[dict[str, Any], int]:
    _bootstrap_project_imports()
    from viewtrust.analysis.subset_scene_bias import (
        ANALYSIS_OUTPUT_FILES,
        build_component_comparison,
        build_pr16_summary,
        build_repeated_false_positive_table,
        build_scene_bias_summary,
        build_scene_subset_condition_results,
        build_subset_bias_summary,
        build_view_identity_bias_table,
        write_artifact_manifest,
        write_csv_rows,
        write_json,
        write_pr16_report,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    plan_rows = _load_plan_subset_manifest(plan_dir) if plan_dir else []
    subset_rows = _minimal_subset_manifest(
        plan_rows=plan_rows,
        scenes=scenes,
        subset_names=subset_names,
    )

    results, missing_rows, condition_dirs = build_scene_subset_condition_results(
        input_root=input_root,
        subset_manifest_rows=subset_rows,
        scenes=scenes,
        conditions=conditions,
        subset_names=subset_names,
        top_k=top_k,
    )
    component_rows = build_component_comparison(condition_dirs=condition_dirs)
    subset_summary_rows = build_subset_bias_summary(results, subset_rows, top_k)
    view_identity_rows = build_view_identity_bias_table(condition_dirs, top_k)
    repeated_false_positive_rows = build_repeated_false_positive_table(
        condition_dirs,
        view_identity_rows,
        top_k,
    )
    scene_summary_rows = build_scene_bias_summary(
        results,
        view_identity_rows,
        repeated_false_positive_rows,
    )
    warnings = [
        str(row.get("warnings"))
        for row in results
        if row.get("warnings")
    ]
    summary = build_pr16_summary(
        scenes=scenes,
        conditions=conditions,
        subset_names=subset_names,
        top_k=top_k,
        results=results,
        scene_rows=scene_summary_rows,
        view_identity_rows=view_identity_rows,
        repeated_false_positive_rows=repeated_false_positive_rows,
        warnings=warnings,
    )

    result_fields = [
        "scene",
        "subset_name",
        "subset_seed",
        "condition",
        "top_k",
        "view_count",
        "corrupted_view_count",
        "corrupted_in_top_k",
        "precision_at_k",
        "recall_at_k",
        "mean_corrupted_risk",
        "mean_uncorrupted_risk",
        "risk_gap_corrupted_minus_uncorrupted",
        "top1_view_name",
        "top1_was_corrupted",
        "top1_risk",
        "top1_main_reason",
        "top5_view_names",
        "top5_corrupted_count",
        "best_ablation_signal",
        "full_signal_precision",
        "full_signal_recall",
        "loss_only_precision",
        "loss_only_recall",
        "lifecycle_only_precision",
        "lifecycle_only_recall",
        "does_full_beat_loss",
        "does_full_beat_lifecycle",
        "status",
        "warnings",
    ]
    subset_fields = [
        "scene",
        "subset_name",
        "subset_seed",
        "condition_count_valid",
        "mean_precision_at_k",
        "mean_recall_at_k",
        "median_precision_at_k",
        "median_recall_at_k",
        "mean_risk_gap",
        "corrupted_view_names",
        "top1_view_names_by_condition",
        "top5_union_view_names",
        "top5_corrupted_union_count",
        "top5_uncorrupted_union_count",
        "full_signal_win_count_over_loss",
        "full_signal_win_count_over_lifecycle",
        "repeated_false_positive_views",
        "status",
        "warnings",
    ]
    scene_fields = [
        "scene",
        "subset_count_valid",
        "condition_count_valid",
        "mean_precision_at_k",
        "mean_recall_at_k",
        "median_precision_at_k",
        "median_recall_at_k",
        "mean_risk_gap",
        "full_signal_win_rate_over_loss",
        "full_signal_win_rate_over_lifecycle",
        "static_top1_rate",
        "repeated_false_positive_count",
        "view_identity_bias_warning",
        "status",
        "warnings",
    ]
    identity_fields = [
        "scene",
        "view_name",
        "observed_condition_count",
        "corrupted_count",
        "uncorrupted_count",
        "top1_count",
        "top3_count",
        "top5_count",
        "top5_when_corrupted_count",
        "top5_when_uncorrupted_count",
        "mean_rank_when_corrupted",
        "mean_rank_when_uncorrupted",
        "mean_risk_when_corrupted",
        "mean_risk_when_uncorrupted",
        "risk_lift_corrupted_minus_uncorrupted",
        "rank_lift_corrupted_minus_uncorrupted",
        "is_repeated_top_view",
        "is_repeated_false_positive",
        "view_identity_bias_flag",
        "conditions_top5_when_uncorrupted",
        "conditions_top5_when_corrupted",
    ]
    false_positive_fields = [
        "scene",
        "view_name",
        "subset_name",
        "condition",
        "rank",
        "risk",
        "main_reason",
        "top5_when_uncorrupted_count",
        "mean_rank_when_uncorrupted",
        "mean_risk_when_uncorrupted",
        "possible_interpretation",
    ]
    component_fields = [
        "scene",
        "subset_name",
        "condition",
        "signal_name",
        "precision_at_k",
        "recall_at_k",
        "risk_gap_corrupted_minus_uncorrupted",
        "corrupted_in_top_k",
        "top1_view_name",
        "top1_was_corrupted",
        "rank_within_condition",
        "status",
    ]
    missing_fields = [
        "scene",
        "subset_name",
        "condition",
        "expected_pattern",
        "status",
        "details",
    ]

    write_json(output_dir / "pr16_bias_probe_summary.json", summary)
    write_csv_rows(output_dir / "pr16_scene_subset_condition_results.csv", results, result_fields)
    write_csv_rows(output_dir / "pr16_subset_bias_summary.csv", subset_summary_rows, subset_fields)
    write_csv_rows(output_dir / "pr16_scene_bias_summary.csv", scene_summary_rows, scene_fields)
    write_csv_rows(output_dir / "pr16_view_identity_bias_table.csv", view_identity_rows, identity_fields)
    write_csv_rows(
        output_dir / "pr16_repeated_false_positive_table.csv",
        repeated_false_positive_rows,
        false_positive_fields,
    )
    write_csv_rows(output_dir / "pr16_component_comparison.csv", component_rows, component_fields)
    write_csv_rows(output_dir / "pr16_missing_outputs.csv", missing_rows, missing_fields)
    write_pr16_report(
        output_dir / "pr16_bias_probe_report.md",
        summary=summary,
        subset_rows=subset_summary_rows,
        scene_rows=scene_summary_rows,
        view_identity_rows=view_identity_rows,
        repeated_false_positive_rows=repeated_false_positive_rows,
        component_rows=component_rows,
        missing_rows=missing_rows,
    )
    write_artifact_manifest(
        output_dir / "artifact_manifest.csv",
        output_dir=output_dir,
        output_files=ANALYSIS_OUTPUT_FILES,
        inputs=[
            ("input_root", input_root, True, "input", "Root containing PR16 offline signal outputs"),
            ("plan_dir", plan_dir, False, "input", "PR16 planner output directory"),
        ],
    )
    if not quiet:
        print(
            f"PR16 valid results: {summary['valid_result_count']} / {len(results)}",
            file=sys.stderr,
        )
    return summary, 0 if summary["valid_result_count"] > 0 else 1


def parse_args() -> argparse.Namespace:
    _bootstrap_project_imports()
    from viewtrust.analysis.subset_scene_bias import (
        DEFAULT_CONDITIONS,
        DEFAULT_SCENES,
        DEFAULT_SUBSET_NAMES,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", default=Path("outputs/reports"), type=Path)
    parser.add_argument("--plan-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--scenes", nargs="+", default=DEFAULT_SCENES)
    parser.add_argument("--conditions", nargs="+", default=DEFAULT_CONDITIONS)
    parser.add_argument("--subset-names", nargs="+", default=DEFAULT_SUBSET_NAMES)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--write-markdown", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> int:
    project_root = _bootstrap_project_imports()
    args = parse_args()
    try:
        summary, exit_code = analyze_pr16_subset_scene_bias(
            input_root=_resolve_path(project_root, args.input_root),
            plan_dir=_resolve_path(project_root, args.plan_dir),
            output_dir=_resolve_path(project_root, args.output_dir),
            scenes=args.scenes,
            conditions=args.conditions,
            subset_names=args.subset_names,
            top_k=args.top_k,
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
