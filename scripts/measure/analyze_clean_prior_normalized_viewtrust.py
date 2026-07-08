#!/usr/bin/env python3
"""Build PR17 clean-prior normalized offline ViewTrust diagnostics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


DEFAULT_CONFIG = Path("configs/offline_viewtrust_signal/default_pr17_clean_prior.json")


def _bootstrap_project_imports() -> Path:
    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root))
    return project_root


def _resolve_path(project_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else project_root / path


def _subset_seed(plan_rows: list[dict[str, str]], scene: str, subset_name: str) -> str:
    for row in plan_rows:
        if row.get("scene") == scene and row.get("subset_name") == subset_name:
            return row.get("subset_seed", "")
    return subset_name.removeprefix("seed_") if subset_name.startswith("seed_") else ""


def analyze_clean_prior_normalized_viewtrust(
    *,
    input_root: Path,
    plan_dir: Path,
    output_dir: Path,
    scenes: list[str],
    conditions: list[str],
    subset_names: list[str],
    top_k: int,
    raw_score_key: str,
    strict: bool,
    allow_missing: bool,
    prior_mode: str,
    config_path: Path,
    write_markdown: bool,
) -> tuple[dict[str, Any], int]:
    _bootstrap_project_imports()
    from viewtrust.analysis.clean_prior_normalization import (
        build_summary,
        compute_component_comparison,
        compute_false_positive_reduction,
        compute_normalized_ablation_metrics,
        compute_normalized_group_metrics,
        compute_normalized_rows,
        compute_view_identity_diagnosis,
        load_json,
        load_offline_signal_dir,
        rank_normalized_rows,
        write_artifact_manifest,
        write_csv_rows,
        write_json,
        write_report,
    )
    from viewtrust.analysis.subset_scene_bias import discover_pr16_condition_output, load_csv_rows

    output_dir.mkdir(parents=True, exist_ok=True)
    config = load_json(config_path)
    if not config:
        raise FileNotFoundError(f"PR17 config not found or empty: {config_path}")
    if config.get("labels", {}).get("use_corruption_labels_for_scoring") is True:
        raise ValueError("PR17 config must not use corruption labels for scoring")
    config["prior_mode"] = prior_mode if prior_mode != "auto" else config.get("prior_mode", "auto")

    plan_rows = load_csv_rows(plan_dir / "pr16_subset_manifest.csv")
    normalized_rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    valid_result_count = 0

    for scene in scenes:
        for subset_name in subset_names:
            subset_seed = _subset_seed(plan_rows, scene, subset_name)
            for condition in conditions:
                signal_dir = discover_pr16_condition_output(input_root, scene, subset_name, condition)
                if signal_dir is None:
                    row = {
                        "scene": scene,
                        "subset_name": subset_name,
                        "condition": condition,
                        "expected_pattern": f"offline_viewtrust_{scene}_{condition}_{subset_name}_pr16_input",
                        "status": "missing",
                        "details": "missing PR16 offline signal directory",
                    }
                    missing_rows.append(row)
                    warnings.append(f"{scene}/{subset_name}/{condition}: missing")
                    continue
                offline = load_offline_signal_dir(signal_dir)
                if offline["missing_files"]:
                    row = {
                        "scene": scene,
                        "subset_name": subset_name,
                        "condition": condition,
                        "expected_pattern": str(signal_dir),
                        "status": "invalid",
                        "details": ";".join(offline["missing_files"]),
                    }
                    missing_rows.append(row)
                    warnings.append(f"{scene}/{subset_name}/{condition}: invalid")
                    continue
                normalized_rows.extend(
                    compute_normalized_rows(
                        offline=offline,
                        scene=scene,
                        subset_name=subset_name,
                        condition=condition,
                        subset_seed=subset_seed,
                        config=config,
                        top_k=top_k,
                        raw_score_key=raw_score_key,
                    )
                )
                valid_result_count += 1

    if valid_result_count == 0 and (strict or not allow_missing):
        raise ValueError("no valid PR17 inputs found")

    ranking_rows = rank_normalized_rows(normalized_rows)
    group_rows = compute_normalized_group_metrics(normalized_rows)
    ablation_rows = compute_normalized_ablation_metrics(normalized_rows, top_k)
    fp_rows = compute_false_positive_reduction(normalized_rows, top_k)
    diagnosis_rows = compute_view_identity_diagnosis(normalized_rows)
    component_rows = compute_component_comparison(ablation_rows)
    summary = build_summary(
        scenes=scenes,
        conditions=conditions,
        subset_names=subset_names,
        top_k=top_k,
        valid_result_count=valid_result_count,
        missing_rows=missing_rows,
        fp_rows=fp_rows,
        diagnosis_rows=diagnosis_rows,
        warnings=warnings,
    )

    normalized_fields = [
        "scene", "subset_name", "subset_seed", "condition", "view_name", "view_split",
        "was_corrupted", "raw_risk", "raw_rank", "clean_prior_risk",
        "clean_prior_rank", "delta_risk", "positive_delta_risk",
        "prior_suppressed_risk", "rank_lift_score", "normalized_viewtrust_risk",
        "normalized_rank", "normalized_consistency", "raw_top_k",
        "normalized_top_k", "raw_false_positive", "normalized_false_positive",
        "prior_source", "component_warnings",
    ]
    ranking_fields = [
        "scene", "subset_name", "subset_seed", "condition", "score_name", "rank",
        "view_name", "was_corrupted", "score", "raw_risk", "clean_prior_risk",
        "delta_risk", "rank_lift_score", "prior_source",
    ]
    group_fields = [
        "scene", "subset_name", "condition", "group", "view_count", "mean_raw_risk",
        "mean_clean_prior_risk", "mean_normalized_viewtrust_risk",
        "median_normalized_viewtrust_risk",
    ]
    ablation_fields = [
        "scene", "subset_name", "condition", "score_name", "top_k",
        "corrupted_in_top_k", "precision_at_k", "recall_at_k",
        "mean_corrupted_score", "mean_uncorrupted_score", "score_gap",
        "top1_view_name", "top1_was_corrupted", "status",
    ]
    fp_fields = [
        "scene", "subset_name", "condition", "raw_false_positive_count_at_k",
        "normalized_false_positive_count_at_k", "false_positive_reduction",
        "raw_recall_at_k", "normalized_recall_at_k", "recall_delta",
        "raw_precision_at_k", "normalized_precision_at_k", "precision_delta",
        "removed_false_positive_views", "newly_added_false_positive_views",
        "recovered_corrupted_views", "lost_corrupted_views",
    ]
    diagnosis_fields = [
        "scene", "view_name", "observed_condition_count", "corrupted_count",
        "uncorrupted_count", "raw_top5_when_uncorrupted_count",
        "normalized_top5_when_uncorrupted_count", "raw_repeated_false_positive",
        "normalized_repeated_false_positive", "raw_mean_rank_when_uncorrupted",
        "normalized_mean_rank_when_uncorrupted", "raw_mean_risk_when_uncorrupted",
        "normalized_mean_risk_when_uncorrupted", "clean_prior_risk_mean",
        "normalized_bias_reduced", "conditions_removed_from_top5_when_uncorrupted",
        "conditions_remaining_top5_when_uncorrupted",
    ]
    component_fields = [
        "scene", "subset_name", "condition", "score_name", "precision_at_k",
        "recall_at_k", "score_gap", "top1_view_name", "top1_was_corrupted", "status",
    ]
    missing_fields = ["scene", "subset_name", "condition", "expected_pattern", "status", "details"]

    write_json(output_dir / "clean_prior_normalized_summary.json", summary)
    write_csv_rows(output_dir / "clean_prior_normalized_rows.csv", normalized_rows, normalized_fields)
    write_csv_rows(output_dir / "clean_prior_normalized_rankings.csv", ranking_rows, ranking_fields)
    write_csv_rows(output_dir / "clean_prior_normalized_group_metrics.csv", group_rows, group_fields)
    write_csv_rows(output_dir / "clean_prior_normalized_ablation.csv", ablation_rows, ablation_fields)
    write_csv_rows(output_dir / "clean_prior_false_positive_reduction.csv", fp_rows, fp_fields)
    write_csv_rows(output_dir / "clean_prior_view_identity_diagnosis.csv", diagnosis_rows, diagnosis_fields)
    write_csv_rows(output_dir / "clean_prior_component_comparison.csv", component_rows, component_fields)
    write_csv_rows(output_dir / "clean_prior_missing_outputs.csv", missing_rows, missing_fields)
    write_report(output_dir / "clean_prior_report.md", summary)
    write_artifact_manifest(output_dir, input_root, plan_dir)

    if strict and missing_rows:
        return summary, 1
    return summary, 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", default=Path("outputs/reports"), type=Path)
    parser.add_argument("--plan-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--scenes", nargs="+", default=["chair", "drums"])
    parser.add_argument("--conditions", nargs="+", default=["corrupt_occluder", "corrupt_noise", "corrupt_mixed"])
    parser.add_argument("--subset-names", nargs="+", default=["original", "seed_20260710"])
    parser.add_argument("--raw-score-key", default="offline_viewtrust_risk")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--write-markdown", action="store_true")
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument(
        "--prior-mode",
        choices=("auto", "clean_components", "clean_columns", "manifest_inputs"),
        default="auto",
    )
    parser.add_argument("--normalization-config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


def main() -> int:
    project_root = _bootstrap_project_imports()
    args = parse_args()
    try:
        summary, exit_code = analyze_clean_prior_normalized_viewtrust(
            input_root=_resolve_path(project_root, args.input_root),
            plan_dir=_resolve_path(project_root, args.plan_dir),
            output_dir=_resolve_path(project_root, args.output_dir),
            scenes=args.scenes,
            conditions=args.conditions,
            subset_names=args.subset_names,
            top_k=args.top_k,
            raw_score_key=args.raw_score_key,
            strict=args.strict,
            allow_missing=args.allow_missing,
            prior_mode=args.prior_mode,
            config_path=_resolve_path(project_root, args.normalization_config),
            write_markdown=args.write_markdown,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
