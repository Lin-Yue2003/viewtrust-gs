#!/usr/bin/env python3
"""Aggregate PR13 offline ViewTrust signals across corruption conditions."""

from __future__ import annotations

import argparse
import csv
import json
import sys
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

OUTPUT_FILES = [
    "offline_viewtrust_multi_condition_summary.json",
    "offline_viewtrust_multi_condition_results.csv",
    "offline_viewtrust_multi_condition_ablation.csv",
    "offline_viewtrust_condition_ranking.csv",
    "offline_viewtrust_failure_cases.csv",
    "offline_viewtrust_multi_condition_report.md",
    "offline_viewtrust_multi_condition_artifact_manifest.csv",
]


def _bootstrap_project_imports() -> Path:
    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root))
    return project_root


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: "" if row.get(field) is None else row.get(field) for field in fields})


def _manifest_rows(output_dir: Path, input_root: Path, condition_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = [
        {
            "relative_path": "input_root",
            "path": str(input_root),
            "exists": str(input_root.exists()).lower(),
            "file_type": "directory" if input_root.is_dir() else "",
            "size_bytes": "",
            "required": "true",
            "artifact_group": "input",
        }
    ]
    for result in condition_results:
        signal_dir = result.get("offline_signal_dir")
        if signal_dir:
            path = Path(str(signal_dir))
            rows.append(
                {
                    "relative_path": f"input_condition/{result.get('condition')}",
                    "path": str(path),
                    "exists": str(path.exists()).lower(),
                    "file_type": "directory" if path.is_dir() else "",
                    "size_bytes": "",
                    "required": "false",
                    "artifact_group": "input_condition",
                }
            )
    for name in OUTPUT_FILES:
        path = output_dir / name
        rows.append(
            {
                "relative_path": name,
                "path": str(path),
                "exists": str(path.exists()).lower(),
                "file_type": path.suffix.lstrip("."),
                "size_bytes": path.stat().st_size if path.is_file() else "",
                "required": "true",
                "artifact_group": "output_pr14",
            }
        )
    return rows


def _write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
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


def _ablation_rows(condition_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    from viewtrust.analysis.offline_multi_condition import ABLATION_SIGNAL_NAMES, load_offline_ablation

    rows: list[dict[str, Any]] = []
    for result in condition_results:
        scene = result.get("scene", "")
        condition = result.get("condition", "")
        status = result.get("status", "")
        if status == "ok":
            for row in load_offline_ablation(Path(str(result.get("offline_signal_dir")))):
                merged = {
                    "scene": scene,
                    "condition": condition,
                    **row,
                    "status": status,
                }
                rows.append(merged)
        else:
            for signal_name in ABLATION_SIGNAL_NAMES:
                rows.append(
                    {
                        "scene": scene,
                        "condition": condition,
                        "signal_name": signal_name,
                        "status": status,
                    }
                )
    return rows


def _markdown(
    *,
    summary: dict[str, Any],
    condition_results: list[dict[str, Any]],
    ranking_rows: list[dict[str, Any]],
    failure_rows: list[dict[str, Any]],
) -> str:
    condition_lines = [
        f"- `{row.get('condition')}` status=`{row.get('status')}` precision@k=`{row.get('precision_at_k', '')}` recall@k=`{row.get('recall_at_k', '')}`"
        for row in condition_results
    ]
    ranking_lines = [
        f"- {row.get('rank')}. `{row.get('condition')}` recall=`{row.get('recall_at_k')}` precision=`{row.get('precision_at_k')}` gap=`{row.get('risk_gap_corrupted_minus_uncorrupted')}`"
        for row in ranking_rows
        if row.get("rank")
    ]
    failure_lines = [
        f"- `{row.get('condition')}` {row.get('failure_type')}: {row.get('details')}"
        for row in failure_rows[:20]
    ]
    return "\n".join(
        [
            "# Offline ViewTrust Multi-Condition Report",
            "",
            "This is offline validation only.",
            "This is not a trust score used during training.",
            "This is not a defense.",
            "This is not a poison classifier.",
            "Corruption labels are used only for post-hoc evaluation.",
            "",
            "## Purpose",
            "PR14 evaluates whether PR13 offline candidate signals rank corrupted views highly across natural corruption conditions.",
            "",
            "## Inputs",
            f"- Scene: `{summary.get('scene')}`",
            f"- Clean condition: `{summary.get('clean_condition')}`",
            f"- Requested conditions: `{summary.get('conditions_requested')}`",
            "",
            "## Observation-Only Guarantee",
            f"- Observation only: `{summary.get('observation_only')}`",
            f"- Uses corruption labels for scoring: `{summary.get('uses_corruption_labels_for_scoring')}`",
            f"- Uses corruption labels for evaluation: `{summary.get('uses_corruption_labels_for_evaluation')}`",
            f"- Training intervention: `{summary.get('training_intervention')}`",
            f"- Defense enabled: `{summary.get('defense_enabled')}`",
            "",
            "## Conditions Evaluated",
            *condition_lines,
            "",
            "## Aggregate Performance",
            f"- Mean precision@k: `{summary.get('mean_precision_at_k')}`",
            f"- Mean recall@k: `{summary.get('mean_recall_at_k')}`",
            f"- Mean risk gap: `{summary.get('mean_risk_gap')}`",
            f"- Conditions with full recall: `{summary.get('conditions_with_full_recall')}`",
            f"- Conditions with positive risk gap: `{summary.get('conditions_with_positive_risk_gap')}`",
            "",
            "## Per-Condition Summary",
            *ranking_lines,
            "" if ranking_lines else "- No valid conditions ranked.",
            "",
            "## Ablation Stability",
            f"- Best ablation signal counts: `{summary.get('best_ablation_signal_counts')}`",
            "",
            "## Failure Cases",
            *failure_lines,
            "" if failure_lines else "- None",
            "",
            "## Interpretation Guidance",
            "- Treat these as candidate offline signal validation results.",
            "- Do not describe the output as detection, operational defense, or view rejection.",
            "",
            "## Known Limitations",
            "- PR14 validates existing PR13 artifacts only; it does not retrain or render.",
            "- Single-seed results are not enough for method claims.",
            "",
            "## Recommended Next Experiments",
            "- Build all-condition PR13 artifacts from existing PR12.1 view influence outputs.",
            "- Extend validation to multiple seeds.",
            "",
        ]
    )


def aggregate_offline_viewtrust_results(
    *,
    input_root: Path,
    output_dir: Path,
    scene: str,
    clean_condition: str,
    conditions: list[str],
    top_k: int,
    require_all_conditions: bool,
    write_markdown: bool,
    quiet: bool,
) -> tuple[dict[str, Any], int]:
    _bootstrap_project_imports()
    from viewtrust.analysis.offline_multi_condition import (
        aggregate_condition_results,
        compute_condition_ranking,
        compute_failure_cases,
        condition_result_from_signal_dir,
        discover_offline_signal_dirs,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    discovered = discover_offline_signal_dirs(input_root, conditions)
    condition_results = [
        condition_result_from_signal_dir(
            scene=scene,
            condition=condition,
            clean_condition=clean_condition,
            signal_dir=discovered.get(condition),
            top_k=top_k,
        )
        for condition in conditions
    ]
    summary = aggregate_condition_results(
        scene=scene,
        clean_condition=clean_condition,
        conditions=conditions,
        condition_results=condition_results,
        top_k=top_k,
    )
    ranking_rows = compute_condition_ranking(condition_results)
    failure_rows = compute_failure_cases(condition_results, top_k)
    ablation_rows = _ablation_rows(condition_results)

    result_fields = [
        "scene", "condition", "clean_condition", "clean_run_id", "corrupt_run_id",
        "offline_signal_dir", "view_count", "corrupted_view_count",
        "uncorrupted_view_count", "top_k", "corrupted_in_top_k",
        "precision_at_k", "recall_at_k", "mean_corrupted_risk",
        "mean_uncorrupted_risk", "risk_gap_corrupted_minus_uncorrupted",
        "best_ablation_signal", "top1_view_name", "top1_was_corrupted",
        "top1_risk", "top1_main_reason", "top3_corrupted_count",
        "top5_corrupted_count", "observation_only",
        "uses_corruption_labels_for_scoring",
        "uses_corruption_labels_for_evaluation", "training_intervention",
        "defense_enabled", "status", "warnings",
    ]
    ablation_fields = [
        "scene", "condition", "signal_name", "top1_view_name", "top1_was_corrupted",
        "topk", "corrupted_in_topk", "precision_at_k", "recall_at_k",
        "mean_corrupted_rank", "median_corrupted_rank", "mean_corrupted_score",
        "mean_uncorrupted_score", "score_gap", "status",
    ]
    ranking_fields = [
        "rank", "scene", "condition", "precision_at_k", "recall_at_k",
        "corrupted_in_top_k", "risk_gap_corrupted_minus_uncorrupted",
        "mean_corrupted_risk", "mean_uncorrupted_risk", "best_ablation_signal",
        "status",
    ]
    failure_fields = [
        "scene", "condition", "failure_type", "view_name", "was_corrupted",
        "rank", "offline_viewtrust_risk", "main_reason", "details",
    ]

    _write_csv(output_dir / "offline_viewtrust_multi_condition_results.csv", condition_results, result_fields)
    _write_csv(output_dir / "offline_viewtrust_multi_condition_ablation.csv", ablation_rows, ablation_fields)
    _write_csv(output_dir / "offline_viewtrust_condition_ranking.csv", ranking_rows, ranking_fields)
    _write_csv(output_dir / "offline_viewtrust_failure_cases.csv", failure_rows, failure_fields)
    (output_dir / "offline_viewtrust_multi_condition_report.md").write_text(
        _markdown(
            summary=summary,
            condition_results=condition_results,
            ranking_rows=ranking_rows,
            failure_rows=failure_rows,
        ),
        encoding="utf-8",
    )
    (output_dir / "offline_viewtrust_multi_condition_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest_path = output_dir / "offline_viewtrust_multi_condition_artifact_manifest.csv"
    _write_manifest(manifest_path, _manifest_rows(output_dir, input_root, condition_results))
    _write_manifest(manifest_path, _manifest_rows(output_dir, input_root, condition_results))

    if not quiet:
        print(
            f"PR14 valid conditions: {summary['condition_count_valid']} / {summary['condition_count_requested']}",
            file=sys.stderr,
        )

    has_failures = any(row.get("status") != "ok" for row in condition_results)
    return summary, 1 if require_all_conditions and has_failures else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--scene", default="chair")
    parser.add_argument("--clean-condition", default="clean")
    parser.add_argument("--conditions", nargs="+", default=DEFAULT_CONDITIONS)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--require-all-conditions", action="store_true")
    parser.add_argument("--write-markdown", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        summary, exit_code = aggregate_offline_viewtrust_results(
            input_root=args.input_root,
            output_dir=args.output_dir,
            scene=args.scene,
            clean_condition=args.clean_condition,
            conditions=args.conditions,
            top_k=args.top_k,
            require_all_conditions=args.require_all_conditions,
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
