#!/usr/bin/env python3
"""Build PR19 Gaussian cluster and event-cluster risk diagnostics."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_CONFIG = Path("configs/offline_viewtrust_signal/default_pr19_gaussian_cluster_risk.json")


def _bootstrap_project_imports() -> Path:
    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root))
    return project_root


def _resolve_path(project_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else project_root / path


def _condition_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return str(row.get("scene", "")), str(row.get("subset_name", "")), str(row.get("condition", ""))


def _fields() -> dict[str, list[str]]:
    return {
        "evidence": [
            "scene", "subset_name", "condition", "signal_dir", "clean_dir",
            "corrupt_dir", "comparison_dir", "exact_gaussian_ids_available",
            "aggregate_event_proxy_available", "evidence_level", "files_used",
            "missing_files", "warnings",
        ],
        "view_group": [
            "scene", "subset_name", "condition", "view_name", "view_group",
            "was_corrupted", "pr17_raw_rank", "pr17_normalized_rank",
            "pr18_spillover_class", "pr18_spillover_confidence",
            "camera_neighbor_evidence", "index_neighbor_evidence",
            "gaussian_overlap_evidence", "included_in_candidate_analysis",
        ],
        "cluster": [
            "scene", "subset_name", "condition", "evidence_level", "cluster_id",
            "cluster_kind", "lifecycle_action", "iteration_bucket",
            "source_view_group", "source_view_names", "source_view_count",
            "direct_corrupted_source_count", "collateral_source_count",
            "clean_prior_source_count", "other_clean_source_count",
            "source_entropy", "source_concentration_score", "weak_support_score",
            "corrupted_plus_collateral_ratio", "clean_prior_ratio",
            "event_count_total", "unique_gaussian_count_total",
            "final_alive_count_total", "final_dead_count_total",
            "clone_birth_count_total", "split_birth_count_total",
            "prune_death_count_total", "birth_ratio", "prune_ratio",
            "death_ratio", "lifecycle_instability_score", "visibility_delta_score",
            "clean_vs_corrupt_delta_score", "gaussian_cluster_risk", "rank",
            "warnings",
        ],
        "ranking": [
            "scene", "subset_name", "condition", "rank", "cluster_id",
            "evidence_level", "gaussian_cluster_risk",
            "corrupted_plus_collateral_ratio", "clean_prior_ratio",
            "source_concentration_score", "lifecycle_instability_score",
            "weak_support_score", "visibility_delta_score", "source_view_names",
            "source_view_groups", "main_reason",
        ],
        "group": [
            "scene", "subset_name", "condition", "evidence_level", "top_k",
            "topk_mean_corrupted_plus_collateral_ratio",
            "topk_mean_clean_prior_ratio", "topk_mean_source_concentration",
            "topk_direct_corrupted_view_count", "topk_collateral_view_count",
            "topk_clean_prior_view_count", "topk_other_clean_view_count",
            "support_concentration_status", "interpretation",
        ],
        "overlap": [
            "scene", "subset_name", "condition", "direct_view", "collateral_view",
            "overlap_type", "shared_cluster_count", "direct_cluster_count",
            "collateral_cluster_count", "overlap_ratio", "mean_shared_cluster_risk",
            "evidence_level", "supported", "warnings",
        ],
        "train013": [
            "scene", "subset_name", "condition", "train013_present",
            "train013_view_group", "train013_cluster_count",
            "train013_high_risk_cluster_count", "train013_mean_cluster_risk",
            "train013_clean_prior_demoted", "train013_low_overlap_with_direct_collateral",
            "control_supported", "interpretation",
        ],
        "preview": [
            "scene", "subset_name", "condition", "cluster_id", "evidence_level",
            "gaussian_cluster_risk", "candidate_reason", "source_view_names",
            "source_view_groups", "corrupted_plus_collateral_ratio",
            "clean_prior_ratio", "weak_support_score", "lifecycle_instability_score",
            "suggested_future_action", "do_not_apply_intervention",
        ],
        "missing": ["scene", "subset_name", "condition", "missing_path", "status", "details"],
    }


def _write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# PR19 Gaussian Cluster Risk",
        "",
        "PR19 is offline observation only. It is not a defense and does not reject views, reweight loss, suppress Gaussian updates, or gate densification.",
        "It does not tune PR17 normalization or PR18 spillover diagnosis.",
        "Corruption labels are used only for post-hoc grouping and evaluation, not for scoring.",
        "",
        "## Evidence Levels",
        "- `exact_gaussian_id`: exact per-Gaussian identifiers were available.",
        "- `aggregate_event_proxy`: only aggregate lifecycle/view influence rows were available.",
        "- `unavailable`: required influence files were missing.",
        "",
        "Proxy evidence supports representation-level suspicion but does not prove exact Gaussian causal overlap.",
        "",
        "## Summary",
        f"- Valid conditions: `{summary.get('valid_condition_count')}`",
        f"- Exact Gaussian conditions: `{summary.get('exact_gaussian_condition_count')}`",
        f"- Aggregate proxy conditions: `{summary.get('aggregate_proxy_condition_count')}`",
        f"- Missing conditions: `{summary.get('missing_condition_count')}`",
        f"- Intervention preview rows: `{summary.get('intervention_candidate_count')}`",
        "",
        "Intervention candidates are preview-only. Every preview row has `do_not_apply_intervention = true`.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _choose_evidence_level(mode: str, availability: dict[str, Any]) -> str:
    exact = availability.get("exact_gaussian_ids_available") is True
    aggregate = availability.get("aggregate_event_proxy_available") is True
    if mode == "exact":
        return "exact_gaussian_id" if exact else "unavailable"
    if mode == "aggregate_proxy":
        return "aggregate_event_proxy" if aggregate else "unavailable"
    if exact:
        return "exact_gaussian_id"
    if aggregate:
        return "aggregate_event_proxy"
    return "unavailable"


def analyze_pr19(
    *,
    input_root: Path,
    plan_dir: Path,
    pr17_dir: Path,
    pr18_dir: Path,
    output_dir: Path,
    scenes: list[str],
    conditions: list[str],
    subset_names: list[str],
    top_k: int,
    config_path: Path,
    mode: str,
    strict: bool,
    allow_missing: bool,
    write_markdown: bool,
    include_clean_prior_controls: bool,
    candidate_only: bool,
    min_cluster_support: int,
) -> tuple[dict[str, Any], int]:
    _bootstrap_project_imports()
    from viewtrust.analysis.clean_prior_normalization import load_json
    from viewtrust.analysis.gaussian_cluster_risk import (
        PR19_OUTPUT_FILES,
        build_aggregate_event_clusters,
        build_exact_gaussian_support_sets,
        build_intervention_preview,
        build_view_group_map,
        compute_cluster_risk_components,
        compute_condition_summary_rows,
        compute_group_overlap_summary,
        compute_summary,
        compute_train013_control_summary,
        inspect_gaussian_id_availability,
        load_pr17_rows,
        load_pr18_classification,
        load_pr18_condition_summary,
        rank_cluster_risks,
        resolve_offline_artifact_inputs,
        write_artifact_manifest,
        write_csv_rows,
        write_json,
    )
    from viewtrust.analysis.subset_scene_bias import discover_pr16_condition_output

    output_dir.mkdir(parents=True, exist_ok=True)
    config = load_json(config_path)
    if not config:
        raise FileNotFoundError(f"PR19 config not found or empty: {config_path}")
    if config.get("labels", {}).get("use_corruption_labels_for_scoring") is True:
        raise ValueError("PR19 config must not use corruption labels for scoring")

    pr17_rows = load_pr17_rows(pr17_dir)
    pr18_rows = load_pr18_classification(pr18_dir)
    _ = load_pr18_condition_summary(pr18_dir)

    evidence_rows: list[dict[str, Any]] = []
    view_group_rows: list[dict[str, Any]] = []
    cluster_rows: list[dict[str, Any]] = []
    ranking_rows: list[dict[str, Any]] = []
    concentration_rows: list[dict[str, Any]] = []
    overlap_rows: list[dict[str, Any]] = []
    train013_rows: list[dict[str, Any]] = []
    preview_rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    warnings: list[str] = []

    event_cfg = config.get("event_cluster", {})
    iteration_bucket_size = int(event_cfg.get("iteration_bucket_size", 100))
    effective_mode = mode if mode != "auto" else str(config.get("mode", "auto"))

    for scene in scenes:
        for subset_name in subset_names:
            for condition in conditions:
                condition_pr17_rows = [
                    row
                    for row in pr17_rows
                    if _condition_key(row) == (scene, subset_name, condition)
                ]
                if not condition_pr17_rows:
                    missing_rows.append(
                        {
                            "scene": scene,
                            "subset_name": subset_name,
                            "condition": condition,
                            "missing_path": str(pr17_dir / "clean_prior_normalized_rows.csv"),
                            "status": "missing",
                            "details": "no PR17 rows for condition",
                        }
                    )
                    evidence_rows.append(_unavailable_evidence(scene, subset_name, condition, "missing PR17 rows"))
                    continue
                condition_view_groups = build_view_group_map(
                    pr17_rows=condition_pr17_rows,
                    pr18_rows=pr18_rows,
                    scene=scene,
                    subset_name=subset_name,
                    condition=condition,
                    candidate_only=candidate_only,
                )
                view_group_rows.extend(condition_view_groups)
                signal_dir = discover_pr16_condition_output(input_root, scene, subset_name, condition)
                if signal_dir is None:
                    missing_rows.append(
                        {
                            "scene": scene,
                            "subset_name": subset_name,
                            "condition": condition,
                            "missing_path": str(input_root),
                            "status": "missing",
                            "details": "offline signal directory unavailable",
                        }
                    )
                    evidence_rows.append(_unavailable_evidence(scene, subset_name, condition, "offline signal directory unavailable"))
                    continue
                paths = resolve_offline_artifact_inputs(signal_dir)
                availability = inspect_gaussian_id_availability(paths)
                evidence_level = _choose_evidence_level(effective_mode, availability)
                evidence_rows.append(
                    {
                        "scene": scene,
                        "subset_name": subset_name,
                        "condition": condition,
                        "signal_dir": str(signal_dir),
                        "clean_dir": str(paths.get("input_clean_dir", "")),
                        "corrupt_dir": str(paths.get("input_corrupt_dir", "")),
                        "comparison_dir": str(paths.get("input_comparison_dir", "")),
                        "exact_gaussian_ids_available": availability["exact_gaussian_ids_available"],
                        "aggregate_event_proxy_available": availability["aggregate_event_proxy_available"],
                        "evidence_level": evidence_level,
                        "files_used": ";".join(availability["files_used"]),
                        "missing_files": ";".join(availability["missing_files"]),
                        "warnings": ";".join(availability["warnings"]),
                    }
                )
                if evidence_level == "unavailable":
                    missing_rows.append(
                        {
                            "scene": scene,
                            "subset_name": subset_name,
                            "condition": condition,
                            "missing_path": str(signal_dir),
                            "status": "unavailable",
                            "details": "no exact Gaussian IDs or aggregate event proxy files available",
                        }
                    )
                    continue
                if evidence_level == "exact_gaussian_id":
                    condition_clusters = build_exact_gaussian_support_sets(
                        scene=scene,
                        subset_name=subset_name,
                        condition=condition,
                        paths=paths,
                        view_group_rows=condition_view_groups,
                        min_cluster_support=min_cluster_support,
                    )
                else:
                    condition_clusters = build_aggregate_event_clusters(
                        scene=scene,
                        subset_name=subset_name,
                        condition=condition,
                        paths=paths,
                        view_group_rows=condition_view_groups,
                        iteration_bucket_size=iteration_bucket_size,
                        min_cluster_support=min_cluster_support,
                    )
                scored = rank_cluster_risks(compute_cluster_risk_components(condition_clusters, config))
                cluster_rows.extend(scored)
                concentration_rows.extend(compute_condition_summary_rows(scored, top_k))
                overlap_rows.extend(compute_group_overlap_summary(scored, condition_view_groups))
                train013_rows.append(compute_train013_control_summary(scored, condition_view_groups, top_k))
                preview_rows.extend(build_intervention_preview(scored, top_k))

    ranking_rows = [_ranking_row(row) for row in cluster_rows]
    summary = compute_summary(
        scenes=scenes,
        conditions=conditions,
        subset_names=subset_names,
        top_k=top_k,
        evidence_rows=evidence_rows,
        cluster_rows=cluster_rows,
        group_rows=concentration_rows,
        train013_rows=train013_rows,
        intervention_rows=preview_rows,
        missing_rows=missing_rows,
        warnings=warnings,
    )

    fields = _fields()
    write_json(output_dir / "pr19_gaussian_cluster_risk_summary.json", summary)
    write_csv_rows(output_dir / "pr19_evidence_availability.csv", evidence_rows, fields["evidence"])
    write_csv_rows(output_dir / "pr19_view_group_map.csv", view_group_rows, fields["view_group"])
    write_csv_rows(output_dir / "pr19_cluster_risk_rows.csv", cluster_rows, fields["cluster"])
    write_csv_rows(output_dir / "pr19_cluster_risk_rankings.csv", ranking_rows, fields["ranking"])
    write_csv_rows(output_dir / "pr19_group_concentration_summary.csv", concentration_rows, fields["group"])
    write_csv_rows(output_dir / "pr19_direct_collateral_overlap.csv", overlap_rows, fields["overlap"])
    write_csv_rows(output_dir / "pr19_train013_control_summary.csv", train013_rows, fields["train013"])
    write_csv_rows(output_dir / "pr19_intervention_candidate_preview.csv", preview_rows, fields["preview"])
    write_csv_rows(output_dir / "pr19_missing_outputs.csv", missing_rows, fields["missing"])
    _write_report(output_dir / "pr19_report.md", summary)
    write_artifact_manifest(output_dir, input_root, plan_dir, pr17_dir, pr18_dir)

    missing_required = [name for name in PR19_OUTPUT_FILES if not (output_dir / name).exists()]
    if missing_required:
        raise RuntimeError(f"missing PR19 outputs: {missing_required}")
    if strict and missing_rows:
        return summary, 1
    if not allow_missing and missing_rows:
        return summary, 1
    return summary, 0


def _unavailable_evidence(scene: str, subset_name: str, condition: str, warning: str) -> dict[str, Any]:
    return {
        "scene": scene,
        "subset_name": subset_name,
        "condition": condition,
        "signal_dir": "",
        "clean_dir": "",
        "corrupt_dir": "",
        "comparison_dir": "",
        "exact_gaussian_ids_available": False,
        "aggregate_event_proxy_available": False,
        "evidence_level": "unavailable",
        "files_used": "",
        "missing_files": "",
        "warnings": warning,
    }


def _number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _ranking_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "scene": row.get("scene", ""),
        "subset_name": row.get("subset_name", ""),
        "condition": row.get("condition", ""),
        "rank": row.get("rank", ""),
        "cluster_id": row.get("cluster_id", ""),
        "evidence_level": row.get("evidence_level", ""),
        "gaussian_cluster_risk": row.get("gaussian_cluster_risk", ""),
        "corrupted_plus_collateral_ratio": row.get("corrupted_plus_collateral_ratio", ""),
        "clean_prior_ratio": row.get("clean_prior_ratio", ""),
        "source_concentration_score": row.get("source_concentration_score", ""),
        "lifecycle_instability_score": row.get("lifecycle_instability_score", ""),
        "weak_support_score": row.get("weak_support_score", ""),
        "visibility_delta_score": row.get("visibility_delta_score", ""),
        "source_view_names": row.get("source_view_names", ""),
        "source_view_groups": row.get("source_view_group", ""),
        "main_reason": _main_reason(row),
    }


def _main_reason(row: dict[str, Any]) -> str:
    reasons = []
    if _number(row.get("corrupted_plus_collateral_ratio")) >= 0.5:
        reasons.append("direct_collateral_concentration")
    if _number(row.get("lifecycle_instability_score")) > 0:
        reasons.append("lifecycle_instability")
    if _number(row.get("visibility_delta_score")) > 0:
        reasons.append("visibility_delta")
    return ";".join(reasons) or "ranked_cluster"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", default=Path("outputs/reports"), type=Path)
    parser.add_argument("--plan-dir", required=True, type=Path)
    parser.add_argument("--pr17-dir", required=True, type=Path)
    parser.add_argument("--pr18-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--scenes", nargs="+", default=["chair", "drums"])
    parser.add_argument("--conditions", nargs="+", default=["corrupt_occluder", "corrupt_noise", "corrupt_mixed"])
    parser.add_argument("--subset-names", nargs="+", default=["original", "seed_20260710"])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--mode", choices=("auto", "exact", "aggregate_proxy"), default="auto")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--write-markdown", action="store_true")
    parser.add_argument("--include-clean-prior-controls", action="store_true")
    parser.add_argument("--candidate-only", action="store_true")
    parser.add_argument("--min-cluster-support", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    project_root = _bootstrap_project_imports()
    args = parse_args()
    try:
        summary, exit_code = analyze_pr19(
            input_root=_resolve_path(project_root, args.input_root),
            plan_dir=_resolve_path(project_root, args.plan_dir),
            pr17_dir=_resolve_path(project_root, args.pr17_dir),
            pr18_dir=_resolve_path(project_root, args.pr18_dir),
            output_dir=_resolve_path(project_root, args.output_dir),
            scenes=args.scenes,
            conditions=args.conditions,
            subset_names=args.subset_names,
            top_k=args.top_k,
            config_path=_resolve_path(project_root, args.config),
            mode=args.mode,
            strict=args.strict,
            allow_missing=args.allow_missing,
            write_markdown=args.write_markdown,
            include_clean_prior_controls=args.include_clean_prior_controls,
            candidate_only=args.candidate_only,
            min_cluster_support=args.min_cluster_support,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
