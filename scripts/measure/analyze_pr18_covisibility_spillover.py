#!/usr/bin/env python3
"""Build PR18 co-visibility spillover diagnostics for normalized ViewTrust."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_CONFIG = Path("configs/offline_viewtrust_signal/default_pr18_covisibility_spillover.json")


def _bootstrap_project_imports() -> Path:
    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root))
    return project_root


def _resolve_path(project_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else project_root / path


def _condition_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return str(row.get("scene", "")), str(row.get("subset_name", "")), str(row.get("condition", ""))


def _truth(value: Any) -> bool:
    text = str(value).strip().lower()
    return value is True or text in {"true", "1", "yes"}


def _condition_suffix(condition: str, subset_name: str) -> str:
    return f"{condition}_{subset_name}"


def _fields() -> dict[str, list[str]]:
    return {
        "candidate": [
            "scene", "subset_name", "subset_seed", "condition", "view_name",
            "was_corrupted", "raw_rank", "normalized_rank", "raw_top_k",
            "normalized_top_k", "raw_false_positive", "normalized_false_positive",
            "raw_risk", "clean_prior_risk", "delta_risk", "positive_delta_risk",
            "prior_suppressed_risk", "rank_lift_score", "normalized_viewtrust_risk",
            "nearest_corrupted_view", "nearest_corrupted_center_distance",
            "nearest_corrupted_rotation_angle_deg", "nearest_corrupted_combined_distance",
            "corrupted_neighbor_rank", "mean_distance_to_corrupted_views",
            "nearest_corrupted_index_gap", "between_corrupted_indices",
            "adjacent_to_corrupted_index", "max_gaussian_support_jaccard_with_corrupted",
            "mean_gaussian_support_jaccard_with_corrupted", "gaussian_overlap_source",
            "stable_prior_pattern", "collateral_lift_pattern", "camera_neighbor_evidence",
            "index_neighbor_evidence", "gaussian_overlap_evidence", "spillover_class",
            "spillover_confidence", "explanation",
        ],
        "camera": [
            "scene", "subset_name", "condition", "view_name", "nearest_corrupted_view",
            "nearest_corrupted_center_distance", "nearest_corrupted_rotation_angle_deg",
            "nearest_corrupted_combined_distance", "corrupted_neighbor_rank",
            "is_camera_neighbor_of_corrupted", "all_corrupted_neighbors_sorted",
        ],
        "pairs": [
            "scene", "view_a", "view_b", "center_distance", "rotation_angle_deg",
            "combined_camera_distance", "index_gap", "view_a_was_corrupted_in_condition",
            "view_b_was_corrupted_in_condition", "scene_median_center_distance",
            "scene_median_rotation_angle_deg",
        ],
        "gaussian": [
            "scene", "subset_name", "condition", "candidate_view", "corrupted_view",
            "overlap_source", "candidate_support_count", "corrupted_support_count",
            "intersection_count", "union_count", "jaccard", "overlap_available", "warnings",
        ],
        "classification": [
            "scene", "subset_name", "condition", "view_name", "spillover_class",
            "spillover_confidence", "normalized_false_positive", "was_corrupted",
            "camera_neighbor_evidence", "index_neighbor_evidence",
            "gaussian_overlap_evidence", "stable_prior_pattern", "collateral_lift_pattern",
            "explanation",
        ],
        "condition": [
            "scene", "subset_name", "condition", "normalized_false_positive_count",
            "clean_prior_false_positive_count", "co_visible_collateral_count",
            "unexplained_false_positive_count", "removed_train_013_count",
            "collateral_views", "unexplained_views", "corrupted_views",
            "top_normalized_views", "status", "warnings",
        ],
        "transition": [
            "scene", "view_name", "raw_false_positive_count",
            "normalized_false_positive_count", "co_visible_collateral_count",
            "clean_prior_false_positive_count", "unexplained_false_positive_count",
            "transition_type", "conditions_as_collateral", "conditions_unexplained",
            "conditions_clean_prior",
        ],
        "missing": ["scene", "subset_name", "condition", "missing_path", "status", "details"],
    }


def _write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# PR18 Co-visibility Spillover Diagnosis",
        "",
        "PR18 is offline observation only. It is not a defense, does not tune PR17 scores, and does not change training or rendering.",
        "Corruption labels are used only after scoring to evaluate direct corrupted views and false positives.",
        "",
        "## Summary",
        f"- Valid conditions: `{summary.get('valid_condition_count')}`",
        f"- Missing conditions: `{summary.get('missing_condition_count')}`",
        f"- Normalized false positives: `{summary.get('normalized_false_positive_count')}`",
        f"- Co-visible collateral: `{summary.get('co_visible_collateral_count')}`",
        f"- Clean-prior false positives: `{summary.get('clean_prior_false_positive_count')}`",
        f"- Unexplained false positives: `{summary.get('unexplained_false_positive_count')}`",
        "",
        "## Interpretation",
        "A clean view in normalized top-k is classified as co-visible collateral only when normalized lift is paired with camera, index, or Gaussian-support evidence.",
        "Unexplained false positives are kept visible for later scientific review rather than hidden by score tuning.",
        "",
        "## Limitations",
        "- Gaussian support overlap is best effort and may be unavailable when per-Gaussian IDs are not present.",
        "- View-index adjacency is auxiliary; camera geometry is the primary neighborhood evidence.",
        "- PR18 is a prerequisite diagnosis before any future intervention, not an intervention.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def analyze_pr18(
    *,
    data_root: Path,
    input_root: Path,
    plan_dir: Path,
    pr17_dir: Path,
    output_dir: Path,
    scenes: list[str],
    conditions: list[str],
    subset_names: list[str],
    top_k: int,
    config_path: Path,
    neighbor_k: int | None,
    strict: bool,
    allow_missing: bool,
    gaussian_overlap_mode: str,
    camera_only: bool,
    write_markdown: bool,
) -> tuple[dict[str, Any], int]:
    _bootstrap_project_imports()
    from viewtrust.analysis.covisibility_spillover import (
        PR18_OUTPUT_FILES,
        build_view_identity_transition,
        classify_spillover_candidates,
        compute_camera_pair_distances,
        compute_corrupted_neighbor_features,
        compute_gaussian_support_overlap_best_effort,
        compute_index_neighbor_features,
        compute_spillover_summary,
        load_camera_poses,
        load_pr16_plan,
        load_pr17_rankings,
        load_pr17_rows,
        write_artifact_manifest,
        write_csv_rows,
        write_json,
    )
    from viewtrust.analysis.clean_prior_normalization import load_json
    from viewtrust.analysis.subset_scene_bias import discover_pr16_condition_output

    output_dir.mkdir(parents=True, exist_ok=True)
    config = load_json(config_path)
    if not config:
        raise FileNotFoundError(f"PR18 config not found or empty: {config_path}")
    if config.get("labels", {}).get("use_corruption_labels_for_scoring") is True:
        raise ValueError("PR18 config must not use corruption labels for scoring")

    camera_cfg = config.get("camera_neighbor", {})
    index_cfg = config.get("index_neighbor", {})
    gaussian_cfg = config.get("gaussian_support", {})
    effective_neighbor_k = neighbor_k or int(camera_cfg.get("neighbor_k", 3))
    effective_gaussian_mode = "off" if camera_only else gaussian_overlap_mode

    pr17_rows = load_pr17_rows(pr17_dir)
    _ = load_pr17_rankings(pr17_dir)
    plan = load_pr16_plan(plan_dir)
    rows_by_condition: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in pr17_rows:
        rows_by_condition[_condition_key(row)].append(dict(row))

    camera_cache: dict[str, tuple[dict[str, dict[str, Any]], list[dict[str, Any]], dict[tuple[str, str], dict[str, Any]], dict[str, float], Path | None]] = {}
    candidate_rows: list[dict[str, Any]] = []
    camera_rows: list[dict[str, Any]] = []
    pair_rows_out: list[dict[str, Any]] = []
    gaussian_rows: list[dict[str, Any]] = []
    classification_rows: list[dict[str, Any]] = []
    condition_rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    warnings: list[str] = []

    for scene in scenes:
        poses, transform_path = load_camera_poses(data_root, scene)
        if poses:
            pair_rows, pair_lookup, pair_stats = compute_camera_pair_distances(
                poses,
                center_weight=float(camera_cfg.get("center_distance_weight", 1.0)),
                rotation_weight=float(camera_cfg.get("rotation_distance_weight", 1.0)),
            )
        else:
            pair_rows, pair_lookup, pair_stats = [], {}, {}
            warnings.append(f"{scene}: transforms_train.json unavailable")
        camera_cache[scene] = (poses, pair_rows, pair_lookup, pair_stats, transform_path)

    for scene in scenes:
        poses, pair_rows, pair_lookup, pair_stats, transform_path = camera_cache[scene]
        for subset_name in subset_names:
            planned = plan.get((scene, subset_name), {})
            subset_seed = str(planned.get("subset_seed", subset_name.removeprefix("seed_") if subset_name.startswith("seed_") else ""))
            for condition in conditions:
                key = (scene, subset_name, condition)
                group_rows = rows_by_condition.get(key, [])
                if not group_rows:
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
                    condition_rows.append(_missing_condition_row(scene, subset_name, condition, "missing_pr17_rows"))
                    continue
                corrupted_views = sorted(
                    {
                        str(row.get("view_name", ""))
                        for row in group_rows
                        if _truth(row.get("was_corrupted"))
                    }
                    or set(planned.get("corrupted_view_names", []))
                )
                view_names = sorted({str(row.get("view_name", "")) for row in group_rows if row.get("view_name")})
                if not poses:
                    missing_rows.append(
                        {
                            "scene": scene,
                            "subset_name": subset_name,
                            "condition": condition,
                            "missing_path": str(data_root),
                            "status": "missing",
                            "details": "camera transforms unavailable",
                        }
                    )
                camera_features = compute_corrupted_neighbor_features(
                    view_names,
                    corrupted_views,
                    pair_lookup,
                    pair_rows,
                    pair_stats,
                    neighbor_k=effective_neighbor_k,
                    median_nn_distance_factor=float(camera_cfg.get("median_nn_distance_factor", 1.5)),
                )
                index_features = compute_index_neighbor_features(
                    view_names,
                    corrupted_views,
                    max_index_gap=int(index_cfg.get("max_index_gap", 2)),
                )
                signal_dir = discover_pr16_condition_output(input_root, scene, subset_name, condition)
                candidates = [
                    dict(row)
                    for row in group_rows
                    if _truth(row.get("raw_false_positive")) or _truth(row.get("normalized_false_positive"))
                ]
                overlap_rows, gaussian_features = compute_gaussian_support_overlap_best_effort(
                    signal_dir=signal_dir,
                    candidate_views=[str(row.get("view_name", "")) for row in candidates],
                    corrupted_views=corrupted_views,
                    mode=effective_gaussian_mode,
                    evidence_threshold=float(gaussian_cfg.get("evidence_jaccard_threshold", 0.25)),
                )
                for overlap in overlap_rows:
                    gaussian_rows.append({"scene": scene, "subset_name": subset_name, "condition": condition, **overlap})
                classified = classify_spillover_candidates(
                    candidate_rows=candidates,
                    group_rows=group_rows,
                    camera_features=camera_features,
                    index_features=index_features,
                    gaussian_features=gaussian_features,
                    config=config,
                )
                classification_rows.extend(classified)
                candidate_rows.extend(classified)
                for view_name, features in sorted(camera_features.items()):
                    camera_rows.append(
                        {
                            "scene": scene,
                            "subset_name": subset_name,
                            "condition": condition,
                            "view_name": view_name,
                            **features,
                        }
                    )
                corrupted_set = set(corrupted_views)
                for row in pair_rows:
                    pair_rows_out.append(
                        {
                            "scene": scene,
                            **row,
                            "view_a_was_corrupted_in_condition": row["view_a"] in corrupted_set,
                            "view_b_was_corrupted_in_condition": row["view_b"] in corrupted_set,
                        }
                    )
                condition_rows.append(
                    _condition_summary_row(
                        scene=scene,
                        subset_name=subset_name,
                        condition=condition,
                        group_rows=group_rows,
                        classified=classified,
                        corrupted_views=corrupted_views,
                        transform_path=transform_path,
                        signal_dir=signal_dir,
                    )
                )

    transition_rows = build_view_identity_transition(classification_rows)
    summary = compute_spillover_summary(
        scenes=scenes,
        conditions=conditions,
        subset_names=subset_names,
        top_k=top_k,
        condition_rows=condition_rows,
        classification_rows=classification_rows,
        missing_rows=missing_rows,
        gaussian_overlap_rows=gaussian_rows,
        warnings=warnings,
    )
    summary.pop("_view_lookup_count", None)

    fields = _fields()
    write_json(output_dir / "pr18_covisibility_spillover_summary.json", summary)
    write_csv_rows(output_dir / "pr18_candidate_false_positive_diagnosis.csv", candidate_rows, fields["candidate"])
    write_csv_rows(output_dir / "pr18_camera_neighbor_table.csv", camera_rows, fields["camera"])
    write_csv_rows(output_dir / "pr18_view_pair_distance_table.csv", pair_rows_out, fields["pairs"])
    write_csv_rows(output_dir / "pr18_gaussian_support_overlap.csv", gaussian_rows, fields["gaussian"])
    write_csv_rows(output_dir / "pr18_spillover_classification.csv", classification_rows, fields["classification"])
    write_csv_rows(output_dir / "pr18_condition_summary.csv", condition_rows, fields["condition"])
    write_csv_rows(output_dir / "pr18_view_identity_transition.csv", transition_rows, fields["transition"])
    write_csv_rows(output_dir / "pr18_missing_outputs.csv", missing_rows, fields["missing"])
    if write_markdown:
        _write_report(output_dir / "pr18_report.md", summary)
    else:
        _write_report(output_dir / "pr18_report.md", summary)
    write_artifact_manifest(output_dir, pr17_dir, plan_dir, data_root, input_root)

    missing_required = [name for name in PR18_OUTPUT_FILES if not (output_dir / name).exists()]
    if missing_required:
        raise RuntimeError(f"missing PR18 outputs: {missing_required}")
    if strict and missing_rows:
        return summary, 1
    if not allow_missing and missing_rows:
        return summary, 1
    return summary, 0


def _missing_condition_row(scene: str, subset_name: str, condition: str, warning: str) -> dict[str, Any]:
    return {
        "scene": scene,
        "subset_name": subset_name,
        "condition": condition,
        "normalized_false_positive_count": 0,
        "clean_prior_false_positive_count": 0,
        "co_visible_collateral_count": 0,
        "unexplained_false_positive_count": 0,
        "removed_train_013_count": 0,
        "collateral_views": "",
        "unexplained_views": "",
        "corrupted_views": "",
        "top_normalized_views": "",
        "status": "missing",
        "warnings": warning,
    }


def _condition_summary_row(
    *,
    scene: str,
    subset_name: str,
    condition: str,
    group_rows: list[dict[str, Any]],
    classified: list[dict[str, Any]],
    corrupted_views: list[str],
    transform_path: Path | None,
    signal_dir: Path | None,
) -> dict[str, Any]:
    normalized_fp = [row for row in classified if _truth(row.get("normalized_false_positive"))]
    collateral = [row for row in normalized_fp if row.get("spillover_class") == "co_visible_collateral"]
    clean_prior = [row for row in normalized_fp if row.get("spillover_class") == "clean_prior_false_positive"]
    unexplained = [row for row in normalized_fp if row.get("spillover_class") == "unexplained_false_positive"]
    train_013_removed = [
        row
        for row in classified
        if row.get("view_name") == "train_013"
        and _truth(row.get("raw_false_positive"))
        and not _truth(row.get("normalized_false_positive"))
    ]
    top_normalized = sorted(
        group_rows,
        key=lambda row: int(float(row.get("normalized_rank", 999999) or 999999)),
    )[:10]
    warnings = []
    if transform_path is None:
        warnings.append("camera_transforms_unavailable")
    if signal_dir is None:
        warnings.append("offline_signal_dir_unavailable_for_gaussian_overlap")
    return {
        "scene": scene,
        "subset_name": subset_name,
        "condition": condition,
        "normalized_false_positive_count": len(normalized_fp),
        "clean_prior_false_positive_count": len(clean_prior),
        "co_visible_collateral_count": len(collateral),
        "unexplained_false_positive_count": len(unexplained),
        "removed_train_013_count": len(train_013_removed),
        "collateral_views": ";".join(sorted(str(row.get("view_name", "")) for row in collateral)),
        "unexplained_views": ";".join(sorted(str(row.get("view_name", "")) for row in unexplained)),
        "corrupted_views": ";".join(corrupted_views),
        "top_normalized_views": ";".join(str(row.get("view_name", "")) for row in top_normalized),
        "status": "ok",
        "warnings": ";".join(warnings),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=Path("data"), type=Path)
    parser.add_argument("--input-root", default=Path("outputs/reports"), type=Path)
    parser.add_argument("--plan-dir", required=True, type=Path)
    parser.add_argument("--pr17-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--scenes", nargs="+", default=["chair", "drums"])
    parser.add_argument("--conditions", nargs="+", default=["corrupt_occluder", "corrupt_noise", "corrupt_mixed"])
    parser.add_argument("--subset-names", nargs="+", default=["original", "seed_20260710"])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--neighbor-k", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--write-markdown", action="store_true")
    parser.add_argument("--gaussian-overlap-mode", choices=("auto", "exact", "proxy", "off"), default="auto")
    parser.add_argument("--camera-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    project_root = _bootstrap_project_imports()
    args = parse_args()
    try:
        summary, exit_code = analyze_pr18(
            data_root=_resolve_path(project_root, args.data_root),
            input_root=_resolve_path(project_root, args.input_root),
            plan_dir=_resolve_path(project_root, args.plan_dir),
            pr17_dir=_resolve_path(project_root, args.pr17_dir),
            output_dir=_resolve_path(project_root, args.output_dir),
            scenes=args.scenes,
            conditions=args.conditions,
            subset_names=args.subset_names,
            top_k=args.top_k,
            config_path=_resolve_path(project_root, args.config),
            neighbor_k=args.neighbor_k,
            strict=args.strict,
            allow_missing=args.allow_missing,
            gaussian_overlap_mode=args.gaussian_overlap_mode,
            camera_only=args.camera_only,
            write_markdown=args.write_markdown,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
