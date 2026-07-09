#!/usr/bin/env python3
"""LOCAL-SAFE smoke test for PR19 Gaussian cluster risk analysis."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path


REQUIRED_OUTPUTS = [
    "pr19_gaussian_cluster_risk_summary.json",
    "pr19_evidence_availability.csv",
    "pr19_view_group_map.csv",
    "pr19_cluster_risk_rows.csv",
    "pr19_cluster_risk_rankings.csv",
    "pr19_group_concentration_summary.csv",
    "pr19_direct_collateral_overlap.csv",
    "pr19_train013_control_summary.csv",
    "pr19_intervention_candidate_preview.csv",
    "pr19_missing_outputs.csv",
    "pr19_report.md",
    "artifact_manifest.csv",
]


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, object]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = fields or list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _condition_rows(condition: str) -> list[dict[str, object]]:
    corrupted = {"train_004", "train_009", "train_012", "train_017"}
    rows = []
    for view in ["train_004", "train_009", "train_012", "train_017", "train_014", "train_007", "train_013", "train_001"]:
        raw_rank = 8
        norm_rank = 8
        if view in corrupted:
            raw_rank = norm_rank = len(rows) + 1
        if view == "train_014":
            norm_rank = 5
        if view == "train_007":
            norm_rank = 6
        if view == "train_013":
            raw_rank = 1
            norm_rank = 12
        rows.append(
            {
                "scene": "chair",
                "subset_name": "seed_20260710",
                "subset_seed": "20260710",
                "condition": condition,
                "view_name": view,
                "view_split": "train",
                "was_corrupted": str(view in corrupted).lower(),
                "raw_risk": 10.0 if view == "train_013" else 5.0,
                "raw_rank": raw_rank,
                "clean_prior_risk": 9.0 if view == "train_013" else 0.1,
                "clean_prior_rank": 1 if view == "train_013" else 10,
                "delta_risk": 0.1 if view == "train_013" else 4.9,
                "positive_delta_risk": 0.1 if view == "train_013" else 4.9,
                "prior_suppressed_risk": 1.0,
                "rank_lift_score": -5.0 if view == "train_013" else 5.0,
                "normalized_viewtrust_risk": 0.1 if view == "train_013" else 5.0,
                "normalized_rank": norm_rank,
                "normalized_consistency": 0.2,
                "raw_top_k": str(raw_rank <= 7),
                "normalized_top_k": str(norm_rank <= 7),
                "raw_false_positive": str(raw_rank <= 7 and view not in corrupted),
                "normalized_false_positive": str(norm_rank <= 7 and view not in corrupted),
                "prior_source": "mock",
                "component_warnings": "",
            }
        )
    return rows


def _make_pr17(pr17_dir: Path) -> None:
    rows = _condition_rows("corrupt_occluder") + _condition_rows("corrupt_noise")
    _write_csv(pr17_dir / "clean_prior_normalized_rows.csv", rows)
    _write_csv(
        pr17_dir / "clean_prior_normalized_rankings.csv",
        [
            {
                "scene": row["scene"],
                "subset_name": row["subset_name"],
                "subset_seed": row["subset_seed"],
                "condition": row["condition"],
                "score_name": "normalized_viewtrust_risk",
                "rank": row["normalized_rank"],
                "view_name": row["view_name"],
                "was_corrupted": row["was_corrupted"],
                "score": row["normalized_viewtrust_risk"],
            }
            for row in rows
        ],
    )
    _write_json(
        pr17_dir / "clean_prior_normalized_summary.json",
        {
            "schema_name": "viewtrust.pr17.clean_prior_normalized.summary",
            "observation_only": True,
            "training_intervention": False,
            "defense_enabled": False,
            "uses_corruption_labels_for_scoring": False,
        },
    )


def _make_pr18(pr18_dir: Path) -> None:
    rows = []
    for condition in ["corrupt_occluder", "corrupt_noise"]:
        for view, klass in [
            ("train_014", "co_visible_collateral"),
            ("train_007", "co_visible_collateral"),
            ("train_013", "clean_prior_false_positive"),
        ]:
            rows.append(
                {
                    "scene": "chair",
                    "subset_name": "seed_20260710",
                    "condition": condition,
                    "view_name": view,
                    "spillover_class": klass,
                    "spillover_confidence": "high",
                    "normalized_false_positive": str(klass == "co_visible_collateral"),
                    "was_corrupted": "false",
                    "camera_neighbor_evidence": str(klass == "co_visible_collateral"),
                    "index_neighbor_evidence": str(klass == "co_visible_collateral"),
                    "gaussian_overlap_evidence": str(klass == "co_visible_collateral"),
                    "stable_prior_pattern": str(klass == "clean_prior_false_positive"),
                    "collateral_lift_pattern": str(klass == "co_visible_collateral"),
                    "explanation": "mock",
                }
            )
    _write_csv(pr18_dir / "pr18_spillover_classification.csv", rows)
    _write_csv(pr18_dir / "pr18_condition_summary.csv", [{"scene": "chair", "subset_name": "seed_20260710", "condition": "corrupt_occluder"}])
    _write_json(
        pr18_dir / "pr18_covisibility_spillover_summary.json",
        {
            "schema_name": "viewtrust.pr18.covisibility_spillover.summary",
            "observation_only": True,
            "training_intervention": False,
            "defense_enabled": False,
        },
    )


def _make_plan(plan_dir: Path) -> None:
    _write_csv(
        plan_dir / "pr16_subset_manifest.csv",
        [
            {
                "scene": "chair",
                "subset_name": "seed_20260710",
                "subset_seed": "20260710",
                "train_view_count": 20,
                "corrupted_view_count": 4,
                "corrupted_view_names": "train_004;train_009;train_012;train_017",
                "status": "ok",
            }
        ],
    )


def _offline_shell(signal_dir: Path, clean_dir: Path, corrupt_dir: Path, comparison_dir: Path, condition: str) -> None:
    rows = [
        {"view_name": "train_004", "was_corrupted": "true", "offline_viewtrust_risk": 5.0, "rank": 1},
        {"view_name": "train_014", "was_corrupted": "false", "offline_viewtrust_risk": 4.0, "rank": 2},
    ]
    _write_json(
        signal_dir / "offline_viewtrust_summary.json",
        {
            "schema_name": "viewtrust.offline_signal.summary",
            "schema_version": 1,
            "scene": "chair",
            "corrupt_condition": condition,
            "observation_only": True,
            "training_intervention": False,
            "defense_enabled": False,
            "uses_corruption_labels_for_scoring": False,
            "uses_corruption_labels_for_evaluation": True,
        },
    )
    _write_csv(signal_dir / "offline_viewtrust_signals.csv", rows)
    _write_csv(signal_dir / "offline_viewtrust_rankings.csv", rows)
    _write_csv(signal_dir / "offline_viewtrust_group_metrics.csv", [{"group": "all", "view_count": 2}])
    _write_csv(signal_dir / "offline_viewtrust_signal_ablation.csv", [{"signal_name": "full_signal", "precision_at_k": 1.0}])
    _write_json(signal_dir / "offline_viewtrust_config.json", {"schema_name": "mock"})
    (signal_dir / "offline_viewtrust_report.md").write_text("mock\n", encoding="utf-8")
    manifest_rows = []
    for group, root, names in [
        ("input_clean", clean_dir, ["view_influence.csv", "view_lifecycle_attribution.csv"]),
        ("input_corrupt", corrupt_dir, ["view_influence.csv", "view_lifecycle_attribution.csv", "view_iteration_events.csv"]),
        ("input_comparison", comparison_dir, ["view_influence_comparison.csv"]),
    ]:
        for name in names:
            path = root / name
            manifest_rows.append(
                {
                    "relative_path": f"{group}/{name}",
                    "path": str(path),
                    "exists": str(path.exists()).lower(),
                    "file_type": path.suffix.lstrip("."),
                    "size_bytes": path.stat().st_size if path.exists() else "",
                    "required": "true",
                    "artifact_group": group,
                }
            )
    _write_csv(signal_dir / "offline_viewtrust_artifact_manifest.csv", manifest_rows)


def _make_exact_condition(input_root: Path) -> None:
    signal_dir = input_root / "offline_viewtrust_chair_corrupt_occluder_seed_20260710_pr16_input"
    clean_dir = input_root / "exact_clean"
    corrupt_dir = input_root / "exact_corrupt"
    comparison_dir = input_root / "exact_comparison"
    _write_csv(clean_dir / "view_influence.csv", [{"view_name": "train_013", "mean_visibility_ratio": 0.1}])
    _write_csv(
        corrupt_dir / "view_lifecycle_attribution.csv",
        [
            {"source_view_name": "train_004", "lifecycle_action": "clone_birth", "source_iteration": 100, "gaussian_id": "g_shared", "event_count": 4, "clone_birth_count": 4, "final_alive_count": 3},
            {"source_view_name": "train_014", "lifecycle_action": "clone_birth", "source_iteration": 100, "gaussian_id": "g_shared", "event_count": 3, "clone_birth_count": 3, "final_alive_count": 2},
            {"source_view_name": "train_007", "lifecycle_action": "split_birth", "source_iteration": 100, "gaussian_id": "g_shared", "event_count": 2, "split_birth_count": 2, "final_dead_count": 1},
            {"source_view_name": "train_013", "lifecycle_action": "clone_birth", "source_iteration": 500, "gaussian_id": "g_prior", "event_count": 2, "clone_birth_count": 2, "final_alive_count": 2},
        ],
    )
    _write_csv(corrupt_dir / "view_influence.csv", [{"view_name": "train_004", "birth_event_count_after_view": 4}])
    _write_csv(corrupt_dir / "view_iteration_events.csv", [{"view_name": "train_004", "iteration": 100, "gaussian_id": "g_shared", "birth_event_count": 1}])
    _write_csv(comparison_dir / "view_influence_comparison.csv", [{"view_name": "train_004", "birth_event_count_delta": 4, "visibility_ratio_delta": 2}])
    _offline_shell(signal_dir, clean_dir, corrupt_dir, comparison_dir, "corrupt_occluder")


def _make_proxy_condition(input_root: Path) -> None:
    signal_dir = input_root / "offline_viewtrust_chair_corrupt_noise_seed_20260710_pr16_input"
    clean_dir = input_root / "proxy_clean"
    corrupt_dir = input_root / "proxy_corrupt"
    comparison_dir = input_root / "proxy_comparison"
    _write_csv(clean_dir / "view_influence.csv", [{"view_name": "train_013", "mean_visibility_ratio": 0.2}])
    _write_csv(
        corrupt_dir / "view_lifecycle_attribution.csv",
        [
            {"source_view_name": "train_004", "lifecycle_action": "clone_birth", "source_iteration": 100, "event_count": 4, "unique_gaussian_count": 4, "clone_birth_count": 4},
            {"source_view_name": "train_014", "lifecycle_action": "clone_birth", "source_iteration": 100, "event_count": 3, "unique_gaussian_count": 3, "clone_birth_count": 3},
            {"source_view_name": "train_013", "lifecycle_action": "prune_death", "source_iteration": 500, "event_count": 1, "unique_gaussian_count": 1, "prune_death_count": 1},
        ],
    )
    _write_csv(corrupt_dir / "view_influence.csv", [{"view_name": "train_004", "birth_event_count_after_view": 4}])
    _write_csv(corrupt_dir / "view_iteration_events.csv", [{"view_name": "train_014", "iteration": 100, "birth_event_count": 2}])
    _write_csv(comparison_dir / "view_influence_comparison.csv", [{"view_name": "train_004", "birth_event_count_delta": 4, "visibility_ratio_delta": 2}])
    _offline_shell(signal_dir, clean_dir, corrupt_dir, comparison_dir, "corrupt_noise")


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def main() -> int:
    project_root = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory(prefix="viewtrust-pr19-") as tmp:
        root = Path(tmp)
        input_root = root / "reports"
        plan_dir = root / "plan"
        pr17_dir = root / "pr17"
        pr18_dir = root / "pr18"
        output_dir = root / "pr19"
        _make_plan(plan_dir)
        _make_pr17(pr17_dir)
        _make_pr18(pr18_dir)
        _make_exact_condition(input_root)
        _make_proxy_condition(input_root)
        result = _run(
            [
                sys.executable,
                str(project_root / "scripts" / "measure" / "analyze_pr19_gaussian_cluster_risk.py"),
                "--input-root",
                str(input_root),
                "--plan-dir",
                str(plan_dir),
                "--pr17-dir",
                str(pr17_dir),
                "--pr18-dir",
                str(pr18_dir),
                "--output-dir",
                str(output_dir),
                "--scenes",
                "chair",
                "--conditions",
                "corrupt_occluder",
                "corrupt_noise",
                "corrupt_mixed",
                "--subset-names",
                "seed_20260710",
                "--top-k",
                "5",
                "--allow-missing",
                "--write-markdown",
            ]
        )
        if result.returncode != 0:
            print(result.stdout)
            print(result.stderr, file=sys.stderr)
            return result.returncode
        for name in REQUIRED_OUTPUTS:
            path = output_dir / name
            assert path.exists(), name
            assert path.stat().st_size > 0, name
        summary = json.loads((output_dir / "pr19_gaussian_cluster_risk_summary.json").read_text())
        assert summary["schema_name"] == "viewtrust.pr19.gaussian_cluster_risk.summary"
        assert summary["observation_only"] is True
        assert summary["training_intervention"] is False
        assert summary["defense_enabled"] is False
        assert summary["uses_corruption_labels_for_scoring"] is False
        assert summary["uses_corruption_labels_for_evaluation"] is True
        assert summary["exact_gaussian_condition_count"] == 1
        assert summary["aggregate_proxy_condition_count"] == 1
        assert summary["missing_condition_count"] > 0
        assert summary["train013_control_supported"] is True
        evidence = _read_csv(output_dir / "pr19_evidence_availability.csv")
        assert any(row["evidence_level"] == "exact_gaussian_id" for row in evidence)
        assert any(row["evidence_level"] == "aggregate_event_proxy" for row in evidence)
        preview = _read_csv(output_dir / "pr19_intervention_candidate_preview.csv")
        assert preview
        assert all(row["do_not_apply_intervention"] == "True" for row in preview)
        clusters = _read_csv(output_dir / "pr19_cluster_risk_rows.csv")
        assert any(float(row["corrupted_plus_collateral_ratio"]) >= 0.5 for row in clusters)
        report = (output_dir / "pr19_report.md").read_text(encoding="utf-8").lower()
        assert "not a defense" in report
        assert "proxy evidence" in report
    print("pr19 gaussian cluster risk smoke test ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
