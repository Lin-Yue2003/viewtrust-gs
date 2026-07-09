#!/usr/bin/env python3
"""LOCAL-SAFE smoke test for PR19.5 attribution semantics audit."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path


REQUIRED_OUTPUTS = [
    "pr195_attribution_semantics_summary.json",
    "pr195_support_mode_failure_analysis.csv",
    "pr195_event_type_group_distribution.csv",
    "pr195_view_group_event_distribution.csv",
    "pr195_high_event_semantics_audit.csv",
    "pr195_suspicious_alive_degeneracy_audit.csv",
    "pr195_birth_prune_semantics_audit.csv",
    "pr195_train013_semantics_audit.csv",
    "pr195_required_attribution_field_gap.csv",
    "pr195_pr20_readiness_assessment.csv",
    "pr195_next_step_recommendation.md",
    "pr195_missing_inputs.csv",
    "pr195_report.md",
    "artifact_manifest.csv",
]


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, object]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = fields or sorted({field for row in rows for field in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def _make_pr193(pr193_dir: Path) -> None:
    view_groups = {
        "train_004": "direct_corrupted",
        "train_014": "co_visible_collateral",
        "train_013": "clean_prior_demoted",
        "train_002": "other_clean",
    }
    _write_csv(
        pr193_dir / "pr193_view_group_map.csv",
        [
            {
                "scene": "chair",
                "condition": "corrupt_occluder",
                "subset_name": "seed_20260710",
                "view_name": view,
                "view_group": group,
                "was_corrupted": str(group == "direct_corrupted").lower(),
            }
            for view, group in view_groups.items()
        ],
    )
    _write_json(
        pr193_dir / "pr193_view_group_binding_summary.json",
        {
            "schema_name": "viewtrust.pr193.exact_view_group_binding.summary",
            "input_exact_evidence_quality": "exact",
            "output_exact_evidence_quality": "exact",
            "observation_only": True,
            "training_intervention": False,
            "defense_enabled": False,
        },
    )
    lifecycle_rows = [
        {
            "scene": "chair",
            "condition": "corrupt_occluder",
            "subset_name": "seed_20260710",
            "event_type": "clone_birth",
            "view_name": "train_013",
            "view_group": "clean_prior_demoted",
            "gaussian_id": "g_birth",
            "contribution_value": 1,
            "is_alive_final": "true",
        },
        {
            "scene": "chair",
            "condition": "corrupt_occluder",
            "subset_name": "seed_20260710",
            "event_type": "prune_death",
            "view_name": "train_013",
            "view_group": "clean_prior_demoted",
            "gaussian_id": "g_prune",
            "contribution_value": 1,
            "is_alive_final": "false",
        },
    ]
    attribution_rows = [
        {
            "scene": "chair",
            "condition": "corrupt_occluder",
            "subset_name": "seed_20260710",
            "event_type": "visibility_observation",
            "view_name": view,
            "view_group": group,
            "gaussian_id": gid,
            "contribution_value": weight,
        }
        for gid, view, group, weight in [
            ("g1", "train_004", "direct_corrupted", 10),
            ("g1", "train_014", "co_visible_collateral", 1),
            ("g2", "train_002", "other_clean", 20),
            ("g_birth", "train_013", "clean_prior_demoted", 4),
        ]
    ]
    support_rows = [
        {"gaussian_id": "g1", "is_alive_final": "true", "view_group": "direct_corrupted"},
        {"gaussian_id": "g2", "is_alive_final": "true", "view_group": "other_clean"},
        {"gaussian_id": "g_birth", "is_alive_final": "true", "view_group": "clean_prior_demoted"},
        {"gaussian_id": "g_prune", "is_alive_final": "false", "view_group": "clean_prior_demoted"},
    ]
    _write_csv(pr193_dir / "gaussian_lifecycle_events_grouped.csv", lifecycle_rows)
    _write_csv(pr193_dir / "view_gaussian_event_attribution_grouped.csv", attribution_rows)
    _write_csv(pr193_dir / "gaussian_support_summary_grouped.csv", support_rows)
    _write_csv(pr193_dir / "pr193_direct_collateral_exact_overlap.csv", [{"exact_overlap_jaccard": 1.0}])
    _write_csv(pr193_dir / "pr193_train013_exact_control.csv", [{"train013_exact_control_supported": "false"}])


def _make_pr194(pr194_dir: Path) -> None:
    _write_json(
        pr194_dir / "pr194_exact_support_filter_summary.json",
        {
            "schema_name": "viewtrust.pr194.exact_support_filter.summary",
            "broad_overlap_degeneracy_detected": True,
            "nontrivial_modes_with_direct_collateral_overlap": [],
            "modes_with_train013_control_supported": ["birth", "prune", "low_entropy"],
            "diagnostic_modes_with_train013_control": ["birth", "prune", "low_entropy"],
            "recommended_pr19_exact_mode": "birth",
            "high_event_threshold": 8,
            "observation_only": True,
            "training_intervention": False,
            "defense_enabled": False,
        },
    )
    comparison_rows = [
        {"support_mode": "broad", "filtered_gaussian_count": 5, "direct_supported_gaussian_count": 5, "collateral_supported_gaussian_count": 5, "clean_prior_supported_gaussian_count": 5, "other_clean_supported_gaussian_count": 5, "direct_collateral_overlap_count": 5, "direct_collateral_jaccard": 1.0, "train013_supported_gaussian_count": 5, "train013_control_supported": "false", "nontrivial_overlap_supported": "false", "broad_overlap_degeneracy_flag": "true"},
        {"support_mode": "suspicious_alive", "filtered_gaussian_count": 5, "direct_supported_gaussian_count": 5, "collateral_supported_gaussian_count": 5, "clean_prior_supported_gaussian_count": 5, "other_clean_supported_gaussian_count": 5, "direct_collateral_overlap_count": 5, "direct_collateral_jaccard": 1.0, "train013_supported_gaussian_count": 5, "train013_control_supported": "false", "nontrivial_overlap_supported": "false", "broad_overlap_degeneracy_flag": "true"},
        {"support_mode": "high_event", "filtered_gaussian_count": 3, "direct_supported_gaussian_count": 3, "collateral_supported_gaussian_count": 0, "clean_prior_supported_gaussian_count": 0, "other_clean_supported_gaussian_count": 0, "direct_collateral_overlap_count": 0, "direct_collateral_jaccard": 0.0, "train013_supported_gaussian_count": 0, "train013_control_supported": "false", "nontrivial_overlap_supported": "false", "broad_overlap_degeneracy_flag": "false"},
        {"support_mode": "dominant_source", "filtered_gaussian_count": 5, "direct_supported_gaussian_count": 0, "collateral_supported_gaussian_count": 0, "clean_prior_supported_gaussian_count": 0, "other_clean_supported_gaussian_count": 5, "direct_collateral_overlap_count": 0, "direct_collateral_jaccard": 0.0, "train013_supported_gaussian_count": 0, "train013_control_supported": "false", "nontrivial_overlap_supported": "false", "broad_overlap_degeneracy_flag": "false"},
        {"support_mode": "birth", "filtered_gaussian_count": 1, "direct_supported_gaussian_count": 0, "collateral_supported_gaussian_count": 0, "clean_prior_supported_gaussian_count": 1, "other_clean_supported_gaussian_count": 0, "direct_collateral_overlap_count": 0, "direct_collateral_jaccard": 0.0, "train013_supported_gaussian_count": 1, "train013_control_supported": "true", "nontrivial_overlap_supported": "false", "broad_overlap_degeneracy_flag": "false"},
        {"support_mode": "prune", "filtered_gaussian_count": 1, "direct_supported_gaussian_count": 0, "collateral_supported_gaussian_count": 0, "clean_prior_supported_gaussian_count": 1, "other_clean_supported_gaussian_count": 0, "direct_collateral_overlap_count": 0, "direct_collateral_jaccard": 0.0, "train013_supported_gaussian_count": 1, "train013_control_supported": "true", "nontrivial_overlap_supported": "false", "broad_overlap_degeneracy_flag": "false"},
        {"support_mode": "low_entropy", "filtered_gaussian_count": 1, "direct_supported_gaussian_count": 0, "collateral_supported_gaussian_count": 0, "clean_prior_supported_gaussian_count": 1, "other_clean_supported_gaussian_count": 0, "direct_collateral_overlap_count": 0, "direct_collateral_jaccard": 0.0, "train013_supported_gaussian_count": 1, "train013_control_supported": "true", "nontrivial_overlap_supported": "false", "broad_overlap_degeneracy_flag": "false"},
    ]
    _write_csv(pr194_dir / "pr194_support_mode_comparison.csv", comparison_rows)
    _write_csv(pr194_dir / "pr194_direct_collateral_overlap_by_mode.csv", comparison_rows)
    _write_csv(
        pr194_dir / "pr194_train013_control_by_mode.csv",
        [
            {
                "support_mode": row["support_mode"],
                "train013_view_group": "clean_prior_demoted",
                "train013_supported_gaussian_count": row["train013_supported_gaussian_count"],
                "train013_direct_collateral_overlap_count": 0,
                "train013_control_supported": row["train013_control_supported"],
                "reason": "mock",
            }
            for row in comparison_rows
        ],
    )
    _write_csv(
        pr194_dir / "pr194_filtered_gaussian_support_by_mode.csv",
        [
            {"support_mode": "high_event", "gaussian_id": "g1", "included_by_filter": "true", "direct_corrupted_support": "true", "collateral_support": "false", "clean_prior_support": "false", "other_clean_support": "false", "dominant_view_group": "direct_corrupted", "dominant_view_name": "train_004"},
            {"support_mode": "dominant_source", "gaussian_id": "g2", "included_by_filter": "true", "direct_corrupted_support": "false", "collateral_support": "false", "clean_prior_support": "false", "other_clean_support": "true", "dominant_view_group": "other_clean", "dominant_view_name": "train_002"},
        ],
    )
    _write_csv(
        pr194_dir / "pr194_view_group_event_concentration.csv",
        [{"gaussian_id": "g1", "direct_corrupted_event_count": 10, "collateral_event_count": 1, "other_clean_event_count": 0}],
    )
    _write_csv(
        pr194_dir / "pr194_nontrivial_overlap_candidates.csv",
        [],
        fields=["scene", "condition", "subset_name", "support_mode", "gaussian_id"],
    )


def main() -> int:
    project_root = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory(prefix="viewtrust-pr195-") as tmp:
        root = Path(tmp)
        pr193_dir = root / "pr193"
        pr194_dir = root / "pr194"
        output_dir = root / "pr195"
        _make_pr193(pr193_dir)
        _make_pr194(pr194_dir)
        result = _run(
            [
                sys.executable,
                str(project_root / "scripts" / "measure" / "audit_pr195_exact_attribution_semantics.py"),
                "--pr193-dir",
                str(pr193_dir),
                "--pr194-dir",
                str(pr194_dir),
                "--scene",
                "chair",
                "--condition",
                "corrupt_occluder",
                "--subset-name",
                "seed_20260710",
                "--output-dir",
                str(output_dir),
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
        summary = json.loads((output_dir / "pr195_attribution_semantics_summary.json").read_text())
        assert summary["observation_only"] is True
        assert summary["training_intervention"] is False
        assert summary["defense_enabled"] is False
        assert summary["corrected_recommended_pr19_exact_mode"] == "none"
        assert summary["pr194_recommendation_is_valid"] is False
        assert summary["pr20_ready_for_intervention"] is False
        assert summary["pr20_requires_sparse_render_contribution"] is True
        failures = {row["support_mode"]: row["failure_type"] for row in _read_csv(output_dir / "pr195_support_mode_failure_analysis.csv")}
        assert failures["broad"] == "broad_degenerate"
        assert failures["suspicious_alive"] == "suspicious_alive_degenerate"
        assert failures["high_event"] == "no_collateral_support"
        assert failures["dominant_source"] == "other_clean_dominant"
        assert failures["birth"] == "train013_only_diagnostic"
    print("pr195 exact attribution semantics smoke test ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
