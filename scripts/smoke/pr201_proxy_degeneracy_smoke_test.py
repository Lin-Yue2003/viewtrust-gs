#!/usr/bin/env python3
"""LOCAL-SAFE smoke test for PR20.1 proxy degeneracy diagnosis."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path


REQUIRED_OUTPUTS = [
    "pr201_proxy_degeneracy_summary.json",
    "pr201_run_summary.csv",
    "pr201_pixel_candidate_reuse.csv",
    "pr201_view_candidate_pool.csv",
    "pr201_view_candidate_pool_overlap.csv",
    "pr201_candidate_weight_uniformity.csv",
    "pr201_group_candidate_pool_audit.csv",
    "pr201_direct_collateral_degeneracy.csv",
    "pr201_train013_proxy_control_audit.csv",
    "pr201_proxy_failure_cases.csv",
    "pr201_recommendations.json",
    "pr201_missing_inputs.csv",
    "pr201_report.md",
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


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def _make_pr200(pr200_dir: Path) -> None:
    _write_json(
        pr200_dir / "pr200_sparse_render_attribution_summary.json",
        {
            "schema_name": "viewtrust.pr200.sparse_render_attribution.summary",
            "scene": "chair",
            "condition": "corrupt_occluder",
            "subset_name": "seed_20260710",
            "selected_view_count": 4,
            "selected_pixel_count": 8,
            "total_pixel_gaussian_contribution_rows": 32,
            "evidence_quality": "approximate_projected_gaussian",
            "attribution_method": "view_event_weighted_gaussian_proxy",
            "direct_collateral_residual_overlap_supported": False,
            "train013_residual_control_supported": True,
            "pr20_ready_for_intervention": False,
            "warnings": [],
        },
    )
    selected = [
        ("train_004", "direct_corrupted", "true"),
        ("train_009", "direct_corrupted", "true"),
        ("train_014", "co_visible_collateral", "false"),
        ("train_013", "clean_prior_demoted", "false"),
    ]
    _write_csv(
        pr200_dir / "pr200_selected_views.csv",
        [
            {
                "scene": "chair",
                "condition": "corrupt_occluder",
                "subset_name": "seed_20260710",
                "view_name": view,
                "view_group": group,
                "was_corrupted": corrupted,
                "included": "true",
            }
            for view, group, corrupted in selected
        ],
    )
    _write_csv(
        pr200_dir / "pr200_view_residual_summary.csv",
        [
            {
                "scene": "chair",
                "condition": "corrupt_occluder",
                "subset_name": "seed_20260710",
                "view_name": view,
                "view_group": group,
                "selected_pixel_count": 2,
                "evidence_quality": "approximate_projected_gaussian",
            }
            for view, group, _ in selected
        ],
    )
    rows = []
    for view, group, _ in selected:
        pool = ["100", "101", "102", "103"] if view == "train_013" else ["0", "1", "2", "3"]
        for pixel_index in range(2):
            for rank, gid in enumerate(pool, start=1):
                rows.append(
                    {
                        "scene": "chair",
                        "condition": "corrupt_occluder",
                        "subset_name": "seed_20260710",
                        "view_name": view,
                        "view_group": group,
                        "pixel_x": pixel_index,
                        "pixel_y": pixel_index + 1,
                        "gaussian_id": gid,
                        "contribution_rank": rank,
                        "splat_weight": 0.25,
                        "alpha_contribution": 0.25,
                        "residual_l1": 0.5,
                        "residual_weighted_splat": 0.125,
                        "evidence_quality": "approximate_projected_gaussian",
                        "attribution_method": "view_event_weighted_gaussian_proxy",
                    }
                )
    _write_csv(pr200_dir / "pr200_pixel_gaussian_contributions.csv", rows)
    _write_csv(
        pr200_dir / "pr200_gaussian_residual_attribution.csv",
        [
            {"gaussian_id": gid, "direct_corrupted_residual_weight": 1, "collateral_residual_weight": 1, "clean_prior_residual_weight": 0}
            for gid in ["0", "1", "2", "3"]
        ]
        + [
            {"gaussian_id": gid, "direct_corrupted_residual_weight": 0, "collateral_residual_weight": 0, "clean_prior_residual_weight": 1}
            for gid in ["100", "101", "102", "103"]
        ],
    )
    _write_csv(
        pr200_dir / "pr200_view_group_residual_attribution.csv",
        [
            {"view_group": "direct_corrupted", "total_residual_weight": 1.0},
            {"view_group": "co_visible_collateral", "total_residual_weight": 1.0},
            {"view_group": "clean_prior_demoted", "total_residual_weight": 1.0},
        ],
    )
    _write_csv(
        pr200_dir / "pr200_direct_collateral_residual_overlap.csv",
        [
            {
                "residual_overlap_gaussian_count": 4,
                "residual_overlap_jaccard": 1.0,
                "nontrivial_residual_overlap_supported": "false",
                "evidence_quality": "approximate_projected_gaussian",
            }
        ],
    )
    _write_csv(
        pr200_dir / "pr200_train013_residual_control.csv",
        [
            {
                "train013_present": "true",
                "train013_view_group": "clean_prior_demoted",
                "train013_overlap_ratio": 0.0,
                "train013_residual_control_supported": "true",
                "evidence_quality": "approximate_projected_gaussian",
            }
        ],
    )
    _write_csv(
        pr200_dir / "pr200_attribution_quality_audit.csv",
        [{"criterion": "exact splat contribution available", "passed": "false"}],
    )


def main() -> int:
    project_root = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory(prefix="viewtrust-pr201-") as tmp:
        root = Path(tmp)
        pr200_dir = root / "pr200"
        output_dir = root / "pr201"
        _make_pr200(pr200_dir)
        result = _run(
            [
                sys.executable,
                str(project_root / "scripts" / "measure" / "analyze_pr201_proxy_degeneracy.py"),
                "--pr200-dir",
                str(pr200_dir),
                "--scene",
                "chair",
                "--condition",
                "corrupt_occluder",
                "--subset-name",
                "seed_20260710",
                "--output-dir",
                str(output_dir),
                "--top-k",
                "4",
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
        summary = json.loads((output_dir / "pr201_proxy_degeneracy_summary.json").read_text())
        assert summary["schema_name"] == "viewtrust.pr201.proxy_degeneracy.summary"
        assert summary["observation_only"] is True
        assert summary["training_intervention"] is False
        assert summary["defense_enabled"] is False
        assert summary["densification_gating_enabled"] is False
        assert summary["pr20_ready_for_intervention"] is False
        assert summary["exact_render_contribution_available"] is False
        assert summary["proxy_degeneracy_confirmed"] is True
        assert summary["pixel_candidate_reuse_degeneracy_confirmed"] is True
        assert summary["candidate_weight_uniformity_confirmed"] is True
        assert summary["direct_collateral_overlap_degenerate"] is True
        assert summary["train013_control_is_proxy_pool_separation"] is True
    print("pr201 proxy degeneracy smoke test ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
