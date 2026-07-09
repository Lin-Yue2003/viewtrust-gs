#!/usr/bin/env python3
"""LOCAL-SAFE smoke test for PR20.0 sparse render attribution."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image


REQUIRED_OUTPUTS = [
    "pr200_sparse_render_attribution_summary.json",
    "pr200_selected_views.csv",
    "pr200_view_residual_summary.csv",
    "pr200_sparse_pixel_residuals.csv",
    "pr200_pixel_gaussian_contributions.csv",
    "pr200_gaussian_residual_attribution.csv",
    "pr200_view_group_residual_attribution.csv",
    "pr200_direct_collateral_residual_overlap.csv",
    "pr200_train013_residual_control.csv",
    "pr200_attribution_quality_audit.csv",
    "pr200_missing_inputs.csv",
    "pr200_report.md",
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


def _write_pair(root: Path, view_name: str, render_color: tuple[int, int, int], gt_color: tuple[int, int, int]) -> None:
    render = root / "renders" / f"{view_name}.png"
    gt = root / "gt" / f"{view_name}.png"
    render.parent.mkdir(parents=True, exist_ok=True)
    gt.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), render_color).save(render)
    Image.new("RGB", (8, 8), gt_color).save(gt)


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def _make_pr193(pr193_dir: Path) -> None:
    view_rows = [
        {"scene": "chair", "condition": "corrupt_occluder", "subset_name": "seed_20260710", "view_name": "train_004", "view_group": "direct_corrupted", "was_corrupted": "true"},
        {"scene": "chair", "condition": "corrupt_occluder", "subset_name": "seed_20260710", "view_name": "train_009", "view_group": "direct_corrupted", "was_corrupted": "true"},
        {"scene": "chair", "condition": "corrupt_occluder", "subset_name": "seed_20260710", "view_name": "train_014", "view_group": "co_visible_collateral", "was_corrupted": "false"},
        {"scene": "chair", "condition": "corrupt_occluder", "subset_name": "seed_20260710", "view_name": "train_013", "view_group": "clean_prior_demoted", "was_corrupted": "false"},
    ]
    _write_csv(pr193_dir / "pr193_view_group_map.csv", view_rows)
    _write_json(pr193_dir / "pr193_view_group_binding_summary.json", {"schema_name": "viewtrust.pr193.exact_view_group_binding.summary"})
    identity = [
        {"gaussian_id": f"g{i}", "root_gaussian_id": f"g{i}", "parent_gaussian_id": "", "is_alive_final": "true"}
        for i in range(1, 6)
    ]
    _write_csv(pr193_dir / "gaussian_identity_table_grouped.csv", identity)
    _write_csv(pr193_dir / "gaussian_support_summary_grouped.csv", identity)
    attribution = [
        {"view_name": "train_004", "view_group": "direct_corrupted", "gaussian_id": "g1", "contribution_value": 10},
        {"view_name": "train_004", "view_group": "direct_corrupted", "gaussian_id": "g2", "contribution_value": 8},
        {"view_name": "train_009", "view_group": "direct_corrupted", "gaussian_id": "g2", "contribution_value": 9},
        {"view_name": "train_009", "view_group": "direct_corrupted", "gaussian_id": "g5", "contribution_value": 4},
        {"view_name": "train_014", "view_group": "co_visible_collateral", "gaussian_id": "g1", "contribution_value": 7},
        {"view_name": "train_014", "view_group": "co_visible_collateral", "gaussian_id": "g3", "contribution_value": 6},
        {"view_name": "train_013", "view_group": "clean_prior_demoted", "gaussian_id": "g4", "contribution_value": 10},
    ]
    _write_csv(pr193_dir / "view_gaussian_event_attribution_grouped.csv", attribution)
    _write_csv(pr193_dir / "gaussian_lifecycle_events_grouped.csv", attribution)


def _make_pr195(pr195_dir: Path) -> None:
    _write_json(
        pr195_dir / "pr195_attribution_semantics_summary.json",
        {
            "schema_name": "viewtrust.pr195.exact_attribution_semantics.summary",
            "pr20_ready_for_intervention": False,
            "pr20_requires_sparse_render_contribution": True,
            "pr20_requires_residual_weighted_attribution": True,
        },
    )
    _write_csv(
        pr195_dir / "pr195_required_attribution_field_gap.csv",
        [{"field_name": "pixel_x", "currently_available": "false"}],
    )
    _write_csv(
        pr195_dir / "pr195_pr20_readiness_assessment.csv",
        [{"criterion": "safe to proceed to densification gating", "passed": "false"}],
    )


def _make_run(run_dir: Path) -> None:
    root = run_dir / "view_evaluation" / "render_models" / "train_test_model" / "train" / "ours_700"
    _write_pair(root, "train_004", (250, 10, 10), (10, 10, 10))
    _write_pair(root, "train_009", (220, 20, 20), (20, 20, 20))
    _write_pair(root, "train_014", (20, 240, 20), (20, 20, 20))
    _write_pair(root, "train_013", (30, 30, 230), (30, 30, 30))


def main() -> int:
    project_root = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory(prefix="viewtrust-pr200-") as tmp:
        root = Path(tmp)
        pr193_dir = root / "pr193"
        pr195_dir = root / "pr195"
        run_dir = root / "run"
        data_root = root / "data"
        output_dir = root / "pr200"
        _make_pr193(pr193_dir)
        _make_pr195(pr195_dir)
        _make_run(run_dir)
        result = _run(
            [
                sys.executable,
                str(project_root / "scripts" / "measure" / "build_pr200_sparse_render_attribution.py"),
                "--pr193-dir",
                str(pr193_dir),
                "--pr195-dir",
                str(pr195_dir),
                "--run-dir",
                str(run_dir),
                "--data-root",
                str(data_root),
                "--scene",
                "chair",
                "--condition",
                "corrupt_occluder",
                "--subset-name",
                "seed_20260710",
                "--output-dir",
                str(output_dir),
                "--top-pixels",
                "4",
                "--top-gaussians-per-pixel",
                "2",
                "--residual-metric",
                "l1",
                "--artifact-mask-mode",
                "top_residual",
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
        summary = json.loads((output_dir / "pr200_sparse_render_attribution_summary.json").read_text())
        assert summary["observation_only"] is True
        assert summary["training_intervention"] is False
        assert summary["defense_enabled"] is False
        assert summary["view_rejection_enabled"] is False
        assert summary["densification_gating_enabled"] is False
        assert summary["training_behavior_modified"] is False
        assert summary["rendering_behavior_modified_for_training"] is False
        assert summary["pr20_ready_for_intervention"] is False
        assert summary["evidence_quality"] != "exact_render_contribution"
        assert summary["selected_pixel_count"] > 0
        assert summary["total_pixel_gaussian_contribution_rows"] > 0
        overlap = _read_csv(output_dir / "pr200_direct_collateral_residual_overlap.csv")[0]
        assert int(overlap["residual_overlap_gaussian_count"]) >= 1
        train013 = _read_csv(output_dir / "pr200_train013_residual_control.csv")[0]
        assert train013["train013_present"] == "true"
        quality = _read_csv(output_dir / "pr200_attribution_quality_audit.csv")
        assert any(row["criterion"] == "exact splat contribution available" and row["passed"] == "false" for row in quality)
    print("pr200 sparse render attribution smoke test ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
