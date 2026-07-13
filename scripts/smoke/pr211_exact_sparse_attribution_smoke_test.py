#!/usr/bin/env python3
"""LOCAL-SAFE smoke test for PR21.1 exact sparse attribution aggregation."""

from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path


REQUIRED_OUTPUTS = [
    "pr211_exact_sparse_attribution_summary.json",
    "pr211_input_readiness_audit.csv",
    "pr211_checkpoint_activation_audit.csv",
    "pr211_selected_pixels.csv",
    "pr211_gsplat_metadata_audit.csv",
    "pr211_exact_pixel_gaussian_contributions.csv",
    "pr211_gaussian_residual_attribution_exact.csv",
    "pr211_view_group_residual_attribution_exact.csv",
    "pr211_direct_collateral_exact_overlap.csv",
    "pr211_train013_exact_control.csv",
    "pr211_exact_vs_proxy_comparison.csv",
    "pr211_weight_nonuniformity_audit.csv",
    "pr211_missing_fields.csv",
    "pr211_blockers.csv",
    "pr211_recommendations.json",
    "pr211_report.md",
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


def _write_fake_ply(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = """ply
format ascii 1.0
element vertex 4
property float x
property float y
property float z
property float f_dc_0
property float f_dc_1
property float f_dc_2
property float opacity
property float scale_0
property float scale_1
property float scale_2
property float rot_0
property float rot_1
property float rot_2
property float rot_3
end_header
"""
    rows = [
        "0 0 3 1 0 0 0 -2 -2 -2 1 0 0 0",
        "1 0 3 0 1 0 0 -2 -2 -2 1 0 0 0",
        "0 1 3 0 0 1 0 -2 -2 -2 1 0 0 0",
        "1 1 3 1 1 1 0 -2 -2 -2 1 0 0 0",
    ]
    path.write_text(header + "\n".join(rows) + "\n", encoding="utf-8")


def _make_inputs(root: Path) -> tuple[Path, Path, Path]:
    run_dir = root / "run"
    pr200_dir = root / "pr200"
    pr210_dir = root / "pr210"
    _write_fake_ply(run_dir / "trainer_output" / "point_cloud" / "iteration_700" / "point_cloud.ply")
    _write_json(
        run_dir / "trainer_output" / "cameras.json",
        [
            {
                "id": 4,
                "img_name": "train_004",
                "width": 8,
                "height": 8,
                "fx": 10.0,
                "fy": 10.0,
                "rotation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                "position": [0, 0, 0],
            },
            {
                "id": 14,
                "img_name": "train_014",
                "width": 8,
                "height": 8,
                "fx": 10.0,
                "fy": 10.0,
                "rotation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                "position": [0, 0, 0],
            },
            {
                "id": 13,
                "img_name": "train_013",
                "width": 8,
                "height": 8,
                "fx": 10.0,
                "fy": 10.0,
                "rotation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                "position": [0, 0, 0],
            },
        ],
    )
    render_root = run_dir / "view_evaluation" / "render_models" / "train_test_model" / "train" / "ours_700" / "renders"
    gt_root = run_dir / "view_evaluation" / "render_models" / "train_test_model" / "train" / "ours_700" / "gt"
    render_root.mkdir(parents=True)
    gt_root.mkdir(parents=True)
    for name in ["train_004.png", "train_014.png", "train_013.png"]:
        (render_root / name).write_bytes(b"fake")
        (gt_root / name).write_bytes(b"fake")

    _write_json(
        pr210_dir / "pr210_gsplat_feasibility_summary.json",
        {
            "schema_name": "viewtrust.pr210.gsplat_feasibility.summary",
            "pr21_ready_for_exact_attribution": True,
            "selected_view_matching_supported": True,
            "selected_view_blocker_count": 0,
        },
    )
    selected_rows = []
    for view in ["train_004", "train_014", "train_013"]:
        selected_rows.append(
            {
                "requested_view_name": view,
                "matched_camera_id": view.split("_")[1],
                "matched_camera_img_name": view,
                "strict_match": "true",
                "split_consistent": "true",
                "valid_for_exact_attribution": "true",
                "official_render_path": str(render_root / f"{view}.png"),
                "official_gt_path": str(gt_root / f"{view}.png"),
            }
        )
    _write_csv(pr210_dir / "pr210_selected_view_audit.csv", selected_rows)

    sparse_rows = []
    proxy_rows = []
    for view, group, pixel_x in [
        ("train_004", "direct_corrupted", 1),
        ("train_014", "co_visible_collateral", 2),
        ("train_013", "clean_prior_demoted", 3),
    ]:
        sparse_rows.append(
            {
                "scene": "chair",
                "condition": "corrupt_occluder",
                "subset_name": "seed_20260710",
                "view_name": view,
                "view_group": group,
                "pixel_x": pixel_x,
                "pixel_y": 1,
                "residual_l1": 0.5,
            }
        )
        for gid in ["1", "2"]:
            proxy_rows.append(
                {
                    "scene": "chair",
                    "condition": "corrupt_occluder",
                    "subset_name": "seed_20260710",
                    "view_name": view,
                    "view_group": group,
                    "pixel_x": pixel_x,
                    "pixel_y": 1,
                    "gaussian_id": gid,
                    "splat_weight": 0.5,
                    "residual_l1": 0.5,
                }
            )
    _write_csv(pr200_dir / "pr200_sparse_pixel_residuals.csv", sparse_rows)
    _write_csv(pr200_dir / "pr200_pixel_gaussian_contributions.csv", proxy_rows)
    return run_dir, pr200_dir, pr210_dir


def _synthetic_exact_rows() -> list[dict[str, object]]:
    rows = []
    specs = [
        ("train_004", "direct_corrupted", 1, [10, 20]),
        ("train_014", "co_visible_collateral", 2, [20, 30]),
        ("train_013", "clean_prior_demoted", 3, [40]),
    ]
    for view, group, x, gids in specs:
        for rank, gid in enumerate(gids, start=1):
            splat = 0.7 if rank == 1 else 0.3
            rows.append(
                {
                    "scene": "chair",
                    "condition": "corrupt_occluder",
                    "subset_name": "seed_20260710",
                    "view_name": view,
                    "view_group": group,
                    "pixel_x": x,
                    "pixel_y": 1,
                    "pixel_id": 8 + x,
                    "gaussian_id": gid,
                    "contributor_rank": rank,
                    "depth_order": rank,
                    "alpha_contribution": splat,
                    "transmittance_before": 1.0 if rank == 1 else 0.3,
                    "splat_weight": splat,
                    "residual_l1": 0.5,
                    "residual_weighted_splat": 0.5 * splat,
                    "evidence_quality": "exact_sparse_render_contribution",
                    "attribution_method": "gsplat_sparse_replay",
                }
            )
    return rows


def _assert_outputs(output_dir: Path) -> None:
    for name in REQUIRED_OUTPUTS:
        path = output_dir / name
        assert path.exists(), name
        assert path.stat().st_size > 0, name


def main() -> int:
    project_root = Path(__file__).resolve().parents[2]
    import sys

    sys.path.insert(0, str(project_root))
    from viewtrust.analysis.gsplat_sparse_attribution import build_pr211_exact_sparse_attribution

    with tempfile.TemporaryDirectory(prefix="viewtrust-pr211-") as tmp:
        root = Path(tmp)
        run_dir, pr200_dir, pr210_dir = _make_inputs(root)
        success_dir = root / "success"
        summary, code = build_pr211_exact_sparse_attribution(
            run_dir=run_dir,
            pr200_dir=pr200_dir,
            pr210_dir=pr210_dir,
            scene="chair",
            condition="corrupt_occluder",
            subset_name="seed_20260710",
            iteration=700,
            split="train",
            views=["train_004", "train_014", "train_013"],
            output_dir=success_dir,
            device="cpu",
            top_pixels_per_view=2,
            max_contributors_per_pixel=2,
            synthetic_exact_rows=_synthetic_exact_rows(),
        )
        assert code == 0
        _assert_outputs(success_dir)
        assert summary["schema_name"] == "viewtrust.pr211.exact_sparse_attribution.summary"
        assert summary["observation_only"] is True
        assert summary["training_intervention"] is False
        assert summary["defense_enabled"] is False
        assert summary["densification_gating_enabled"] is False
        assert summary["third_party_modified"] is False
        assert summary["ready_for_intervention"] is False
        assert summary["exact_attribution_succeeded"] is True
        assert summary["evidence_quality"] == "exact_sparse_render_contribution"
        direct = list(csv.DictReader((success_dir / "pr211_direct_collateral_exact_overlap.csv").open()))
        assert direct[0]["overlap_count"] == "1"
        train = list(csv.DictReader((success_dir / "pr211_train013_exact_control.csv").open()))
        assert train[0]["train013_exact_control_supported"] == "true"
        comparison = list(csv.DictReader((success_dir / "pr211_exact_vs_proxy_comparison.csv").open()))
        assert comparison
        manifest = list(csv.DictReader((success_dir / "artifact_manifest.csv").open()))
        assert any(row["relative_path"] == "pr211_exact_sparse_attribution_summary.json" for row in manifest)

        failure_dir = root / "failure"
        failure_summary, failure_code = build_pr211_exact_sparse_attribution(
            run_dir=run_dir,
            pr200_dir=pr200_dir,
            pr210_dir=pr210_dir,
            scene="chair",
            condition="corrupt_occluder",
            subset_name="seed_20260710",
            iteration=700,
            split="train",
            views=["train_004", "train_014", "train_013"],
            output_dir=failure_dir,
            device="cpu",
            top_pixels_per_view=2,
            max_contributors_per_pixel=2,
            force_failure=True,
        )
        assert failure_code == 0
        _assert_outputs(failure_dir)
        assert failure_summary["exact_attribution_succeeded"] is False
        assert failure_summary["ready_for_intervention"] is False
        exact_rows = list(csv.DictReader((failure_dir / "pr211_exact_pixel_gaussian_contributions.csv").open()))
        assert not exact_rows
        blockers = (failure_dir / "pr211_blockers.csv").read_text(encoding="utf-8")
        assert "exact contributor IDs unavailable" in blockers
    print("pr211 exact sparse attribution smoke test ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
