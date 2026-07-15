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
    "pr211_gsplat_contributor_api_audit.csv",
    "pr211_transmittance_audit.csv",
    "pr211_gsplat_rasterization_output_audit.csv",
    "pr211_gsplat_source_audit.csv",
    "pr211_internal_loop_shape_audit.csv",
    "pr211_internal_loop_attempts.csv",
    "pr211_accumulation_audit.csv",
    "pr211_contributor_path_decision.json",
    "pr211_contributor_path_attempts.csv",
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


def _synthetic_id_only_rows() -> list[dict[str, object]]:
    rows = []
    for rank, gid in enumerate([100, 101], start=1):
        rows.append(
            {
                "scene": "chair",
                "condition": "corrupt_occluder",
                "subset_name": "seed_20260710",
                "view_name": "train_004",
                "view_group": "direct_corrupted",
                "pixel_x": 1,
                "pixel_y": 1,
                "pixel_id": 9,
                "gaussian_id": gid,
                "contributor_rank": rank,
                "depth_order": rank,
                "residual_l1": 0.5,
                "evidence_quality": "exact_sparse_contributor_id_only",
                "attribution_method": "gsplat_sparse_contributor_id_replay",
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
    from viewtrust.analysis.gsplat_sparse_attribution import (
        _call_rasterize_to_indices_in_range_safely,
        _extract_contributors_with_gsplat_api,
        recover_contributors_by_gsplat_internal_loop,
        build_pr211_exact_sparse_attribution,
    )

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
        assert summary["exact_contributor_ids_available"] is True
        assert summary["exact_contribution_row_count"] > 0
        assert summary["transmittance_resolution_succeeded"] is True
        assert summary["contributor_api_dry_run_succeeded"] is True
        assert summary["rasterize_to_indices_call_succeeded"] is True
        assert summary["selected_pixel_hit_count"] > 0
        assert summary["source_audit_completed"] is True
        assert summary["contributor_path_selected"] == "synthetic_exact_rows"
        assert summary["exact_render_contribution_succeeded"] is True
        direct = list(csv.DictReader((success_dir / "pr211_direct_collateral_exact_overlap.csv").open()))
        assert direct[0]["overlap_count"] == "1"
        train = list(csv.DictReader((success_dir / "pr211_train013_exact_control.csv").open()))
        assert train[0]["train013_exact_control_supported"] == "true"
        comparison = list(csv.DictReader((success_dir / "pr211_exact_vs_proxy_comparison.csv").open()))
        assert comparison
        manifest = list(csv.DictReader((success_dir / "artifact_manifest.csv").open()))
        assert any(row["relative_path"] == "pr211_exact_sparse_attribution_summary.json" for row in manifest)
        decision = json.loads((success_dir / "pr211_contributor_path_decision.json").read_text(encoding="utf-8"))
        assert decision["modifies_gsplat_or_third_party"] is False
        assert list(csv.DictReader((success_dir / "pr211_gsplat_source_audit.csv").open()))
        assert list(csv.DictReader((success_dir / "pr211_contributor_path_attempts.csv").open()))

        id_only_dir = root / "id_only"
        id_summary, id_code = build_pr211_exact_sparse_attribution(
            run_dir=run_dir,
            pr200_dir=pr200_dir,
            pr210_dir=pr210_dir,
            scene="chair",
            condition="corrupt_occluder",
            subset_name="seed_20260710",
            iteration=700,
            split="train",
            views=["train_004", "train_014", "train_013"],
            output_dir=id_only_dir,
            device="cpu",
            top_pixels_per_view=2,
            max_contributors_per_pixel=2,
            synthetic_exact_rows=_synthetic_id_only_rows(),
        )
        assert id_code == 0
        _assert_outputs(id_only_dir)
        assert id_summary["exact_attribution_succeeded"] is True
        assert id_summary["evidence_quality"] == "exact_sparse_contributor_id_only"
        assert id_summary["exact_contributor_id_only_succeeded"] is True
        assert id_summary["exact_render_contribution_succeeded"] is False
        assert id_summary["exact_alpha_available"] is False
        assert id_summary["exact_transmittance_available"] is False
        assert id_summary["exact_splat_weight_available"] is False
        assert id_summary["ready_for_intervention"] is False

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
        assert failure_summary["transmittance_resolution_succeeded"] is False
        assert failure_summary["source_audit_completed"] is False
        assert failure_summary["contributor_path_selected"] == "source_level_failure"
        assert failure_summary["exact_contribution_row_count"] == 0
        assert failure_summary["ready_for_intervention"] is False
        exact_rows = list(csv.DictReader((failure_dir / "pr211_exact_pixel_gaussian_contributions.csv").open()))
        assert not exact_rows
        blockers = (failure_dir / "pr211_blockers.csv").read_text(encoding="utf-8")
        assert "gsplat_transmittance_resolution" in blockers
        assert "exact contributor IDs unavailable" in blockers
        comparison = list(csv.DictReader((failure_dir / "pr211_exact_vs_proxy_comparison.csv").open()))
        assert comparison
        assert all(row["interpretation"] == "exact unavailable due to failed sparse replay" for row in comparison)
        trans_rows = list(csv.DictReader((failure_dir / "pr211_transmittance_audit.csv").open()))
        assert trans_rows
        assert trans_rows[0]["selected_as_transmittance"] == "false"
        assert "view_event_weighted_gaussian_proxy" not in (failure_dir / "pr211_exact_pixel_gaussian_contributions.csv").read_text(encoding="utf-8")

        recorded: dict[str, object] = {}

        def fake_rasterize_to_indices_in_range(**kwargs: object) -> tuple[list[int], list[int], list[int]]:
            recorded.update(kwargs)
            if "transmittances" not in kwargs or kwargs["transmittances"] is None:
                raise AssertionError("transmittances missing")
            return [1], [2], [0]

        _call_rasterize_to_indices_in_range_safely(
            fake_rasterize_to_indices_in_range,
            range_start=0,
            range_end=1,
            transmittances=[0.5],
            means2d=[0.0],
            conics=[0.0],
            opacities=[0.9],
            image_width=8,
            image_height=8,
            tile_size=16,
            isect_offsets=[0],
            flatten_ids=[1],
        )
        assert "transmittances" in recorded

        def fake_compact_api(**kwargs: object) -> tuple[list[int], list[int], list[int]]:
            assert "transmittances" in kwargs
            return [0], [9], [0]

        compact_rows = _extract_contributors_with_gsplat_api(
            api=fake_compact_api,
            meta={
                "gaussian_ids": [777],
                "means2d": [0.0],
                "conics": [0.0],
                "opacities": [0.9],
                "isect_offsets": [0],
                "flatten_ids": [0],
                "tile_size": 16,
            },
            transmittances=[0.5],
            image_width=8,
            image_height=8,
            selected_views=[{"requested_view_name": "train_004"}],
            selected_pixels=[
                {
                    "scene": "chair",
                    "condition": "corrupt_occluder",
                    "subset_name": "seed_20260710",
                    "view_name": "train_004",
                    "view_group": "direct_corrupted",
                    "pixel_x": 1,
                    "pixel_y": 1,
                    "pixel_id": 9,
                    "residual_l1": 0.5,
                }
            ],
            max_contributors_per_pixel=1,
            torch=None,
        )
        assert compact_rows
        assert compact_rows[0]["gaussian_id"] == 777
        assert compact_rows[0]["evidence_quality"] == "exact_sparse_contributor_id_only"
        assert compact_rows[0]["_compact_gaussian_id_mapping_used"] is True

        import numpy as np

        class NumpyTorch:
            @staticmethod
            def zeros(shape: tuple[int, ...], device: object = None) -> np.ndarray:
                del device
                return np.zeros(shape, dtype=np.float32)

            @staticmethod
            def tensor(values: list[int], device: object = None, dtype: object = None) -> np.ndarray:
                del device
                return np.array(values, dtype=dtype)

            @staticmethod
            def cat(values: list[np.ndarray]) -> np.ndarray:
                return np.concatenate(values)

            @staticmethod
            def exp(value: object) -> object:
                return np.exp(value)

        internal_meta = {
            "means2d": np.zeros((2, 3, 2), dtype=np.float32),
            "conics": np.zeros((2, 3, 3), dtype=np.float32),
            "opacities": np.ones((2, 3), dtype=np.float32),
            "isect_offsets": np.array([[[0]], [[2]]], dtype=np.int64),
            "flatten_ids": np.array([0, 1, 2, 3], dtype=np.int64),
            "tile_size": 1,
        }
        colors = np.ones((2, 3, 3), dtype=np.float32)
        source_rows = [
            {
                "item": "pattern:alpha",
                "path": "/fake/gsplat/cuda/_torch_impl.py",
                "line_number": 123,
                "symbol": "",
                "signature": "",
                "snippet": "alpha = torch.clamp(opacity * torch.exp(-sigma), max=0.999)",
                "notes": "source grep",
            },
            {
                "item": "pattern:conic",
                "path": "/fake/gsplat/cuda/_torch_impl.py",
                "line_number": 122,
                "symbol": "",
                "signature": "",
                "snippet": "sigma = 0.5 * (conic[0] * delta_x * delta_x + conic[2] * delta_y * delta_y) + conic[1] * delta_x * delta_y",
                "notes": "source grep",
            },
        ]
        loop_calls: list[tuple[int, tuple[int, ...], float]] = []
        accumulate_calls = {"count": 0}

        def fake_internal_rasterize(**kwargs: object) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            transmittances = kwargs["transmittances"]
            assert getattr(transmittances, "shape", None) == (2, 4, 4)
            step = int(kwargs["range_start"])
            loop_calls.append((step, tuple(transmittances.shape), float(transmittances[0, 1, 1])))
            if step > 0:
                return np.array([], dtype=np.int64), np.array([], dtype=np.int64), np.array([], dtype=np.int64)
            return np.array([1, 2], dtype=np.int64), np.array([5, 5], dtype=np.int64), np.array([0, 0], dtype=np.int64)

        def fake_accumulate(*args: object) -> tuple[np.ndarray, np.ndarray]:
            del args
            accumulate_calls["count"] += 1
            acc = np.zeros((2, 4, 4, 1), dtype=np.float32)
            acc[0, 1, 1, 0] = 0.25
            return np.zeros((2, 4, 4, 3), dtype=np.float32), acc

        internal_result = recover_contributors_by_gsplat_internal_loop(
            rasterize_api=fake_internal_rasterize,
            accumulate_api=fake_accumulate,
            meta=internal_meta,
            colors=colors,
            image_width=4,
            image_height=4,
            selected_views=[{"requested_view_name": "train_004"}, {"requested_view_name": "train_014"}],
            selected_pixels=[
                {
                    "scene": "chair",
                    "condition": "corrupt_occluder",
                    "subset_name": "seed_20260710",
                    "view_name": "train_004",
                    "view_group": "direct_corrupted",
                    "pixel_x": 1,
                    "pixel_y": 1,
                    "pixel_id": 5,
                    "residual_l1": 0.5,
                }
            ],
            max_contributors_per_pixel=2,
            torch=NumpyTorch,
            packed=False,
            source_rows=source_rows,
            batch_per_iter=1,
        )
        assert loop_calls
        assert len(loop_calls) >= 2
        assert loop_calls[0][2] == 1.0
        assert loop_calls[1][2] < 1.0
        assert accumulate_calls["count"] == 1
        assert internal_result["attempt_row"]["succeeded"] == "true"
        assert internal_result["status"]["internal_loop_shape_validation_succeeded"] is True
        assert internal_result["status"]["rasterize_to_indices_call_succeeded"] is True
        assert internal_result["status"]["accumulate_succeeded"] is True
        assert internal_result["status"]["total_contributor_rows_before_filter"] == 2
        assert internal_result["status"]["selected_pixel_hit_count"] == 1
        assert internal_result["status"]["gaussian_id_mapping_mode"] == "unpacked_direct_gaussian_index"
        assert len(internal_result["rows"]) == 2
        assert internal_result["rows"][0]["evidence_quality"] == "exact_sparse_contributor_id_only"
        assert internal_result["rows"][0]["attribution_method"] == "gsplat_internal_loop_contributor_id_replay"
        assert internal_result["rows"][0]["splat_weight"] == ""

        bad_shape_meta = dict(internal_meta)
        bad_shape_meta["conics"] = np.zeros((2, 2, 3), dtype=np.float32)
        bad_shape = recover_contributors_by_gsplat_internal_loop(
            rasterize_api=fake_internal_rasterize,
            accumulate_api=fake_accumulate,
            meta=bad_shape_meta,
            colors=colors,
            image_width=4,
            image_height=4,
            selected_views=[{"requested_view_name": "train_004"}],
            selected_pixels=[],
            max_contributors_per_pixel=2,
            torch=NumpyTorch,
            packed=False,
            source_rows=source_rows,
            batch_per_iter=1,
        )
        assert not bad_shape["rows"]
        assert bad_shape["attempt_row"]["error"] == "internal loop shape validation failed"

        def fake_zero_hit_rasterize(**kwargs: object) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            if int(kwargs["range_start"]) > 0:
                return np.array([], dtype=np.int64), np.array([], dtype=np.int64), np.array([], dtype=np.int64)
            return np.array([1], dtype=np.int64), np.array([6], dtype=np.int64), np.array([0], dtype=np.int64)

        zero_hit = recover_contributors_by_gsplat_internal_loop(
            rasterize_api=fake_zero_hit_rasterize,
            accumulate_api=fake_accumulate,
            meta=internal_meta,
            colors=colors,
            image_width=4,
            image_height=4,
            selected_views=[{"requested_view_name": "train_004"}],
            selected_pixels=[
                {
                    "scene": "chair",
                    "condition": "corrupt_occluder",
                    "subset_name": "seed_20260710",
                    "view_name": "train_004",
                    "view_group": "direct_corrupted",
                    "pixel_x": 1,
                    "pixel_y": 1,
                    "pixel_id": 5,
                    "residual_l1": 0.5,
                }
            ],
            max_contributors_per_pixel=2,
            torch=NumpyTorch,
            packed=False,
            source_rows=source_rows,
            batch_per_iter=1,
        )
        assert not zero_hit["rows"]
        assert zero_hit["attempt_row"]["total_contributor_rows_before_filter"] == 1
        assert zero_hit["attempt_row"]["error"] == "selected_pixel_filtering_failed"

        fallback_calls: list[tuple[int, float]] = []

        def fake_nerfacc_accumulate(*args: object) -> tuple[np.ndarray, np.ndarray]:
            del args
            raise RuntimeError("Error building extension 'nerfacc_cuda': Please install nerfacc")

        def fake_fallback_rasterize(**kwargs: object) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            step = int(kwargs["range_start"])
            transmittances = kwargs["transmittances"]
            fallback_calls.append((step, float(transmittances[0, 1, 1])))
            if step > 0:
                assert float(transmittances[0, 1, 1]) < 1.0
                return np.array([], dtype=np.int64), np.array([], dtype=np.int64), np.array([], dtype=np.int64)
            return np.array([1], dtype=np.int64), np.array([5], dtype=np.int64), np.array([0], dtype=np.int64)

        fallback_result = recover_contributors_by_gsplat_internal_loop(
            rasterize_api=fake_fallback_rasterize,
            accumulate_api=fake_nerfacc_accumulate,
            meta=internal_meta,
            colors=colors,
            image_width=4,
            image_height=4,
            selected_views=[{"requested_view_name": "train_004"}],
            selected_pixels=[
                {
                    "scene": "chair",
                    "condition": "corrupt_occluder",
                    "subset_name": "seed_20260710",
                    "view_name": "train_004",
                    "view_group": "direct_corrupted",
                    "pixel_x": 1,
                    "pixel_y": 1,
                    "pixel_id": 5,
                    "residual_l1": 0.5,
                }
            ],
            max_contributors_per_pixel=2,
            torch=NumpyTorch,
            packed=False,
            source_rows=source_rows,
            batch_per_iter=1,
        )
        assert fallback_result["attempt_row"]["succeeded"] == "true"
        assert fallback_result["status"]["gsplat_accumulate_attempted"] is True
        assert fallback_result["status"]["gsplat_accumulate_succeeded"] is False
        assert "nerfacc_cuda" in fallback_result["status"]["gsplat_accumulate_error"]
        assert fallback_result["status"]["pure_torch_accumulate_attempted"] is True
        assert fallback_result["status"]["pure_torch_accumulate_succeeded"] is True
        assert fallback_result["status"]["pure_torch_accumulate_source_verified"] is True
        assert fallback_result["status"]["accumulation_source_selected"] == "pure_torch_alpha_accumulate"
        assert fallback_result["rows"]
        assert fallback_result["rows"][0]["evidence_quality"] == "exact_sparse_contributor_id_only"
        assert fallback_result["rows"][0]["splat_weight"] == ""
        assert fallback_calls[0][1] == 1.0
        assert fallback_calls[1][1] < 1.0
        accumulation_audit = fallback_result["accumulation_rows"]
        assert any(row["accumulation_source"] == "gsplat_accumulate" and row["succeeded"] == "false" for row in accumulation_audit)
        assert any(row["accumulation_source"] == "pure_torch_alpha_accumulate" and row["succeeded"] == "true" for row in accumulation_audit)
    print("pr211 exact sparse attribution smoke test ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
