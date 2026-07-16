#!/usr/bin/env python3
"""LOCAL-SAFE smoke test for PR21.1f drums selected-pixel alignment audit."""

from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_png(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (16, 16), color=(value, value, value))
    image.save(path)


def _make_inputs(root: Path) -> tuple[Path, Path, Path]:
    run_dir = root / "official_run"
    pr200 = root / "pr200"
    pr211 = root / "pr211"
    run_dir.mkdir(parents=True)
    proxy_rows = [
        {"view_name": "train_004", "pixel_x": 10, "pixel_y": 20, "gaussian_id": 1, "splat_weight": 0.5},
        {"view_name": "train_004", "pixel_x": 10, "pixel_y": 20, "gaussian_id": 2, "splat_weight": 0.5},
        {"view_name": "train_004", "pixel_x": 11, "pixel_y": 21, "gaussian_id": 3, "splat_weight": 1.0},
        {"view_name": "train_009", "pixel_x": 30, "pixel_y": 40, "gaussian_id": 4, "splat_weight": 1.0},
        {"view_name": "train_017", "pixel_x": 50, "pixel_y": 60, "gaussian_id": 5, "splat_weight": 1.0},
    ]
    _write_csv(pr200 / "pr200_pixel_gaussian_contributions.csv", proxy_rows)
    _write_png(pr200 / "renders" / "train_004_render.png", 16)
    _write_png(pr200 / "gt" / "train_004_gt.png", 48)
    (pr200 / "residuals").mkdir(parents=True, exist_ok=True)
    np.save(pr200 / "residuals" / "train_004_residual_l1.npy", np.ones((16, 16), dtype=np.float32))
    _write_png(pr211 / "renders" / "train_009_render.png", 64)
    _write_json(pr200 / "selected_pixels" / "train_004_selected_pixels.json", {"view_name": "train_004"})
    _write_json(
        pr211 / "pr211_exact_sparse_attribution_summary.json",
        {
            "scene": "drums",
            "per_view_replay_enabled": True,
            "multi_view_image_id_mapping_used": False,
            "pure_torch_accumulate_succeeded": True,
            "per_view_total_contributor_rows_before_filter": 42,
            "selected_pixel_hit_count": 0,
            "exact_contributor_id_row_count": 0,
        },
    )
    _write_csv(
        pr211 / "pr211_per_view_replay_audit.csv",
        [
            {
                "view_name": "train_004",
                "view_group": "direct_corrupted",
                "attempted": "true",
                "succeeded": "true",
                "raw_contributor_rows_before_filter": 20,
                "unique_raw_pixel_count": 15,
                "selected_pixel_hit_count_normal": 0,
                "y_flip_hit_count": 2,
                "x_flip_hit_count": 0,
                "xy_swap_hit_count": 0,
                "xy_swap_y_flip_hit_count": 0,
                "xy_swap_x_flip_hit_count": 0,
                "neighborhood_r1_hit_count": 0,
                "neighborhood_r2_hit_count": 0,
                "neighborhood_r4_hit_count": 1,
                "neighborhood_r8_hit_count": 2,
                "image_ids_seen": "0",
                "unexpected_image_id_count": 0,
                "accumulation_source_selected": "pure_torch_alpha",
            },
            {
                "view_name": "train_009",
                "view_group": "direct_corrupted",
                "attempted": "true",
                "succeeded": "true",
                "raw_contributor_rows_before_filter": 12,
                "unique_raw_pixel_count": 8,
                "selected_pixel_hit_count_normal": 0,
                "y_flip_hit_count": 0,
                "x_flip_hit_count": 1,
                "xy_swap_hit_count": 0,
                "xy_swap_y_flip_hit_count": 0,
                "xy_swap_x_flip_hit_count": 0,
                "neighborhood_r1_hit_count": 0,
                "neighborhood_r2_hit_count": 0,
                "neighborhood_r4_hit_count": 0,
                "neighborhood_r8_hit_count": 1,
                "image_ids_seen": "0",
                "unexpected_image_id_count": 0,
                "accumulation_source_selected": "pure_torch_alpha",
            },
            {
                "view_name": "train_012",
                "view_group": "direct_corrupted",
                "attempted": "true",
                "succeeded": "true",
                "raw_contributor_rows_before_filter": 0,
                "unique_raw_pixel_count": 0,
                "selected_pixel_hit_count_normal": 0,
                "y_flip_hit_count": 0,
                "x_flip_hit_count": 0,
                "xy_swap_hit_count": 0,
                "xy_swap_y_flip_hit_count": 0,
                "xy_swap_x_flip_hit_count": 0,
                "neighborhood_r1_hit_count": 0,
                "neighborhood_r2_hit_count": 0,
                "neighborhood_r4_hit_count": 0,
                "neighborhood_r8_hit_count": 0,
                "image_ids_seen": "",
                "unexpected_image_id_count": 0,
                "accumulation_source_selected": "",
                "error": "no raw contributors in fake view",
            },
            {
                "view_name": "train_017",
                "view_group": "direct_corrupted",
                "attempted": "true",
                "succeeded": "true",
                "raw_contributor_rows_before_filter": 0,
                "unique_raw_pixel_count": 0,
                "selected_pixel_hit_count_normal": 0,
                "y_flip_hit_count": 0,
                "x_flip_hit_count": 0,
                "xy_swap_hit_count": 0,
                "xy_swap_y_flip_hit_count": 0,
                "xy_swap_x_flip_hit_count": 0,
                "neighborhood_r1_hit_count": 0,
                "neighborhood_r2_hit_count": 0,
                "neighborhood_r4_hit_count": 0,
                "neighborhood_r8_hit_count": 0,
                "image_ids_seen": "0",
                "unexpected_image_id_count": 0,
                "accumulation_source_selected": "pure_torch_alpha",
            },
        ],
    )
    return run_dir, pr200, pr211


def main() -> int:
    project_root = Path(__file__).resolve().parents[2]
    import sys

    sys.path.insert(0, str(project_root))
    from viewtrust.analysis.pr211f_alignment import build_pr211f_drums_alignment_audit

    with tempfile.TemporaryDirectory(prefix="viewtrust-pr211f-") as tmp:
        root = Path(tmp)
        run_dir, pr200, pr211 = _make_inputs(root)
        output = root / "out"
        summary, code = build_pr211f_drums_alignment_audit(
            run_dir=run_dir,
            pr200_dir=pr200,
            pr211_dir=pr211,
            output_dir=output,
            scene="drums",
            condition="corrupt_occluder",
            subset_name="seed_20260710",
            write_markdown=True,
        )
        assert code == 0
        assert summary["schema_name"] == "viewtrust.pr211f.drums_selected_pixel_alignment.summary"
        assert summary["observation_only"] is True
        assert summary["training_intervention"] is False
        assert summary["defense_enabled"] is False
        assert summary["view_rejection_enabled"] is False
        assert summary["densification_gating_enabled"] is False
        assert summary["third_party_modified"] is False
        assert summary["exact_evidence_allowed_for_drums"] is False
        assert summary["drums_ready_for_pr212"] is False
        assert summary["normal_exact_hit_total"] == 0
        assert summary["best_diagnostic_hit_total"] > 0
        assert summary["raw_contributor_total"] > 0
        assert summary["likely_failure_mode_overall"] == "mixed_coordinate_candidate_and_no_raw_contributors"
        assert summary["likely_failure_mode_overall"] != "exact_replay_has_no_raw_contributors"
        assert summary["view_count_with_raw_contributors"] == 2
        assert summary["view_count_without_raw_contributors"] == 4
        assert summary["view_count_with_normal_hits"] == 0
        assert summary["view_count_with_diagnostic_hits"] == 2
        assert summary["views_with_raw_contributors"] == "train_004;train_009"
        assert summary["source_search_paths_written"] is True
        assert summary["source_inventory_written"] is True
        assert summary["source_candidate_file_count"] > 0
        assert summary["source_inventory_file_count"] > 0

        expected = [
            "pr211f_drums_selected_pixel_alignment_summary.json",
            "pr211f_drums_pr20_selected_pixel_audit.csv",
            "pr211f_drums_exact_replay_raw_pixel_coverage.csv",
            "pr211f_drums_coordinate_convention_audit.csv",
            "pr211f_drums_residual_source_alignment_audit.csv",
            "pr211f_drums_top_residual_crosscheck.csv",
            "pr211f_drums_alignment_diagnosis.csv",
            "pr211f_drums_source_search_paths.csv",
            "pr211f_drums_source_file_inventory.csv",
            "pr211f_drums_selected_pixel_alignment_report.md",
            "artifact_manifest.csv",
        ]
        for name in expected:
            assert (output / name).exists(), name

        selected_rows = _read_csv(output / "pr211f_drums_pr20_selected_pixel_audit.csv")
        train004 = next(row for row in selected_rows if row["view_name"] == "train_004")
        assert train004["proxy_row_count"] == "3"
        assert train004["unique_selected_pixel_count"] == "2"
        assert float(train004["duplicate_proxy_rows_per_pixel"]) > 1.0

        coord_rows = _read_csv(output / "pr211f_drums_coordinate_convention_audit.csv")
        normal004 = next(row for row in coord_rows if row["view_name"] == "train_004" and row["convention"] == "normal")
        yflip004 = next(row for row in coord_rows if row["view_name"] == "train_004" and row["convention"] == "y_flip")
        assert normal004["hit_count"] == "0"
        assert yflip004["hit_count"] == "2"
        assert yflip004["diagnostic_only"] == "true"
        assert yflip004["can_be_exact_evidence"] == "false"
        assert yflip004["interpretation"] == "diagnostic_only_not_exact_evidence"

        residual_rows = _read_csv(output / "pr211f_drums_top_residual_crosscheck.csv")
        assert any(row["reconstructed_residual_available"] == "true" for row in residual_rows)
        assert any(row["interpretation"] == "residual_source_unavailable" for row in residual_rows)
        assert any(row["interpretation"] == "source_candidates_found_but_residual_not_reconstructable" for row in residual_rows)

        source_paths = _read_csv(output / "pr211f_drums_source_search_paths.csv")
        assert source_paths
        assert any(row["source_root_name"] == "pr20" and row["exists"] == "true" for row in source_paths)
        inventory = _read_csv(output / "pr211f_drums_source_file_inventory.csv")
        categories = {row["matched_category"] for row in inventory}
        assert "render_candidate" in categories
        assert "gt_candidate" in categories
        assert "residual_candidate" in categories
        assert "selected_pixel_candidate" in categories

        residual_source = _read_csv(output / "pr211f_drums_residual_source_alignment_audit.csv")
        train004_source = next(row for row in residual_source if row["view_name"] == "train_004")
        assert train004_source["pr20_render_candidate_count"] != "0"
        assert train004_source["pr20_gt_candidate_count"] != "0"
        assert train004_source["pr20_residual_candidate_count"] != "0"

        diagnosis = _read_csv(output / "pr211f_drums_alignment_diagnosis.csv")
        modes = {row["likely_failure_mode"] for row in diagnosis}
        assert "selected_pixel_coordinate_convention_candidate" in modes
        assert "exact_replay_has_no_raw_contributors_for_view" in modes
        assert all(row["exact_evidence_allowed"] == "false" for row in diagnosis)

        manifest = _read_csv(output / "artifact_manifest.csv")
        assert any(row["relative_path"] == "pr211f_drums_selected_pixel_alignment_summary.json" and row["exists"] == "true" for row in manifest)
        assert any(row["relative_path"] == "pr211f_drums_source_search_paths.csv" and row["exists"] == "true" for row in manifest)
        assert any(row["relative_path"] == "pr211f_drums_source_file_inventory.csv" and row["exists"] == "true" for row in manifest)
        assert not (output / "pr211f_exact_pixel_gaussian_contributions.csv").exists()
        report = (output / "pr211f_drums_selected_pixel_alignment_report.md").read_text(encoding="utf-8")
        assert "Diagnostic flip/swap/neighborhood hits are not promoted to exact evidence." in report

    print("pr211f drums selected-pixel alignment smoke test ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
