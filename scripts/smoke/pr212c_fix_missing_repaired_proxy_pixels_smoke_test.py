#!/usr/bin/env python3
"""LOCAL-SAFE smoke test for PR21.2c-fix missing repaired proxy diagnostics."""

from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, object]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = fields or sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _make_inputs(root: Path) -> tuple[Path, Path, Path, Path, Path, Path, Path]:
    pr200 = root / "pr200"
    pr211 = root / "pr211"
    pr212 = root / "pr212"
    pr212a = root / "pr212a"
    pr212b = root / "pr212b"
    pr212c = root / "pr212c"
    pr213 = root / "pr213"
    exact_rows = [
        {"view_name": "train_009", "view_group": "direct_corrupted", "pixel_x": 1, "pixel_y": 2, "gaussian_id": 10},
        {"view_name": "train_012", "view_group": "direct_corrupted", "pixel_x": 3, "pixel_y": 4, "gaussian_id": 11},
        {"view_name": "train_012", "view_group": "direct_corrupted", "pixel_x": 5, "pixel_y": 6, "gaussian_id": 12},
        {"view_name": "train_012", "view_group": "direct_corrupted", "pixel_x": 7, "pixel_y": 8, "gaussian_id": 13},
        {"view_name": "train_012", "view_group": "direct_corrupted", "pixel_x": 9, "pixel_y": 10, "gaussian_id": 14},
    ]
    pr20_rows = [
        {"view_name": "train_009", "view_group": "direct_corrupted", "pixel_x": 1, "pixel_y": 2, "gaussian_id": 100},
        {"view_name": "train_012", "view_group": "direct_corrupted", "pixel_x": 5, "pixel_y": 6, "gaussian_id": 101},
        {"view_name": "train_012", "view_group": "direct_corrupted", "pixel_x": 7, "pixel_y": 8, "gaussian_id": 102},
        {"view_name": "train_012", "view_group": "direct_corrupted", "pixel_x": "9.0", "pixel_y": "10.0", "gaussian_id": 103},
    ]
    repaired_rows = [
        {
            "scene": "chair",
            "condition": "corrupt_occluder",
            "subset_name": "seed_20260710",
            "view_name": "train_009",
            "view_group": "direct_corrupted",
            "pixel_x": 1,
            "pixel_y": 2,
            "original_gaussian_id": 100,
            "verified_final_gaussian_index": 1,
            "mapping_confidence": "high",
            "mapping_status": "verified_mapping_candidate",
            "repair_warning": "",
        },
        {
            "scene": "chair",
            "condition": "corrupt_occluder",
            "subset_name": "seed_20260710",
            "view_name": "train_012",
            "view_group": "direct_corrupted",
            "pixel_x": 5,
            "pixel_y": 6,
            "original_gaussian_id": 101,
            "verified_final_gaussian_index": 2,
            "mapping_confidence": "high",
            "mapping_status": "verified_mapping_candidate",
            "repair_warning": "",
        },
        {
            "scene": "chair",
            "condition": "corrupt_occluder",
            "subset_name": "seed_20260710",
            "view_name": "train_012",
            "view_group": "direct_corrupted",
            "pixel_x": 7,
            "pixel_y": 8,
            "original_gaussian_id": 102,
            "verified_final_gaussian_index": 3,
            "mapping_confidence": "medium",
            "mapping_status": "ambiguous_mapping_candidate",
            "repair_warning": "mapping_not_high_confidence",
        },
    ]
    pr212c_rows = [
        {"scene": "chair", "condition": "corrupt_occluder", "subset_name": "seed_20260710", "view_name": "train_009", "view_group": "direct_corrupted", "pixel_x": 1, "pixel_y": 2, "exact_id_count": 1, "repaired_proxy_id_count": 1, "intersection_count": 0, "jaccard": 0.0, "comparison_status": "repaired_comparison_valid", "interpretation": "repaired_zero_overlap"},
        {"scene": "chair", "condition": "corrupt_occluder", "subset_name": "seed_20260710", "view_name": "train_012", "view_group": "direct_corrupted", "pixel_x": 3, "pixel_y": 4, "exact_id_count": 1, "repaired_proxy_id_count": 0, "intersection_count": 0, "jaccard": 0.0, "comparison_status": "missing_repaired_proxy_ids", "interpretation": "repaired_comparison_incomplete"},
        {"scene": "chair", "condition": "corrupt_occluder", "subset_name": "seed_20260710", "view_name": "train_012", "view_group": "direct_corrupted", "pixel_x": 5, "pixel_y": 6, "exact_id_count": 1, "repaired_proxy_id_count": 0, "intersection_count": 0, "jaccard": 0.0, "comparison_status": "missing_repaired_proxy_ids", "interpretation": "repaired_comparison_incomplete"},
        {"scene": "chair", "condition": "corrupt_occluder", "subset_name": "seed_20260710", "view_name": "train_012", "view_group": "direct_corrupted", "pixel_x": 7, "pixel_y": 8, "exact_id_count": 1, "repaired_proxy_id_count": 0, "intersection_count": 0, "jaccard": 0.0, "comparison_status": "missing_repaired_proxy_ids", "interpretation": "repaired_comparison_incomplete"},
        {"scene": "chair", "condition": "corrupt_occluder", "subset_name": "seed_20260710", "view_name": "train_012", "view_group": "direct_corrupted", "pixel_x": 9, "pixel_y": 10, "exact_id_count": 1, "repaired_proxy_id_count": 0, "intersection_count": 0, "jaccard": 0.0, "comparison_status": "missing_repaired_proxy_ids", "interpretation": "repaired_comparison_incomplete"},
    ]
    _write_json(pr200 / "pr200_sparse_render_attribution_summary.json", {"scene": "chair"})
    _write_csv(pr200 / "pr200_pixel_gaussian_contributions.csv", pr20_rows)
    _write_csv(pr200 / "pr200_sparse_pixel_residuals.csv", [{"view_name": "train_009", "pixel_x": 1, "pixel_y": 2}])
    _write_json(pr211 / "pr211_exact_sparse_attribution_summary.json", {"evidence_quality": "exact_sparse_contributor_id_only"})
    _write_csv(pr211 / "pr211_exact_pixel_gaussian_contributions.csv", exact_rows)
    _write_json(pr212 / "pr212_chair_exact_vs_proxy_summary.json", {"mean_pixel_jaccard": 0.0})
    _write_json(pr212a / "pr212a_chair_id_namespace_audit_summary.json", {"same_global_gaussian_id_namespace_supported": False})
    _write_json(
        pr212b / "pr212b_pr20_proxy_id_source_audit_summary.json",
        {
            "explicit_mapping_available": True,
            "mapping_confidence": "high",
            "all_pr20_proxy_rows_repair_feasible": True,
            "exact_available_proxy_rows_repair_feasible": True,
            "exact_available_mapping_coverage_rate": 1.0,
            "proxy_safe_for_intervention": False,
        },
    )
    _write_csv(pr212b / "pr212b_pr20_proxy_repaired_preview.csv", repaired_rows)
    _write_csv(pr212b / "pr212b_repaired_exact_vs_proxy_preview.csv", [{"view_name": "train_012", "pixel_x": 5, "pixel_y": 6}])
    _write_json(
        pr212c / "pr212c_repaired_exact_vs_proxy_summary.json",
        {
            "exact_pixel_count": 43,
            "repaired_mean_pixel_jaccard": 0.0,
            "repaired_mean_exact_recall_by_proxy": 0.0,
            "repaired_mean_proxy_precision_against_exact": 0.0,
            "repaired_zero_overlap_claim_safe_within_exact_available_scope": False,
            "pr212c_ready_for_pr214": False,
            "recommended_next_step": "Fix repaired proxy coverage before PR21.4.",
        },
    )
    _write_csv(pr212c / "pr212c_repaired_pixel_exact_vs_proxy.csv", pr212c_rows)
    _write_json(pr213 / "pr213_chair_exact_evidence_positioning_summary.json", {"scene": "chair"})
    return pr200, pr211, pr212, pr212a, pr212b, pr212c, pr213


def main() -> int:
    project_root = Path(__file__).resolve().parents[2]
    import sys

    sys.path.insert(0, str(project_root))
    from viewtrust.analysis.pr212c_fix_missing_proxy import build_pr212c_fix_missing_repaired_proxy_pixels

    with tempfile.TemporaryDirectory(prefix="viewtrust-pr212cfix-") as tmp:
        root = Path(tmp)
        pr200, pr211, pr212, pr212a, pr212b, pr212c, pr213 = _make_inputs(root)
        output = root / "out"
        summary, code = build_pr212c_fix_missing_repaired_proxy_pixels(
            pr200_chair_dir=pr200,
            pr211_chair_dir=pr211,
            pr212_chair_dir=pr212,
            pr212a_chair_dir=pr212a,
            pr212b_chair_dir=pr212b,
            pr212c_chair_dir=pr212c,
            pr213_chair_dir=pr213,
            output_dir=output,
            scene="chair",
            condition="corrupt_occluder",
            subset_name="seed_20260710",
            write_markdown=True,
        )
        assert code == 0
        assert summary["schema_name"] == "viewtrust.pr212cfix.missing_repaired_proxy_pixels.summary"
        assert summary["observation_only"] is True
        assert summary["training_intervention"] is False
        assert summary["defense_enabled"] is False
        assert summary["view_rejection_enabled"] is False
        assert summary["densification_gating_enabled"] is False
        assert summary["third_party_modified"] is False
        assert summary["proxy_safe_for_intervention"] is False
        assert summary["exact_contribution_magnitude_available"] is False
        assert summary["drums_used_as_exact_evidence"] is False
        assert summary["coverage_problem_resolved"] is False
        assert summary["corrected_zero_overlap_claim_safe_within_exact_available_scope"] is False
        assert summary["corrected_zero_overlap_preserved_on_covered_pixels"] is True
        assert summary["pr212cfix_ready_for_pr214"] is False

        missing = _read_csv(output / "pr212cfix_missing_pixel_list.csv")
        assert len(missing) == 4
        trace = _read_csv(output / "pr212cfix_missing_pixel_trace.csv")
        assert any(row["inferred_failure_mode"] == "absent_from_pr20_original_proxy" for row in trace)
        assert any(row["inferred_failure_mode"] == "present_in_pr212b_but_filtered_by_pr212c" for row in trace)
        assert any(row["inferred_failure_mode"] == "mapping_status_too_strict" for row in trace)

        coord = _read_csv(output / "pr212cfix_coordinate_format_audit.csv")
        type_row = next(row for row in coord if row["pixel_x"] == "9" and row["pixel_y"] == "10")
        assert type_row["normal_match_pr20"] == "false"
        assert type_row["int_cast_match_pr20"] == "true"

        corrected = _read_csv(output / "pr212cfix_corrected_repaired_pixel_comparison_preview.csv")
        absent = next(row for row in corrected if row["pixel_x"] == "3" and row["pixel_y"] == "4")
        assert absent["comparison_status"] == "missing_repaired_proxy_ids"
        assert absent["corrected_repaired_proxy_id_count"] == "0"
        recovered = next(row for row in corrected if row["pixel_x"] == "5" and row["pixel_y"] == "6")
        assert recovered["comparison_status"] == "repaired_comparison_valid"

        claims = _read_csv(output / "pr212cfix_claim_scope_recommendation.csv")
        full = next(row for row in claims if row["scope_name"] == "full_exact_available_scope")
        covered = next(row for row in claims if row["scope_name"] == "repaired_proxy_covered_exact_scope")
        assert full["support_status"] == "unsupported"
        assert covered["support_status"] == "supported"
        assert "repaired-proxy-covered" in covered["paper_safe_wording"]
        report = (output / "pr212cfix_missing_repaired_proxy_pixels_report.md").read_text(encoding="utf-8")
        assert "Observation-only" in report
        manifest = _read_csv(output / "artifact_manifest.csv")
        assert any(row["relative_path"] == "pr212cfix_corrected_summary.json" and row["exists"] == "true" for row in manifest)

    print("pr212c-fix missing repaired proxy pixels smoke test ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
