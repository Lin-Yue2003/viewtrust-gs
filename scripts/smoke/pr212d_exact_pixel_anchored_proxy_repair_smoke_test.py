#!/usr/bin/env python3
"""LOCAL-SAFE smoke test for PR21.2d exact-pixel-anchored proxy repair."""

from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path


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


def _make_inputs(root: Path, *, mode: str) -> tuple[Path, Path, Path, Path, Path, Path, Path, Path, Path]:
    pr200 = root / "pr200"
    pr211 = root / "pr211"
    pr212 = root / "pr212"
    pr212a = root / "pr212a"
    pr212b = root / "pr212b"
    pr212c = root / "pr212c"
    pr212cfix = root / "pr212cfix"
    pr213 = root / "pr213"
    run_dir = root / "run"
    anchors = [("train_009", idx + 1, idx + 2) for idx in range(8)]
    anchors.extend(("train_012", idx + 20, idx + 30) for idx in range(35))
    exact_rows = [
        {"view_name": view, "view_group": "direct_corrupted", "pixel_x": x, "pixel_y": y, "gaussian_id": idx}
        for idx, (view, x, y) in enumerate(anchors, start=10)
    ]
    pr20_rows = []
    mapping_rows = []
    for idx, (view, x, y) in enumerate(anchors):
        if mode == "partial_missing_proxy" and idx == len(anchors) - 1:
            continue
        proxy_id = 100 + idx
        pr20_rows.append(
            {
                "view_name": view,
                "view_group": "direct_corrupted",
                "pixel_x": x,
                "pixel_y": y,
                "gaussian_id": proxy_id,
                "root_gaussian_id": proxy_id,
                "parent_gaussian_id": "",
                "contribution_rank": 1,
                "splat_weight": round(0.1 + idx * 0.001, 4),
            }
        )
        if mode == "partial_unmapped" and idx == len(anchors) - 1:
            continue
        if mode == "nonzero" and idx == 0:
            final_index = 10
        elif mode == "out_of_range" and idx == len(anchors) - 1:
            final_index = 9999
        else:
            final_index = 1000 + idx
        mapping_rows.append({"gaussian_id": proxy_id, "final_index": final_index, "alive": "true"})
    _write_csv(pr200 / "pr200_pixel_gaussian_contributions.csv", pr20_rows)
    _write_json(pr200 / "pr200_sparse_render_attribution_summary.json", {"scene": "chair"})
    _write_csv(pr200 / "pr200_sparse_pixel_residuals.csv", [{"view_name": "train_009", "pixel_x": 1, "pixel_y": 2}])
    _write_json(
        pr211 / "pr211_exact_sparse_attribution_summary.json",
        {
            "scene": "chair",
            "condition": "corrupt_occluder",
            "evidence_quality": "exact_sparse_contributor_id_only",
            "exact_render_contribution_succeeded": False,
        },
    )
    _write_csv(pr211 / "pr211_exact_pixel_gaussian_contributions.csv", exact_rows)
    _write_json(pr212 / "pr212_chair_exact_vs_proxy_summary.json", {"schema_name": "viewtrust.pr212.chair_exact_vs_proxy.summary"})
    _write_json(pr212a / "pr212a_chair_id_namespace_audit_summary.json", {"checkpoint_gaussian_count": 2000})
    _write_json(
        pr212b / "pr212b_pr20_proxy_id_source_audit_summary.json",
        {
            "explicit_mapping_available": True,
            "mapping_source": "tables/gaussian_lifecycle_final.csv",
            "mapping_confidence": "high",
            "all_pr20_proxy_rows_repair_feasible": mode not in {"partial_unmapped", "out_of_range"},
            "exact_available_proxy_rows_repair_feasible": mode not in {"partial_unmapped", "out_of_range"},
            "exact_available_mapping_coverage_rate": 1.0 if mode not in {"partial_unmapped"} else 0.98,
            "proxy_safe_for_intervention": False,
        },
    )
    _write_csv(pr212b / "pr212b_mapping_candidate_table.csv", [{"mapping_status": "verified_mapping_candidate"}])
    _write_csv(pr212b / "pr212b_repair_feasibility_summary.csv", [{"repair_scope": "exact_available_pr20_proxy_rows"}])
    _write_json(
        pr212c / "pr212c_repaired_exact_vs_proxy_summary.json",
        {
            "exact_pixel_count": 43,
            "repaired_mean_pixel_jaccard": 0.0,
            "repaired_proxy_unique_id_count_on_exact_pixels": 36,
            "pr212c_ready_for_pr214": False,
        },
    )
    _write_csv(
        pr212c / "pr212c_repaired_pixel_exact_vs_proxy.csv",
        [
            {"view_name": view, "pixel_x": x, "pixel_y": y, "comparison_status": "repaired_comparison_valid" if idx < 36 else "missing_repaired_proxy_ids", "repaired_proxy_id_count": 1 if idx < 36 else 0}
            for idx, (view, x, y) in enumerate(anchors)
        ],
    )
    _write_json(
        pr212cfix / "pr212cfix_corrected_summary.json",
        {
            "exact_pixel_count": 43,
            "missing_pixel_count": 7,
            "exact_pixels_present_in_pr20_proxy_count": 43,
            "exact_pixels_present_in_pr212b_repaired_preview_count": 36,
            "missing_pixel_failure_mode_counts": {"present_in_pr20_but_missing_from_pr212b_repaired_preview": 7},
            "coverage_problem_resolved": False,
        },
    )
    _write_json(pr213 / "pr213_chair_exact_evidence_positioning_summary.json", {"scene": "chair"})
    _write_csv(run_dir / "tables" / "gaussian_lifecycle_final.csv", mapping_rows)
    return pr200, pr211, pr212, pr212a, pr212b, pr212c, pr212cfix, pr213, run_dir


def _run(root: Path, *, mode: str) -> tuple[dict[str, object], Path]:
    project_root = Path(__file__).resolve().parents[2]
    import sys

    sys.path.insert(0, str(project_root))
    from viewtrust.analysis.pr212d_exact_pixel_repair import build_pr212d_exact_pixel_anchored_proxy_repair

    pr200, pr211, pr212, pr212a, pr212b, pr212c, pr212cfix, pr213, run_dir = _make_inputs(root, mode=mode)
    output = root / "out"
    summary, code = build_pr212d_exact_pixel_anchored_proxy_repair(
        pr200_chair_dir=pr200,
        pr211_chair_dir=pr211,
        pr212_chair_dir=pr212,
        pr212a_chair_dir=pr212a,
        pr212b_chair_dir=pr212b,
        pr212c_chair_dir=pr212c,
        pr212cfix_chair_dir=pr212cfix,
        pr213_chair_dir=pr213,
        run_dir=run_dir,
        output_dir=output,
        scene="chair",
        condition="corrupt_occluder",
        subset_name="seed_20260710",
        write_markdown=True,
    )
    assert code == 0
    return summary, output


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="viewtrust-pr212d-") as tmp:
        root = Path(tmp)
        zero_summary, zero_output = _run(root / "zero", mode="zero")
        assert zero_summary["schema_name"] == "viewtrust.pr212d.exact_pixel_anchored_proxy_repair.summary"
        assert zero_summary["observation_only"] is True
        assert zero_summary["training_intervention"] is False
        assert zero_summary["defense_enabled"] is False
        assert zero_summary["view_rejection_enabled"] is False
        assert zero_summary["densification_gating_enabled"] is False
        assert zero_summary["third_party_modified"] is False
        assert zero_summary["exact_pixel_count"] == 43
        assert zero_summary["exact_row_count"] == 43
        assert zero_summary["exact_view_count_with_rows"] == 2
        assert zero_summary["exact_pixels_with_pr20_proxy_rows"] == 43
        assert zero_summary["complete_valid_pixel_count"] == 43
        assert zero_summary["complete_missing_pixel_count"] == 0
        assert zero_summary["complete_zero_overlap_preserved"] is True
        assert zero_summary["complete_zero_overlap_claim_safe_within_exact_available_scope"] is True
        assert zero_summary["complete_nonzero_overlap_found"] is False
        assert zero_summary["proxy_safe_for_intervention"] is False
        assert zero_summary["exact_contribution_magnitude_available"] is False
        assert zero_summary["drums_used_as_exact_evidence"] is False
        assert zero_summary["pr212d_ready_for_pr214"] is True

        anchor = _read_csv(zero_output / "pr212d_exact_pixel_anchor.csv")
        assert len(anchor) == 43
        pr20_anchor = _read_csv(zero_output / "pr212d_pr20_proxy_rows_on_exact_pixels.csv")
        assert len(pr20_anchor) == 43
        assert all(row["exact_anchor_present"] == "true" for row in pr20_anchor)
        coverage = _read_csv(zero_output / "pr212d_anchor_proxy_coverage.csv")
        assert all(row["coverage_status"] == "pr20_proxy_present" for row in coverage)
        audit = _read_csv(zero_output / "pr212d_verified_mapping_audit.csv")[0]
        assert audit["source_id_column"] == "gaussian_id"
        assert audit["target_index_column"] == "final_index"
        assert audit["final_index_in_checkpoint_range"] == "true"
        pixel = _read_csv(zero_output / "pr212d_complete_repaired_pixel_exact_vs_proxy.csv")
        assert all(row["comparison_status"] == "complete_repaired_comparison_valid" for row in pixel)
        assert all(row["intersection_count"] == "0" for row in pixel)
        claims = _read_csv(zero_output / "pr212d_claim_status_table.csv")
        assert next(row for row in claims if row["claim_id"] == "C3")["status"] == "unsupported"

        nonzero_summary, nonzero_output = _run(root / "nonzero", mode="nonzero")
        assert nonzero_summary["complete_zero_overlap_preserved"] is False
        assert nonzero_summary["complete_nonzero_overlap_found"] is True
        assert nonzero_summary["pr212d_ready_for_pr214"] is True
        nonzero_pixel = _read_csv(nonzero_output / "pr212d_complete_repaired_pixel_exact_vs_proxy.csv")
        assert any(row["interpretation"] == "complete_repaired_nonzero_overlap" for row in nonzero_pixel)

        partial_summary, partial_output = _run(root / "partial", mode="partial_unmapped")
        assert partial_summary["complete_zero_overlap_claim_safe_within_exact_available_scope"] is False
        assert partial_summary["pr212d_ready_for_pr214"] is False
        partial_audit = _read_csv(partial_output / "pr212d_verified_mapping_audit.csv")[0]
        assert partial_audit["unmapped_pr20_exact_pixel_proxy_row_count"] == "1"

        missing_summary, _ = _run(root / "missing", mode="partial_missing_proxy")
        assert missing_summary["exact_pixels_missing_pr20_proxy_rows"] == 1
        assert missing_summary["complete_missing_pixel_count"] == 1
        assert missing_summary["pr212d_ready_for_pr214"] is False

        out_of_range_summary, out_of_range_output = _run(root / "range", mode="out_of_range")
        assert out_of_range_summary["pr212d_ready_for_pr214"] is False
        range_audit = _read_csv(out_of_range_output / "pr212d_verified_mapping_audit.csv")[0]
        assert range_audit["final_index_in_checkpoint_range"] == "false"

    print("pr212d exact-pixel-anchored proxy repair smoke test ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
