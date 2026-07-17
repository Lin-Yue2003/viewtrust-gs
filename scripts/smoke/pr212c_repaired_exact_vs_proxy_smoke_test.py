#!/usr/bin/env python3
"""LOCAL-SAFE smoke test for PR21.2c repaired exact-vs-proxy comparison."""

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


def _make_inputs(root: Path, *, mode: str) -> tuple[Path, Path, Path, Path, Path, Path]:
    pr200 = root / "pr200"
    pr211 = root / "pr211"
    pr212 = root / "pr212"
    pr212a = root / "pr212a"
    pr212b = root / "pr212b"
    pr213 = root / "pr213"
    exact_rows = [
        {"view_name": "train_004", "view_group": "direct_corrupted", "pixel_x": 1, "pixel_y": 2, "gaussian_id": 10},
        {"view_name": "train_004", "view_group": "direct_corrupted", "pixel_x": 1, "pixel_y": 2, "gaussian_id": 11},
        {"view_name": "train_009", "view_group": "direct_corrupted", "pixel_x": 3, "pixel_y": 4, "gaussian_id": 12},
    ]
    if mode == "partial":
        exact_rows.append({"view_name": "train_012", "view_group": "direct_corrupted", "pixel_x": 5, "pixel_y": 6, "gaussian_id": 13})
    repaired_rows = [
        {
            "scene": "chair",
            "condition": "corrupt_occluder",
            "subset_name": "seed_20260710",
            "view_name": "train_004",
            "view_group": "direct_corrupted",
            "pixel_x": 1,
            "pixel_y": 2,
            "original_gaussian_id": 100,
            "original_root_gaussian_id": 100,
            "original_parent_gaussian_id": "",
            "verified_final_gaussian_index": 10 if mode == "nonzero" else 1,
            "mapping_name": "gaussian_id_to_final_index",
            "mapping_source": "run/gaussian_identity_table.csv",
            "mapping_confidence": "high",
            "mapping_status": "verified_mapping_candidate",
            "repair_warning": "",
        },
        {
            "scene": "chair",
            "condition": "corrupt_occluder",
            "subset_name": "seed_20260710",
            "view_name": "train_004",
            "view_group": "direct_corrupted",
            "pixel_x": 1,
            "pixel_y": 2,
            "original_gaussian_id": 101,
            "original_root_gaussian_id": 101,
            "original_parent_gaussian_id": "",
            "verified_final_gaussian_index": 2,
            "mapping_name": "gaussian_id_to_final_index",
            "mapping_source": "run/gaussian_identity_table.csv",
            "mapping_confidence": "high",
            "mapping_status": "verified_mapping_candidate",
            "repair_warning": "",
        },
        {
            "scene": "chair",
            "condition": "corrupt_occluder",
            "subset_name": "seed_20260710",
            "view_name": "train_009",
            "view_group": "direct_corrupted",
            "pixel_x": 3,
            "pixel_y": 4,
            "original_gaussian_id": 102,
            "original_root_gaussian_id": 102,
            "original_parent_gaussian_id": "",
            "verified_final_gaussian_index": 3,
            "mapping_name": "gaussian_id_to_final_index",
            "mapping_source": "run/gaussian_identity_table.csv",
            "mapping_confidence": "high",
            "mapping_status": "verified_mapping_candidate",
            "repair_warning": "",
        },
    ]
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
    _write_json(
        pr212 / "pr212_chair_exact_vs_proxy_summary.json",
        {
            "scene": "chair",
            "condition": "corrupt_occluder",
            "exact_pixel_count": len({(r["view_name"], r["pixel_x"], r["pixel_y"]) for r in exact_rows}),
            "exact_row_count": len(exact_rows),
            "exact_view_count_with_rows": len({r["view_name"] for r in exact_rows}),
            "mean_pixel_jaccard": 0.0,
            "median_pixel_jaccard": 0.0,
            "mean_exact_recall_by_proxy": 0.0,
            "mean_proxy_precision_against_exact": 0.0,
            "view_mean_jaccard_for_exact_available_views": 0.0,
        },
    )
    _write_csv(
        pr212 / "pr212_chair_pixel_exact_vs_proxy.csv",
        [
            {
                "view_name": "train_004",
                "pixel_x": 1,
                "pixel_y": 2,
                "proxy_gaussian_ids_semicolon": "100;101",
            }
        ],
    )
    _write_json(
        pr212a / "pr212a_chair_id_namespace_audit_summary.json",
        {"same_global_gaussian_id_namespace_supported": False},
    )
    _write_json(
        pr212b / "pr212b_pr20_proxy_id_source_audit_summary.json",
        {
            "scene": "chair",
            "condition": "corrupt_occluder",
            "explicit_mapping_available": True,
            "mapping_source": "run/gaussian_identity_table.csv",
            "mapping_confidence": "high",
            "exact_available_proxy_rows_repair_feasible": True,
            "exact_available_mapping_coverage_rate": 1.0,
            "repaired_zero_overlap_claim_safe": True,
            "pr212b_ready_for_pr212c": True,
            "proxy_safe_for_intervention": False,
            "pr212b_ready_for_pr214": False,
        },
    )
    _write_csv(pr212b / "pr212b_pr20_proxy_repaired_preview.csv", repaired_rows)
    _write_csv(pr212b / "pr212b_repaired_exact_vs_proxy_preview.csv", [{"comparison_status": "repaired_comparison_valid"}])
    _write_json(pr213 / "pr213_chair_exact_evidence_positioning_summary.json", {"schema_name": "viewtrust.pr213.chair_exact_evidence_positioning.summary"})
    _write_json(pr200 / "pr200_sparse_render_attribution_summary.json", {"scene": "chair"})
    return pr200, pr211, pr212, pr212a, pr212b, pr213


def _run(root: Path, *, mode: str) -> tuple[dict[str, object], Path]:
    project_root = Path(__file__).resolve().parents[2]
    import sys

    sys.path.insert(0, str(project_root))
    from viewtrust.analysis.pr212c_repaired_comparison import build_pr212c_repaired_exact_vs_proxy_comparison

    pr200, pr211, pr212, pr212a, pr212b, pr213 = _make_inputs(root, mode=mode)
    output = root / "out"
    summary, code = build_pr212c_repaired_exact_vs_proxy_comparison(
        pr200_chair_dir=pr200,
        pr211_chair_dir=pr211,
        pr212_chair_dir=pr212,
        pr212a_chair_dir=pr212a,
        pr212b_chair_dir=pr212b,
        pr213_chair_dir=pr213,
        output_dir=output,
        scene="chair",
        condition="corrupt_occluder",
        subset_name="seed_20260710",
        write_markdown=True,
    )
    assert code == 0
    return summary, output


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="viewtrust-pr212c-") as tmp:
        root = Path(tmp)
        zero_summary, zero_out = _run(root / "zero", mode="zero")
        assert zero_summary["schema_name"] == "viewtrust.pr212c.repaired_exact_vs_proxy.summary"
        assert zero_summary["observation_only"] is True
        assert zero_summary["training_intervention"] is False
        assert zero_summary["defense_enabled"] is False
        assert zero_summary["view_rejection_enabled"] is False
        assert zero_summary["densification_gating_enabled"] is False
        assert zero_summary["third_party_modified"] is False
        assert zero_summary["proxy_safe_for_intervention"] is False
        assert zero_summary["exact_contribution_magnitude_available"] is False
        assert zero_summary["drums_used_as_exact_evidence"] is False
        assert zero_summary["repaired_zero_overlap_preserved"] is True
        assert zero_summary["repaired_nonzero_overlap_found"] is False
        assert zero_summary["pr212c_ready_for_pr214"] is True
        assert zero_summary["repaired_mapping_confidence"] == "high"

        expected = [
            "pr212c_repaired_exact_vs_proxy_summary.json",
            "pr212c_repaired_pixel_exact_vs_proxy.csv",
            "pr212c_repaired_view_exact_vs_proxy.csv",
            "pr212c_repaired_group_exact_vs_proxy.csv",
            "pr212c_original_vs_repaired_delta.csv",
            "pr212c_repaired_proxy_degeneracy_reassessment.csv",
            "pr212c_claim_status_table.csv",
            "pr212c_repaired_exact_vs_proxy_report.md",
            "pr212c_paper_wording.md",
            "pr212c_next_step_decision_memo.md",
            "artifact_manifest.csv",
        ]
        for name in expected:
            assert (zero_out / name).exists(), name
        delta = _read_csv(zero_out / "pr212c_original_vs_repaired_delta.csv")
        assert any(row["metric"] == "pixel_mean_jaccard" and row["interpretation"] == "unchanged_zero_overlap_after_repair" for row in delta)
        manifest = _read_csv(zero_out / "artifact_manifest.csv")
        assert any(row["relative_path"] == "pr212c_repaired_exact_vs_proxy_summary.json" and row["exists"] == "true" for row in manifest)

        nonzero_summary, nonzero_out = _run(root / "nonzero", mode="nonzero")
        assert nonzero_summary["repaired_zero_overlap_preserved"] is False
        assert nonzero_summary["repaired_nonzero_overlap_found"] is True
        assert nonzero_summary["pr212c_ready_for_pr214"] is True
        nonzero_pixels = _read_csv(nonzero_out / "pr212c_repaired_pixel_exact_vs_proxy.csv")
        assert any(row["interpretation"] == "repaired_nonzero_overlap" for row in nonzero_pixels)

        partial_summary, partial_out = _run(root / "partial", mode="partial")
        assert partial_summary["comparison_partial"] is True
        assert partial_summary["repaired_zero_overlap_claim_safe_within_exact_available_scope"] is False
        assert partial_summary["pr212c_ready_for_pr214"] is False
        partial_pixels = _read_csv(partial_out / "pr212c_repaired_pixel_exact_vs_proxy.csv")
        assert any(row["comparison_status"] == "missing_repaired_proxy_ids" for row in partial_pixels)

    print("pr212c repaired exact-vs-proxy smoke test ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
