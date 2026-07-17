#!/usr/bin/env python3
"""LOCAL-SAFE smoke test for PR21.2b PR20 proxy ID source audit."""

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


def _write_ply(path: Path, count: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "ply\n"
        "format ascii 1.0\n"
        f"element vertex {count}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "end_header\n",
        encoding="ascii",
    )


def _make_inputs(root: Path, *, explicit_mapping: bool) -> tuple[Path, Path, Path, Path, Path, Path]:
    pr200 = root / "pr200"
    pr211 = root / "pr211"
    pr212 = root / "pr212"
    pr212a = root / "pr212a"
    pr213 = root / "pr213"
    run_dir = root / "run"
    proxy_rows = [
        {"view_name": "train_004", "view_group": "direct_corrupted", "pixel_x": 1, "pixel_y": 2, "gaussian_id": 57, "root_gaussian_id": 57, "parent_gaussian_id": "", "contribution_rank": 1},
        {"view_name": "train_004", "view_group": "direct_corrupted", "pixel_x": 1, "pixel_y": 2, "gaussian_id": 58, "root_gaussian_id": 58, "parent_gaussian_id": "", "contribution_rank": 2},
        {"view_name": "train_009", "view_group": "direct_corrupted", "pixel_x": 3, "pixel_y": 4, "gaussian_id": 59, "root_gaussian_id": 59, "parent_gaussian_id": "", "contribution_rank": 1},
        {"view_name": "train_013", "view_group": "clean_prior_demoted", "pixel_x": 5, "pixel_y": 6, "gaussian_id": 100000, "root_gaussian_id": 100000, "parent_gaussian_id": "", "contribution_rank": 1},
        {"view_name": "train_013", "view_group": "clean_prior_demoted", "pixel_x": 5, "pixel_y": 6, "gaussian_id": 100016, "root_gaussian_id": 100016, "parent_gaussian_id": "", "contribution_rank": 2},
    ]
    exact_rows = [
        {"view_name": "train_004", "pixel_x": 1, "pixel_y": 2, "gaussian_id": 7},
        {"view_name": "train_004", "pixel_x": 1, "pixel_y": 2, "gaussian_id": 8},
        {"view_name": "train_009", "pixel_x": 3, "pixel_y": 4, "gaussian_id": 9},
    ]
    _write_csv(pr200 / "pr200_pixel_gaussian_contributions.csv", proxy_rows)
    _write_json(pr200 / "pr200_sparse_render_attribution_summary.json", {"scene": "chair"})
    _write_csv(pr200 / "pr200_sparse_pixel_residuals.csv", [{"view_name": "train_004", "pixel_x": 1, "pixel_y": 2}])
    _write_json(pr211 / "pr211_exact_sparse_attribution_summary.json", {"scene": "chair"})
    _write_csv(pr211 / "pr211_exact_pixel_gaussian_contributions.csv", exact_rows)
    _write_json(pr212 / "pr212_chair_exact_vs_proxy_summary.json", {"scene": "chair", "mean_pixel_jaccard": 0.0})
    _write_csv(pr212 / "pr212_chair_pixel_exact_vs_proxy.csv", [{"view_name": "train_004", "pixel_x": 1, "pixel_y": 2}])
    _write_json(
        pr212a / "pr212a_chair_id_namespace_audit_summary.json",
        {
            "same_global_gaussian_id_namespace_supported": False,
            "zero_overlap_claim_safe_within_exact_available_scope": False,
            "pr20_proxy_ids_in_checkpoint_range": False,
            "pr211_exact_ids_in_checkpoint_range": True,
            "checkpoint_gaussian_count": 10,
            "checkpoint_gaussian_count_source": str(run_dir / "point_cloud" / "iteration_700" / "point_cloud.ply"),
            "checkpoint_count_confidence": "high",
            "proxy_safe_for_intervention": False,
        },
    )
    _write_json(pr213 / "pr213_chair_exact_evidence_positioning_summary.json", {"scene": "chair"})
    _write_ply(run_dir / "point_cloud" / "iteration_700" / "point_cloud.ply", 10)
    if explicit_mapping:
        _write_csv(
            run_dir / "gaussian_identity_table.csv",
            [
                {"gaussian_id": 57, "root_gaussian_id": 57, "parent_gaussian_id": "", "final_index": 0, "status": "alive"},
                {"gaussian_id": 58, "root_gaussian_id": 58, "parent_gaussian_id": "", "final_index": 1, "status": "alive"},
                {"gaussian_id": 59, "root_gaussian_id": 59, "parent_gaussian_id": "", "final_index": 2, "status": "alive"},
                {"gaussian_id": 100000, "root_gaussian_id": 100000, "parent_gaussian_id": "", "final_index": 3, "status": "alive"},
                {"gaussian_id": 100016, "root_gaussian_id": 100016, "parent_gaussian_id": "", "final_index": 4, "status": "alive"},
            ],
        )
    return pr200, pr211, pr212, pr212a, pr213, run_dir


def _run(root: Path, *, explicit_mapping: bool) -> tuple[dict[str, object], Path]:
    project_root = Path(__file__).resolve().parents[2]
    import sys

    sys.path.insert(0, str(project_root))
    from viewtrust.analysis.pr212b_proxy_source import build_pr212b_pr20_proxy_id_source_audit

    pr200, pr211, pr212, pr212a, pr213, run_dir = _make_inputs(root, explicit_mapping=explicit_mapping)
    output = root / "out"
    summary, code = build_pr212b_pr20_proxy_id_source_audit(
        pr200_chair_dir=pr200,
        pr211_chair_dir=pr211,
        pr212_chair_dir=pr212,
        pr212a_chair_dir=pr212a,
        pr213_chair_dir=pr213,
        run_dir=run_dir,
        output_dir=output,
        scene="chair",
        condition="corrupt_occluder",
        subset_name="seed_20260710",
        sample_id_count=40,
        write_markdown=True,
    )
    assert code == 0
    return summary, output


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="viewtrust-pr212b-") as tmp:
        root = Path(tmp)
        summary, output = _run(root / "nomap", explicit_mapping=False)
        assert summary["schema_name"] == "viewtrust.pr212b.pr20_proxy_id_source_audit.summary"
        assert summary["observation_only"] is True
        assert summary["training_intervention"] is False
        assert summary["defense_enabled"] is False
        assert summary["view_rejection_enabled"] is False
        assert summary["densification_gating_enabled"] is False
        assert summary["third_party_modified"] is False
        assert summary["explicit_mapping_available"] is False
        assert summary["all_pr20_proxy_rows_repair_feasible"] is False
        assert summary["exact_available_proxy_rows_repair_feasible"] is False
        assert summary["repaired_zero_overlap_claim_safe"] is False
        assert summary["proxy_safe_for_intervention"] is False
        assert summary["pr212b_ready_for_pr212c"] is False
        assert summary["pr212b_ready_for_pr214"] is False

        expected = [
            "pr212b_pr20_proxy_id_source_audit_summary.json",
            "pr212b_pr20_proxy_id_profile.csv",
            "pr212b_identity_mapping_inventory.csv",
            "pr212b_proxy_id_lookup_across_identity_sources.csv",
            "pr212b_pr20_id_semantics_diagnosis.csv",
            "pr212b_mapping_candidate_table.csv",
            "pr212b_repair_feasibility_summary.csv",
            "pr212b_pr20_proxy_repaired_preview.csv",
            "pr212b_repaired_exact_vs_proxy_preview.csv",
            "pr212b_code_proxy_id_source_audit.csv",
            "pr212b_code_proxy_id_source_summary.json",
            "pr212b_pr20_proxy_id_source_audit_report.md",
            "pr212b_proxy_namespace_wording.md",
            "pr212b_next_step_decision_memo.md",
            "artifact_manifest.csv",
        ]
        for name in expected:
            assert (output / name).exists(), name

        profile = _read_csv(output / "pr212b_pr20_proxy_id_profile.csv")
        train013 = next(row for row in profile if row["view_name"] == "train_013")
        assert "train013_100000_range" in train013["suspicious_id_pattern"]
        assert train013["final_checkpoint_range_status"] == "out_of_final_checkpoint_range"

        mapping = _read_csv(output / "pr212b_mapping_candidate_table.csv")
        assert mapping[0]["mapping_status"] == "no_mapping_available"
        repair = _read_csv(output / "pr212b_repair_feasibility_summary.csv")
        assert any(row["repair_scope"] == "exact_available_pr20_proxy_rows" and row["repair_feasible"] == "false" for row in repair)
        preview = _read_csv(output / "pr212b_pr20_proxy_repaired_preview.csv")
        assert preview
        assert all(row["verified_final_gaussian_index"] == "" for row in preview)

        mapped_summary, mapped_output = _run(root / "mapped", explicit_mapping=True)
        assert mapped_summary["explicit_mapping_available"] is True
        assert mapped_summary["train013_100000_ids_explained"] is True
        assert mapped_summary["all_pr20_proxy_rows_repair_feasible"] is True
        assert mapped_summary["exact_available_proxy_rows_repair_feasible"] is True
        assert mapped_summary["repaired_exact_vs_proxy_preview_available"] is True
        assert mapped_summary["pr212b_ready_for_pr212c"] is True
        assert mapped_summary["pr212b_ready_for_pr214"] is False
        assert mapped_summary["proxy_safe_for_intervention"] is False
        mapped_inventory = _read_csv(mapped_output / "pr212b_identity_mapping_inventory.csv")
        assert any(row["candidate_mapping_type"] == "persistent_to_final_index" for row in mapped_inventory)
        lookup = _read_csv(mapped_output / "pr212b_proxy_id_lookup_across_identity_sources.csv")
        assert any(row["queried_id"] == "100000" and row["found"] == "true" for row in lookup)
        mapped_candidates = _read_csv(mapped_output / "pr212b_mapping_candidate_table.csv")
        assert any(row["mapping_status"] == "verified_mapping_candidate" for row in mapped_candidates)
        mapped_preview = _read_csv(mapped_output / "pr212b_pr20_proxy_repaired_preview.csv")
        assert any(row["verified_final_gaussian_index"] == "0" and row["original_gaussian_id"] == "57" for row in mapped_preview)
        repaired_compare = _read_csv(mapped_output / "pr212b_repaired_exact_vs_proxy_preview.csv")
        assert any(row["comparison_status"] == "repaired_comparison_valid" for row in repaired_compare)

    print("pr212b PR20 proxy ID source audit smoke test ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
