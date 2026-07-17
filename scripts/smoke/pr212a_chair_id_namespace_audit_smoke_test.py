#!/usr/bin/env python3
"""LOCAL-SAFE smoke test for PR21.2a chair ID namespace audit."""

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


def _write_ply(path: Path, vertex_count: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        (
            "ply\n"
            "format binary_little_endian 1.0\n"
            f"element vertex {vertex_count}\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "end_header\n"
        ).encode("ascii")
    )


def _make_inputs(root: Path, *, rank_like_proxy: bool = False, out_of_range_exact: bool = False) -> tuple[Path, Path, Path, Path, Path]:
    pr200 = root / "pr200"
    pr211 = root / "pr211"
    pr212 = root / "pr212"
    pr213 = root / "pr213"
    run_dir = root / "run"
    _write_ply(run_dir / "point_cloud" / "iteration_700" / "point_cloud.ply", 100)
    exact_gids = [10, 20] if not out_of_range_exact else [10, 200]
    exact_rows = [
        {"view_name": "train_009", "pixel_x": 1, "pixel_y": 2, "gaussian_id": exact_gids[0]},
        {"view_name": "train_009", "pixel_x": 1, "pixel_y": 2, "gaussian_id": exact_gids[1]},
        {"view_name": "train_012", "pixel_x": 3, "pixel_y": 4, "gaussian_id": 30},
    ]
    if rank_like_proxy:
        proxy_rows = []
        for view, x, y in [("train_009", 1, 2), ("train_012", 3, 4), ("train_013", 5, 6)]:
            for gid in [0, 1]:
                proxy_rows.append({"view_name": view, "pixel_x": x, "pixel_y": y, "gaussian_id": gid, "contribution_rank": gid})
    else:
        proxy_rows = [
            {"view_name": "train_009", "pixel_x": 1, "pixel_y": 2, "gaussian_id": 40},
            {"view_name": "train_009", "pixel_x": 1, "pixel_y": 2, "gaussian_id": 41},
            {"view_name": "train_012", "pixel_x": 3, "pixel_y": 4, "gaussian_id": 50},
            {"view_name": "train_012", "pixel_x": 3, "pixel_y": 4, "gaussian_id": 51},
        ]
    _write_csv(pr200 / "pr200_pixel_gaussian_contributions.csv", proxy_rows)
    _write_json(pr200 / "pr200_sparse_render_attribution_summary.json", {"scene": "chair"})
    _write_json(
        pr211 / "pr211_exact_sparse_attribution_summary.json",
        {
            "scene": "chair",
            "condition": "corrupt_occluder",
            "evidence_quality": "exact_sparse_contributor_id_only",
            "exact_contributor_id_only_succeeded": True,
            "exact_contributor_id_row_count": len(exact_rows),
            "exact_render_contribution_succeeded": False,
            "ready_for_intervention": False,
        },
    )
    _write_csv(pr211 / "pr211_exact_pixel_gaussian_contributions.csv", exact_rows)
    _write_csv(pr211 / "pr211_selected_pixels.csv", [{"view_name": "train_009", "pixel_x": 1, "pixel_y": 2}])
    _write_csv(pr211 / "pr211_per_view_replay_audit.csv", [{"view_name": "train_009"}])
    _write_json(
        pr212 / "pr212_chair_exact_vs_proxy_summary.json",
        {
            "scene": "chair",
            "condition": "corrupt_occluder",
            "exact_contributor_id_only_available": True,
            "exact_render_contribution_available": False,
            "exact_pixel_count": 2,
            "exact_row_count": len(exact_rows),
            "mean_pixel_jaccard": 0.0,
            "mean_exact_recall_by_proxy": 0.0,
            "mean_proxy_precision_against_exact": 0.0,
            "pr212_ready_for_interpretation": True,
            "pr212_ready_for_intervention": False,
        },
    )
    _write_csv(
        pr212 / "pr212_chair_pixel_exact_vs_proxy.csv",
        [
            {
                "view_name": "train_009",
                "pixel_x": 1,
                "pixel_y": 2,
                "exact_gaussian_ids_semicolon": ";".join(str(v) for v in exact_gids),
                "proxy_gaussian_ids_semicolon": "0;1" if rank_like_proxy else "40;41",
            }
        ],
    )
    _write_json(
        pr213 / "pr213_chair_exact_evidence_positioning_summary.json",
        {
            "scene": "chair",
            "positioning_status": "exact_attribution_trust_signal_validation_not_defense",
            "proxy_degeneracy_supported_by_exact": False,
            "proxy_safe_for_intervention": False,
            "drums_used_as_exact_evidence": False,
        },
    )
    return pr200, pr211, pr212, pr213, run_dir


def _run(root: Path, *, rank_like_proxy: bool = False, out_of_range_exact: bool = False) -> tuple[dict[str, object], Path]:
    project_root = Path(__file__).resolve().parents[2]
    import sys

    sys.path.insert(0, str(project_root))
    from viewtrust.analysis.pr212a_namespace import build_pr212a_chair_id_namespace_audit

    pr200, pr211, pr212, pr213, run_dir = _make_inputs(root, rank_like_proxy=rank_like_proxy, out_of_range_exact=out_of_range_exact)
    output = root / "out"
    summary, code = build_pr212a_chair_id_namespace_audit(
        pr200_chair_dir=pr200,
        pr211_chair_dir=pr211,
        pr212_chair_dir=pr212,
        pr213_chair_dir=pr213,
        run_dir=run_dir,
        output_dir=output,
        scene="chair",
        condition="corrupt_occluder",
        subset_name="seed_20260710",
        sample_id_count=2,
        write_markdown=True,
    )
    assert code == 0
    return summary, output


def main() -> int:
    project_root = Path(__file__).resolve().parents[2]
    import sys

    sys.path.insert(0, str(project_root))
    from viewtrust.analysis.pr212a_namespace import read_ply_vertex_count

    with tempfile.TemporaryDirectory(prefix="viewtrust-pr212a-") as tmp:
        root = Path(tmp)
        ply = root / "header_only.ply"
        _write_ply(ply, 123)
        assert read_ply_vertex_count(ply) == 123

        summary, output = _run(root / "global")
        assert summary["schema_name"] == "viewtrust.pr212a.chair_id_namespace_audit.summary"
        assert summary["observation_only"] is True
        assert summary["training_intervention"] is False
        assert summary["defense_enabled"] is False
        assert summary["view_rejection_enabled"] is False
        assert summary["densification_gating_enabled"] is False
        assert summary["third_party_modified"] is False
        assert summary["checkpoint_gaussian_count"] == 100
        assert summary["pr20_proxy_ids_numeric"] is True
        assert summary["pr211_exact_ids_numeric"] is True
        assert summary["pr20_proxy_ids_in_checkpoint_range"] is True
        assert summary["pr211_exact_ids_in_checkpoint_range"] is True
        assert summary["same_global_gaussian_id_namespace_supported"] is True
        assert summary["zero_overlap_claim_safe_within_exact_available_scope"] is True
        assert summary["proxy_safe_for_intervention"] is False
        assert summary["drums_used_as_exact_evidence"] is False
        assert summary["pr212a_ready_for_pr214"] is True

        expected = [
            "pr212a_chair_id_namespace_audit_summary.json",
            "pr212a_checkpoint_gaussian_inventory.csv",
            "pr212a_id_source_schema_audit.csv",
            "pr212a_id_range_audit.csv",
            "pr212a_same_pixel_id_namespace_comparison.csv",
            "pr212a_sample_id_lookup.csv",
            "pr212a_proxy_id_semantics_audit.csv",
            "pr212a_code_id_semantics_audit.csv",
            "pr212a_code_id_semantics_summary.json",
            "pr212a_id_namespace_diagnosis.csv",
            "pr212a_chair_id_namespace_audit_report.md",
            "pr212a_zero_overlap_namespace_wording.md",
            "pr212a_next_step_decision_memo.md",
            "artifact_manifest.csv",
        ]
        for name in expected:
            assert (output / name).exists(), name

        ranges = _read_csv(output / "pr212a_id_range_audit.csv")
        assert any(row["id_group"] == "pr20_proxy_all" and row["range_status"] == "zero_based_global_index_plausible" for row in ranges)
        assert any(row["id_group"] == "pr211_exact_all" and row["range_status"] == "zero_based_global_index_plausible" for row in ranges)

        same_pixel = _read_csv(output / "pr212a_same_pixel_id_namespace_comparison.csv")
        assert any(row["intersection_count"] == "0" and row["jaccard"] == "0.0" and row["interpretation"] == "zero_overlap_same_namespace_plausible" for row in same_pixel)

        semantics = _read_csv(output / "pr212a_proxy_id_semantics_audit.csv")
        assert any(row["conclusion"] == "proxy_ids_plausibly_global_checkpoint_indices" for row in semantics)
        samples = _read_csv(output / "pr212a_sample_id_lookup.csv")
        assert samples
        assert all(row["zero_based_index_exists"] == "true" for row in samples)
        diagnosis = _read_csv(output / "pr212a_id_namespace_diagnosis.csv")
        assert any(row["diagnosis_id"] == "pr20_vs_pr211_common_namespace" and row["status"] == "same_global_gaussian_id_namespace_supported" for row in diagnosis)
        wording = (output / "pr212a_zero_overlap_namespace_wording.md").read_text(encoding="utf-8")
        assert "If Namespace Supported" in wording
        memo = (output / "pr212a_next_step_decision_memo.md").read_text(encoding="utf-8")
        assert "Recommend PR21.4 exact contribution magnitude" in memo

        rank_summary, rank_output = _run(root / "rank_like", rank_like_proxy=True)
        assert rank_summary["pr20_proxy_ids_in_checkpoint_range"] is True
        assert rank_summary["pr20_gaussian_id_semantics"] == "local_candidate_rank"
        assert rank_summary["same_global_gaussian_id_namespace_supported"] is False
        assert rank_summary["zero_overlap_claim_safe_within_exact_available_scope"] is False
        rank_semantics = _read_csv(rank_output / "pr212a_proxy_id_semantics_audit.csv")
        assert any(row["local_rank_plausible"] == "true" and row["conclusion"] == "proxy_ids_plausibly_local_candidate_ranks" for row in rank_semantics)

        bad_summary, bad_output = _run(root / "out_of_range", out_of_range_exact=True)
        assert bad_summary["pr211_exact_ids_in_checkpoint_range"] is False
        assert bad_summary["same_global_gaussian_id_namespace_supported"] is False
        bad_ranges = _read_csv(bad_output / "pr212a_id_range_audit.csv")
        assert any(row["id_group"] == "pr211_exact_all" and row["range_status"] == "out_of_checkpoint_range" for row in bad_ranges)

    print("pr212a chair ID namespace audit smoke test ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
