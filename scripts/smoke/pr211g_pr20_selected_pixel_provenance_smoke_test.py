#!/usr/bin/env python3
"""LOCAL-SAFE smoke test for PR21.1g PR20 selected-pixel provenance audit."""

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


def _make_inputs(root: Path, *, reproducible: bool) -> tuple[Path, Path, Path, Path]:
    pr200 = root / "pr200"
    pr211f = root / "pr211f"
    pr211 = root / "pr211"
    run_dir = root / "run"
    run_dir.mkdir(parents=True)
    pr211.mkdir(parents=True)
    views = ["train_004", "train_009", "train_012"]
    proxy_rows = []
    selected = {
        "train_004": [(1, 2), (3, 4)],
        "train_009": [(10, 20), (11, 21)],
        "train_012": [(7, 8), (9, 10)],
    }
    for view, pixels in selected.items():
        for x, y in pixels:
            for gid in [100, 200]:
                proxy_rows.append(
                    {
                        "scene": "drums",
                        "view_name": view,
                        "view_group": "direct_corrupted",
                        "pixel_x": x,
                        "pixel_y": y,
                        "gaussian_id": gid,
                        "residual_l1": 1.0 + x / 100.0,
                        "residual_weighted_splat": 0.5 + y / 100.0,
                    }
                )
    _write_csv(pr200 / "pr200_pixel_gaussian_contributions.csv", proxy_rows)

    residual_rows = []
    if reproducible:
        residual_rows.extend(
            [
                {"view_name": "train_004", "pixel_x": 1, "pixel_y": 2, "pixel_id": 801, "residual_l1": 9.0},
                {"view_name": "train_004", "pixel_x": 3, "pixel_y": 4, "pixel_id": 1603, "residual_l1": 8.0},
                {"view_name": "train_004", "pixel_x": 99, "pixel_y": 99, "pixel_id": 39699, "residual_l1": 0.1},
                {"view_name": "train_009", "pixel_x": 300, "pixel_y": 20, "pixel_id": 8300, "residual_l1": 9.0},
                {"view_name": "train_009", "pixel_x": 301, "pixel_y": 21, "pixel_id": 8701, "residual_l1": 8.0},
                {"view_name": "train_012", "pixel_x": 7, "pixel_y": 391, "pixel_id": 156407, "residual_l1": 9.0},
                {"view_name": "train_012", "pixel_x": 9, "pixel_y": 389, "pixel_id": 155609, "residual_l1": 8.0},
            ]
        )
    else:
        for view in views:
            residual_rows.extend(
                [
                    {"view_name": view, "pixel_x": 300, "pixel_y": 300, "pixel_id": 120300, "residual_l1": 9.0},
                    {"view_name": view, "pixel_x": 301, "pixel_y": 301, "pixel_id": 120701, "residual_l1": 8.0},
                ]
            )
    _write_csv(pr200 / "pr200_sparse_pixel_residuals.csv", residual_rows)
    _write_csv(pr200 / "pr200_gaussian_residual_attribution.csv", [{"view_name": "train_004", "gaussian_id": 100, "residual_weighted_splat": 1.0}])
    if reproducible:
        _write_csv(
            pr200 / "pr200_selected_pixel_echo.csv",
            [
                {"view_name": "train_004", "pixel_x": 1, "pixel_y": 2},
                {"view_name": "train_004", "pixel_x": 3, "pixel_y": 4},
            ],
        )
    _write_json(
        pr211f / "pr211f_drums_selected_pixel_alignment_summary.json",
        {
            "scene": "drums",
            "likely_failure_mode_overall": "mixed_coordinate_candidate_and_no_raw_contributors",
            "exact_evidence_allowed_for_drums": False,
            "drums_ready_for_pr212": False,
        },
    )
    return pr200, pr211f, pr211, run_dir


def _run_builder(root: Path, *, reproducible: bool) -> tuple[dict[str, object], Path]:
    project_root = Path(__file__).resolve().parents[2]
    import sys

    sys.path.insert(0, str(project_root))
    from viewtrust.analysis.pr211g_provenance import build_pr211g_pr20_selected_pixel_provenance_audit

    pr200, pr211f, pr211, run_dir = _make_inputs(root, reproducible=reproducible)
    output = root / "out"
    summary, code = build_pr211g_pr20_selected_pixel_provenance_audit(
        pr200_dir=pr200,
        pr211f_dir=pr211f,
        pr211_dir=pr211,
        run_dir=run_dir,
        output_dir=output,
        scene="drums",
        condition="corrupt_occluder",
        subset_name="seed_20260710",
        views=["train_004", "train_009", "train_012"],
        top_pixels_per_view=2,
        max_contributors_per_pixel=2,
        write_markdown=True,
    )
    assert code == 0
    return summary, output


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="viewtrust-pr211g-") as tmp:
        root = Path(tmp)
        summary, output = _run_builder(root / "match", reproducible=True)
        assert summary["schema_name"] == "viewtrust.pr211g.pr20_selected_pixel_provenance.summary"
        assert summary["observation_only"] is True
        assert summary["training_intervention"] is False
        assert summary["defense_enabled"] is False
        assert summary["view_rejection_enabled"] is False
        assert summary["densification_gating_enabled"] is False
        assert summary["third_party_modified"] is False
        assert summary["drums_ready_for_pr212"] is False
        assert summary["exact_evidence_allowed_for_drums"] is False
        assert summary["proxy_selected_pixel_total"] == 6
        assert summary["residual_csv_available"] is True
        assert summary["residual_csv_schema_inferred"] is True

        expected = [
            "pr211g_pr20_selected_pixel_provenance_summary.json",
            "pr211g_pr20_selected_from_proxy_contributions.csv",
            "pr211g_pr20_residual_csv_schema_audit.csv",
            "pr211g_pr20_residual_to_selected_reproduction.csv",
            "pr211g_pr20_selected_pixel_membership_in_residual_csv.csv",
            "pr211g_pr20_code_provenance_audit.csv",
            "pr211g_pr20_code_provenance_summary.json",
            "pr211g_pr20_pixel_set_hash_comparison.csv",
            "pr211g_pr20_selected_pixel_provenance_diagnosis.csv",
            "pr211g_pr20_selected_pixel_provenance_report.md",
            "artifact_manifest.csv",
        ]
        for name in expected:
            assert (output / name).exists(), name

        proxy_rows = _read_csv(output / "pr211g_pr20_selected_from_proxy_contributions.csv")
        train004_proxy = next(row for row in proxy_rows if row["view_name"] == "train_004")
        assert train004_proxy["proxy_row_count"] == "4"
        assert train004_proxy["unique_selected_pixel_count"] == "2"
        assert train004_proxy["contributors_per_pixel_min"] == "2"
        assert train004_proxy["contributors_per_pixel_max"] == "2"

        schema_rows = _read_csv(output / "pr211g_pr20_residual_csv_schema_audit.csv")
        schema = schema_rows[0]
        assert schema["candidate_view_column"] == "view_name"
        assert schema["candidate_pixel_x_column"] == "pixel_x"
        assert schema["candidate_pixel_y_column"] == "pixel_y"
        assert "residual_l1" in schema["candidate_residual_score_columns_semicolon"]

        repro_rows = _read_csv(output / "pr211g_pr20_residual_to_selected_reproduction.csv")
        assert any(row["view_name"] == "train_004" and row["exact_reproduction"] == "true" for row in repro_rows)
        assert any(row["view_name"] == "train_012" and row["best_convention"] == "y_flip" and row["best_overlap_count"] == "2" for row in repro_rows)

        membership = _read_csv(output / "pr211g_pr20_selected_pixel_membership_in_residual_csv.csv")
        train004_member = next(row for row in membership if row["view_name"] == "train_004")
        train012_member = next(row for row in membership if row["view_name"] == "train_012")
        assert train004_member["residual_csv_same_pixel_count"] == "2"
        assert train004_member["best_membership_convention"] == "normal"
        assert train012_member["residual_csv_y_flip_count"] == "2"
        assert train012_member["best_membership_convention"] == "y_flip"

        hash_rows = _read_csv(output / "pr211g_pr20_pixel_set_hash_comparison.csv")
        assert any(row["view_name"] == "train_004" and row["hash_match"] == "true" for row in hash_rows)
        assert any("pr200_sparse_pixel_residuals.csv" in row["source_file"] and row["view_name"] == "train_009" and row["hash_match"] == "false" for row in hash_rows)

        diagnosis = _read_csv(output / "pr211g_pr20_selected_pixel_provenance_diagnosis.csv")
        assert any(row["view_name"] == "train_004" and row["provenance_status"] == "provenance_verified_from_residual_csv" for row in diagnosis)
        assert all(row["exact_evidence_allowed"] == "false" for row in diagnosis)
        assert all(row["drums_ready_for_pr212"] == "false" for row in diagnosis)

        code_summary = json.loads((output / "pr211g_pr20_code_provenance_summary.json").read_text(encoding="utf-8"))
        assert code_summary["provenance_confidence"] in {"high", "low"}
        assert "selected_pixel_generation_file" in code_summary

        manifest = _read_csv(output / "artifact_manifest.csv")
        assert any(row["relative_path"] == "pr211g_pr20_selected_pixel_provenance_summary.json" and row["exists"] == "true" for row in manifest)
        report = (output / "pr211g_pr20_selected_pixel_provenance_report.md").read_text(encoding="utf-8")
        assert "PR20 proxy rows are not exact contributor rows" in report

        bad_summary, bad_output = _run_builder(root / "mismatch", reproducible=False)
        assert bad_summary["provenance_status_overall"] == "provenance_unresolved_selected_pixels_not_reproduced"
        assert bad_summary["best_reproduction_overlap_total"] == 0
        assert bad_summary["drums_ready_for_pr212"] is False
        bad_diag = _read_csv(bad_output / "pr211g_pr20_selected_pixel_provenance_diagnosis.csv")
        assert all(row["provenance_status"] in {"provenance_verified_from_code_but_not_csv", "provenance_unresolved_selected_pixels_not_reproduced"} for row in bad_diag)

    print("pr211g PR20 selected-pixel provenance smoke test ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
