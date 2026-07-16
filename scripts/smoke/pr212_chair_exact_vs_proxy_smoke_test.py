#!/usr/bin/env python3
"""LOCAL-SAFE smoke test for PR21.2 chair exact-vs-proxy comparison."""

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


def _make_inputs(root: Path, *, valid_exact: bool = True) -> tuple[Path, Path, Path]:
    pr200 = root / "pr200"
    pr211 = root / "pr211"
    pr201 = root / "pr201"
    _write_json(
        pr211 / "pr211_exact_sparse_attribution_summary.json",
        {
            "scene": "chair",
            "condition": "corrupt_occluder",
            "exact_attribution_succeeded": valid_exact,
            "evidence_quality": "exact_sparse_contributor_id_only",
            "exact_contributor_id_only_succeeded": valid_exact,
            "exact_contributor_id_row_count": 3 if valid_exact else 0,
            "exact_render_contribution_succeeded": False,
            "ready_for_intervention": False,
        },
    )
    exact_rows = [
        {
            "scene": "chair",
            "condition": "corrupt_occluder",
            "subset_name": "seed_20260710",
            "view_name": "train_009",
            "view_group": "direct_corrupted",
            "pixel_x": 1,
            "pixel_y": 2,
            "pixel_id": 801,
            "gaussian_id": 10,
            "residual_l1": 0.5,
            "evidence_quality": "exact_sparse_contributor_id_only",
        },
        {
            "scene": "chair",
            "condition": "corrupt_occluder",
            "subset_name": "seed_20260710",
            "view_name": "train_009",
            "view_group": "direct_corrupted",
            "pixel_x": 1,
            "pixel_y": 2,
            "pixel_id": 801,
            "gaussian_id": 20,
            "residual_l1": 0.5,
            "evidence_quality": "exact_sparse_contributor_id_only",
        },
        {
            "scene": "chair",
            "condition": "corrupt_occluder",
            "subset_name": "seed_20260710",
            "view_name": "train_012",
            "view_group": "direct_corrupted",
            "pixel_x": 3,
            "pixel_y": 4,
            "pixel_id": 1603,
            "gaussian_id": 30,
            "residual_l1": 0.7,
            "evidence_quality": "exact_sparse_contributor_id_only",
        },
    ]
    _write_csv(pr211 / "pr211_exact_pixel_gaussian_contributions.csv", exact_rows if valid_exact else [])
    _write_csv(pr211 / "pr211_per_view_replay_audit.csv", [{"view_name": "train_009", "succeeded": "true"}])
    proxy_rows = [
        {"view_name": "train_009", "pixel_x": 1, "pixel_y": 2, "gaussian_id": 10, "splat_weight": 0.5},
        {"view_name": "train_009", "pixel_x": 1, "pixel_y": 2, "gaussian_id": 99, "splat_weight": 0.5},
        {"view_name": "train_012", "pixel_x": 3, "pixel_y": 4, "gaussian_id": 40, "splat_weight": 1.0},
        {"view_name": "train_014", "pixel_x": 5, "pixel_y": 6, "gaussian_id": 999, "splat_weight": 1.0},
    ]
    _write_csv(pr200 / "pr200_pixel_gaussian_contributions.csv", proxy_rows)
    _write_json(pr201 / "pr201_proxy_degeneracy_summary.json", {"proxy_degeneracy_confirmed": True})
    return pr200, pr211, pr201


def main() -> int:
    project_root = Path(__file__).resolve().parents[2]
    import sys

    sys.path.insert(0, str(project_root))
    from viewtrust.analysis.pr212_exact_vs_proxy import build_pr212_chair_exact_vs_proxy_comparison

    with tempfile.TemporaryDirectory(prefix="viewtrust-pr212-") as tmp:
        root = Path(tmp)
        pr200, pr211, pr201 = _make_inputs(root)
        output = root / "out"
        summary, code = build_pr212_chair_exact_vs_proxy_comparison(
            pr200_dir=pr200,
            pr211_dir=pr211,
            pr201_dir=pr201,
            output_dir=output,
            scene="chair",
            condition="corrupt_occluder",
            subset_name="seed_20260710",
            write_markdown=True,
        )
        assert code == 0
        assert summary["schema_name"] == "viewtrust.pr212.chair_exact_vs_proxy.summary"
        assert summary["observation_only"] is True
        assert summary["training_intervention"] is False
        assert summary["defense_enabled"] is False
        assert summary["view_rejection_enabled"] is False
        assert summary["densification_gating_enabled"] is False
        assert summary["third_party_modified"] is False
        assert summary["exact_render_contribution_available"] is False
        assert summary["exact_contributor_id_only_available"] is True
        assert summary["pr212_ready_for_intervention"] is False
        assert summary["exact_pixel_count"] == 2
        assert summary["exact_row_count"] == 3
        assert summary["proxy_pixel_count_for_exact_pixels"] == 2

        pixel_rows = list(csv.DictReader((output / "pr212_chair_pixel_exact_vs_proxy.csv").open()))
        assert len(pixel_rows) == 2
        first = next(row for row in pixel_rows if row["view_name"] == "train_009")
        assert first["exact_count"] == "2"
        assert first["proxy_count"] == "2"
        assert first["intersection_count"] == "1"
        assert abs(float(first["jaccard"]) - (1 / 3)) < 1e-9
        assert first["exact_recall_by_proxy"] == "0.5"
        assert first["proxy_precision_against_exact"] == "0.5"
        assert first["evidence_quality_exact"] == "exact_sparse_contributor_id_only"
        assert first["evidence_quality_proxy"] == "view_event_weighted_gaussian_proxy"

        view_rows = list(csv.DictReader((output / "pr212_chair_view_exact_vs_proxy.csv").open()))
        train014 = next(row for row in view_rows if row["view_name"] == "train_014")
        assert train014["exact_view_has_rows"] == "false"
        assert train014["interpretation"] == "exact_unavailable_for_view_proxy_not_validated"
        assert train014["proxy_unique_gaussian_count"] == "1"

        group_rows = list(csv.DictReader((output / "pr212_chair_group_exact_overlap.csv").open()))
        assert any(row["interpretation"] == "exact_evidence_unavailable_for_one_or_both_groups" for row in group_rows)

        reassessment = list(csv.DictReader((output / "pr212_chair_proxy_degeneracy_reassessment.csv").open()))
        assert reassessment
        assert any(row["interpretation"] == "proxy_degeneracy_not_supported_by_exact_rows" for row in reassessment)

        manifest = list(csv.DictReader((output / "artifact_manifest.csv").open()))
        assert any(row["relative_path"] == "pr212_chair_exact_vs_proxy_summary.json" and row["exists"] == "true" for row in manifest)
        report = (output / "pr212_chair_exact_vs_proxy_report.md").read_text(encoding="utf-8")
        assert "contributor-ID-only exact evidence" in report

        bad_pr200, bad_pr211, _ = _make_inputs(root / "bad", valid_exact=False)
        try:
            build_pr212_chair_exact_vs_proxy_comparison(
                pr200_dir=bad_pr200,
                pr211_dir=bad_pr211,
                output_dir=root / "bad_out",
                scene="chair",
                condition="corrupt_occluder",
                subset_name="seed_20260710",
            )
        except ValueError as exc:
            assert "exact_attribution_succeeded must be true" in str(exc)
        else:
            raise AssertionError("invalid exact input should fail")

    print("pr212 chair exact-vs-proxy smoke test ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
