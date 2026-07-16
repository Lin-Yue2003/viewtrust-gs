#!/usr/bin/env python3
"""LOCAL-SAFE smoke test for PR21.3 chair exact-evidence positioning."""

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


def _make_inputs(root: Path) -> tuple[Path, Path, Path, Path]:
    pr211 = root / "pr211e_chair"
    pr212 = root / "pr212_chair"
    pr200 = root / "pr200_chair"
    pr201 = root / "pr201_chair"
    _write_json(
        pr211 / "pr211_exact_sparse_attribution_summary.json",
        {
            "scene": "chair",
            "condition": "corrupt_occluder",
            "evidence_quality": "exact_sparse_contributor_id_only",
            "exact_contributor_id_only_succeeded": True,
            "exact_contributor_id_row_count": 4,
            "exact_render_contribution_succeeded": False,
            "ready_for_intervention": False,
            "pr211_ready_for_pr212_comparison": True,
        },
    )
    _write_json(
        pr212 / "pr212_chair_exact_vs_proxy_summary.json",
        {
            "scene": "chair",
            "condition": "corrupt_occluder",
            "exact_evidence_quality": "exact_sparse_contributor_id_only",
            "exact_contributor_id_only_available": True,
            "exact_render_contribution_available": False,
            "exact_pixel_count": 2,
            "exact_row_count": 4,
            "exact_view_count_with_rows": 2,
            "mean_pixel_jaccard": 0.0,
            "mean_exact_recall_by_proxy": 0.0,
            "mean_proxy_precision_against_exact": 0.0,
            "direct_exact_unique_gaussian_count": 4,
            "collateral_exact_unique_gaussian_count": 0,
            "control_exact_unique_gaussian_count": 0,
            "pr212_ready_for_interpretation": True,
            "pr212_ready_for_intervention": False,
        },
    )
    _write_csv(
        pr212 / "pr212_chair_group_exact_overlap.csv",
        [
            {
                "group_a": "direct_corrupted",
                "group_b": "co_visible_collateral",
                "interpretation": "exact_evidence_unavailable_for_one_or_both_groups",
            },
            {
                "group_a": "direct_corrupted",
                "group_b": "clean_prior_demoted",
                "interpretation": "exact_evidence_unavailable_for_one_or_both_groups",
            },
        ],
    )
    _write_csv(
        pr212 / "pr212_chair_proxy_degeneracy_reassessment.csv",
        [
            {
                "claim_name": "proxy_degeneracy_confirmed",
                "exact_supports_proxy_claim": "false",
                "interpretation": "proxy_degeneracy_not_supported_by_exact_rows",
            }
        ],
    )
    _write_json(pr201 / "pr201_proxy_degeneracy_summary.json", {"proxy_degeneracy_confirmed": True})
    pr200.mkdir(parents=True, exist_ok=True)
    return pr211, pr212, pr200, pr201


def main() -> int:
    project_root = Path(__file__).resolve().parents[2]
    import sys

    sys.path.insert(0, str(project_root))
    from viewtrust.analysis.pr213_positioning import build_pr213_chair_exact_evidence_positioning

    with tempfile.TemporaryDirectory(prefix="viewtrust-pr213-") as tmp:
        root = Path(tmp)
        pr211, pr212, pr200, pr201 = _make_inputs(root)
        output = root / "out"
        summary, code = build_pr213_chair_exact_evidence_positioning(
            pr211_chair_dir=pr211,
            pr212_chair_dir=pr212,
            pr200_chair_dir=pr200,
            pr201_chair_dir=pr201,
            output_dir=output,
            scene="chair",
            condition="corrupt_occluder",
            subset_name="seed_20260710",
            write_markdown=True,
        )
        assert code == 0
        assert summary["schema_name"] == "viewtrust.pr213.chair_exact_evidence_positioning.summary"
        assert summary["observation_only"] is True
        assert summary["training_intervention"] is False
        assert summary["defense_enabled"] is False
        assert summary["view_rejection_enabled"] is False
        assert summary["densification_gating_enabled"] is False
        assert summary["third_party_modified"] is False
        assert summary["exact_contributor_id_only_available"] is True
        assert summary["exact_render_contribution_available"] is False
        assert summary["proxy_degeneracy_supported_by_exact"] is False
        assert summary["proxy_safe_for_intervention"] is False
        assert summary["drums_used_as_exact_evidence"] is False
        assert summary["direct_collateral_exact_overlap_established"] is False
        assert summary["train013_exact_control_separation_established"] is False
        assert summary["positioning_status"] == "exact_attribution_trust_signal_validation_not_defense"

        expected = [
            "pr213_chair_exact_evidence_positioning_summary.json",
            "pr213_chair_claim_table.csv",
            "pr213_chair_limitation_table.csv",
            "pr213_chair_exact_evidence_positioning_report.md",
            "pr213_paper_wording_snippets.md",
            "pr213_next_step_decision_memo.md",
            "artifact_manifest.csv",
        ]
        for name in expected:
            assert (output / name).exists(), name

        claims = _read_csv(output / "pr213_chair_claim_table.csv")
        assert len(claims) >= 6
        assert all(row["paper_safe_wording"] for row in claims)
        assert all(row["unsafe_wording_to_avoid"] for row in claims)
        c1 = next(row for row in claims if row["claim_id"] == "C1")
        assert c1["supported"] == "true"
        assert "chair exact-available pixels/views only" in c1["scope"]

        limitations = _read_csv(output / "pr213_chair_limitation_table.csv")
        assert any("contributor-ID only" in row["limitation"] for row in limitations)
        assert any("No intervention experiment" in row["limitation"] for row in limitations)

        snippets = (output / "pr213_paper_wording_snippets.md").read_text(encoding="utf-8").lower()
        forbidden = ["defense_enabled true", "reject views", "rejects bad views", "prevents poisoning", "detects poison", "removes attack"]
        assert not any(term in snippets for term in forbidden)
        assert "exact sparse contributor-id evidence" in snippets

        report = (output / "pr213_chair_exact_evidence_positioning_report.md").read_text(encoding="utf-8")
        assert "## What We Can Claim" in report
        assert "## What We Cannot Claim Yet" in report
        assert "ViewTrust-GS studies whether trust signals" in report

        memo = (output / "pr213_next_step_decision_memo.md").read_text(encoding="utf-8")
        assert "Do PR21.2a first, then PR21.4" in memo

        manifest = _read_csv(output / "artifact_manifest.csv")
        assert any(row["relative_path"] == "pr213_chair_exact_evidence_positioning_summary.json" and row["exists"] == "true" for row in manifest)

    print("pr213 chair exact-evidence positioning smoke test ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
