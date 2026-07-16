"""PR21.3 chair exact-evidence interpretation and positioning report."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from viewtrust.analysis.clean_prior_normalization import load_csv_rows, load_json, write_csv_rows, write_json


OUTPUT_FILES = [
    "pr213_chair_exact_evidence_positioning_summary.json",
    "pr213_chair_claim_table.csv",
    "pr213_chair_limitation_table.csv",
    "pr213_chair_exact_evidence_positioning_report.md",
    "pr213_paper_wording_snippets.md",
    "pr213_next_step_decision_memo.md",
    "artifact_manifest.csv",
]

CLAIM_FIELDS = [
    "claim_id",
    "claim",
    "evidence_source",
    "evidence_strength",
    "supported",
    "scope",
    "caveat",
    "paper_safe_wording",
    "unsafe_wording_to_avoid",
]

LIMITATION_FIELDS = ["limitation_id", "limitation", "why_it_matters", "affected_claims", "mitigation_or_next_step", "severity"]
MANIFEST_FIELDS = ["relative_path", "path", "exists", "file_type", "size_bytes", "required", "artifact_group"]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _bool_text(value: Any) -> str:
    return "true" if bool(value) else "false"


def _truth(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        if value in ("", None):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _warn_if(warnings: list[str], condition: bool, message: str) -> None:
    if condition:
        warnings.append(message)


def _validate_pr211(summary: dict[str, Any], scene: str, warnings: list[str]) -> None:
    _warn_if(warnings, summary.get("scene") != scene, f"PR21.1e scene is {summary.get('scene')!r}; expected {scene!r}")
    _warn_if(warnings, summary.get("evidence_quality") != "exact_sparse_contributor_id_only", "PR21.1e evidence_quality is not exact_sparse_contributor_id_only")
    _warn_if(warnings, not _truth(summary.get("exact_contributor_id_only_succeeded")), "PR21.1e exact_contributor_id_only_succeeded is not true")
    _warn_if(warnings, _safe_int(summary.get("exact_contributor_id_row_count")) <= 0, "PR21.1e exact_contributor_id_row_count is not > 0")
    _warn_if(warnings, _truth(summary.get("exact_render_contribution_succeeded")), "PR21.1e exact_render_contribution_succeeded should be false")
    _warn_if(warnings, _truth(summary.get("ready_for_intervention")), "PR21.1e ready_for_intervention should be false")
    if "pr211_ready_for_pr212_comparison" in summary:
        _warn_if(warnings, not _truth(summary.get("pr211_ready_for_pr212_comparison")), "PR21.1e pr211_ready_for_pr212_comparison is not true")


def _validate_pr212(summary: dict[str, Any], scene: str, warnings: list[str]) -> None:
    _warn_if(warnings, summary.get("scene") != scene, f"PR21.2 scene is {summary.get('scene')!r}; expected {scene!r}")
    _warn_if(warnings, not _truth(summary.get("exact_contributor_id_only_available")), "PR21.2 exact_contributor_id_only_available is not true")
    _warn_if(warnings, _truth(summary.get("exact_render_contribution_available")), "PR21.2 exact_render_contribution_available should be false")
    _warn_if(warnings, _safe_int(summary.get("exact_pixel_count")) <= 0, "PR21.2 exact_pixel_count is not > 0")
    _warn_if(warnings, _safe_int(summary.get("exact_row_count")) <= 0, "PR21.2 exact_row_count is not > 0")
    for key in ["mean_pixel_jaccard", "mean_exact_recall_by_proxy", "mean_proxy_precision_against_exact"]:
        value = _safe_float(summary.get(key))
        _warn_if(warnings, value is None or abs(value) > 1e-12, f"PR21.2 {key} is not zero")
    _warn_if(warnings, not _truth(summary.get("pr212_ready_for_interpretation")), "PR21.2 pr212_ready_for_interpretation is not true")
    _warn_if(warnings, _truth(summary.get("pr212_ready_for_intervention")), "PR21.2 pr212_ready_for_intervention should be false")


def _claim_rows() -> list[dict[str, Any]]:
    return [
        {
            "claim_id": "C1",
            "claim": "PR20 proxy degeneracy is not supported by chair exact contributor-ID evidence.",
            "evidence_source": "PR21.1e chair exact contributor-ID rows and PR21.2 exact-vs-proxy comparison",
            "evidence_strength": "strong within exact-available scope",
            "supported": "true",
            "scope": "chair exact-available pixels/views only",
            "caveat": "Does not prove every PR20 proxy row is globally wrong.",
            "paper_safe_wording": "For chair exact-available pixels/views, the proxy Gaussian pool identified by PR20/PR20.1 does not overlap with exact contributor IDs recovered by PR21.1e/PR21.2.",
            "unsafe_wording_to_avoid": "The proxy method is always wrong.",
        },
        {
            "claim_id": "C2",
            "claim": "PR20 proxy IDs should not be used for intervention.",
            "evidence_source": "Chair zero-overlap exact evidence plus unresolved drums alignment/provenance",
            "evidence_strength": "strong safety constraint",
            "supported": "true",
            "scope": "all current ViewTrust-GS experiments",
            "caveat": "Proxy rows remain useful as diagnostics, not action targets.",
            "paper_safe_wording": "Because proxy IDs do not match exact contributors in chair exact-available evidence and drums remains unresolved, proxy attribution should remain diagnostic-only and should not be used for reweighting, rejection, or densification gating.",
            "unsafe_wording_to_avoid": "Proxy candidates are confirmed false positives in every view.",
        },
        {
            "claim_id": "C3",
            "claim": "ViewTrust-GS is currently an exact attribution and trust-signal validation framework, not a defense.",
            "evidence_source": "Observation-only PR20 through PR21.3 artifacts",
            "evidence_strength": "method-positioning",
            "supported": "true",
            "scope": "current method version",
            "caveat": "No training-time intervention has been evaluated.",
            "paper_safe_wording": "The current contribution is an observation-only framework for validating trust signals against exact sparse contributor-ID evidence.",
            "unsafe_wording_to_avoid": "ViewTrust-GS defends 3DGS against attacks.",
        },
        {
            "claim_id": "C4",
            "claim": "Exact contributor-ID replay can falsify approximate proxy evidence.",
            "evidence_source": "PR21.2 chair exact-vs-proxy zero-overlap metrics",
            "evidence_strength": "strong within exact-available scope",
            "supported": "true",
            "scope": "chair exact-available pixels/views",
            "caveat": "Contributor magnitudes are not available yet.",
            "paper_safe_wording": "The chair result shows that exact contributor-ID replay can challenge proxy-based conclusions that looked plausible under approximate attribution.",
            "unsafe_wording_to_avoid": "Exact replay fully solves view trust.",
        },
        {
            "claim_id": "C5",
            "claim": "Direct-collateral exact overlap is not established.",
            "evidence_source": "PR21.2 exact rows availability",
            "evidence_strength": "supported limitation",
            "supported": "true",
            "scope": "chair current exact rows",
            "caveat": "Exact rows are unavailable for one or more compared groups.",
            "paper_safe_wording": "Because exact rows are unavailable for collateral/control views in the current chair result, direct-collateral and direct-control exact overlap claims should be deferred.",
            "unsafe_wording_to_avoid": "Collateral views share no exact contributors with direct views.",
        },
        {
            "claim_id": "C6",
            "claim": "The method is not intervention-ready.",
            "evidence_source": "PR21.1e/PR21.2 readiness flags and contributor-ID-only evidence",
            "evidence_strength": "strong safety constraint",
            "supported": "true",
            "scope": "current PR21.3",
            "caveat": "Future PRs may evaluate magnitude-aware attribution or interventions.",
            "paper_safe_wording": "PR21.3 remains interpretation-only and is not ready for training intervention.",
            "unsafe_wording_to_avoid": "We can now reject or downweight views.",
        },
    ]


def _limitation_rows() -> list[dict[str, Any]]:
    return [
        {
            "limitation_id": "L1",
            "limitation": "Exact evidence is contributor-ID only, without weights or magnitude.",
            "why_it_matters": "Contributor identity does not quantify alpha, transmittance, splat weight, or residual magnitude contribution.",
            "affected_claims": "C1;C4;C6",
            "mitigation_or_next_step": "PR21.4 exact contribution magnitude / alpha / transmittance-aware replay.",
            "severity": "high",
        },
        {
            "limitation_id": "L2",
            "limitation": "Exact rows are available only for a subset of chair views/pixels.",
            "why_it_matters": "The zero-overlap result is scoped to exact-available sparse evidence.",
            "affected_claims": "C1;C4",
            "mitigation_or_next_step": "Expand exact replay coverage after ID namespace validation.",
            "severity": "high",
        },
        {
            "limitation_id": "L3",
            "limitation": "No exact collateral/control rows are available for current direct-collateral/control claims.",
            "why_it_matters": "Group overlap and train013 separation claims cannot be made from absent exact rows.",
            "affected_claims": "C5",
            "mitigation_or_next_step": "Recover exact rows for collateral and control views or defer group claims.",
            "severity": "high",
        },
        {
            "limitation_id": "L4",
            "limitation": "Drums remains unresolved and excluded.",
            "why_it_matters": "Drums coordinate/provenance/exact alignment cannot support chair-only exact claims.",
            "affected_claims": "C1;C4",
            "mitigation_or_next_step": "Continue PR21.1f/PR21.1g drums alignment and provenance work separately.",
            "severity": "medium",
        },
        {
            "limitation_id": "L5",
            "limitation": "PR20 proxy rows cannot be used as exact evidence.",
            "why_it_matters": "Proxy candidate pools may look plausible but can disagree with exact contributor IDs.",
            "affected_claims": "C1;C2;C4",
            "mitigation_or_next_step": "Use proxy rows only as diagnostics until exact replay validates them.",
            "severity": "high",
        },
        {
            "limitation_id": "L6",
            "limitation": "Exact replay currently validates selected sparse pixels, not full image or full training trajectory evidence.",
            "why_it_matters": "Sparse validation is not equivalent to full-scene proof.",
            "affected_claims": "C1;C4",
            "mitigation_or_next_step": "Scale exact sparse replay or design representative sampling audits.",
            "severity": "medium",
        },
        {
            "limitation_id": "L7",
            "limitation": "No intervention experiment has been performed.",
            "why_it_matters": "Observation quality does not imply training-time utility or safety.",
            "affected_claims": "C2;C3;C6",
            "mitigation_or_next_step": "Only after exact evidence matures, design separate intervention experiments.",
            "severity": "high",
        },
    ]


def _write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# PR21.3 Chair Exact Evidence Positioning",
        "",
        "## Executive Summary",
        "",
        "PR20 proxy attribution produced plausible but approximate Gaussian candidate pools. PR21.1e then produced exact sparse contributor-ID evidence for chair through per-view replay, and PR21.2 found zero overlap between exact contributors and PR20 proxy candidates on exact-available chair pixels/views.",
        "",
        "This means the current ViewTrust-GS contribution should be positioned as exact trust-signal validation rather than training-time defense. The evidence can challenge proxy-based conclusions, but it does not yet justify view rejection, reweighting, densification gating, or any other intervention.",
        "",
        "## Background: Why Proxy Attribution Was Not Enough",
        "",
        "PR20 proxy attribution was approximate: it connected sparse residual pixels to view/event-weighted Gaussian candidate pools rather than replaying the renderer to recover actual contributing Gaussian IDs. PR20.1 showed proxy degeneracy, but proxy degeneracy alone could not establish actual rendered contributors. Proxy evidence must therefore remain diagnostic and must not drive intervention.",
        "",
        "## Exact Evidence from PR21.1e",
        "",
        "PR21.1e enabled per-view exact sparse contributor-ID replay for chair. The available evidence quality is contributor-ID-only: exact Gaussian IDs are recovered for selected pixels, but exact contribution magnitudes, alpha values, transmittance, and splat weights are not yet available.",
        "",
        "## Exact-vs-Proxy Result from PR21.2",
        "",
        f"PR21.2 reports `{summary.get('exact_pixel_count')}` exact pixels, `{summary.get('exact_row_count')}` exact rows, and `{summary.get('exact_view_count_with_rows')}` views with exact rows. On exact-available chair pixels/views, mean pixel Jaccard, exact recall by proxy, and proxy precision against exact are all `{summary.get('mean_pixel_jaccard')}`.",
        "",
        "The proxy degeneracy observed in PR20/PR20.1 is not supported by the exact contributor-ID evidence for the exact-available chair scope.",
        "",
        "## What We Can Claim",
        "",
        "- For chair exact-available pixels/views, PR20 proxy candidate IDs do not match exact contributor IDs.",
        "- PR20/PR20.1 proxy degeneracy is not supported by chair exact contributor-ID evidence.",
        "- Exact replay is useful for validating or falsifying trust signals.",
        "- ViewTrust-GS is better framed as trust-aware exact attribution and evidence validation.",
        "",
        "## What We Cannot Claim Yet",
        "",
        "- No defense or intervention claim.",
        "- No exact contribution magnitude claim.",
        "- No full-scene or full-pixel proof.",
        "- No direct-collateral exact overlap claim.",
        "- No train013 exact separation claim.",
        "- No drums exact claim.",
        "",
        "## Updated ViewTrust-GS Positioning",
        "",
        "ViewTrust-GS studies whether trust signals derived from suspicious views correspond to actual rendered Gaussian contributors. Instead of treating view-level residuals or approximate proxy candidate pools as direct evidence, it introduces an exact sparse contributor-ID replay audit that tests whether proxy-attributed Gaussians are actual participants in the render. The chair result shows that plausible proxy degeneracy can disappear under exact contributor-ID evidence, motivating exact trust-signal validation before any intervention.",
        "",
        "## Relation to RobustNeRF / RobustGS / Poison-3DGS",
        "",
        "RobustNeRF and RobustGS focus on robust reconstruction or natural distractors, while Poison-3DGS studies poisoning and attack behavior. At this stage, ViewTrust-GS is not a poison classifier and not a robust training defense. It provides evidence-level validation of which Gaussians actually participate in suspicious render residuals.",
        "",
        "## Next Step",
        "",
        "Recommended default: run PR21.2a ID namespace audit first, then PR21.4 exact contribution magnitude if the namespace is validated. Zero-overlap is central to the chair story, so ID namespace validation should be closed before building a stronger magnitude-based attribution layer.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_snippets(path: Path) -> None:
    lines = [
        "# PR21.3 Paper Wording Snippets",
        "",
        "## Abstract-Style Contribution",
        "",
        "ViewTrust-GS introduces an observation-only audit for validating trust signals in 3D Gaussian Splatting against exact sparse contributor-ID evidence. On a chair scene with exact-available pixels/views, the Gaussian IDs recovered by exact replay show zero overlap with an approximate proxy candidate pool, demonstrating that plausible proxy evidence can be falsified by renderer-grounded contributor IDs.",
        "",
        "## Method Positioning",
        "",
        "Rather than treating residual-based proxy candidates as direct evidence, ViewTrust-GS separates approximate trust-signal generation from exact sparse contributor-ID validation. This separation lets the analysis test whether suspicious-view signals correspond to Gaussians that actually participate in the render.",
        "",
        "## Limitation",
        "",
        "The current exact evidence is contributor-ID-only and sparse. It does not provide alpha, transmittance, splat-weight, or residual-weighted contribution magnitudes, and it does not yet cover every view or pixel.",
        "",
        "## Chair Result",
        "",
        "For chair exact-available pixels/views, PR21.2 finds zero overlap between PR20 proxy Gaussian candidates and exact contributor IDs recovered by PR21.1e. This result indicates that the approximate proxy pool is not supported by exact contributor-ID evidence in the evaluated scope.",
        "",
        "## Conservative Conclusion",
        "",
        "These results motivate exact trust-signal validation before any training-time action. The current evidence supports interpretation and auditing, while stronger contribution magnitudes and broader coverage remain future work.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_next_step_memo(path: Path) -> None:
    lines = [
        "# PR21.3 Next-Step Decision Memo",
        "",
        "## Option A: PR21.2a ID Namespace Audit",
        "",
        "Purpose: Verify exact IDs and proxy IDs refer to the same Gaussian index namespace before turning zero-overlap into a paper claim.",
        "",
        "Pros:",
        "- Cheap.",
        "- High value.",
        "- Strengthens PR21.2 claim.",
        "- Avoids reviewer concern.",
        "",
        "Cons:",
        "- Does not improve exact contribution magnitude.",
        "- Still chair-only.",
        "",
        "## Option B: PR21.4 Exact Contribution Magnitude",
        "",
        "Purpose: Move from contributor-ID-only to alpha/transmittance/splat-weight-aware exact contributions.",
        "",
        "Pros:",
        "- Stronger technical contribution.",
        "- More useful for future intervention research.",
        "- Moves closer to trust scoring.",
        "",
        "Cons:",
        "- More implementation risk.",
        "- Requires more careful gsplat internal replay.",
        "- May be slower.",
        "",
        "## Recommendation",
        "",
        "Do PR21.2a first, then PR21.4. Zero-overlap is central to the current chair story, so ID namespace validation should be closed before building a stronger magnitude-based attribution layer.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _manifest_rows(output_dir: Path, inputs: list[tuple[str, Path | None, bool]]) -> list[dict[str, Any]]:
    items: list[tuple[str, Path | None, bool, str]] = [(name, path, required, "input" if required else "input_optional") for name, path, required in inputs]
    items.extend((name, output_dir / name, True, "output_pr213") for name in OUTPUT_FILES)
    rows = []
    for relative, path, required, group in items:
        exists = bool(path and path.exists())
        rows.append(
            {
                "relative_path": relative,
                "path": str(path or ""),
                "exists": _bool_text(exists),
                "file_type": "directory" if path and path.is_dir() else path.suffix.lstrip(".") if path else "",
                "size_bytes": path.stat().st_size if path and path.is_file() else "",
                "required": _bool_text(required),
                "artifact_group": group,
            }
        )
    return rows


def build_pr213_chair_exact_evidence_positioning(
    *,
    pr211_chair_dir: Path,
    pr212_chair_dir: Path,
    pr200_chair_dir: Path,
    output_dir: Path,
    pr201_chair_dir: Path | None = None,
    pr211_drums_dir: Path | None = None,
    pr211fa_drums_dir: Path | None = None,
    pr211g_drums_dir: Path | None = None,
    scene: str = "chair",
    condition: str = "corrupt_occluder",
    subset_name: str = "seed_20260710",
    write_markdown: bool = False,
) -> tuple[dict[str, Any], int]:
    del write_markdown
    output_dir.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []
    pr211_summary = load_json(pr211_chair_dir / "pr211_exact_sparse_attribution_summary.json")
    pr212_summary = load_json(pr212_chair_dir / "pr212_chair_exact_vs_proxy_summary.json")
    _validate_pr211(pr211_summary, scene, warnings)
    _validate_pr212(pr212_summary, scene, warnings)
    claim_rows = _claim_rows()
    limitation_rows = _limitation_rows()
    group_rows = load_csv_rows(pr212_chair_dir / "pr212_chair_group_exact_overlap.csv")
    exact_direct_collateral = any(
        row.get("group_a") == "direct_corrupted"
        and row.get("group_b") == "co_visible_collateral"
        and row.get("interpretation") != "exact_evidence_unavailable_for_one_or_both_groups"
        for row in group_rows
    )
    train013_established = _safe_int(pr212_summary.get("control_exact_unique_gaussian_count")) > 0
    mean_jaccard = _safe_float(pr212_summary.get("mean_pixel_jaccard"), 0.0)
    mean_recall = _safe_float(pr212_summary.get("mean_exact_recall_by_proxy"), 0.0)
    mean_precision = _safe_float(pr212_summary.get("mean_proxy_precision_against_exact"), 0.0)
    proxy_supported = not (mean_jaccard == 0.0 and mean_recall == 0.0 and mean_precision == 0.0)
    summary = {
        "schema_name": "viewtrust.pr213.chair_exact_evidence_positioning.summary",
        "schema_version": 1,
        "created_at_utc": _utc_now(),
        "scene": scene,
        "condition": condition,
        "subset_name": subset_name,
        "observation_only": True,
        "training_intervention": False,
        "defense_enabled": False,
        "view_rejection_enabled": False,
        "densification_gating_enabled": False,
        "third_party_modified": False,
        "pr211_chair_input_dir": str(pr211_chair_dir),
        "pr212_chair_input_dir": str(pr212_chair_dir),
        "pr200_chair_input_dir": str(pr200_chair_dir),
        "pr201_chair_input_dir": str(pr201_chair_dir or ""),
        "exact_evidence_quality": pr212_summary.get("exact_evidence_quality") or pr211_summary.get("evidence_quality", ""),
        "exact_contributor_id_only_available": _truth(pr212_summary.get("exact_contributor_id_only_available")),
        "exact_render_contribution_available": _truth(pr212_summary.get("exact_render_contribution_available")),
        "exact_pixel_count": _safe_int(pr212_summary.get("exact_pixel_count")),
        "exact_row_count": _safe_int(pr212_summary.get("exact_row_count")),
        "exact_view_count_with_rows": _safe_int(pr212_summary.get("exact_view_count_with_rows")),
        "mean_pixel_jaccard": mean_jaccard,
        "mean_exact_recall_by_proxy": mean_recall,
        "mean_proxy_precision_against_exact": mean_precision,
        "proxy_degeneracy_supported_by_exact": proxy_supported,
        "proxy_safe_for_intervention": False,
        "direct_collateral_exact_overlap_established": exact_direct_collateral,
        "train013_exact_control_separation_established": train013_established,
        "drums_used_as_exact_evidence": False,
        "claim_count": len(claim_rows),
        "supported_claim_count": sum(1 for row in claim_rows if row["supported"] == "true"),
        "limitation_count": len(limitation_rows),
        "positioning_status": "exact_attribution_trust_signal_validation_not_defense",
        "recommended_next_step": "Run PR21.2a ID namespace audit first, then PR21.4 exact contribution magnitude if the namespace is validated.",
        "warnings": warnings,
    }
    write_json(output_dir / "pr213_chair_exact_evidence_positioning_summary.json", summary)
    write_csv_rows(output_dir / "pr213_chair_claim_table.csv", claim_rows, CLAIM_FIELDS)
    write_csv_rows(output_dir / "pr213_chair_limitation_table.csv", limitation_rows, LIMITATION_FIELDS)
    _write_report(output_dir / "pr213_chair_exact_evidence_positioning_report.md", summary)
    _write_snippets(output_dir / "pr213_paper_wording_snippets.md")
    _write_next_step_memo(output_dir / "pr213_next_step_decision_memo.md")
    inputs = [
        ("pr211_chair_dir", pr211_chair_dir, True),
        ("pr212_chair_dir", pr212_chair_dir, True),
        ("pr200_chair_dir", pr200_chair_dir, True),
        ("pr201_chair_dir", pr201_chair_dir, False),
        ("pr211_drums_dir", pr211_drums_dir, False),
        ("pr211fa_drums_dir", pr211fa_drums_dir, False),
        ("pr211g_drums_dir", pr211g_drums_dir, False),
    ]
    manifest = output_dir / "artifact_manifest.csv"
    write_csv_rows(manifest, _manifest_rows(output_dir, inputs), MANIFEST_FIELDS)
    write_csv_rows(manifest, _manifest_rows(output_dir, inputs), MANIFEST_FIELDS)
    return summary, 0
