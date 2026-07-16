"""PR21.2 chair-only exact-vs-proxy contributor-ID comparison."""

from __future__ import annotations

import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from viewtrust.analysis.clean_prior_normalization import load_csv_rows, load_json, write_csv_rows, write_json


PIXEL_FIELDS = [
    "scene",
    "condition",
    "subset_name",
    "view_name",
    "view_group",
    "pixel_x",
    "pixel_y",
    "pixel_id",
    "exact_gaussian_ids_semicolon",
    "proxy_gaussian_ids_semicolon",
    "intersection_gaussian_ids_semicolon",
    "exact_only_gaussian_ids_semicolon",
    "proxy_only_gaussian_ids_semicolon",
    "exact_count",
    "proxy_count",
    "intersection_count",
    "exact_only_count",
    "proxy_only_count",
    "jaccard",
    "exact_recall_by_proxy",
    "proxy_precision_against_exact",
    "residual_l1",
    "evidence_quality_exact",
    "evidence_quality_proxy",
    "interpretation",
]

VIEW_FIELDS = [
    "scene",
    "condition",
    "subset_name",
    "view_name",
    "view_group",
    "exact_view_has_rows",
    "exact_unique_gaussian_count",
    "proxy_unique_gaussian_count",
    "intersection_count",
    "exact_only_count",
    "proxy_only_count",
    "jaccard",
    "exact_recall_by_proxy",
    "proxy_precision_against_exact",
    "exact_gaussian_ids_semicolon",
    "proxy_gaussian_ids_semicolon",
    "intersection_gaussian_ids_semicolon",
    "exact_only_gaussian_ids_semicolon",
    "proxy_only_gaussian_ids_semicolon",
    "interpretation",
]

GROUP_FIELDS = [
    "scene",
    "condition",
    "subset_name",
    "group_a",
    "group_b",
    "exact_group_a_count",
    "exact_group_b_count",
    "exact_overlap_count",
    "exact_jaccard",
    "proxy_group_a_count",
    "proxy_group_b_count",
    "proxy_overlap_count",
    "proxy_jaccard",
    "interpretation",
]

DEGENERACY_FIELDS = [
    "scene",
    "condition",
    "subset_name",
    "claim_name",
    "proxy_claim_value",
    "exact_observation_value",
    "exact_evidence_available",
    "exact_supports_proxy_claim",
    "interpretation",
    "caveat",
]

MANIFEST_FIELDS = ["relative_path", "path", "exists", "file_type", "size_bytes", "required", "artifact_group"]

OUTPUT_FILES = [
    "pr212_chair_exact_vs_proxy_summary.json",
    "pr212_chair_pixel_exact_vs_proxy.csv",
    "pr212_chair_view_exact_vs_proxy.csv",
    "pr212_chair_group_exact_overlap.csv",
    "pr212_chair_proxy_degeneracy_reassessment.csv",
    "pr212_chair_exact_vs_proxy_report.md",
    "artifact_manifest.csv",
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _bool_text(value: Any) -> str:
    return "true" if bool(value) else "false"


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def _fmt_ids(values: set[str]) -> str:
    def key(value: str) -> tuple[int, Any]:
        try:
            return (0, int(value))
        except ValueError:
            return (1, value)

    return ";".join(sorted((str(value) for value in values if str(value) != ""), key=key))


def _ratio(num: int, den: int) -> float | None:
    return num / den if den else None


def _jaccard(a: set[str], b: set[str]) -> float | None:
    union = a | b
    return len(a & b) / len(union) if union else None


def _group_for_view(view: str, direct_views: list[str], collateral_views: list[str], control_views: list[str]) -> str:
    if view in direct_views:
        return "direct_corrupted"
    if view in collateral_views:
        return "co_visible_collateral"
    if view in control_views:
        return "clean_prior_demoted"
    return "other_clean"


def _pixel_key(row: dict[str, Any]) -> tuple[str, int, int]:
    return str(row.get("view_name", "")), _safe_int(row.get("pixel_x")), _safe_int(row.get("pixel_y"))


def _validate_exact_summary(summary: dict[str, Any], scene: str, condition: str) -> None:
    checks = [
        (summary.get("scene") == scene, f"exact summary scene is {summary.get('scene')}, expected {scene}"),
        (summary.get("condition") == condition, f"exact summary condition is {summary.get('condition')}, expected {condition}"),
        (summary.get("exact_attribution_succeeded") is True, "exact_attribution_succeeded must be true"),
        (summary.get("evidence_quality") == "exact_sparse_contributor_id_only", "evidence_quality must be exact_sparse_contributor_id_only"),
        (summary.get("exact_contributor_id_only_succeeded") is True, "exact_contributor_id_only_succeeded must be true"),
        (_safe_int(summary.get("exact_contributor_id_row_count")) > 0, "exact_contributor_id_row_count must be > 0"),
        (summary.get("exact_render_contribution_succeeded") is False, "exact_render_contribution_succeeded must be false"),
        (summary.get("ready_for_intervention") is False, "ready_for_intervention must be false"),
    ]
    failed = [message for ok, message in checks if not ok]
    if failed:
        raise ValueError("PR21.2 exact input validation failed: " + "; ".join(failed))


def _build_sets(rows: list[dict[str, Any]]) -> tuple[dict[tuple[str, int, int], set[str]], dict[str, set[str]], dict[tuple[str, int, int], dict[str, Any]]]:
    by_pixel: dict[tuple[str, int, int], set[str]] = {}
    by_view: dict[str, set[str]] = {}
    first_row: dict[tuple[str, int, int], dict[str, Any]] = {}
    for row in rows:
        gid = str(row.get("gaussian_id", ""))
        if gid == "":
            continue
        key = _pixel_key(row)
        by_pixel.setdefault(key, set()).add(gid)
        by_view.setdefault(key[0], set()).add(gid)
        first_row.setdefault(key, row)
    return by_pixel, by_view, first_row


def _compare_sets(exact: set[str], proxy: set[str]) -> dict[str, Any]:
    inter = exact & proxy
    exact_only = exact - proxy
    proxy_only = proxy - exact
    return {
        "intersection": inter,
        "exact_only": exact_only,
        "proxy_only": proxy_only,
        "jaccard": _jaccard(exact, proxy),
        "exact_recall_by_proxy": _ratio(len(inter), len(exact)),
        "proxy_precision_against_exact": _ratio(len(inter), len(proxy)),
    }


def _pixel_interpretation(exact: set[str], proxy: set[str]) -> str:
    inter = exact & proxy
    if not inter:
        return "proxy_no_overlap_with_exact"
    if inter == exact and proxy <= exact:
        return "proxy_matches_exact_subset"
    if exact - proxy and proxy - exact:
        return "proxy_partial_overlap"
    if exact - proxy:
        return "proxy_misses_exact_contributors"
    return "proxy_contains_non_exact_candidates"


def _artifact_rows(output_dir: Path, pr200_dir: Path, pr211_dir: Path, pr201_dir: Path | None) -> list[dict[str, Any]]:
    items: list[tuple[str, Path, bool, str]] = [
        ("pr200_dir", pr200_dir, True, "input"),
        ("pr211_dir", pr211_dir, True, "input"),
    ]
    if pr201_dir is not None:
        items.append(("pr201_dir", pr201_dir, False, "input_optional"))
    items.extend((name, output_dir / name, True, "output_pr212") for name in OUTPUT_FILES)
    rows = []
    for relative, path, required, group in items:
        rows.append(
            {
                "relative_path": relative,
                "path": str(path),
                "exists": _bool_text(path.exists()),
                "file_type": "directory" if path.is_dir() else path.suffix.lstrip("."),
                "size_bytes": path.stat().st_size if path.is_file() else "",
                "required": _bool_text(required),
                "artifact_group": group,
            }
        )
    return rows


def _write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# PR21.2 Chair Exact-vs-Proxy Contributor-ID Comparison",
        "",
        "PR21.2 is observation-only. It does not implement defense, view rejection, reweighting, update suppression, or densification gating.",
        "",
        "This remains contributor-ID-only exact evidence. It does not provide exact splat weights, alpha contributions, transmittance, or render contribution magnitudes.",
        "",
        "## Inputs",
        f"- Exact input: `{summary.get('exact_input_dir')}`",
        f"- Proxy input: `{summary.get('proxy_input_dir')}`",
        f"- PR20.1 input: `{summary.get('pr201_input_dir')}`",
        "",
        "## Pixel-Level Findings",
        f"- Exact pixels: `{summary.get('exact_pixel_count')}`",
        f"- Exact rows: `{summary.get('exact_row_count')}`",
        f"- Mean pixel Jaccard: `{summary.get('mean_pixel_jaccard')}`",
        f"- Mean exact recall by proxy: `{summary.get('mean_exact_recall_by_proxy')}`",
        f"- Mean proxy precision against exact: `{summary.get('mean_proxy_precision_against_exact')}`",
        "",
        "## View-Level Findings",
        f"- Exact view count with rows: `{summary.get('exact_view_count_with_rows')}`",
        f"- View mean Jaccard for exact-available views: `{summary.get('view_mean_jaccard_for_exact_available_views')}`",
        "",
        "## Direct / Collateral / Control",
        f"- Direct exact unique Gaussians: `{summary.get('direct_exact_unique_gaussian_count')}`",
        f"- Collateral exact unique Gaussians: `{summary.get('collateral_exact_unique_gaussian_count')}`",
        f"- Control exact unique Gaussians: `{summary.get('control_exact_unique_gaussian_count')}`",
        f"- Direct/collateral exact Jaccard: `{summary.get('direct_collateral_exact_jaccard')}`",
        "",
        "## Proxy Degeneracy Reassessment",
        "See `pr212_chair_proxy_degeneracy_reassessment.csv`.",
        "",
        "## Caveats",
        "Views without exact rows cannot validate or falsify proxy candidates for that view. Drums is excluded from PR21.2 exact evidence until coordinate alignment is resolved.",
        "",
        "## Recommended Next Step",
        str(summary.get("recommended_next_step", "")),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def build_pr212_chair_exact_vs_proxy_comparison(
    *,
    pr200_dir: Path,
    pr211_dir: Path,
    output_dir: Path,
    scene: str = "chair",
    condition: str = "corrupt_occluder",
    subset_name: str = "seed_20260710",
    pr201_dir: Path | None = None,
    direct_views: list[str] | None = None,
    collateral_views: list[str] | None = None,
    control_views: list[str] | None = None,
    write_markdown: bool = False,
) -> tuple[dict[str, Any], int]:
    del write_markdown
    direct_views = direct_views or ["train_004", "train_009", "train_012", "train_017"]
    collateral_views = collateral_views or ["train_014"]
    control_views = control_views or ["train_013"]
    selected_views = direct_views + collateral_views + control_views
    output_dir.mkdir(parents=True, exist_ok=True)

    exact_summary = load_json(pr211_dir / "pr211_exact_sparse_attribution_summary.json")
    _validate_exact_summary(exact_summary, scene, condition)
    exact_rows = load_csv_rows(pr211_dir / "pr211_exact_pixel_gaussian_contributions.csv")
    proxy_rows = load_csv_rows(pr200_dir / "pr200_pixel_gaussian_contributions.csv")
    if not (pr211_dir / "pr211_per_view_replay_audit.csv").is_file():
        raise ValueError("missing required exact input: pr211_per_view_replay_audit.csv")

    exact_by_pixel, exact_by_view, exact_first = _build_sets(exact_rows)
    proxy_by_pixel, proxy_by_view, _ = _build_sets(proxy_rows)

    pixel_rows = []
    pixel_metrics: list[float] = []
    recall_metrics: list[float] = []
    precision_metrics: list[float] = []
    for key in sorted(exact_by_pixel):
        view, x, y = key
        exact = exact_by_pixel[key]
        proxy = proxy_by_pixel.get(key, set())
        cmp = _compare_sets(exact, proxy)
        first = exact_first.get(key, {})
        pixel_id = first.get("pixel_id", _safe_int(y) * 400 + _safe_int(x))
        row = {
            "scene": scene,
            "condition": condition,
            "subset_name": subset_name,
            "view_name": view,
            "view_group": first.get("view_group") or _group_for_view(view, direct_views, collateral_views, control_views),
            "pixel_x": x,
            "pixel_y": y,
            "pixel_id": pixel_id,
            "exact_gaussian_ids_semicolon": _fmt_ids(exact),
            "proxy_gaussian_ids_semicolon": _fmt_ids(proxy),
            "intersection_gaussian_ids_semicolon": _fmt_ids(cmp["intersection"]),
            "exact_only_gaussian_ids_semicolon": _fmt_ids(cmp["exact_only"]),
            "proxy_only_gaussian_ids_semicolon": _fmt_ids(cmp["proxy_only"]),
            "exact_count": len(exact),
            "proxy_count": len(proxy),
            "intersection_count": len(cmp["intersection"]),
            "exact_only_count": len(cmp["exact_only"]),
            "proxy_only_count": len(cmp["proxy_only"]),
            "jaccard": cmp["jaccard"],
            "exact_recall_by_proxy": cmp["exact_recall_by_proxy"],
            "proxy_precision_against_exact": cmp["proxy_precision_against_exact"],
            "residual_l1": first.get("residual_l1", ""),
            "evidence_quality_exact": "exact_sparse_contributor_id_only",
            "evidence_quality_proxy": "view_event_weighted_gaussian_proxy",
            "interpretation": _pixel_interpretation(exact, proxy),
        }
        pixel_rows.append(row)
        if cmp["jaccard"] is not None:
            pixel_metrics.append(float(cmp["jaccard"]))
        if cmp["exact_recall_by_proxy"] is not None:
            recall_metrics.append(float(cmp["exact_recall_by_proxy"]))
        if cmp["proxy_precision_against_exact"] is not None:
            precision_metrics.append(float(cmp["proxy_precision_against_exact"]))

    view_rows = []
    view_jaccards = []
    for view in selected_views:
        exact = exact_by_view.get(view, set())
        proxy = proxy_by_view.get(view, set())
        cmp = _compare_sets(exact, proxy)
        has_exact = bool(exact)
        if has_exact and cmp["jaccard"] is not None:
            view_jaccards.append(float(cmp["jaccard"]))
        interpretation = "exact_unavailable_for_view_proxy_not_validated" if not has_exact and proxy else _pixel_interpretation(exact, proxy) if has_exact else "no_exact_or_proxy_rows_for_view"
        view_rows.append(
            {
                "scene": scene,
                "condition": condition,
                "subset_name": subset_name,
                "view_name": view,
                "view_group": _group_for_view(view, direct_views, collateral_views, control_views),
                "exact_view_has_rows": _bool_text(has_exact),
                "exact_unique_gaussian_count": len(exact),
                "proxy_unique_gaussian_count": len(proxy),
                "intersection_count": len(cmp["intersection"]),
                "exact_only_count": len(cmp["exact_only"]),
                "proxy_only_count": len(cmp["proxy_only"]),
                "jaccard": cmp["jaccard"] if has_exact else "",
                "exact_recall_by_proxy": cmp["exact_recall_by_proxy"] if has_exact else "",
                "proxy_precision_against_exact": cmp["proxy_precision_against_exact"] if has_exact else "",
                "exact_gaussian_ids_semicolon": _fmt_ids(exact),
                "proxy_gaussian_ids_semicolon": _fmt_ids(proxy),
                "intersection_gaussian_ids_semicolon": _fmt_ids(cmp["intersection"]),
                "exact_only_gaussian_ids_semicolon": _fmt_ids(cmp["exact_only"]),
                "proxy_only_gaussian_ids_semicolon": _fmt_ids(cmp["proxy_only"]),
                "interpretation": interpretation,
            }
        )

    def union_for(views: list[str], source: dict[str, set[str]]) -> set[str]:
        out: set[str] = set()
        for view in views:
            out |= source.get(view, set())
        return out

    groups = {
        "direct_corrupted": direct_views,
        "co_visible_collateral": collateral_views,
        "clean_prior_demoted": control_views,
    }
    exact_group = {name: union_for(views, exact_by_view) for name, views in groups.items()}
    proxy_group = {name: union_for(views, proxy_by_view) for name, views in groups.items()}
    group_rows = []
    for a, b in [("direct_corrupted", "co_visible_collateral"), ("direct_corrupted", "clean_prior_demoted"), ("co_visible_collateral", "clean_prior_demoted")]:
        exact_a, exact_b = exact_group[a], exact_group[b]
        proxy_a, proxy_b = proxy_group[a], proxy_group[b]
        exact_available = bool(exact_a) and bool(exact_b)
        group_rows.append(
            {
                "scene": scene,
                "condition": condition,
                "subset_name": subset_name,
                "group_a": a,
                "group_b": b,
                "exact_group_a_count": len(exact_a),
                "exact_group_b_count": len(exact_b),
                "exact_overlap_count": len(exact_a & exact_b) if exact_available else "",
                "exact_jaccard": _jaccard(exact_a, exact_b) if exact_available else "",
                "proxy_group_a_count": len(proxy_a),
                "proxy_group_b_count": len(proxy_b),
                "proxy_overlap_count": len(proxy_a & proxy_b),
                "proxy_jaccard": _jaccard(proxy_a, proxy_b),
                "interpretation": "exact_evidence_unavailable_for_one_or_both_groups" if not exact_available else "exact_group_overlap_detected" if exact_a & exact_b else "no_exact_group_overlap_detected",
            }
        )

    pr201_summary = load_json(pr201_dir / "pr201_proxy_degeneracy_summary.json") if pr201_dir and (pr201_dir / "pr201_proxy_degeneracy_summary.json").is_file() else {}
    direct_exact = exact_group["direct_corrupted"]
    direct_proxy = proxy_group["direct_corrupted"]
    proxy_in_exact = len(direct_exact & direct_proxy)
    degeneracy_rows = [
        {
            "scene": scene,
            "condition": condition,
            "subset_name": subset_name,
            "claim_name": "proxy_degeneracy_confirmed",
            "proxy_claim_value": pr201_summary.get("proxy_degeneracy_confirmed", "optional_pr201_missing"),
            "exact_observation_value": f"exact_direct_unique={len(direct_exact)};proxy_direct_unique={len(direct_proxy)};intersection={proxy_in_exact}",
            "exact_evidence_available": _bool_text(bool(direct_exact)),
            "exact_supports_proxy_claim": _bool_text(False),
            "interpretation": "proxy_degeneracy_not_supported_by_exact_rows" if direct_exact else "exact_rows_available_only_for_subset_of_views",
            "caveat": "exact evidence is contributor-ID-only and available only for views with PR21.1 exact rows",
        },
        {
            "scene": scene,
            "condition": condition,
            "subset_name": subset_name,
            "claim_name": "proxy_pool_contains_exact_contributors",
            "proxy_claim_value": f"proxy_direct_unique={len(direct_proxy)}",
            "exact_observation_value": f"intersection={proxy_in_exact};exact_direct_unique={len(direct_exact)}",
            "exact_evidence_available": _bool_text(bool(direct_exact)),
            "exact_supports_proxy_claim": _bool_text(bool(proxy_in_exact)),
            "interpretation": "exact_contributor_ids_refine_proxy_analysis" if proxy_in_exact else "proxy_pool_contains_non_exact_candidates",
            "caveat": "proxy-only IDs are not false positives for views without exact rows",
        },
        {
            "scene": scene,
            "condition": condition,
            "subset_name": subset_name,
            "claim_name": "train013_control",
            "proxy_claim_value": "train013 proxy-pool separation if reported by PR20.1",
            "exact_observation_value": f"control_exact_unique={len(exact_group['clean_prior_demoted'])}",
            "exact_evidence_available": _bool_text(bool(exact_group["clean_prior_demoted"])),
            "exact_supports_proxy_claim": "",
            "interpretation": "exact_evidence_insufficient_for_collateral_or_control" if not exact_group["clean_prior_demoted"] else "exact_control_rows_available",
            "caveat": "do not claim train013 clean control separation without exact rows for train013",
        },
    ]

    direct_collateral = next(row for row in group_rows if row["group_a"] == "direct_corrupted" and row["group_b"] == "co_visible_collateral")
    direct_control = next(row for row in group_rows if row["group_a"] == "direct_corrupted" and row["group_b"] == "clean_prior_demoted")
    collateral_control = next(row for row in group_rows if row["group_a"] == "co_visible_collateral" and row["group_b"] == "clean_prior_demoted")

    exact_pixel_keys = set(exact_by_pixel)
    summary = {
        "schema_name": "viewtrust.pr212.chair_exact_vs_proxy.summary",
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
        "exact_input_dir": str(pr211_dir),
        "proxy_input_dir": str(pr200_dir),
        "pr201_input_dir": str(pr201_dir) if pr201_dir else "",
        "exact_evidence_quality": "exact_sparse_contributor_id_only",
        "exact_render_contribution_available": False,
        "exact_contributor_id_only_available": True,
        "exact_pixel_count": len(exact_pixel_keys),
        "exact_row_count": len(exact_rows),
        "exact_view_count_with_rows": len([view for view, ids in exact_by_view.items() if ids]),
        "proxy_pixel_count_for_exact_pixels": len([key for key in exact_pixel_keys if key in proxy_by_pixel]),
        "mean_pixel_jaccard": _mean(pixel_metrics),
        "median_pixel_jaccard": _median(pixel_metrics),
        "mean_exact_recall_by_proxy": _mean(recall_metrics),
        "mean_proxy_precision_against_exact": _mean(precision_metrics),
        "view_mean_jaccard_for_exact_available_views": _mean(view_jaccards),
        "direct_exact_unique_gaussian_count": len(exact_group["direct_corrupted"]),
        "collateral_exact_unique_gaussian_count": len(exact_group["co_visible_collateral"]),
        "control_exact_unique_gaussian_count": len(exact_group["clean_prior_demoted"]),
        "direct_collateral_exact_jaccard": direct_collateral["exact_jaccard"],
        "direct_control_exact_jaccard": direct_control["exact_jaccard"],
        "collateral_control_exact_jaccard": collateral_control["exact_jaccard"],
        "proxy_degeneracy_reassessed": True,
        "pr212_ready_for_interpretation": True,
        "pr212_ready_for_intervention": False,
        "recommended_next_step": "Use PR21.2 chair exact-vs-proxy results to write the first exact-evidence analysis section; keep drums as coordinate-alignment unresolved until PR21.1f.",
    }

    write_json(output_dir / "pr212_chair_exact_vs_proxy_summary.json", summary)
    write_csv_rows(output_dir / "pr212_chair_pixel_exact_vs_proxy.csv", pixel_rows, PIXEL_FIELDS)
    write_csv_rows(output_dir / "pr212_chair_view_exact_vs_proxy.csv", view_rows, VIEW_FIELDS)
    write_csv_rows(output_dir / "pr212_chair_group_exact_overlap.csv", group_rows, GROUP_FIELDS)
    write_csv_rows(output_dir / "pr212_chair_proxy_degeneracy_reassessment.csv", degeneracy_rows, DEGENERACY_FIELDS)
    _write_report(output_dir / "pr212_chair_exact_vs_proxy_report.md", summary)
    manifest = output_dir / "artifact_manifest.csv"
    write_csv_rows(manifest, _artifact_rows(output_dir, pr200_dir, pr211_dir, pr201_dir), MANIFEST_FIELDS)
    write_csv_rows(manifest, _artifact_rows(output_dir, pr200_dir, pr211_dir, pr201_dir), MANIFEST_FIELDS)
    return summary, 0
