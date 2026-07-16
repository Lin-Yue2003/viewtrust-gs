"""PR21.1f drums selected-pixel source alignment audit."""

from __future__ import annotations

import hashlib
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from viewtrust.analysis.clean_prior_normalization import load_csv_rows, load_json, write_csv_rows, write_json


OUTPUT_FILES = [
    "pr211f_drums_selected_pixel_alignment_summary.json",
    "pr211f_drums_pr20_selected_pixel_audit.csv",
    "pr211f_drums_exact_replay_raw_pixel_coverage.csv",
    "pr211f_drums_coordinate_convention_audit.csv",
    "pr211f_drums_residual_source_alignment_audit.csv",
    "pr211f_drums_top_residual_crosscheck.csv",
    "pr211f_drums_alignment_diagnosis.csv",
    "pr211f_drums_selected_pixel_alignment_report.md",
    "artifact_manifest.csv",
]

SELECTED_PIXEL_AUDIT_FIELDS = [
    "scene",
    "view_name",
    "view_group",
    "proxy_row_count",
    "unique_selected_pixel_count",
    "min_x",
    "max_x",
    "min_y",
    "max_y",
    "duplicate_proxy_rows_per_pixel",
    "source_file",
    "notes",
]

RAW_COVERAGE_FIELDS = [
    "scene",
    "view_name",
    "view_group",
    "attempted",
    "succeeded",
    "raw_contributor_row_count",
    "unique_raw_pixel_count",
    "min_raw_x",
    "max_raw_x",
    "min_raw_y",
    "max_raw_y",
    "image_ids_seen",
    "unexpected_image_id_count",
    "accumulation_source_selected",
    "error",
    "notes",
]

COORDINATE_FIELDS = [
    "scene",
    "view_name",
    "view_group",
    "convention",
    "selected_pixel_count",
    "raw_unique_pixel_count",
    "hit_count",
    "hit_rate",
    "diagnostic_only",
    "can_be_exact_evidence",
    "interpretation",
]

RESIDUAL_SOURCE_FIELDS = [
    "scene",
    "view_name",
    "pr20_render_path",
    "pr20_gt_path",
    "pr20_residual_path",
    "pr21_render_path",
    "pr21_gt_path",
    "pr20_render_shape",
    "pr20_gt_shape",
    "pr20_residual_shape",
    "pr21_render_shape",
    "pr21_gt_shape",
    "render_shape_match",
    "gt_shape_match",
    "residual_shape_match",
    "selected_pixels_high_residual_normal",
    "selected_pixels_high_residual_y_flip",
    "selected_pixels_high_residual_x_flip",
    "selected_pixels_high_residual_xy_swap",
    "best_residual_convention",
    "best_residual_convention_score",
    "notes",
]

TOP_RESIDUAL_FIELDS = [
    "scene",
    "view_name",
    "reconstructed_residual_available",
    "pr20_selected_count",
    "reconstructed_topk_count",
    "normal_overlap_count",
    "y_flip_overlap_count",
    "x_flip_overlap_count",
    "xy_swap_overlap_count",
    "xy_swap_y_flip_overlap_count",
    "xy_swap_x_flip_overlap_count",
    "best_overlap_convention",
    "best_overlap_count",
    "interpretation",
]

DIAGNOSIS_FIELDS = [
    "scene",
    "view_name",
    "view_group",
    "raw_contributors_available",
    "normal_exact_hit_count",
    "best_diagnostic_convention",
    "best_diagnostic_hit_count",
    "best_diagnostic_hit_rate",
    "residual_source_available",
    "best_residual_convention",
    "best_residual_overlap_count",
    "likely_failure_mode",
    "exact_evidence_allowed",
    "recommended_fix",
    "caveat",
]

MANIFEST_FIELDS = ["relative_path", "path", "exists", "file_type", "size_bytes", "required", "artifact_group"]

CONVENTIONS = ["normal", "y_flip", "x_flip", "xy_swap", "xy_swap_y_flip", "xy_swap_x_flip", "neighborhood_r1", "neighborhood_r2", "neighborhood_r4", "neighborhood_r8"]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _bool_text(value: Any) -> str:
    return "true" if bool(value) else "false"


def _truth(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() == "true"


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _group_for_view(view: str) -> str:
    if view in {"train_004", "train_009", "train_012", "train_017"}:
        return "direct_corrupted"
    if view == "train_007":
        return "co_visible_collateral"
    if view == "train_013":
        return "clean_prior_demoted"
    return "other_clean"


def _pixel_id(x: int, y: int, width: int) -> int:
    return int(y) * int(width) + int(x)


def _transform_pixel(x: int, y: int, width: int, height: int, convention: str) -> tuple[int, int] | None:
    if convention == "normal":
        tx, ty = x, y
    elif convention == "y_flip":
        tx, ty = x, height - 1 - y
    elif convention == "x_flip":
        tx, ty = width - 1 - x, y
    elif convention == "xy_swap":
        tx, ty = y, x
    elif convention == "xy_swap_y_flip":
        tx, ty = y, height - 1 - x
    elif convention == "xy_swap_x_flip":
        tx, ty = width - 1 - y, x
    else:
        tx, ty = x, y
    if tx < 0 or ty < 0 or tx >= width or ty >= height:
        return None
    return tx, ty


def _load_selected_pixels(pr200_dir: Path, views: list[str]) -> tuple[dict[str, set[tuple[int, int]]], dict[str, list[dict[str, Any]]]]:
    rows = load_csv_rows(pr200_dir / "pr200_pixel_gaussian_contributions.csv")
    selected: dict[str, set[tuple[int, int]]] = {view: set() for view in views}
    by_view: dict[str, list[dict[str, Any]]] = {view: [] for view in views}
    for row in rows:
        view = str(row.get("view_name", ""))
        if view not in selected:
            continue
        pixel = (_safe_int(row.get("pixel_x")), _safe_int(row.get("pixel_y")))
        selected[view].add(pixel)
        by_view[view].append(row)
    return selected, by_view


def _selected_pixel_audit(scene: str, pr200_dir: Path, views: list[str], selected: dict[str, set[tuple[int, int]]], rows_by_view: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    out = []
    for view in views:
        pixels = selected.get(view, set())
        xs = [x for x, _ in pixels]
        ys = [y for _, y in pixels]
        proxy_count = len(rows_by_view.get(view, []))
        out.append(
            {
                "scene": scene,
                "view_name": view,
                "view_group": _group_for_view(view),
                "proxy_row_count": proxy_count,
                "unique_selected_pixel_count": len(pixels),
                "min_x": min(xs) if xs else "",
                "max_x": max(xs) if xs else "",
                "min_y": min(ys) if ys else "",
                "max_y": max(ys) if ys else "",
                "duplicate_proxy_rows_per_pixel": proxy_count / len(pixels) if pixels else "",
                "source_file": str(pr200_dir / "pr200_pixel_gaussian_contributions.csv"),
                "notes": "deduplicated proxy rows by view_name,pixel_x,pixel_y",
            }
        )
    return out


def _validate_pr211_summary(summary: dict[str, Any], scene: str) -> None:
    checks = [
        (summary.get("scene") == scene, f"scene must be {scene}"),
        (_truth(summary.get("per_view_replay_enabled")), "per_view_replay_enabled must be true"),
        (not _truth(summary.get("multi_view_image_id_mapping_used")), "multi_view_image_id_mapping_used must be false"),
        (_truth(summary.get("pure_torch_accumulate_succeeded")), "pure_torch_accumulate_succeeded must be true"),
        (_safe_int(summary.get("per_view_total_contributor_rows_before_filter")) > 0, "per_view_total_contributor_rows_before_filter must be > 0"),
        (_safe_int(summary.get("selected_pixel_hit_count")) == 0, "selected_pixel_hit_count must be 0"),
        (_safe_int(summary.get("exact_contributor_id_row_count")) == 0, "exact_contributor_id_row_count must be 0"),
    ]
    failed = [message for ok, message in checks if not ok]
    if failed:
        raise ValueError("PR21.1f expected failure-mode validation failed: " + "; ".join(failed))


def _raw_coverage(scene: str, pr211_dir: Path, views: list[str]) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    rows = load_csv_rows(pr211_dir / "pr211_per_view_replay_audit.csv")
    by_view = {str(row.get("view_name", "")): row for row in rows}
    out = []
    for view in views:
        row = by_view.get(view, {})
        out.append(
            {
                "scene": scene,
                "view_name": view,
                "view_group": row.get("view_group") or _group_for_view(view),
                "attempted": row.get("attempted", ""),
                "succeeded": row.get("succeeded", ""),
                "raw_contributor_row_count": row.get("raw_contributor_rows_before_filter", 0),
                "unique_raw_pixel_count": row.get("unique_raw_pixel_count", 0),
                "min_raw_x": "",
                "max_raw_x": "",
                "min_raw_y": "",
                "max_raw_y": "",
                "image_ids_seen": row.get("image_ids_seen", ""),
                "unexpected_image_id_count": row.get("unexpected_image_id_count", 0),
                "accumulation_source_selected": row.get("accumulation_source_selected", ""),
                "error": row.get("error", ""),
                "notes": row.get("notes", "loaded from PR21.1e per-view replay audit"),
            }
        )
    return out, by_view


def _coordinate_audit(scene: str, views: list[str], selected: dict[str, set[tuple[int, int]]], raw_by_view: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    field_for = {
        "normal": "selected_pixel_hit_count_normal",
        "y_flip": "y_flip_hit_count",
        "x_flip": "x_flip_hit_count",
        "xy_swap": "xy_swap_hit_count",
        "xy_swap_y_flip": "xy_swap_y_flip_hit_count",
        "xy_swap_x_flip": "xy_swap_x_flip_hit_count",
        "neighborhood_r1": "neighborhood_r1_hit_count",
        "neighborhood_r2": "neighborhood_r2_hit_count",
        "neighborhood_r4": "neighborhood_r4_hit_count",
        "neighborhood_r8": "neighborhood_r8_hit_count",
    }
    for view in views:
        raw = raw_by_view.get(view, {})
        selected_count = len(selected.get(view, set()))
        raw_count = _safe_int(raw.get("unique_raw_pixel_count"))
        for convention in CONVENTIONS:
            hits = _safe_int(raw.get(field_for[convention]))
            diagnostic_only = convention != "normal"
            out.append(
                {
                    "scene": scene,
                    "view_name": view,
                    "view_group": raw.get("view_group") or _group_for_view(view),
                    "convention": convention,
                    "selected_pixel_count": selected_count,
                    "raw_unique_pixel_count": raw_count,
                    "hit_count": hits,
                    "hit_rate": hits / selected_count if selected_count else "",
                    "diagnostic_only": _bool_text(diagnostic_only),
                    "can_be_exact_evidence": "false",
                    "interpretation": "normal_hits_require_source_alignment_validation"
                    if convention == "normal" and hits
                    else "diagnostic_only_not_exact_evidence"
                    if diagnostic_only and hits
                    else "no_hits",
                }
            )
    return out


def _shape(path: Path | None) -> str:
    if path is None or not path.is_file():
        return ""
    try:
        with Image.open(path) as image:
            return str(tuple(image.size[::-1]) + ((len(image.getbands()),) if image.getbands() else ()))
    except Exception:
        return ""


def _read_image(path: Path | None) -> np.ndarray | None:
    if path is None or not path.is_file():
        return None
    try:
        with Image.open(path) as image:
            return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    except Exception:
        return None


def _find_image(root: Path, view: str, hints: list[str]) -> Path | None:
    if not root.exists():
        return None
    names = [f"{view}.png", f"{view}.jpg", f"{view}.jpeg", f"{view.split('_')[-1]}.png"]
    candidates = []
    for name in names:
        candidates.extend(root.rglob(name))
    for hint in hints:
        hinted = [path for path in candidates if hint.lower() in str(path).lower()]
        if hinted:
            return sorted(hinted, key=lambda p: len(str(p)))[0]
    return sorted(candidates, key=lambda p: len(str(p)))[0] if candidates else None


def _checksum(path: Path | None) -> str:
    if path is None or not path.is_file():
        return ""
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            digest.update(handle.read(1024 * 1024))
        return digest.hexdigest()[:16]
    except Exception:
        return ""


def _best_overlap(selected: set[tuple[int, int]], top_pixels: set[tuple[int, int]], width: int, height: int) -> tuple[str, int, dict[str, int]]:
    counts: dict[str, int] = {}
    for convention in ["normal", "y_flip", "x_flip", "xy_swap", "xy_swap_y_flip", "xy_swap_x_flip"]:
        transformed = {_transform_pixel(x, y, width, height, convention) for x, y in selected}
        transformed.discard(None)
        counts[convention] = len(transformed & top_pixels)  # type: ignore[arg-type]
    best = max(counts, key=lambda key: counts[key]) if counts else "unavailable"
    return best, counts.get(best, 0), counts


def _source_alignment(scene: str, run_dir: Path, pr200_dir: Path, views: list[str], selected: dict[str, set[tuple[int, int]]], top_k: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    residual_rows = []
    top_rows = []
    for view in views:
        pr20_render = _find_image(pr200_dir, view, ["render"])
        pr20_gt = _find_image(pr200_dir, view, ["gt"])
        pr20_residual = _find_image(pr200_dir, view, ["residual", "heatmap"])
        pr21_render = _find_image(run_dir, view, ["renders", "render"])
        pr21_gt = _find_image(run_dir, view, ["gt"])
        pr20_render_shape = _shape(pr20_render)
        pr20_gt_shape = _shape(pr20_gt)
        pr20_residual_shape = _shape(pr20_residual)
        pr21_render_shape = _shape(pr21_render)
        pr21_gt_shape = _shape(pr21_gt)
        render = _read_image(pr20_render)
        if render is None:
            render = _read_image(pr21_render)
        gt = _read_image(pr20_gt)
        if gt is None:
            gt = _read_image(pr21_gt)
        residual_available = render is not None and gt is not None and render.shape == gt.shape
        best_convention = ""
        best_score: int | str = ""
        top_counts = {key: 0 for key in ["normal", "y_flip", "x_flip", "xy_swap", "xy_swap_y_flip", "xy_swap_x_flip"]}
        if residual_available:
            residual = np.mean(np.abs(render - gt), axis=2)
            height, width = residual.shape
            flat = np.argsort(residual.reshape(-1))[-top_k:]
            top_pixels = {(int(idx % width), int(idx // width)) for idx in flat}
            best_convention, best_score, top_counts = _best_overlap(selected.get(view, set()), top_pixels, width, height)
        notes = []
        if pr20_render:
            notes.append(f"pr20_render_sha256_1mb={_checksum(pr20_render)}")
        if pr21_render:
            notes.append(f"pr21_render_sha256_1mb={_checksum(pr21_render)}")
        if not residual_available:
            notes.append("residual source unavailable or render/gt shapes differ")
        residual_rows.append(
            {
                "scene": scene,
                "view_name": view,
                "pr20_render_path": str(pr20_render or ""),
                "pr20_gt_path": str(pr20_gt or ""),
                "pr20_residual_path": str(pr20_residual or ""),
                "pr21_render_path": str(pr21_render or ""),
                "pr21_gt_path": str(pr21_gt or ""),
                "pr20_render_shape": pr20_render_shape,
                "pr20_gt_shape": pr20_gt_shape,
                "pr20_residual_shape": pr20_residual_shape,
                "pr21_render_shape": pr21_render_shape,
                "pr21_gt_shape": pr21_gt_shape,
                "render_shape_match": _bool_text(bool(pr20_render_shape and pr21_render_shape and pr20_render_shape == pr21_render_shape)),
                "gt_shape_match": _bool_text(bool(pr20_gt_shape and pr21_gt_shape and pr20_gt_shape == pr21_gt_shape)),
                "residual_shape_match": _bool_text(bool(pr20_residual_shape and pr20_render_shape and pr20_residual_shape[:8] == pr20_render_shape[:8])),
                "selected_pixels_high_residual_normal": top_counts["normal"],
                "selected_pixels_high_residual_y_flip": top_counts["y_flip"],
                "selected_pixels_high_residual_x_flip": top_counts["x_flip"],
                "selected_pixels_high_residual_xy_swap": top_counts["xy_swap"],
                "best_residual_convention": best_convention,
                "best_residual_convention_score": best_score,
                "notes": "; ".join(notes),
            }
        )
        top_rows.append(
            {
                "scene": scene,
                "view_name": view,
                "reconstructed_residual_available": _bool_text(residual_available),
                "pr20_selected_count": len(selected.get(view, set())),
                "reconstructed_topk_count": top_k if residual_available else 0,
                "normal_overlap_count": top_counts["normal"],
                "y_flip_overlap_count": top_counts["y_flip"],
                "x_flip_overlap_count": top_counts["x_flip"],
                "xy_swap_overlap_count": top_counts["xy_swap"],
                "xy_swap_y_flip_overlap_count": top_counts["xy_swap_y_flip"],
                "xy_swap_x_flip_overlap_count": top_counts["xy_swap_x_flip"],
                "best_overlap_convention": best_convention or "unavailable",
                "best_overlap_count": best_score if best_score != "" else 0,
                "interpretation": "residual_source_unavailable"
                if not residual_available
                else f"pr20_selected_pixels_match_reconstructed_residual_{best_convention}"
                if best_score
                else "pr20_selected_pixels_do_not_match_reconstructed_residual",
            }
        )
    return residual_rows, top_rows


def _diagnosis(scene: str, views: list[str], coord_rows: list[dict[str, Any]], raw_by_view: dict[str, dict[str, Any]], residual_rows: list[dict[str, Any]], top_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    coord_by_view: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in coord_rows:
        coord_by_view[str(row["view_name"])].append(row)
    residual_by_view = {str(row["view_name"]): row for row in residual_rows}
    top_by_view = {str(row["view_name"]): row for row in top_rows}
    out = []
    best_totals: dict[str, int] = defaultdict(int)
    normal_total = 0
    raw_total = 0
    for view in views:
        raw = raw_by_view.get(view, {})
        coord = coord_by_view.get(view, [])
        normal = next((row for row in coord if row["convention"] == "normal"), {})
        diagnostics = [row for row in coord if row["convention"] != "normal"]
        best = max(diagnostics, key=lambda row: _safe_int(row.get("hit_count"))) if diagnostics else {}
        best_name = str(best.get("convention", ""))
        best_hits = _safe_int(best.get("hit_count"))
        best_totals[best_name] += best_hits
        normal_hits = _safe_int(normal.get("hit_count"))
        normal_total += normal_hits
        raw_count = _safe_int(raw.get("raw_contributor_rows_before_filter"))
        raw_total += raw_count
        residual = residual_by_view.get(view, {})
        top = top_by_view.get(view, {})
        residual_available = _truth(top.get("reconstructed_residual_available"))
        if raw_count == 0:
            failure = "exact_replay_has_no_raw_contributors"
            fix = "inspect_pr21_replay_for_view"
        elif normal_hits > 0:
            failure = "normal_hits_require_validation"
            fix = "validate_coordinate_convention_before_exact_rows"
        elif best_hits > 0 and "flip" in best_name:
            failure = "selected_pixel_coordinate_flip_candidate"
            fix = "validate_coordinate_convention_before_exact_rows"
        elif best_hits > 0 and "swap" in best_name:
            failure = "selected_pixel_xy_swap_candidate"
            fix = "validate_coordinate_convention_before_exact_rows"
        elif residual_available and not top.get("normal_overlap_count"):
            failure = "residual_source_mismatch"
            fix = "regenerate_pr20_selected_pixels_from_same_render_gt_used_by_pr21"
        elif not residual_available:
            failure = "insufficient_source_files_to_decide"
            fix = "inspect_pr20_selected_pixel_generation"
        else:
            failure = "selected_pixels_outside_exact_support"
            fix = "keep_drums_excluded_from_pr212_until_alignment_resolved"
        out.append(
            {
                "scene": scene,
                "view_name": view,
                "view_group": raw.get("view_group") or _group_for_view(view),
                "raw_contributors_available": _bool_text(raw_count > 0),
                "normal_exact_hit_count": normal_hits,
                "best_diagnostic_convention": best_name,
                "best_diagnostic_hit_count": best_hits,
                "best_diagnostic_hit_rate": best.get("hit_rate", ""),
                "residual_source_available": _bool_text(residual_available),
                "best_residual_convention": residual.get("best_residual_convention", ""),
                "best_residual_overlap_count": top.get("best_overlap_count", ""),
                "likely_failure_mode": failure,
                "exact_evidence_allowed": "false",
                "recommended_fix": fix,
                "caveat": "Diagnostic flip/swap/neighborhood hits are not promoted to exact evidence.",
            }
        )
    best_overall = max(best_totals, key=lambda key: best_totals[key]) if best_totals else ""
    return out, {"normal_total": normal_total, "raw_total": raw_total, "best_overall": best_overall, "best_total": best_totals.get(best_overall, 0)}


def _manifest_rows(output_dir: Path, run_dir: Path, pr200_dir: Path, pr211_dir: Path, pr210_dir: Path | None) -> list[dict[str, Any]]:
    items = [("run_dir", run_dir, True, "input"), ("pr200_dir", pr200_dir, True, "input"), ("pr211_dir", pr211_dir, True, "input")]
    if pr210_dir:
        items.append(("pr210_dir", pr210_dir, False, "input_optional"))
    items.extend((name, output_dir / name, True, "output_pr211f") for name in OUTPUT_FILES)
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
        "# PR21.1f Drums Selected-Pixel Alignment Audit",
        "",
        "PR21.1f is observation-only. It does not implement defense, rejection, reweighting, update suppression, or densification gating.",
        "",
        "Diagnostic flip/swap/neighborhood hits are not promoted to exact evidence.",
        "",
        "## Why Drums Was Excluded From PR21.2",
        "PR21.1e recovered raw contributors for drums but normal selected-pixel exact hits remained zero.",
        "",
        "## Summary",
        f"- Selected views: `{summary.get('selected_view_count')}`",
        f"- PR20 selected pixels: `{summary.get('pr20_selected_pixel_total')}`",
        f"- Raw contributors: `{summary.get('raw_contributor_total')}`",
        f"- Normal exact hits: `{summary.get('normal_exact_hit_total')}`",
        f"- Best diagnostic convention: `{summary.get('best_diagnostic_convention_overall')}`",
        f"- Likely failure mode: `{summary.get('likely_failure_mode_overall')}`",
        f"- Drums ready for PR21.2: `{summary.get('drums_ready_for_pr212')}`",
        "",
        "## Recommended Next Step",
        str(summary.get("recommended_next_step", "")),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def build_pr211f_drums_alignment_audit(
    *,
    run_dir: Path,
    pr200_dir: Path,
    pr211_dir: Path,
    output_dir: Path,
    pr210_dir: Path | None = None,
    scene: str = "drums",
    condition: str = "corrupt_occluder",
    subset_name: str = "seed_20260710",
    views: list[str] | None = None,
    top_pixels_per_view: int = 128,
    write_markdown: bool = False,
) -> tuple[dict[str, Any], int]:
    del write_markdown
    views = views or ["train_004", "train_009", "train_012", "train_017", "train_007", "train_013"]
    output_dir.mkdir(parents=True, exist_ok=True)
    summary211 = load_json(pr211_dir / "pr211_exact_sparse_attribution_summary.json")
    _validate_pr211_summary(summary211, scene)

    selected, rows_by_view = _load_selected_pixels(pr200_dir, views)
    selected_audit = _selected_pixel_audit(scene, pr200_dir, views, selected, rows_by_view)
    raw_rows, raw_by_view = _raw_coverage(scene, pr211_dir, views)
    coord_rows = _coordinate_audit(scene, views, selected, raw_by_view)
    residual_rows, top_rows = _source_alignment(scene, run_dir, pr200_dir, views, selected, top_pixels_per_view)
    diagnosis_rows, totals = _diagnosis(scene, views, coord_rows, raw_by_view, residual_rows, top_rows)

    likely_modes = [row["likely_failure_mode"] for row in diagnosis_rows]
    likely_overall = statistics.mode(likely_modes) if likely_modes else "insufficient_source_files_to_decide"
    residual_available = any(_truth(row.get("reconstructed_residual_available")) for row in top_rows)
    selected_total = sum(len(selected.get(view, set())) for view in views)
    exact_allowed = False
    summary = {
        "schema_name": "viewtrust.pr211f.drums_selected_pixel_alignment.summary",
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
        "pr20_input_dir": str(pr200_dir),
        "pr211_input_dir": str(pr211_dir),
        "pr210_input_dir": str(pr210_dir) if pr210_dir else "",
        "run_dir": str(run_dir),
        "selected_view_count": len(views),
        "pr20_selected_pixel_total": selected_total,
        "raw_contributor_total": totals["raw_total"],
        "normal_exact_hit_total": totals["normal_total"],
        "best_diagnostic_convention_overall": totals["best_overall"],
        "best_diagnostic_hit_total": totals["best_total"],
        "residual_source_alignment_available": residual_available,
        "likely_failure_mode_overall": likely_overall,
        "exact_evidence_allowed_for_drums": exact_allowed,
        "drums_ready_for_pr212": False,
        "recommended_next_step": "Validate coordinate convention or regenerate PR20 selected pixels from the same render/GT source before using drums in PR21.2.",
    }

    write_json(output_dir / "pr211f_drums_selected_pixel_alignment_summary.json", summary)
    write_csv_rows(output_dir / "pr211f_drums_pr20_selected_pixel_audit.csv", selected_audit, SELECTED_PIXEL_AUDIT_FIELDS)
    write_csv_rows(output_dir / "pr211f_drums_exact_replay_raw_pixel_coverage.csv", raw_rows, RAW_COVERAGE_FIELDS)
    write_csv_rows(output_dir / "pr211f_drums_coordinate_convention_audit.csv", coord_rows, COORDINATE_FIELDS)
    write_csv_rows(output_dir / "pr211f_drums_residual_source_alignment_audit.csv", residual_rows, RESIDUAL_SOURCE_FIELDS)
    write_csv_rows(output_dir / "pr211f_drums_top_residual_crosscheck.csv", top_rows, TOP_RESIDUAL_FIELDS)
    write_csv_rows(output_dir / "pr211f_drums_alignment_diagnosis.csv", diagnosis_rows, DIAGNOSIS_FIELDS)
    _write_report(output_dir / "pr211f_drums_selected_pixel_alignment_report.md", summary)
    manifest = output_dir / "artifact_manifest.csv"
    write_csv_rows(manifest, _manifest_rows(output_dir, run_dir, pr200_dir, pr211_dir, pr210_dir), MANIFEST_FIELDS)
    write_csv_rows(manifest, _manifest_rows(output_dir, run_dir, pr200_dir, pr211_dir, pr210_dir), MANIFEST_FIELDS)
    return summary, 0
