"""PR21.1g PR20 selected-pixel provenance audit."""

from __future__ import annotations

import hashlib
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from viewtrust.analysis.clean_prior_normalization import load_csv_rows, load_json, write_csv_rows, write_json


OUTPUT_FILES = [
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

PROXY_FIELDS = [
    "scene",
    "view_name",
    "view_group",
    "proxy_row_count",
    "unique_selected_pixel_count",
    "expected_unique_selected_pixel_count",
    "expected_proxy_row_count",
    "contributors_per_pixel_min",
    "contributors_per_pixel_max",
    "contributors_per_pixel_mean",
    "residual_l1_min",
    "residual_l1_max",
    "residual_l1_mean",
    "residual_weighted_splat_min",
    "residual_weighted_splat_max",
    "residual_weighted_splat_mean",
    "selected_pixel_hash",
    "source_file",
    "notes",
]

SCHEMA_FIELDS = [
    "source_file",
    "exists",
    "row_count",
    "column_count",
    "columns_semicolon",
    "candidate_view_column",
    "candidate_pixel_x_column",
    "candidate_pixel_y_column",
    "candidate_pixel_id_column",
    "candidate_residual_score_columns_semicolon",
    "candidate_gaussian_columns_semicolon",
    "notes",
]

REPRO_FIELDS = [
    "scene",
    "view_name",
    "candidate_source_file",
    "strategy_name",
    "grouping_mode",
    "coordinate_mode",
    "score_column",
    "sort_direction",
    "top_k",
    "selected_count",
    "reproduced_count",
    "normal_overlap_count",
    "y_flip_overlap_count",
    "x_flip_overlap_count",
    "xy_swap_overlap_count",
    "xy_swap_y_flip_overlap_count",
    "xy_swap_x_flip_overlap_count",
    "best_convention",
    "best_overlap_count",
    "best_overlap_rate",
    "exact_reproduction",
    "interpretation",
]

MEMBERSHIP_FIELDS = [
    "scene",
    "view_name",
    "view_group",
    "selected_pixel_count",
    "residual_csv_same_pixel_count",
    "residual_csv_same_pixel_rate",
    "residual_csv_y_flip_count",
    "residual_csv_x_flip_count",
    "residual_csv_xy_swap_count",
    "residual_csv_xy_swap_y_flip_count",
    "residual_csv_xy_swap_x_flip_count",
    "best_membership_convention",
    "best_membership_count",
    "best_membership_rate",
    "interpretation",
]

CODE_AUDIT_FIELDS = ["matched_file", "line_number", "matched_text", "inferred_role", "confidence", "notes"]

HASH_FIELDS = [
    "scene",
    "source_file",
    "view_name",
    "coordinate_mode",
    "row_count",
    "unique_pixel_count",
    "selected_proxy_hash",
    "candidate_file_hash",
    "hash_match",
    "overlap_count",
    "overlap_rate",
    "best_transform",
    "notes",
]

DIAGNOSIS_FIELDS = [
    "scene",
    "view_name",
    "view_group",
    "selected_pixels_from_proxy_count",
    "residual_csv_available",
    "selected_pixels_present_in_residual_csv",
    "best_residual_reproduction_strategy",
    "best_residual_reproduction_overlap",
    "best_residual_reproduction_rate",
    "best_residual_reproduction_convention",
    "pixel_set_hash_match_found",
    "code_provenance_confidence",
    "likely_pr20_selected_pixel_source",
    "provenance_status",
    "exact_evidence_allowed",
    "drums_ready_for_pr212",
    "recommended_next_step",
    "caveat",
]

MANIFEST_FIELDS = ["relative_path", "path", "exists", "file_type", "size_bytes", "required", "artifact_group"]

CONVENTIONS = ["normal", "y_flip", "x_flip", "xy_swap", "xy_swap_y_flip", "xy_swap_x_flip"]
SCORE_TERMS = ["residual_l1", "residual", "residual_weighted_splat", "l1", "error", "score", "weighted", "attribution", "rank"]
CODE_TERMS = [
    "pr200_pixel_gaussian_contributions",
    "pr200_sparse_pixel_residuals",
    "sparse_pixel_residuals",
    "pixel_gaussian_contributions",
    "top_pixels",
    "top-pixels-per-view",
    "residual_l1",
    "residual_weighted_splat",
    "gaussian_residual_attribution",
    "selected pixels",
    "selected_pixel",
    "topk",
    "argsort",
    "sort_values",
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


def _safe_float(value: Any) -> float | None:
    try:
        if value in ("", None):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _group_for_view(view: str) -> str:
    if view in {"train_004", "train_009", "train_012", "train_017"}:
        return "direct_corrupted"
    if view == "train_007":
        return "co_visible_collateral"
    if view == "train_013":
        return "clean_prior_demoted"
    return "other_clean"


def _pixel_hash(pixels: set[tuple[int, int]]) -> str:
    payload = ";".join(f"{x},{y}" for x, y in sorted(pixels))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


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


def _overlap_counts(selected: set[tuple[int, int]], candidate: set[tuple[int, int]], width: int = 400, height: int = 400) -> dict[str, int]:
    counts: dict[str, int] = {}
    for convention in CONVENTIONS:
        transformed = {_transform_pixel(x, y, width, height, convention) for x, y in selected}
        transformed.discard(None)
        counts[convention] = len(transformed & candidate)  # type: ignore[arg-type]
    return counts


def _best_count(counts: dict[str, int]) -> tuple[str, int]:
    if not counts:
        return "", 0
    best = max(counts, key=lambda key: counts[key])
    return best, counts[best]


def _infer_schema(rows: list[dict[str, str]], path: Path) -> dict[str, Any]:
    columns = list(rows[0].keys()) if rows else []

    def first(names: list[str], contains: list[str] | None = None) -> str:
        lowered = {col.lower(): col for col in columns}
        for name in names:
            if name.lower() in lowered:
                return lowered[name.lower()]
        if contains:
            for col in columns:
                text = col.lower()
                if any(term in text for term in contains):
                    return col
        return ""

    score_cols = [col for col in columns if any(term in col.lower() for term in SCORE_TERMS)]
    gaussian_cols = [col for col in columns if "gaussian" in col.lower() or col.lower() in {"gid", "root_gaussian_id", "parent_gaussian_id"}]
    return {
        "source_file": str(path),
        "exists": _bool_text(path.exists()),
        "row_count": len(rows),
        "column_count": len(columns),
        "columns_semicolon": ";".join(columns),
        "candidate_view_column": first(["view_name", "view", "image_name", "image_id"], ["view", "image"]),
        "candidate_pixel_x_column": first(["pixel_x", "x", "col", "column"]),
        "candidate_pixel_y_column": first(["pixel_y", "y", "row"]),
        "candidate_pixel_id_column": first(["pixel_id", "flat_index", "index"]),
        "candidate_residual_score_columns_semicolon": ";".join(score_cols),
        "candidate_gaussian_columns_semicolon": ";".join(gaussian_cols),
        "notes": "schema inferred from column names; unknown columns are reported rather than assumed",
    }


def _row_view(row: dict[str, str], view_col: str, views: list[str]) -> str:
    if not view_col:
        return ""
    value = str(row.get(view_col, ""))
    if value in views:
        return value
    lowered = value.lower()
    for view in views:
        if view.lower() in lowered or view.split("_")[-1] in lowered:
            return view
    return value


def _coords_from_row(row: dict[str, str], mode: str, width: int = 400) -> tuple[int, int] | None:
    try:
        if mode == "pixel_x+pixel_y":
            return _safe_int(row.get("pixel_x")), _safe_int(row.get("pixel_y"))
        if mode == "x+y":
            return _safe_int(row.get("x")), _safe_int(row.get("y"))
        if mode == "col+row":
            return _safe_int(row.get("col")), _safe_int(row.get("row"))
        if mode.startswith("columns:"):
            _, x_col, y_col = mode.split(":", 2)
            return _safe_int(row.get(x_col)), _safe_int(row.get(y_col))
        if mode == "pixel_id_width_400":
            pid = _safe_int(row.get("pixel_id"))
            return pid % width, pid // width
    except Exception:
        return None
    return None


def _coordinate_modes(columns: list[str]) -> list[str]:
    modes = []
    if "pixel_x" in columns and "pixel_y" in columns:
        modes.append("pixel_x+pixel_y")
    if "x" in columns and "y" in columns:
        modes.append("x+y")
    if "col" in columns and "row" in columns:
        modes.append("col+row")
    if "pixel_id" in columns:
        modes.append("pixel_id_width_400")
    if not modes:
        x_cols = [col for col in columns if col.lower() in {"pixel_x", "x", "col", "column"}]
        y_cols = [col for col in columns if col.lower() in {"pixel_y", "y", "row"}]
        for x_col in x_cols:
            for y_col in y_cols:
                modes.append(f"columns:{x_col}:{y_col}")
    return modes


def _selected_from_proxy(
    *,
    scene: str,
    pr200_dir: Path,
    views: list[str],
    top_pixels_per_view: int,
    max_contributors_per_pixel: int,
) -> tuple[list[dict[str, Any]], dict[str, set[tuple[int, int]]], dict[str, list[dict[str, str]]], list[str]]:
    path = pr200_dir / "pr200_pixel_gaussian_contributions.csv"
    rows = load_csv_rows(path)
    warnings = []
    by_view = {view: [] for view in views}
    pixels = {view: set() for view in views}
    for row in rows:
        view = str(row.get("view_name", ""))
        if view in by_view:
            by_view[view].append(row)
            pixels[view].add((_safe_int(row.get("pixel_x")), _safe_int(row.get("pixel_y"))))
    audit_rows = []
    expected_proxy_rows = top_pixels_per_view * max_contributors_per_pixel
    for view in views:
        view_rows = by_view[view]
        counts = Counter((_safe_int(row.get("pixel_x")), _safe_int(row.get("pixel_y"))) for row in view_rows)
        residual_l1 = [value for value in (_safe_float(row.get("residual_l1")) for row in view_rows) if value is not None]
        weighted = [value for value in (_safe_float(row.get("residual_weighted_splat")) for row in view_rows) if value is not None]
        if len(view_rows) != expected_proxy_rows:
            warnings.append(f"{view}: proxy row count {len(view_rows)} != expected {expected_proxy_rows}")
        if len(pixels[view]) != top_pixels_per_view:
            warnings.append(f"{view}: unique selected pixel count {len(pixels[view])} != expected {top_pixels_per_view}")
        audit_rows.append(
            {
                "scene": scene,
                "view_name": view,
                "view_group": _group_for_view(view),
                "proxy_row_count": len(view_rows),
                "unique_selected_pixel_count": len(pixels[view]),
                "expected_unique_selected_pixel_count": top_pixels_per_view,
                "expected_proxy_row_count": expected_proxy_rows,
                "contributors_per_pixel_min": min(counts.values()) if counts else "",
                "contributors_per_pixel_max": max(counts.values()) if counts else "",
                "contributors_per_pixel_mean": _mean(list(counts.values())),
                "residual_l1_min": min(residual_l1) if residual_l1 else "",
                "residual_l1_max": max(residual_l1) if residual_l1 else "",
                "residual_l1_mean": _mean(residual_l1),
                "residual_weighted_splat_min": min(weighted) if weighted else "",
                "residual_weighted_splat_max": max(weighted) if weighted else "",
                "residual_weighted_splat_mean": _mean(weighted),
                "selected_pixel_hash": _pixel_hash(pixels[view]),
                "source_file": str(path),
                "notes": "selected pixels deduplicated from proxy contribution rows",
            }
        )
    return audit_rows, pixels, by_view, warnings


def _residual_rows_by_view(rows: list[dict[str, str]], view_col: str, views: list[str]) -> dict[str, list[dict[str, str]]]:
    by_view = {view: [] for view in views}
    if view_col:
        for row in rows:
            view = _row_view(row, view_col, views)
            if view in by_view:
                by_view[view].append(row)
    else:
        for view in views:
            by_view[view] = list(rows)
    return by_view


def _pixels_from_rows(rows: list[dict[str, str]], mode: str) -> set[tuple[int, int]]:
    pixels = set()
    for row in rows:
        coords = _coords_from_row(row, mode)
        if coords is not None:
            pixels.add(coords)
    return pixels


def _reproduction_rows(
    *,
    scene: str,
    residual_path: Path,
    residual_rows: list[dict[str, str]],
    schema: dict[str, Any],
    selected: dict[str, set[tuple[int, int]]],
    views: list[str],
    top_k: int,
) -> list[dict[str, Any]]:
    if not residual_rows:
        return []
    columns = list(residual_rows[0].keys())
    coord_modes = _coordinate_modes(columns)
    score_cols = [col for col in str(schema.get("candidate_residual_score_columns_semicolon", "")).split(";") if col]
    view_col = str(schema.get("candidate_view_column", ""))
    grouped = _residual_rows_by_view(residual_rows, view_col, views)
    strategies: list[tuple[str, str, str]] = []
    for score_col in score_cols:
        strategies.append((f"top_desc_{score_col}", score_col, "desc"))
        strategies.append((f"top_asc_{score_col}", score_col, "asc"))
    strategies.append(("file_order_first", "", "first"))
    strategies.append(("file_order_last", "", "last"))
    out = []
    for view in views:
        view_rows = grouped.get(view, [])
        grouping_mode = f"by_{view_col}" if view_col else "no_view_grouping_weak"
        for coord_mode in coord_modes:
            for strategy_name, score_col, direction in strategies:
                rows = list(view_rows)
                if score_col:
                    rows = [row for row in rows if _safe_float(row.get(score_col)) is not None]
                    reverse = direction == "desc"
                    rows = sorted(rows, key=lambda row: (_safe_float(row.get(score_col)) or 0.0, _safe_int(row.get("pixel_id"))), reverse=reverse)
                elif direction == "last":
                    rows = list(reversed(rows))
                top_rows = rows[:top_k]
                reproduced = _pixels_from_rows(top_rows, coord_mode)
                counts = _overlap_counts(selected.get(view, set()), reproduced)
                best_conv, best = _best_count(counts)
                selected_count = len(selected.get(view, set()))
                exact = best == selected_count and selected_count > 0
                if not coord_modes:
                    interpretation = "residual_csv_missing_required_columns"
                elif not view_col:
                    interpretation = "residual_csv_view_mapping_uncertain"
                elif exact:
                    interpretation = "selected_pixels_reproduced_from_residual_csv"
                elif best > 0:
                    interpretation = "selected_pixels_partially_reproduced_from_residual_csv"
                else:
                    interpretation = "selected_pixels_not_reproduced_from_residual_csv"
                out.append(
                    {
                        "scene": scene,
                        "view_name": view,
                        "candidate_source_file": str(residual_path),
                        "strategy_name": strategy_name,
                        "grouping_mode": grouping_mode,
                        "coordinate_mode": coord_mode,
                        "score_column": score_col,
                        "sort_direction": direction,
                        "top_k": top_k,
                        "selected_count": selected_count,
                        "reproduced_count": len(reproduced),
                        "normal_overlap_count": counts.get("normal", 0),
                        "y_flip_overlap_count": counts.get("y_flip", 0),
                        "x_flip_overlap_count": counts.get("x_flip", 0),
                        "xy_swap_overlap_count": counts.get("xy_swap", 0),
                        "xy_swap_y_flip_overlap_count": counts.get("xy_swap_y_flip", 0),
                        "xy_swap_x_flip_overlap_count": counts.get("xy_swap_x_flip", 0),
                        "best_convention": best_conv,
                        "best_overlap_count": best,
                        "best_overlap_rate": best / selected_count if selected_count else "",
                        "exact_reproduction": _bool_text(exact),
                        "interpretation": interpretation,
                    }
                )
    return out


def _membership_rows(scene: str, residual_rows: list[dict[str, str]], schema: dict[str, Any], selected: dict[str, set[tuple[int, int]]], views: list[str]) -> list[dict[str, Any]]:
    columns = list(residual_rows[0].keys()) if residual_rows else []
    coord_modes = _coordinate_modes(columns)
    coord_mode = coord_modes[0] if coord_modes else ""
    view_col = str(schema.get("candidate_view_column", ""))
    grouped = _residual_rows_by_view(residual_rows, view_col, views) if coord_mode else {view: [] for view in views}
    out = []
    for view in views:
        residual_pixels = _pixels_from_rows(grouped.get(view, []), coord_mode) if coord_mode else set()
        counts = _overlap_counts(selected.get(view, set()), residual_pixels)
        best_conv, best = _best_count(counts)
        selected_count = len(selected.get(view, set()))
        rate = best / selected_count if selected_count else 0.0
        out.append(
            {
                "scene": scene,
                "view_name": view,
                "view_group": _group_for_view(view),
                "selected_pixel_count": selected_count,
                "residual_csv_same_pixel_count": counts.get("normal", 0),
                "residual_csv_same_pixel_rate": counts.get("normal", 0) / selected_count if selected_count else "",
                "residual_csv_y_flip_count": counts.get("y_flip", 0),
                "residual_csv_x_flip_count": counts.get("x_flip", 0),
                "residual_csv_xy_swap_count": counts.get("xy_swap", 0),
                "residual_csv_xy_swap_y_flip_count": counts.get("xy_swap_y_flip", 0),
                "residual_csv_xy_swap_x_flip_count": counts.get("xy_swap_x_flip", 0),
                "best_membership_convention": best_conv,
                "best_membership_count": best,
                "best_membership_rate": rate,
                "interpretation": "selected_pixels_present_in_residual_csv"
                if rate == 1.0
                else "selected_pixels_partially_present_in_residual_csv"
                if best > 0
                else "selected_pixels_not_present_in_residual_csv",
            }
        )
    return out


def _hash_rows(scene: str, pr200_dir: Path, selected: dict[str, set[tuple[int, int]]], views: list[str]) -> list[dict[str, Any]]:
    out = []
    for path in sorted(pr200_dir.glob("pr200_*.csv")):
        if path.name == "pr200_pixel_gaussian_contributions.csv":
            continue
        rows = load_csv_rows(path)
        if not rows:
            continue
        columns = list(rows[0].keys())
        coord_modes = _coordinate_modes(columns)
        if not coord_modes:
            continue
        schema = _infer_schema(rows, path)
        view_col = str(schema.get("candidate_view_column", ""))
        grouped = _residual_rows_by_view(rows, view_col, views)
        for coord_mode in coord_modes:
            for view in views:
                candidate = _pixels_from_rows(grouped.get(view, []), coord_mode)
                counts = _overlap_counts(selected.get(view, set()), candidate)
                best_conv, best = _best_count(counts)
                selected_count = len(selected.get(view, set()))
                selected_hash = _pixel_hash(selected.get(view, set()))
                candidate_hash = _pixel_hash(candidate)
                out.append(
                    {
                        "scene": scene,
                        "source_file": str(path),
                        "view_name": view,
                        "coordinate_mode": coord_mode,
                        "row_count": len(grouped.get(view, [])),
                        "unique_pixel_count": len(candidate),
                        "selected_proxy_hash": selected_hash,
                        "candidate_file_hash": candidate_hash,
                        "hash_match": _bool_text(selected_hash == candidate_hash and selected_count == len(candidate)),
                        "overlap_count": best,
                        "overlap_rate": best / selected_count if selected_count else "",
                        "best_transform": best_conv,
                        "notes": "hash compares full per-view pixel set for this file and coordinate mode",
                    }
                )
    return out


def _code_provenance(project_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    for base in [project_root / "viewtrust", project_root / "scripts", project_root / "docs"]:
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if path.suffix not in {".py", ".md"} or not path.is_file():
                continue
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except UnicodeDecodeError:
                continue
            for line_no, text in enumerate(lines, start=1):
                lowered = text.lower()
                matches = [term for term in CODE_TERMS if term.lower() in lowered]
                if not matches:
                    continue
                if "write_csv_rows" in lowered and "pr200_pixel_gaussian_contributions" in lowered:
                    role, confidence = "pixel_gaussian_contribution_generation", "high"
                elif "write_csv_rows" in lowered and "pr200_sparse_pixel_residuals" in lowered:
                    role, confidence = "residual_csv_generation", "high"
                elif "argsort" in lowered or "top_pixels" in lowered:
                    role, confidence = "selected_pixel_generation", "medium"
                elif "residual_l1" in lowered or "residual_weighted_splat" in lowered:
                    role, confidence = "score_column_reference", "medium"
                else:
                    role, confidence = "related_reference", "low"
                rows.append(
                    {
                        "matched_file": str(path.relative_to(project_root)),
                        "line_number": line_no,
                        "matched_text": text.strip(),
                        "inferred_role": role,
                        "confidence": confidence,
                        "notes": "source text match; inspect manually before treating as specification",
                    }
                )
    selected_file = ""
    contribution_file = ""
    residual_file = ""
    for row in rows:
        if row["inferred_role"] == "selected_pixel_generation" and not selected_file:
            selected_file = str(row["matched_file"])
        if row["inferred_role"] == "pixel_gaussian_contribution_generation" and not contribution_file:
            contribution_file = str(row["matched_file"])
        if row["inferred_role"] == "residual_csv_generation" and not residual_file:
            residual_file = str(row["matched_file"])
    summary = {
        "selected_pixel_generation_file": selected_file or "unknown",
        "selected_pixel_generation_function": "_top_residual_pixels" if any("_top_residual_pixels" in str(row["matched_text"]) for row in rows) else "unknown",
        "selected_pixel_sort_column": "residual metric from np.argsort(flat)[::-1]" if any("np.argsort(flat)[::-1]" in str(row["matched_text"]) for row in rows) else "unknown",
        "selected_pixel_sort_direction": "descending" if any("np.argsort(flat)[::-1]" in str(row["matched_text"]) for row in rows) else "unknown",
        "selected_pixel_grouping": "per selected view" if selected_file else "unknown",
        "selected_pixel_coordinate_source": "pixel_x/pixel_y from residual metric indices" if selected_file else "unknown",
        "pixel_gaussian_contribution_generation_file": contribution_file or "unknown",
        "residual_csv_generation_file": residual_file or "unknown",
        "provenance_confidence": "high" if selected_file and contribution_file and residual_file else "low",
        "caveats": "code provenance describes intended PR20 writer; CSV reproduction remains the stronger artifact-level check",
    }
    return rows, summary


def _diagnosis_rows(
    *,
    scene: str,
    views: list[str],
    selected: dict[str, set[tuple[int, int]]],
    membership_rows: list[dict[str, Any]],
    reproduction_rows: list[dict[str, Any]],
    hash_rows: list[dict[str, Any]],
    code_confidence: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    membership_by_view = {str(row["view_name"]): row for row in membership_rows}
    repro_by_view: dict[str, dict[str, Any]] = {}
    for view in views:
        candidates = [row for row in reproduction_rows if row.get("view_name") == view]
        if candidates:
            repro_by_view[view] = max(candidates, key=lambda row: _safe_float(row.get("best_overlap_rate")) or 0.0)
    hash_match_by_view = {
        view: any(row.get("view_name") == view and row.get("hash_match") == "true" for row in hash_rows)
        for view in views
    }
    rows = []
    verified = partial = unresolved = 0
    best_total = 0
    selected_total = 0
    best_strategy_counts: Counter[str] = Counter()
    for view in views:
        selected_count = len(selected.get(view, set()))
        selected_total += selected_count
        membership = membership_by_view.get(view, {})
        repro = repro_by_view.get(view, {})
        overlap = _safe_int(repro.get("best_overlap_count"))
        best_total += overlap
        rate = _safe_float(repro.get("best_overlap_rate")) or 0.0
        hash_match = hash_match_by_view[view]
        if selected_count and (rate == 1.0 or hash_match):
            status = "provenance_verified_from_residual_csv"
            likely = "pr200_sparse_pixel_residuals.csv"
            verified += 1
        elif overlap > 0 or _safe_int(membership.get("best_membership_count")) > 0:
            status = "provenance_partially_verified_from_residual_csv"
            likely = "pr200_sparse_pixel_residuals.csv_partial"
            partial += 1
        elif code_confidence == "high":
            status = "provenance_verified_from_code_but_not_csv"
            likely = "pr20_code_path_residual_top_pixels_but_artifact_not_reproduced"
            unresolved += 1
        else:
            status = "provenance_unresolved_selected_pixels_not_reproduced"
            likely = "unknown"
            unresolved += 1
        if repro:
            best_strategy_counts[str(repro.get("strategy_name", ""))] += 1
        rows.append(
            {
                "scene": scene,
                "view_name": view,
                "view_group": _group_for_view(view),
                "selected_pixels_from_proxy_count": selected_count,
                "residual_csv_available": _bool_text(bool(reproduction_rows)),
                "selected_pixels_present_in_residual_csv": _bool_text(_safe_int(membership.get("residual_csv_same_pixel_count")) == selected_count and selected_count > 0),
                "best_residual_reproduction_strategy": repro.get("strategy_name", ""),
                "best_residual_reproduction_overlap": overlap,
                "best_residual_reproduction_rate": rate,
                "best_residual_reproduction_convention": repro.get("best_convention", ""),
                "pixel_set_hash_match_found": _bool_text(hash_match),
                "code_provenance_confidence": code_confidence,
                "likely_pr20_selected_pixel_source": likely,
                "provenance_status": status,
                "exact_evidence_allowed": "false",
                "drums_ready_for_pr212": "false",
                "recommended_next_step": "verify PR20 selected-pixel generation against residual CSV and normal-coordinate exact replay before using drums in PR21.2",
                "caveat": "Provenance alone is not exact evidence; PR20 proxy rows are not exact contributor rows.",
            }
        )
    best_strategy = best_strategy_counts.most_common(1)[0][0] if best_strategy_counts else ""
    if verified == len(views) and views:
        overall = "provenance_verified_from_residual_csv"
    elif verified or partial:
        overall = "provenance_partially_verified_from_residual_csv"
    else:
        overall = "provenance_unresolved_selected_pixels_not_reproduced"
    return rows, {
        "verified": verified,
        "partial": partial,
        "unresolved": unresolved,
        "best_total": best_total,
        "selected_total": selected_total,
        "best_rate": best_total / selected_total if selected_total else 0.0,
        "best_strategy": best_strategy,
        "overall": overall,
    }


def _validate_pr211f(summary: dict[str, Any], warnings: list[str]) -> None:
    expected = {
        "scene": "drums",
        "likely_failure_mode_overall": "mixed_coordinate_candidate_and_no_raw_contributors",
        "exact_evidence_allowed_for_drums": False,
        "drums_ready_for_pr212": False,
    }
    for key, value in expected.items():
        if summary.get(key) != value:
            warnings.append(f"PR21.1f-a {key}={summary.get(key)!r}; expected {value!r}")


def _manifest_rows(output_dir: Path, pr200_dir: Path, pr211f_dir: Path, pr211_dir: Path, run_dir: Path) -> list[dict[str, Any]]:
    items = [
        ("pr200_dir", pr200_dir, True, "input"),
        ("pr211f_dir", pr211f_dir, True, "input"),
        ("pr211_dir", pr211_dir, True, "input"),
        ("run_dir", run_dir, True, "input"),
    ]
    items.extend((name, output_dir / name, True, "output_pr211g") for name in OUTPUT_FILES)
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
        "# PR21.1g PR20 Selected-Pixel Provenance Audit",
        "",
        "PR21.1g is observation-only. It does not implement defense, rejection, reweighting, update suppression, or densification gating.",
        "",
        "Diagnostic coordinate transforms do not create exact evidence, and PR20 proxy rows are not exact contributor rows.",
        "",
        "## Purpose",
        "PR21.1g checks whether PR20 selected pixels in `pr200_pixel_gaussian_contributions.csv` can be reproduced from PR20 residual/proxy CSV artifacts and source-code provenance.",
        "",
        "## Summary",
        f"- Proxy selected pixels: `{summary.get('proxy_selected_pixel_total')}`",
        f"- Residual CSV available: `{summary.get('residual_csv_available')}`",
        f"- Residual CSV rows: `{summary.get('residual_csv_row_count')}`",
        f"- Best reproduction strategy: `{summary.get('best_reproduction_strategy_overall')}`",
        f"- Best reproduction overlap rate: `{summary.get('best_reproduction_overlap_rate')}`",
        f"- Provenance status: `{summary.get('provenance_status_overall')}`",
        f"- Code provenance confidence: `{summary.get('code_provenance_confidence')}`",
        f"- Drums ready for PR21.2: `{summary.get('drums_ready_for_pr212')}`",
        "",
        "## Why Drums Remains Excluded",
        "Drums remains excluded unless PR20 selected-pixel provenance is verified and PR21 exact replay validates normal-coordinate selected-pixel hits.",
        "",
        "## Recommended Next Step",
        str(summary.get("recommended_next_step", "")),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def build_pr211g_pr20_selected_pixel_provenance_audit(
    *,
    pr200_dir: Path,
    pr211f_dir: Path,
    pr211_dir: Path,
    run_dir: Path,
    output_dir: Path,
    scene: str = "drums",
    condition: str = "corrupt_occluder",
    subset_name: str = "seed_20260710",
    views: list[str] | None = None,
    top_pixels_per_view: int = 128,
    max_contributors_per_pixel: int = 16,
    write_markdown: bool = False,
) -> tuple[dict[str, Any], int]:
    del write_markdown
    project_root = Path(__file__).resolve().parents[2]
    views = views or ["train_004", "train_009", "train_012", "train_017", "train_007", "train_013"]
    output_dir.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []

    proxy_path = pr200_dir / "pr200_pixel_gaussian_contributions.csv"
    if not proxy_path.exists():
        raise FileNotFoundError(f"missing required PR20 proxy contribution CSV: {proxy_path}")
    pr211f_summary = load_json(pr211f_dir / "pr211f_drums_selected_pixel_alignment_summary.json")
    _validate_pr211f(pr211f_summary, warnings)

    proxy_audit, selected, _proxy_by_view, proxy_warnings = _selected_from_proxy(
        scene=scene,
        pr200_dir=pr200_dir,
        views=views,
        top_pixels_per_view=top_pixels_per_view,
        max_contributors_per_pixel=max_contributors_per_pixel,
    )
    warnings.extend(proxy_warnings)

    residual_path = pr200_dir / "pr200_sparse_pixel_residuals.csv"
    residual_rows = load_csv_rows(residual_path)
    schema = _infer_schema(residual_rows, residual_path)
    reproduction = _reproduction_rows(
        scene=scene,
        residual_path=residual_path,
        residual_rows=residual_rows,
        schema=schema,
        selected=selected,
        views=views,
        top_k=top_pixels_per_view,
    )
    membership = _membership_rows(scene, residual_rows, schema, selected, views)
    hash_rows = _hash_rows(scene, pr200_dir, selected, views)
    code_rows, code_summary = _code_provenance(project_root)
    diagnosis, diag_summary = _diagnosis_rows(
        scene=scene,
        views=views,
        selected=selected,
        membership_rows=membership,
        reproduction_rows=reproduction,
        hash_rows=hash_rows,
        code_confidence=str(code_summary.get("provenance_confidence", "unknown")),
    )
    best_repro = max(reproduction, key=lambda row: _safe_float(row.get("best_overlap_rate")) or 0.0) if reproduction else {}
    proxy_total = sum(len(pixels) for pixels in selected.values())
    residual_available = bool(residual_rows)
    summary = {
        "schema_name": "viewtrust.pr211g.pr20_selected_pixel_provenance.summary",
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
        "pr211f_input_dir": str(pr211f_dir),
        "pr211_input_dir": str(pr211_dir),
        "run_dir": str(run_dir),
        "selected_view_count": len(views),
        "proxy_selected_pixel_total": proxy_total,
        "residual_csv_available": residual_available,
        "residual_csv_row_count": len(residual_rows),
        "residual_csv_schema_inferred": bool(schema.get("candidate_pixel_x_column") or schema.get("candidate_pixel_id_column")),
        "best_reproduction_strategy_overall": best_repro.get("strategy_name", ""),
        "best_reproduction_overlap_total": diag_summary["best_total"],
        "best_reproduction_overlap_rate": diag_summary["best_rate"],
        "views_with_verified_provenance": diag_summary["verified"],
        "views_with_partial_provenance": diag_summary["partial"],
        "views_with_unresolved_provenance": diag_summary["unresolved"],
        "code_provenance_confidence": code_summary.get("provenance_confidence", "unknown"),
        "likely_pr20_selected_pixel_source_overall": "pr200_sparse_pixel_residuals.csv"
        if diag_summary["verified"] or diag_summary["partial"]
        else "unknown",
        "provenance_status_overall": diag_summary["overall"],
        "exact_evidence_allowed_for_drums": False,
        "drums_ready_for_pr212": False,
        "recommended_next_step": "Resolve PR20 selected-pixel provenance and validate normal-coordinate exact replay before including drums in PR21.2.",
        "warnings": warnings,
    }

    write_json(output_dir / "pr211g_pr20_selected_pixel_provenance_summary.json", summary)
    write_csv_rows(output_dir / "pr211g_pr20_selected_from_proxy_contributions.csv", proxy_audit, PROXY_FIELDS)
    write_csv_rows(output_dir / "pr211g_pr20_residual_csv_schema_audit.csv", [schema], SCHEMA_FIELDS)
    write_csv_rows(output_dir / "pr211g_pr20_residual_to_selected_reproduction.csv", reproduction, REPRO_FIELDS)
    write_csv_rows(output_dir / "pr211g_pr20_selected_pixel_membership_in_residual_csv.csv", membership, MEMBERSHIP_FIELDS)
    write_csv_rows(output_dir / "pr211g_pr20_code_provenance_audit.csv", code_rows, CODE_AUDIT_FIELDS)
    write_json(output_dir / "pr211g_pr20_code_provenance_summary.json", code_summary)
    write_csv_rows(output_dir / "pr211g_pr20_pixel_set_hash_comparison.csv", hash_rows, HASH_FIELDS)
    write_csv_rows(output_dir / "pr211g_pr20_selected_pixel_provenance_diagnosis.csv", diagnosis, DIAGNOSIS_FIELDS)
    _write_report(output_dir / "pr211g_pr20_selected_pixel_provenance_report.md", summary)
    manifest = output_dir / "artifact_manifest.csv"
    write_csv_rows(manifest, _manifest_rows(output_dir, pr200_dir, pr211f_dir, pr211_dir, run_dir), MANIFEST_FIELDS)
    write_csv_rows(manifest, _manifest_rows(output_dir, pr200_dir, pr211f_dir, pr211_dir, run_dir), MANIFEST_FIELDS)
    return summary, 0
