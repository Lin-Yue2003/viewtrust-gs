"""PR21.2a chair exact-vs-proxy ID namespace audit."""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from viewtrust.analysis.clean_prior_normalization import load_csv_rows, load_json, write_csv_rows, write_json


OUTPUT_FILES = [
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

CHECKPOINT_FIELDS = [
    "source_path",
    "source_type",
    "exists",
    "preferred",
    "detected_gaussian_count",
    "detection_method",
    "iteration_hint",
    "load_status",
    "notes",
]

SCHEMA_FIELDS = [
    "source_name",
    "source_file",
    "exists",
    "row_count",
    "columns_semicolon",
    "gaussian_id_column",
    "view_column",
    "pixel_x_column",
    "pixel_y_column",
    "id_parse_success_count",
    "id_parse_failure_count",
    "notes",
]

RANGE_FIELDS = [
    "id_group",
    "source_file",
    "view_name",
    "row_count",
    "unique_id_count",
    "min_id",
    "max_id",
    "negative_id_count",
    "non_integer_id_count",
    "zero_based_in_range_count",
    "zero_based_out_of_range_count",
    "one_based_in_range_count",
    "one_based_out_of_range_count",
    "checkpoint_gaussian_count",
    "likely_index_base",
    "range_status",
    "notes",
]

SAME_PIXEL_FIELDS = [
    "scene",
    "view_name",
    "pixel_x",
    "pixel_y",
    "exact_id_count",
    "proxy_id_count",
    "intersection_count",
    "exact_only_count",
    "proxy_only_count",
    "jaccard",
    "exact_recall_by_proxy",
    "proxy_precision_against_exact",
    "exact_ids_in_checkpoint_range",
    "proxy_ids_in_checkpoint_range",
    "both_sources_same_namespace_plausible",
    "interpretation",
]

SAMPLE_FIELDS = [
    "sample_group",
    "gaussian_id",
    "source",
    "zero_based_index_exists",
    "one_based_index_exists",
    "checkpoint_gaussian_count",
    "x",
    "y",
    "z",
    "opacity",
    "lookup_status",
    "notes",
]

PROXY_SEMANTICS_FIELDS = [
    "view_name",
    "pixel_count",
    "row_count",
    "unique_proxy_id_count",
    "contributors_per_pixel_min",
    "contributors_per_pixel_max",
    "contributors_per_pixel_mean",
    "min_proxy_id",
    "max_proxy_id",
    "small_integer_rank_like",
    "repeated_same_id_pool_across_pixels",
    "repeated_same_id_pool_across_views",
    "rank_column_present",
    "gaussian_id_equals_rank_pattern",
    "global_index_plausible",
    "local_rank_plausible",
    "conclusion",
    "notes",
]

CODE_AUDIT_FIELDS = ["matched_file", "line_number", "matched_text", "inferred_role", "confidence", "notes"]

DIAGNOSIS_FIELDS = [
    "diagnosis_id",
    "status",
    "evidence",
    "confidence",
    "paper_safe_wording",
    "unsafe_wording_to_avoid",
    "recommended_next_step",
]

MANIFEST_FIELDS = ["relative_path", "path", "exists", "file_type", "size_bytes", "required", "artifact_group"]

CODE_TERMS = [
    "gaussian_id",
    "pr200_pixel_gaussian_contributions",
    "PIXEL_GAUSSIAN_FIELDS",
    "pr211_exact_pixel_gaussian_contributions",
    "exact_pixel_gaussian",
    "unpacked_direct_gaussian_index",
    "gaussian_id_mapping_mode",
    "flatten_ids",
    "gaussian_ids",
    "direct_gaussian_index",
]


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


def read_ply_vertex_count(path: Path) -> int | None:
    try:
        with path.open("rb") as handle:
            for _ in range(4096):
                raw = handle.readline()
                if not raw:
                    break
                line = raw.decode("utf-8", errors="replace").strip()
                parts = line.split()
                if len(parts) >= 3 and parts[0] == "element" and parts[1] == "vertex":
                    return int(parts[2])
                if line == "end_header":
                    break
    except (OSError, ValueError):
        return None
    return None


def _warn_if(warnings: list[str], condition: bool, message: str) -> None:
    if condition:
        warnings.append(message)


def _id_value(value: Any) -> int | None:
    text = str(value).strip()
    if text == "":
        return None
    try:
        number = int(text)
    except ValueError:
        try:
            as_float = float(text)
        except ValueError:
            return None
        if not as_float.is_integer():
            return None
        number = int(as_float)
    return number


def _id_column(columns: list[str]) -> str:
    for name in ["gaussian_id", "gid", "global_gaussian_id"]:
        if name in columns:
            return name
    for col in columns:
        if "gaussian" in col.lower() and "id" in col.lower():
            return col
    return ""


def _column(columns: list[str], names: list[str]) -> str:
    lowered = {col.lower(): col for col in columns}
    for name in names:
        if name.lower() in lowered:
            return lowered[name.lower()]
    return ""


def _pixel_key(row: dict[str, Any]) -> tuple[str, int, int]:
    return str(row.get("view_name", "")), _safe_int(row.get("pixel_x")), _safe_int(row.get("pixel_y"))


def _load_id_rows(path: Path, source_name: str) -> tuple[list[dict[str, str]], dict[str, Any], list[int], int]:
    rows = load_csv_rows(path)
    columns = list(rows[0].keys()) if rows else []
    id_col = _id_column(columns)
    view_col = _column(columns, ["view_name", "view", "image_name"])
    x_col = _column(columns, ["pixel_x", "x", "col"])
    y_col = _column(columns, ["pixel_y", "y", "row"])
    ids: list[int] = []
    failures = 0
    for row in rows:
        value = _id_value(row.get(id_col, "")) if id_col else None
        if value is None:
            failures += 1
        else:
            ids.append(value)
    schema = {
        "source_name": source_name,
        "source_file": str(path),
        "exists": _bool_text(path.exists()),
        "row_count": len(rows),
        "columns_semicolon": ";".join(columns),
        "gaussian_id_column": id_col,
        "view_column": view_col,
        "pixel_x_column": x_col,
        "pixel_y_column": y_col,
        "id_parse_success_count": len(ids),
        "id_parse_failure_count": failures,
        "notes": "column inference is name-based",
    }
    return rows, schema, ids, failures


def _discover_checkpoint_inventory(run_dir: Path) -> tuple[list[dict[str, Any]], int | None, str, str]:
    rows = []
    best_count: int | None = None
    best_path = ""
    best_score = -1
    for path in sorted(run_dir.rglob("*")) if run_dir.exists() else []:
        if not path.is_file() or path.suffix.lower() not in {".ply", ".pt", ".pth", ".npz", ".npy", ".json"}:
            continue
        suffix = path.suffix.lower().lstrip(".")
        iteration_hint = "iteration_700" if "iteration_700" in str(path) or "ours_700" in str(path) else ""
        count: int | None = None
        method = ""
        status = "metadata_only"
        notes = ""
        if suffix == "ply":
            count = read_ply_vertex_count(path)
            method = "ply_header_element_vertex"
            status = "loaded_header" if count is not None else "failed_header"
        elif suffix == "json":
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                for key in ["gaussian_count", "num_gaussians", "point_count", "num_points"]:
                    if isinstance(data, dict) and key in data:
                        count = _safe_int(data[key])
                        method = f"json_key:{key}"
                        break
                status = "loaded_json"
            except Exception as exc:
                status = f"json_load_failed:{type(exc).__name__}"
        else:
            status = "not_loaded_checkpoint_like_file"
            notes = "torch/numpy checkpoint loading intentionally avoided unless metadata format is known"
        preferred_score = (100 if iteration_hint else 0) + (10 if suffix == "ply" else 0) + (1 if count is not None else 0)
        preferred = count is not None and preferred_score > best_score
        if preferred:
            best_score = preferred_score
            best_count = count
            best_path = str(path)
        rows.append(
            {
                "source_path": str(path),
                "source_type": suffix,
                "exists": "true",
                "preferred": "false",
                "detected_gaussian_count": count if count is not None else "",
                "detection_method": method,
                "iteration_hint": iteration_hint,
                "load_status": status,
                "notes": notes,
            }
        )
    for row in rows:
        row["preferred"] = _bool_text(row["source_path"] == best_path and best_count is not None)
    confidence = "high" if best_count is not None and best_path.endswith(".ply") else "medium" if best_count is not None else "none"
    return rows, best_count, best_path, confidence


def _range_status(ids: list[int], failures: int, checkpoint_count: int | None) -> tuple[str, str, dict[str, int]]:
    unique = sorted(set(ids))
    negative = sum(1 for value in ids if value < 0)
    if not ids:
        return "", "empty_id_set", {"negative": negative, "z_in": 0, "z_out": 0, "o_in": 0, "o_out": 0}
    if failures and failures >= len(ids):
        return "", "ids_not_numeric", {"negative": negative, "z_in": 0, "z_out": 0, "o_in": 0, "o_out": 0}
    if checkpoint_count is None:
        return "unknown", "no_checkpoint_count_available", {"negative": negative, "z_in": 0, "z_out": 0, "o_in": 0, "o_out": 0}
    z_in = sum(1 for value in ids if 0 <= value < checkpoint_count)
    z_out = len(ids) - z_in
    o_in = sum(1 for value in ids if 1 <= value <= checkpoint_count)
    o_out = len(ids) - o_in
    if z_in == len(ids):
        return "zero_based", "zero_based_global_index_plausible", {"negative": negative, "z_in": z_in, "z_out": z_out, "o_in": o_in, "o_out": o_out}
    if o_in == len(ids):
        return "one_based", "one_based_global_index_plausible", {"negative": negative, "z_in": z_in, "z_out": z_out, "o_in": o_in, "o_out": o_out}
    return "unknown", "out_of_checkpoint_range", {"negative": negative, "z_in": z_in, "z_out": z_out, "o_in": o_in, "o_out": o_out}


def _range_row(id_group: str, source_file: Path, view_name: str, ids: list[int], failures: int, checkpoint_count: int | None, row_count: int) -> dict[str, Any]:
    base, status, counts = _range_status(ids, failures, checkpoint_count)
    return {
        "id_group": id_group,
        "source_file": str(source_file),
        "view_name": view_name,
        "row_count": row_count,
        "unique_id_count": len(set(ids)),
        "min_id": min(ids) if ids else "",
        "max_id": max(ids) if ids else "",
        "negative_id_count": counts["negative"],
        "non_integer_id_count": failures,
        "zero_based_in_range_count": counts["z_in"],
        "zero_based_out_of_range_count": counts["z_out"],
        "one_based_in_range_count": counts["o_in"],
        "one_based_out_of_range_count": counts["o_out"],
        "checkpoint_gaussian_count": checkpoint_count if checkpoint_count is not None else "",
        "likely_index_base": base,
        "range_status": status,
        "notes": "range audit only; code provenance and rank heuristics decide namespace confidence",
    }


def _ids_by_pixel(rows: list[dict[str, str]]) -> dict[tuple[str, int, int], set[int]]:
    out: dict[tuple[str, int, int], set[int]] = defaultdict(set)
    for row in rows:
        gid = _id_value(row.get("gaussian_id", ""))
        if gid is None:
            continue
        out[_pixel_key(row)].add(gid)
    return out


def _ratio(num: int, den: int) -> float | str:
    return num / den if den else ""


def _in_range(ids: set[int], checkpoint_count: int | None) -> bool:
    if checkpoint_count is None or not ids:
        return False
    return all(0 <= value < checkpoint_count for value in ids)


def _same_pixel_rows(scene: str, exact_rows: list[dict[str, str]], proxy_rows: list[dict[str, str]], checkpoint_count: int | None, namespace_plausible: bool) -> list[dict[str, Any]]:
    exact_by_pixel = _ids_by_pixel(exact_rows)
    proxy_by_pixel = _ids_by_pixel(proxy_rows)
    out = []
    for key in sorted(exact_by_pixel):
        view, x, y = key
        exact = exact_by_pixel[key]
        proxy = proxy_by_pixel.get(key, set())
        inter = exact & proxy
        union = exact | proxy
        exact_range = _in_range(exact, checkpoint_count)
        proxy_range = _in_range(proxy, checkpoint_count)
        if not exact:
            interp = "missing_exact_ids"
        elif not proxy:
            interp = "missing_proxy_ids"
        elif not exact_range or not proxy_range:
            interp = "id_range_problem"
        elif inter:
            interp = "nonzero_overlap_same_namespace_plausible" if namespace_plausible else "zero_overlap_but_namespace_unverified"
        elif namespace_plausible:
            interp = "zero_overlap_same_namespace_plausible"
        else:
            interp = "zero_overlap_but_namespace_unverified"
        out.append(
            {
                "scene": scene,
                "view_name": view,
                "pixel_x": x,
                "pixel_y": y,
                "exact_id_count": len(exact),
                "proxy_id_count": len(proxy),
                "intersection_count": len(inter),
                "exact_only_count": len(exact - proxy),
                "proxy_only_count": len(proxy - exact),
                "jaccard": len(inter) / len(union) if union else "",
                "exact_recall_by_proxy": _ratio(len(inter), len(exact)),
                "proxy_precision_against_exact": _ratio(len(inter), len(proxy)),
                "exact_ids_in_checkpoint_range": _bool_text(exact_range),
                "proxy_ids_in_checkpoint_range": _bool_text(proxy_range),
                "both_sources_same_namespace_plausible": _bool_text(namespace_plausible and exact_range and proxy_range),
                "interpretation": interp,
            }
        )
    return out


def _proxy_semantics_rows(proxy_rows: list[dict[str, str]], checkpoint_count: int | None) -> tuple[list[dict[str, Any]], str]:
    by_view: dict[str, list[dict[str, str]]] = defaultdict(list)
    all_pixel_pools = []
    for row in proxy_rows:
        by_view[str(row.get("view_name", ""))].append(row)
    for rows in by_view.values():
        pools: dict[tuple[str, int, int], set[int]] = defaultdict(set)
        for row in rows:
            gid = _id_value(row.get("gaussian_id", ""))
            if gid is not None:
                pools[_pixel_key(row)].add(gid)
        all_pixel_pools.extend(frozenset(pool) for pool in pools.values())
    repeated_across_views = bool(all_pixel_pools and len(set(all_pixel_pools)) <= max(1, len(all_pixel_pools) // 3))
    out = []
    conclusions = []
    for view, rows in sorted(by_view.items()):
        pools: dict[tuple[str, int, int], list[int]] = defaultdict(list)
        for row in rows:
            gid = _id_value(row.get("gaussian_id", ""))
            if gid is not None:
                pools[_pixel_key(row)].append(gid)
        pool_sets = [frozenset(ids) for ids in pools.values()]
        ids = [gid for vals in pools.values() for gid in vals]
        counts = [len(vals) for vals in pools.values()]
        rank_present = any("rank" in key.lower() for row in rows[:1] for key in row)
        equals_rank = False
        if rank_present:
            for row in rows:
                gid = _id_value(row.get("gaussian_id", ""))
                ranks = [_id_value(value) for key, value in row.items() if "rank" in key.lower()]
                if gid is not None and gid in [rank for rank in ranks if rank is not None]:
                    equals_rank = True
                    break
        unique = set(ids)
        small_rank_like = bool(unique and min(unique) >= 0 and max(unique) <= 32 and len(unique) <= 33)
        repeated_pixel_pool = bool(pool_sets and len(set(pool_sets)) <= max(1, len(pool_sets) // 3))
        global_ok = bool(ids and checkpoint_count is not None and all(0 <= gid < checkpoint_count for gid in ids) and not (small_rank_like and repeated_pixel_pool))
        local_rank = bool(small_rank_like and (repeated_pixel_pool or repeated_across_views or equals_rank))
        if global_ok:
            conclusion = "proxy_ids_plausibly_global_checkpoint_indices"
        elif local_rank:
            conclusion = "proxy_ids_plausibly_local_candidate_ranks"
        elif ids and checkpoint_count is not None and not all(0 <= gid < checkpoint_count for gid in ids):
            conclusion = "proxy_ids_invalid"
        else:
            conclusion = "proxy_ids_ambiguous"
        conclusions.append(conclusion)
        out.append(
            {
                "view_name": view,
                "pixel_count": len(pools),
                "row_count": len(rows),
                "unique_proxy_id_count": len(unique),
                "contributors_per_pixel_min": min(counts) if counts else "",
                "contributors_per_pixel_max": max(counts) if counts else "",
                "contributors_per_pixel_mean": sum(counts) / len(counts) if counts else "",
                "min_proxy_id": min(ids) if ids else "",
                "max_proxy_id": max(ids) if ids else "",
                "small_integer_rank_like": _bool_text(small_rank_like),
                "repeated_same_id_pool_across_pixels": _bool_text(repeated_pixel_pool),
                "repeated_same_id_pool_across_views": _bool_text(repeated_across_views),
                "rank_column_present": _bool_text(rank_present),
                "gaussian_id_equals_rank_pattern": _bool_text(equals_rank),
                "global_index_plausible": _bool_text(global_ok),
                "local_rank_plausible": _bool_text(local_rank),
                "conclusion": conclusion,
                "notes": "heuristic only; final namespace diagnosis also uses range and code provenance",
            }
        )
    if conclusions and all(item == "proxy_ids_plausibly_global_checkpoint_indices" for item in conclusions):
        overall = "global_checkpoint_gaussian_index"
    elif any(item == "proxy_ids_plausibly_local_candidate_ranks" for item in conclusions):
        overall = "local_candidate_rank"
    elif any(item == "proxy_ids_invalid" for item in conclusions):
        overall = "unknown"
    else:
        overall = "unknown"
    return out, overall


def _code_provenance(project_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    for base in [project_root / "viewtrust", project_root / "scripts"]:
        if not base.exists():
            continue
        for path in base.rglob("*.py"):
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except UnicodeDecodeError:
                continue
            for line_no, text in enumerate(lines, start=1):
                lowered = text.lower()
                if not any(term.lower() in lowered for term in CODE_TERMS):
                    continue
                if "pr200_pixel_gaussian_contributions.csv" in text:
                    role, confidence = "pr20_gaussian_id_writer", "high"
                elif "unpacked_direct_gaussian_index" in text or "direct gaussian ids" in text:
                    role, confidence = "pr211_exact_direct_global_index", "high"
                elif "metadata.gaussian_ids compact mapping" in text:
                    role, confidence = "pr211_exact_compact_mapping_branch", "medium"
                elif "flatten_ids" in text or "gaussian_ids" in text:
                    role, confidence = "exact_replay_id_source", "medium"
                else:
                    role, confidence = "related_reference", "low"
                rows.append(
                    {
                        "matched_file": str(path.relative_to(project_root)),
                        "line_number": line_no,
                        "matched_text": text.strip(),
                        "inferred_role": role,
                        "confidence": confidence,
                        "notes": "source text match; inspect manually before treating as formal proof",
                    }
                )
    pr20_file = next((row["matched_file"] for row in rows if row["inferred_role"] == "pr20_gaussian_id_writer"), "unknown")
    pr211_file = next((row["matched_file"] for row in rows if row["inferred_role"] == "pr211_exact_direct_global_index"), "unknown")
    pr20_sem = "global_checkpoint_gaussian_index" if pr20_file != "unknown" else "unknown"
    pr211_sem = "global_checkpoint_gaussian_index" if pr211_file != "unknown" else "unknown"
    summary = {
        "pr20_gaussian_id_writer_file": pr20_file,
        "pr20_gaussian_id_semantics_from_code": pr20_sem,
        "pr20_code_confidence": "medium" if pr20_file != "unknown" else "low",
        "pr211_exact_gaussian_id_writer_file": pr211_file,
        "pr211_exact_gaussian_id_semantics_from_code": pr211_sem,
        "pr211_code_confidence": "high" if pr211_file != "unknown" else "low",
        "code_namespace_conclusion": "common_global_namespace_supported_by_code" if pr20_sem == pr211_sem == "global_checkpoint_gaussian_index" else "code_namespace_inconclusive",
        "caveats": "PR20 proxy rows may inherit IDs from upstream grouped Gaussian identity tables; range and rank heuristics remain necessary.",
    }
    return rows, summary


def _sample_lookup_rows(exact_ids: set[int], proxy_ids: set[int], checkpoint_count: int | None, sample_count: int) -> list[dict[str, Any]]:
    rng = random.Random(0)
    groups = {
        "exact_only": sorted(exact_ids - proxy_ids),
        "proxy_only": sorted(proxy_ids - exact_ids),
        "intersection": sorted(exact_ids & proxy_ids),
        "exact_random": sorted(exact_ids),
        "proxy_random": sorted(proxy_ids),
    }
    out = []
    for group, ids in groups.items():
        sample = ids if len(ids) <= sample_count else rng.sample(ids, sample_count)
        for gid in sorted(sample):
            z_exists = checkpoint_count is not None and 0 <= gid < checkpoint_count
            o_exists = checkpoint_count is not None and 1 <= gid <= checkpoint_count
            out.append(
                {
                    "sample_group": group,
                    "gaussian_id": gid,
                    "source": "pr211_exact" if group.startswith("exact") else "pr20_proxy" if group.startswith("proxy") else "both",
                    "zero_based_index_exists": _bool_text(z_exists),
                    "one_based_index_exists": _bool_text(o_exists),
                    "checkpoint_gaussian_count": checkpoint_count if checkpoint_count is not None else "",
                    "x": "",
                    "y": "",
                    "z": "",
                    "opacity": "",
                    "lookup_status": "range_exists" if z_exists else "range_missing_or_no_checkpoint_count",
                    "notes": "PLY header count validates index existence only; full vertex attributes not loaded",
                }
            )
    return out


def _diagnosis_rows(common_supported: bool, common_conf: str, zero_safe: bool, checkpoint_count: int | None, pr20_sem: str, pr211_sem: str) -> list[dict[str, Any]]:
    if checkpoint_count is None:
        common_status = "namespace_inconclusive_missing_checkpoint_count"
    elif pr20_sem == "local_candidate_rank":
        common_status = "proxy_ids_not_global_namespace_zero_overlap_claim_not_safe"
    elif pr211_sem != "global_checkpoint_gaussian_index":
        common_status = "exact_ids_not_global_namespace_zero_overlap_claim_not_safe"
    elif common_supported and common_conf == "high":
        common_status = "same_global_gaussian_id_namespace_supported"
    else:
        common_status = "same_namespace_plausible_but_code_semantics_needs_review"
    zero_status = "zero_overlap_claim_supported_within_exact_available_scope" if zero_safe else "zero_overlap_claim_not_safe_until_namespace_resolved"
    return [
        {
            "diagnosis_id": "pr20_proxy_id_namespace",
            "status": pr20_sem,
            "evidence": "PR20 range audit, proxy ID semantics heuristic, and code provenance",
            "confidence": common_conf if pr20_sem == "global_checkpoint_gaussian_index" else "medium",
            "paper_safe_wording": "PR20 proxy IDs are treated as global checkpoint indices only if range and semantics checks support that interpretation.",
            "unsafe_wording_to_avoid": "PR20 proxy IDs are exact contributors.",
            "recommended_next_step": "Inspect PR20 upstream candidate source if proxy semantics remain ambiguous.",
        },
        {
            "diagnosis_id": "pr211_exact_id_namespace",
            "status": pr211_sem,
            "evidence": "PR21 exact range audit and exact replay code provenance",
            "confidence": common_conf if pr211_sem == "global_checkpoint_gaussian_index" else "medium",
            "paper_safe_wording": "PR21 exact IDs are contributor IDs from exact sparse replay under the audited namespace.",
            "unsafe_wording_to_avoid": "PR21 exact rows include contribution magnitude.",
            "recommended_next_step": "Proceed to contribution magnitude only after ID namespace is validated.",
        },
        {
            "diagnosis_id": "pr20_vs_pr211_common_namespace",
            "status": common_status,
            "evidence": "Checkpoint range checks plus code and proxy-rank heuristics",
            "confidence": common_conf,
            "paper_safe_wording": "PR20 proxy IDs and PR21 exact IDs are in the same global Gaussian namespace only within the audited evidence constraints.",
            "unsafe_wording_to_avoid": "All proxy IDs are globally false positives.",
            "recommended_next_step": "Run PR21.4 if supported, otherwise repair namespace export.",
        },
        {
            "diagnosis_id": "pr212_zero_overlap_claim",
            "status": zero_status,
            "evidence": "PR21.2 zero-overlap metrics and PR21.2a namespace audit",
            "confidence": common_conf if zero_safe else "low",
            "paper_safe_wording": "Zero overlap is safe only within chair exact-available pixels/views under the verified namespace.",
            "unsafe_wording_to_avoid": "The proxy method is always wrong.",
            "recommended_next_step": "Use paper-safe scoped wording if namespace is supported.",
        },
        {
            "diagnosis_id": "proxy_intervention_safety",
            "status": "proxy_ids_not_safe_for_intervention",
            "evidence": "Observation-only ID audit; no contribution magnitude or intervention experiment",
            "confidence": "high",
            "paper_safe_wording": "Proxy IDs remain diagnostic-only.",
            "unsafe_wording_to_avoid": "Proxy IDs can now be used to reject or downweight views.",
            "recommended_next_step": "Keep intervention work separate from namespace interpretation.",
        },
    ]


def _write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# PR21.2a Chair ID Namespace Audit",
        "",
        "## Purpose",
        "PR21.2 zero-overlap requires namespace validation because exact and proxy IDs must refer to the same checkpoint Gaussian index space before mismatch can be interpreted.",
        "",
        "## Checkpoint Gaussian Count",
        f"Detected checkpoint Gaussian count: `{summary.get('checkpoint_gaussian_count')}` from `{summary.get('checkpoint_gaussian_count_source')}` with confidence `{summary.get('checkpoint_count_confidence')}`.",
        "",
        "## PR20 Proxy ID Range",
        f"PR20 proxy IDs numeric: `{summary.get('pr20_proxy_ids_numeric')}`; in range: `{summary.get('pr20_proxy_ids_in_checkpoint_range')}`; likely base: `{summary.get('pr20_proxy_likely_index_base')}`.",
        "",
        "## PR21 Exact ID Range",
        f"PR21 exact IDs numeric: `{summary.get('pr211_exact_ids_numeric')}`; in range: `{summary.get('pr211_exact_ids_in_checkpoint_range')}`; likely base: `{summary.get('pr211_exact_likely_index_base')}`.",
        "",
        "## Same-Pixel Exact-vs-Proxy Namespace Check",
        f"Same global namespace supported: `{summary.get('same_global_gaussian_id_namespace_supported')}`. Zero-overlap claim safe within exact-available scope: `{summary.get('zero_overlap_claim_safe_within_exact_available_scope')}`.",
        "",
        "## Proxy ID Semantics",
        f"PR20 proxy semantics: `{summary.get('pr20_gaussian_id_semantics')}`.",
        "",
        "## Code Provenance",
        f"PR20 code semantics: `{summary.get('pr20_gaussian_id_semantics')}`. PR21 exact code semantics: `{summary.get('pr211_exact_gaussian_id_semantics')}`.",
        "",
        "## Conclusion",
        "The PR21.2 zero-overlap result is paper-safe only if the common namespace flag is true, and only for chair exact-available pixels/views.",
        "",
        "## Safety Boundary",
        "- This is observation-only.",
        "- Proxy IDs remain unsafe for intervention.",
        "- No exact contribution magnitude is available.",
        "- Drums is not used as exact evidence.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_wording(path: Path) -> None:
    lines = [
        "# PR21.2a Zero-Overlap Namespace Wording",
        "",
        "## A. If Namespace Supported",
        "",
        "For chair exact-available pixels/views, PR20 proxy Gaussian IDs and PR21 exact contributor IDs are verified to lie in the same global Gaussian index namespace. Under this validated namespace, their zero overlap indicates that the proxy Gaussian pool is not supported by exact contributor-ID evidence in the evaluated scope.",
        "",
        "## B. If Namespace Not Supported",
        "",
        "The observed zero overlap between PR20 proxy IDs and PR21 exact contributor IDs cannot yet be interpreted as a contributor mismatch because the ID namespace is not fully validated. The result should be treated as an unresolved ID-semantics issue until the namespace is verified.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_next_step(path: Path, namespace_supported: bool) -> None:
    if namespace_supported:
        text = "Recommend PR21.4 exact contribution magnitude. ID identity is now validated; the next weakness is contributor-ID-only evidence without alpha/transmittance/splat-weight magnitude."
    else:
        text = "Recommend PR21.2b namespace repair or re-export exact/proxy IDs in a verified global-index format. The zero-overlap claim is not paper-safe until ID namespace is fixed."
    path.write_text("# PR21.2a Next-Step Decision Memo\n\n" + text + "\n", encoding="utf-8")


def _manifest_rows(output_dir: Path, inputs: list[tuple[str, Path, bool]]) -> list[dict[str, Any]]:
    items = [(name, path, required, "input") for name, path, required in inputs]
    items.extend((name, output_dir / name, True, "output_pr212a") for name in OUTPUT_FILES)
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


def build_pr212a_chair_id_namespace_audit(
    *,
    pr200_chair_dir: Path,
    pr211_chair_dir: Path,
    pr212_chair_dir: Path,
    pr213_chair_dir: Path,
    run_dir: Path,
    output_dir: Path,
    scene: str = "chair",
    condition: str = "corrupt_occluder",
    subset_name: str = "seed_20260710",
    sample_id_count: int = 20,
    write_markdown: bool = False,
) -> tuple[dict[str, Any], int]:
    del write_markdown
    output_dir.mkdir(parents=True, exist_ok=True)
    project_root = Path(__file__).resolve().parents[2]
    warnings: list[str] = []

    pr211_summary = load_json(pr211_chair_dir / "pr211_exact_sparse_attribution_summary.json")
    pr212_summary = load_json(pr212_chair_dir / "pr212_chair_exact_vs_proxy_summary.json")
    pr213_summary = load_json(pr213_chair_dir / "pr213_chair_exact_evidence_positioning_summary.json")
    _warn_if(warnings, pr211_summary.get("evidence_quality") != "exact_sparse_contributor_id_only", "PR21.1e evidence quality is unexpected")
    _warn_if(warnings, not _truth(pr211_summary.get("exact_contributor_id_only_succeeded")), "PR21.1e exact ID-only success is not true")
    _warn_if(warnings, _safe_int(pr211_summary.get("exact_contributor_id_row_count")) <= 0, "PR21.1e exact row count is not > 0")
    _warn_if(warnings, _truth(pr211_summary.get("exact_render_contribution_succeeded")), "PR21.1e exact render contribution should be false")
    _warn_if(warnings, _truth(pr211_summary.get("ready_for_intervention")), "PR21.1e ready_for_intervention should be false")
    _warn_if(warnings, not _truth(pr212_summary.get("exact_contributor_id_only_available")), "PR21.2 exact ID-only available is not true")
    _warn_if(warnings, _truth(pr212_summary.get("exact_render_contribution_available")), "PR21.2 exact render contribution should be false")
    _warn_if(warnings, _safe_int(pr212_summary.get("exact_pixel_count")) <= 0, "PR21.2 exact_pixel_count is not > 0")
    _warn_if(warnings, _safe_int(pr212_summary.get("exact_row_count")) <= 0, "PR21.2 exact_row_count is not > 0")
    for key in ["mean_pixel_jaccard", "mean_exact_recall_by_proxy", "mean_proxy_precision_against_exact"]:
        _warn_if(warnings, (_safe_float(pr212_summary.get(key), 0.0) or 0.0) != 0.0, f"PR21.2 {key} is not zero")
    _warn_if(warnings, pr213_summary.get("positioning_status") != "exact_attribution_trust_signal_validation_not_defense", "PR21.3 positioning status is unexpected")
    _warn_if(warnings, _truth(pr213_summary.get("proxy_degeneracy_supported_by_exact")), "PR21.3 proxy degeneracy should not be supported by exact")
    _warn_if(warnings, _truth(pr213_summary.get("proxy_safe_for_intervention")), "PR21.3 proxy_safe_for_intervention should be false")
    _warn_if(warnings, _truth(pr213_summary.get("drums_used_as_exact_evidence")), "PR21.3 drums_used_as_exact_evidence should be false")

    checkpoint_rows, checkpoint_count, checkpoint_source, checkpoint_conf = _discover_checkpoint_inventory(run_dir)
    proxy_path = pr200_chair_dir / "pr200_pixel_gaussian_contributions.csv"
    exact_path = pr211_chair_dir / "pr211_exact_pixel_gaussian_contributions.csv"
    pr212_pixel_path = pr212_chair_dir / "pr212_chair_pixel_exact_vs_proxy.csv"
    proxy_rows, proxy_schema, proxy_ids, proxy_failures = _load_id_rows(proxy_path, "pr20_proxy")
    exact_rows, exact_schema, exact_ids, exact_failures = _load_id_rows(exact_path, "pr211_exact")
    pr212_pixel_rows = load_csv_rows(pr212_pixel_path)
    schema_rows = [proxy_schema, exact_schema]

    exact_keys = {_pixel_key(row) for row in exact_rows}
    proxy_exact_pixel_ids = [_id_value(row.get("gaussian_id")) for row in proxy_rows if _pixel_key(row) in exact_keys]
    proxy_exact_pixel_ids = [value for value in proxy_exact_pixel_ids if value is not None]
    range_rows = [
        _range_row("pr20_proxy_all", proxy_path, "", proxy_ids, proxy_failures, checkpoint_count, len(proxy_rows)),
        _range_row("pr20_proxy_exact_available_pixels_only", proxy_path, "", proxy_exact_pixel_ids, 0, checkpoint_count, len(proxy_exact_pixel_ids)),
        _range_row("pr211_exact_all", exact_path, "", exact_ids, exact_failures, checkpoint_count, len(exact_rows)),
    ]
    for view in sorted({row.get("view_name", "") for row in exact_rows if row.get("view_name")}):
        ids = [_id_value(row.get("gaussian_id")) for row in exact_rows if row.get("view_name") == view]
        ids = [value for value in ids if value is not None]
        range_rows.append(_range_row(f"pr211_exact_by_view_{view}", exact_path, view, ids, 0, checkpoint_count, len(ids)))
    for view in sorted({row.get("view_name", "") for row in proxy_rows if row.get("view_name")}):
        ids = [_id_value(row.get("gaussian_id")) for row in proxy_rows if row.get("view_name") == view]
        ids = [value for value in ids if value is not None]
        range_rows.append(_range_row(f"pr20_proxy_by_view_{view}", proxy_path, view, ids, 0, checkpoint_count, len(ids)))
    if pr212_pixel_rows and "exact_gaussian_ids_semicolon" in pr212_pixel_rows[0]:
        ids = []
        for row in pr212_pixel_rows:
            for text in str(row.get("exact_gaussian_ids_semicolon", "")).split(";"):
                value = _id_value(text)
                if value is not None:
                    ids.append(value)
        range_rows.append(_range_row("pr212_pixel_comparison_exact_available", pr212_pixel_path, "", ids, 0, checkpoint_count, len(ids)))

    proxy_semantics_rows, proxy_semantics = _proxy_semantics_rows(proxy_rows, checkpoint_count)
    code_rows, code_summary = _code_provenance(project_root)
    exact_range = _range_status(exact_ids, exact_failures, checkpoint_count)
    proxy_range = _range_status(proxy_ids, proxy_failures, checkpoint_count)
    pr20_code_sem = code_summary.get("pr20_gaussian_id_semantics_from_code", "unknown")
    pr211_code_sem = code_summary.get("pr211_exact_gaussian_id_semantics_from_code", "unknown")
    pr20_sem = "local_candidate_rank" if proxy_semantics == "local_candidate_rank" else pr20_code_sem if pr20_code_sem != "unknown" else proxy_semantics
    pr211_sem = pr211_code_sem
    proxy_in_range = proxy_range[1] == "zero_based_global_index_plausible"
    exact_in_range = exact_range[1] == "zero_based_global_index_plausible"
    same_supported = bool(checkpoint_count is not None and proxy_in_range and exact_in_range and pr20_sem == "global_checkpoint_gaussian_index" and pr211_sem == "global_checkpoint_gaussian_index")
    same_conf = "high" if same_supported and checkpoint_conf == "high" else "medium" if same_supported else "low"
    same_pixel_rows = _same_pixel_rows(scene, exact_rows, proxy_rows, checkpoint_count, same_supported)
    zero_metrics = all((_safe_float(pr212_summary.get(key), 0.0) or 0.0) == 0.0 for key in ["mean_pixel_jaccard", "mean_exact_recall_by_proxy", "mean_proxy_precision_against_exact"])
    zero_safe = bool(same_supported and zero_metrics)
    exact_id_set = set(exact_ids)
    proxy_id_set = set(proxy_ids)
    sample_rows = _sample_lookup_rows(exact_id_set, proxy_id_set, checkpoint_count, sample_id_count)
    diagnosis = _diagnosis_rows(same_supported, same_conf, zero_safe, checkpoint_count, pr20_sem, pr211_sem)
    summary = {
        "schema_name": "viewtrust.pr212a.chair_id_namespace_audit.summary",
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
        "pr200_chair_input_dir": str(pr200_chair_dir),
        "pr211_chair_input_dir": str(pr211_chair_dir),
        "pr212_chair_input_dir": str(pr212_chair_dir),
        "pr213_chair_input_dir": str(pr213_chair_dir),
        "run_dir": str(run_dir),
        "checkpoint_gaussian_count": checkpoint_count,
        "checkpoint_gaussian_count_source": checkpoint_source,
        "checkpoint_count_confidence": checkpoint_conf,
        "pr20_proxy_id_count": len(proxy_ids),
        "pr20_proxy_unique_id_count": len(proxy_id_set),
        "pr211_exact_id_count": len(exact_ids),
        "pr211_exact_unique_id_count": len(exact_id_set),
        "pr20_proxy_ids_numeric": proxy_failures == 0 and bool(proxy_ids),
        "pr211_exact_ids_numeric": exact_failures == 0 and bool(exact_ids),
        "pr20_proxy_ids_in_checkpoint_range": proxy_in_range,
        "pr211_exact_ids_in_checkpoint_range": exact_in_range,
        "pr20_proxy_likely_index_base": proxy_range[0],
        "pr211_exact_likely_index_base": exact_range[0],
        "pr20_gaussian_id_semantics": pr20_sem,
        "pr211_exact_gaussian_id_semantics": pr211_sem,
        "same_global_gaussian_id_namespace_supported": same_supported,
        "same_namespace_confidence": same_conf,
        "zero_overlap_claim_safe_within_exact_available_scope": zero_safe,
        "proxy_safe_for_intervention": False,
        "exact_render_contribution_available": False,
        "exact_contribution_magnitude_available": False,
        "drums_used_as_exact_evidence": False,
        "pr212a_ready_for_pr214": zero_safe,
        "recommended_next_step": "Run PR21.4 exact contribution magnitude / alpha-transmittance-aware replay."
        if zero_safe
        else "Resolve gaussian_id namespace before using PR21.2 zero-overlap as a paper claim.",
        "warnings": warnings,
    }

    write_json(output_dir / "pr212a_chair_id_namespace_audit_summary.json", summary)
    write_csv_rows(output_dir / "pr212a_checkpoint_gaussian_inventory.csv", checkpoint_rows, CHECKPOINT_FIELDS)
    write_csv_rows(output_dir / "pr212a_id_source_schema_audit.csv", schema_rows, SCHEMA_FIELDS)
    write_csv_rows(output_dir / "pr212a_id_range_audit.csv", range_rows, RANGE_FIELDS)
    write_csv_rows(output_dir / "pr212a_same_pixel_id_namespace_comparison.csv", same_pixel_rows, SAME_PIXEL_FIELDS)
    write_csv_rows(output_dir / "pr212a_sample_id_lookup.csv", sample_rows, SAMPLE_FIELDS)
    write_csv_rows(output_dir / "pr212a_proxy_id_semantics_audit.csv", proxy_semantics_rows, PROXY_SEMANTICS_FIELDS)
    write_csv_rows(output_dir / "pr212a_code_id_semantics_audit.csv", code_rows, CODE_AUDIT_FIELDS)
    write_json(output_dir / "pr212a_code_id_semantics_summary.json", code_summary)
    write_csv_rows(output_dir / "pr212a_id_namespace_diagnosis.csv", diagnosis, DIAGNOSIS_FIELDS)
    _write_report(output_dir / "pr212a_chair_id_namespace_audit_report.md", summary)
    _write_wording(output_dir / "pr212a_zero_overlap_namespace_wording.md")
    _write_next_step(output_dir / "pr212a_next_step_decision_memo.md", zero_safe)
    inputs = [
        ("pr200_chair_dir", pr200_chair_dir, True),
        ("pr211_chair_dir", pr211_chair_dir, True),
        ("pr212_chair_dir", pr212_chair_dir, True),
        ("pr213_chair_dir", pr213_chair_dir, True),
        ("run_dir", run_dir, True),
    ]
    manifest = output_dir / "artifact_manifest.csv"
    write_csv_rows(manifest, _manifest_rows(output_dir, inputs), MANIFEST_FIELDS)
    write_csv_rows(manifest, _manifest_rows(output_dir, inputs), MANIFEST_FIELDS)
    return summary, 0
