"""PR21.2b PR20 proxy ID namespace source audit and repair feasibility."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from viewtrust.analysis.clean_prior_normalization import load_csv_rows, load_json, write_csv_rows, write_json
from viewtrust.analysis.pr212a_namespace import read_ply_vertex_count


OUTPUT_FILES = [
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

PROFILE_FIELDS = [
    "view_name",
    "view_group",
    "row_count",
    "pixel_count",
    "gaussian_id_unique_count",
    "gaussian_id_min",
    "gaussian_id_max",
    "root_gaussian_id_unique_count",
    "root_gaussian_id_min",
    "root_gaussian_id_max",
    "parent_gaussian_id_unique_count",
    "parent_gaussian_id_min",
    "parent_gaussian_id_max",
    "contribution_rank_min",
    "contribution_rank_max",
    "id_pool_signature",
    "id_pool_reused_across_pixels",
    "id_pool_reused_across_views",
    "final_checkpoint_range_status",
    "suspicious_id_pattern",
    "notes",
]

INVENTORY_FIELDS = [
    "source_path",
    "source_root",
    "file_type",
    "exists",
    "row_count_or_json_size",
    "columns_or_keys_semicolon",
    "has_gaussian_id",
    "has_root_gaussian_id",
    "has_parent_gaussian_id",
    "has_final_index",
    "has_alive_index",
    "has_compact_index",
    "has_current_index",
    "has_source_child_mapping",
    "candidate_mapping_type",
    "load_status",
    "notes",
]

LOOKUP_FIELDS = [
    "queried_id",
    "id_role",
    "source_path",
    "found",
    "matched_column_or_key",
    "matched_row_count",
    "associated_root_gaussian_id",
    "associated_parent_gaussian_id",
    "associated_final_index",
    "associated_alive_index",
    "associated_compact_index",
    "associated_current_index",
    "associated_status",
    "evidence_snippet",
    "notes",
]

SEMANTICS_FIELDS = [
    "diagnosis_target",
    "inferred_semantics",
    "confidence",
    "evidence",
    "blocking_issue",
    "repair_possible",
    "repair_strategy",
    "paper_safe_wording",
    "unsafe_wording_to_avoid",
]

MAPPING_FIELDS = [
    "mapping_name",
    "source_path",
    "source_columns",
    "source_id_column",
    "target_index_column",
    "source_id_count",
    "target_index_count",
    "duplicate_source_id_count",
    "duplicate_target_index_count",
    "target_index_in_checkpoint_range_rate",
    "covers_pr20_proxy_id_count",
    "covers_pr20_proxy_unique_id_count",
    "covers_pr20_exact_available_proxy_unique_id_count",
    "covers_train013_suspicious_id_count",
    "mapping_confidence",
    "mapping_status",
    "notes",
]

REPAIR_FIELDS = [
    "repair_scope",
    "row_count",
    "unique_proxy_id_count",
    "mapped_row_count",
    "unmapped_row_count",
    "mapped_unique_id_count",
    "unmapped_unique_id_count",
    "mapping_coverage_rate",
    "all_mapped_ids_in_checkpoint_range",
    "repair_feasible",
    "repair_confidence",
    "blocker",
    "recommended_action",
]

REPAIRED_PREVIEW_FIELDS = [
    "scene",
    "condition",
    "subset_name",
    "view_name",
    "view_group",
    "pixel_x",
    "pixel_y",
    "original_gaussian_id",
    "original_root_gaussian_id",
    "original_parent_gaussian_id",
    "verified_final_gaussian_index",
    "mapping_name",
    "mapping_source",
    "mapping_confidence",
    "mapping_status",
    "repair_warning",
]

REPAIRED_COMPARE_FIELDS = [
    "view_name",
    "pixel_x",
    "pixel_y",
    "exact_id_count",
    "repaired_proxy_id_count",
    "unmapped_proxy_id_count",
    "intersection_count",
    "jaccard",
    "exact_recall_by_repaired_proxy",
    "repaired_proxy_precision_against_exact",
    "repair_confidence",
    "comparison_status",
    "interpretation",
]

CODE_FIELDS = ["matched_file", "line_number", "matched_text", "inferred_role", "confidence", "notes"]
MANIFEST_FIELDS = ["relative_path", "path", "exists", "file_type", "size_bytes", "required", "artifact_group"]

CODE_TERMS = [
    "gaussian_id",
    "root_gaussian_id",
    "parent_gaussian_id",
    "gaussian_identity_table_grouped",
    "gaussian_identity_table",
    "current_ids",
    "final_index",
    "alive_index",
    "compact_index",
    "child_gaussian_id",
    "source_gaussian_id",
    "PR20",
    "pr200_pixel_gaussian_contributions",
    "PIXEL_GAUSSIAN_FIELDS",
    "contribution_rank",
    "train013",
    "clean_prior_demoted",
    "100000",
]

ID_COLUMNS = ["gaussian_id", "root_gaussian_id", "parent_gaussian_id", "source_gaussian_id", "child_gaussian_id"]
TARGET_COLUMNS = ["final_index", "alive_index", "compact_index", "current_index", "verified_final_gaussian_index"]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _bool_text(value: Any) -> str:
    return "true" if bool(value) else "false"


def _truth(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def _safe_int(value: Any) -> int | None:
    try:
        if value in ("", None):
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _ratio(num: int, den: int) -> float | str:
    return num / den if den else ""


def _id_values(rows: list[dict[str, Any]], column: str) -> list[int]:
    values = []
    for row in rows:
        value = _safe_int(row.get(column))
        if value is not None:
            values.append(value)
    return values


def _range_status(ids: list[int], checkpoint_count: int | None) -> str:
    if not ids:
        return "empty_id_set"
    if checkpoint_count is None:
        return "no_checkpoint_count_available"
    return "in_final_checkpoint_range" if all(0 <= value < checkpoint_count for value in ids) else "out_of_final_checkpoint_range"


def _pixel_key(row: dict[str, Any]) -> tuple[str, int, int]:
    return str(row.get("view_name", "")), _safe_int(row.get("pixel_x")) or 0, _safe_int(row.get("pixel_y")) or 0


def _load_checkpoint_count(run_dir: Path) -> tuple[int | None, str]:
    candidates = sorted(run_dir.rglob("*.ply")) if run_dir.exists() else []
    candidates = sorted(candidates, key=lambda p: (0 if "iteration_700" in str(p) else 1, len(str(p))))
    for path in candidates:
        count = read_ply_vertex_count(path)
        if count is not None:
            return count, str(path)
    return None, ""


def _profile_proxy_rows(rows: list[dict[str, str]], checkpoint_count: int | None) -> list[dict[str, Any]]:
    by_view: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_view[str(row.get("view_name", ""))].append(row)
    all_pool_counts: Counter[str] = Counter()
    view_pools: dict[str, list[str]] = {}
    for view, view_rows in by_view.items():
        pools: dict[tuple[str, int, int], set[int]] = defaultdict(set)
        for row in view_rows:
            gid = _safe_int(row.get("gaussian_id"))
            if gid is not None:
                pools[_pixel_key(row)].add(gid)
        sigs = [";".join(str(v) for v in sorted(pool)) for pool in pools.values()]
        view_pools[view] = sigs
        all_pool_counts.update(sigs)
    out = []
    for view, view_rows in sorted(by_view.items()):
        gids = _id_values(view_rows, "gaussian_id")
        roots = _id_values(view_rows, "root_gaussian_id")
        parents = _id_values(view_rows, "parent_gaussian_id")
        ranks = _id_values(view_rows, "contribution_rank")
        pixel_count = len({_pixel_key(row) for row in view_rows})
        pool_counts = Counter(view_pools.get(view, []))
        reused_pixels = bool(pool_counts and pool_counts.most_common(1)[0][1] > 1)
        reused_views = any(count > pool_counts.get(sig, 0) for sig, count in all_pool_counts.items() if sig in pool_counts)
        suspicious = []
        if checkpoint_count is not None and any(value >= checkpoint_count for value in gids):
            suspicious.append("exceeds_final_checkpoint_count")
        if view == "train_013" and any(100000 <= value <= 100016 for value in gids):
            suspicious.append("train013_100000_range")
        out.append(
            {
                "view_name": view,
                "view_group": view_rows[0].get("view_group", ""),
                "row_count": len(view_rows),
                "pixel_count": pixel_count,
                "gaussian_id_unique_count": len(set(gids)),
                "gaussian_id_min": min(gids) if gids else "",
                "gaussian_id_max": max(gids) if gids else "",
                "root_gaussian_id_unique_count": len(set(roots)),
                "root_gaussian_id_min": min(roots) if roots else "",
                "root_gaussian_id_max": max(roots) if roots else "",
                "parent_gaussian_id_unique_count": len(set(parents)),
                "parent_gaussian_id_min": min(parents) if parents else "",
                "parent_gaussian_id_max": max(parents) if parents else "",
                "contribution_rank_min": min(ranks) if ranks else "",
                "contribution_rank_max": max(ranks) if ranks else "",
                "id_pool_signature": pool_counts.most_common(1)[0][0] if pool_counts else "",
                "id_pool_reused_across_pixels": _bool_text(reused_pixels),
                "id_pool_reused_across_views": _bool_text(reused_views),
                "final_checkpoint_range_status": _range_status(gids, checkpoint_count),
                "suspicious_id_pattern": ";".join(suspicious),
                "notes": "PR20 proxy gaussian_id profile; not exact contributor evidence",
            }
        )
    return out


def _csv_columns(path: Path) -> tuple[list[str], int, str]:
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
            return list(reader.fieldnames or []), len(rows), "loaded_csv"
    except Exception as exc:
        return [], 0, f"csv_load_failed:{type(exc).__name__}"


def _json_keys(path: Path) -> tuple[list[str], int, str]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return list(data.keys()), len(json.dumps(data)), "loaded_json"
        if isinstance(data, list):
            keys = sorted({key for item in data if isinstance(item, dict) for key in item})
            return keys, len(data), "loaded_json_list"
    except Exception as exc:
        return [], 0, f"json_load_failed:{type(exc).__name__}"
    return [], 0, "json_unknown_shape"


def _mapping_type(columns: list[str]) -> str:
    cols = set(columns)
    has_target = bool(cols & set(TARGET_COLUMNS))
    if "gaussian_id" in cols and has_target:
        return "persistent_to_final_index"
    if "root_gaussian_id" in cols and has_target:
        return "root_to_final_index"
    if {"parent_gaussian_id", "child_gaussian_id"} & cols:
        return "parent_to_child"
    if "current_ids" in cols:
        return "lifecycle_state_table"
    if {"created_ids", "cloned_ids", "split_ids"} & cols:
        return "creation_event_log"
    if "pruned_ids" in cols:
        return "prune_event_log"
    if cols & {"gaussian_id", "root_gaussian_id", "parent_gaussian_id"}:
        return "identity_metadata_only"
    return "unknown"


def _inventory_sources(roots: list[tuple[str, Path]]) -> list[dict[str, Any]]:
    wanted = ["gaussian", "identity", "lifecycle", "training_events", "current_ids", "final_index", "compact_index"]
    rows = []
    for root_name, root in roots:
        if not root or not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in {".csv", ".json"}:
                continue
            text = str(path).lower()
            if not any(term in text for term in wanted):
                continue
            if path.suffix.lower() == ".csv":
                columns, count, status = _csv_columns(path)
                ftype = "csv"
            else:
                columns, count, status = _json_keys(path)
                ftype = "json"
            cols = set(columns)
            rows.append(
                {
                    "source_path": str(path),
                    "source_root": root_name,
                    "file_type": ftype,
                    "exists": "true",
                    "row_count_or_json_size": count,
                    "columns_or_keys_semicolon": ";".join(columns),
                    "has_gaussian_id": _bool_text("gaussian_id" in cols),
                    "has_root_gaussian_id": _bool_text("root_gaussian_id" in cols),
                    "has_parent_gaussian_id": _bool_text("parent_gaussian_id" in cols),
                    "has_final_index": _bool_text("final_index" in cols),
                    "has_alive_index": _bool_text("alive_index" in cols),
                    "has_compact_index": _bool_text("compact_index" in cols),
                    "has_current_index": _bool_text("current_index" in cols),
                    "has_source_child_mapping": _bool_text(bool(cols & {"source_gaussian_id", "child_gaussian_id"})),
                    "candidate_mapping_type": _mapping_type(columns),
                    "load_status": status,
                    "notes": "candidate identity/lifecycle mapping inventory",
                }
            )
    return rows


def _candidate_rows(path: Path) -> list[dict[str, str]]:
    return load_csv_rows(path) if path.suffix.lower() == ".csv" else []


def _lookup_ids(query: dict[str, set[int]], inventory: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for role, ids in query.items():
        for qid in sorted(ids):
            found_any = False
            for inv in inventory:
                path = Path(str(inv["source_path"]))
                if path.suffix.lower() != ".csv":
                    continue
                rows = _candidate_rows(path)
                columns = list(rows[0].keys()) if rows else []
                match_cols = [col for col in columns if col in ID_COLUMNS + TARGET_COLUMNS]
                for col in match_cols:
                    matches = [row for row in rows if _safe_int(row.get(col)) == qid]
                    if not matches:
                        continue
                    found_any = True
                    first = matches[0]
                    out.append(
                        {
                            "queried_id": qid,
                            "id_role": role,
                            "source_path": str(path),
                            "found": "true",
                            "matched_column_or_key": col,
                            "matched_row_count": len(matches),
                            "associated_root_gaussian_id": first.get("root_gaussian_id", ""),
                            "associated_parent_gaussian_id": first.get("parent_gaussian_id", ""),
                            "associated_final_index": first.get("final_index", ""),
                            "associated_alive_index": first.get("alive_index", ""),
                            "associated_compact_index": first.get("compact_index", ""),
                            "associated_current_index": first.get("current_index", ""),
                            "associated_status": first.get("status", first.get("lifecycle_status", "")),
                            "evidence_snippet": str({key: first.get(key, "") for key in columns[:8]}),
                            "notes": "explicit CSV match",
                        }
                    )
            if not found_any:
                out.append(
                    {
                        "queried_id": qid,
                        "id_role": role,
                        "source_path": "",
                        "found": "false",
                        "matched_column_or_key": "",
                        "matched_row_count": 0,
                        "associated_root_gaussian_id": "",
                        "associated_parent_gaussian_id": "",
                        "associated_final_index": "",
                        "associated_alive_index": "",
                        "associated_compact_index": "",
                        "associated_current_index": "",
                        "associated_status": "",
                        "evidence_snippet": "",
                        "notes": "not found in candidate identity/lifecycle sources",
                    }
                )
    return out


def _build_mapping_candidates(
    inventory: list[dict[str, Any]],
    proxy_ids: set[int],
    exact_pixel_proxy_ids: set[int],
    train013_ids: set[int],
    checkpoint_count: int | None,
) -> tuple[list[dict[str, Any]], dict[int, int], dict[str, str]]:
    rows = []
    best_map: dict[int, int] = {}
    best_meta = {"mapping_name": "", "source_path": "", "mapping_confidence": "", "mapping_status": "no_mapping_available"}
    for inv in inventory:
        path = Path(str(inv["source_path"]))
        if path.suffix.lower() != ".csv":
            continue
        data = _candidate_rows(path)
        if not data:
            continue
        columns = list(data[0].keys())
        source_cols = [col for col in ["gaussian_id", "root_gaussian_id", "parent_gaussian_id", "source_gaussian_id"] if col in columns]
        target_cols = [col for col in TARGET_COLUMNS if col in columns]
        for source_col in source_cols:
            for target_col in target_cols:
                mapping: dict[int, int] = {}
                source_counts: Counter[int] = Counter()
                target_counts: Counter[int] = Counter()
                for row in data:
                    source = _safe_int(row.get(source_col))
                    target = _safe_int(row.get(target_col))
                    if source is None or target is None:
                        continue
                    source_counts[source] += 1
                    target_counts[target] += 1
                    mapping.setdefault(source, target)
                if not mapping:
                    continue
                target_in_range = [0 <= target < checkpoint_count for target in mapping.values()] if checkpoint_count is not None else []
                in_rate = sum(1 for item in target_in_range if item) / len(target_in_range) if target_in_range else 0.0
                covers = proxy_ids & set(mapping)
                covers_exact = exact_pixel_proxy_ids & set(mapping)
                covers_train013 = train013_ids & set(mapping)
                dup_source = sum(1 for value in source_counts.values() if value > 1)
                dup_target = sum(1 for value in target_counts.values() if value > 1)
                if covers and in_rate == 1.0 and dup_source == 0:
                    status = "verified_mapping_candidate"
                    confidence = "high"
                elif covers and in_rate > 0:
                    status = "partial_mapping_candidate"
                    confidence = "medium"
                elif covers:
                    status = "invalid_mapping_candidate"
                    confidence = "low"
                else:
                    status = "ambiguous_mapping_candidate"
                    confidence = "low"
                row_out = {
                    "mapping_name": f"{source_col}_to_{target_col}",
                    "source_path": str(path),
                    "source_columns": ";".join(columns),
                    "source_id_column": source_col,
                    "target_index_column": target_col,
                    "source_id_count": len(mapping),
                    "target_index_count": len(set(mapping.values())),
                    "duplicate_source_id_count": dup_source,
                    "duplicate_target_index_count": dup_target,
                    "target_index_in_checkpoint_range_rate": in_rate,
                    "covers_pr20_proxy_id_count": sum(1 for value in proxy_ids if value in mapping),
                    "covers_pr20_proxy_unique_id_count": len(covers),
                    "covers_pr20_exact_available_proxy_unique_id_count": len(covers_exact),
                    "covers_train013_suspicious_id_count": len(covers_train013),
                    "mapping_confidence": confidence,
                    "mapping_status": status,
                    "notes": "explicit mapping candidate from identity/lifecycle table",
                }
                rows.append(row_out)
                if status == "verified_mapping_candidate" and len(covers) > len(best_map):
                    best_map = mapping
                    best_meta = {key: str(row_out[key]) for key in ["mapping_name", "source_path", "mapping_confidence", "mapping_status"]}
    if not rows:
        rows.append(
            {
                "mapping_name": "none",
                "source_path": "",
                "source_columns": "",
                "source_id_column": "",
                "target_index_column": "",
                "source_id_count": 0,
                "target_index_count": 0,
                "duplicate_source_id_count": 0,
                "duplicate_target_index_count": 0,
                "target_index_in_checkpoint_range_rate": 0,
                "covers_pr20_proxy_id_count": 0,
                "covers_pr20_proxy_unique_id_count": 0,
                "covers_pr20_exact_available_proxy_unique_id_count": 0,
                "covers_train013_suspicious_id_count": 0,
                "mapping_confidence": "none",
                "mapping_status": "no_mapping_available",
                "notes": "no explicit source_id to final/compact/current index mapping found",
            }
        )
    return rows, best_map, best_meta


def _repair_rows_for_scope(scope: str, rows: list[dict[str, str]], mapping: dict[int, int], checkpoint_count: int | None, confidence: str) -> dict[str, Any]:
    ids = [_safe_int(row.get("gaussian_id")) for row in rows]
    ids = [value for value in ids if value is not None]
    mapped = [mapping[value] for value in ids if value in mapping]
    unmapped = [value for value in ids if value not in mapping]
    all_range = bool(mapped and checkpoint_count is not None and all(0 <= value < checkpoint_count for value in mapped))
    feasible = bool(rows and not unmapped and all_range and confidence == "high")
    return {
        "repair_scope": scope,
        "row_count": len(rows),
        "unique_proxy_id_count": len(set(ids)),
        "mapped_row_count": len(mapped),
        "unmapped_row_count": len(unmapped),
        "mapped_unique_id_count": len(set(mapped)),
        "unmapped_unique_id_count": len(set(unmapped)),
        "mapping_coverage_rate": _ratio(len(mapped), len(ids)),
        "all_mapped_ids_in_checkpoint_range": _bool_text(all_range),
        "repair_feasible": _bool_text(feasible),
        "repair_confidence": confidence if mapped else "none",
        "blocker": "" if feasible else "missing_explicit_complete_mapping",
        "recommended_action": "run_pr212c_repaired_exact_vs_proxy" if feasible else "re_export_proxy_with_final_checkpoint_indices",
    }


def _repair_feasibility(proxy_rows: list[dict[str, str]], exact_keys: set[tuple[str, int, int]], mapping: dict[int, int], checkpoint_count: int | None, confidence: str) -> list[dict[str, Any]]:
    scopes = {
        "all_pr20_proxy_rows": proxy_rows,
        "exact_available_pr20_proxy_rows": [row for row in proxy_rows if _pixel_key(row) in exact_keys],
    }
    for view in ["train_004", "train_009", "train_012", "train_014", "train_017", "train_013"]:
        scopes[view] = [row for row in proxy_rows if row.get("view_name") == view]
    return [_repair_rows_for_scope(scope, rows, mapping, checkpoint_count, confidence) for scope, rows in scopes.items()]


def _repaired_preview(scene: str, condition: str, subset_name: str, proxy_rows: list[dict[str, str]], mapping: dict[int, int], meta: dict[str, str], limit: int = 5000) -> list[dict[str, Any]]:
    out = []
    for row in proxy_rows[:limit]:
        gid = _safe_int(row.get("gaussian_id"))
        mapped = mapping.get(gid) if gid is not None else None
        out.append(
            {
                "scene": scene,
                "condition": condition,
                "subset_name": subset_name,
                "view_name": row.get("view_name", ""),
                "view_group": row.get("view_group", ""),
                "pixel_x": row.get("pixel_x", ""),
                "pixel_y": row.get("pixel_y", ""),
                "original_gaussian_id": row.get("gaussian_id", ""),
                "original_root_gaussian_id": row.get("root_gaussian_id", ""),
                "original_parent_gaussian_id": row.get("parent_gaussian_id", ""),
                "verified_final_gaussian_index": mapped if mapped is not None and meta.get("mapping_confidence") == "high" else "",
                "mapping_name": meta.get("mapping_name", ""),
                "mapping_source": meta.get("source_path", ""),
                "mapping_confidence": meta.get("mapping_confidence", ""),
                "mapping_status": meta.get("mapping_status", ""),
                "repair_warning": "" if mapped is not None and meta.get("mapping_confidence") == "high" else "unmapped_or_mapping_not_verified",
            }
        )
    return out


def _repaired_compare(exact_rows: list[dict[str, str]], proxy_rows: list[dict[str, str]], mapping: dict[int, int], confidence: str) -> list[dict[str, Any]]:
    exact_by_pixel: dict[tuple[str, int, int], set[int]] = defaultdict(set)
    proxy_by_pixel: dict[tuple[str, int, int], list[int]] = defaultdict(list)
    for row in exact_rows:
        gid = _safe_int(row.get("gaussian_id"))
        if gid is not None:
            exact_by_pixel[_pixel_key(row)].add(gid)
    for row in proxy_rows:
        gid = _safe_int(row.get("gaussian_id"))
        if gid is not None:
            proxy_by_pixel[_pixel_key(row)].append(gid)
    out = []
    for key in sorted(exact_by_pixel):
        exact = exact_by_pixel[key]
        raw_proxy = proxy_by_pixel.get(key, [])
        repaired = {mapping[value] for value in raw_proxy if value in mapping}
        unmapped = [value for value in raw_proxy if value not in mapping]
        inter = exact & repaired
        union = exact | repaired
        valid = bool(raw_proxy and not unmapped and confidence == "high")
        out.append(
            {
                "view_name": key[0],
                "pixel_x": key[1],
                "pixel_y": key[2],
                "exact_id_count": len(exact),
                "repaired_proxy_id_count": len(repaired),
                "unmapped_proxy_id_count": len(unmapped),
                "intersection_count": len(inter),
                "jaccard": len(inter) / len(union) if union else "",
                "exact_recall_by_repaired_proxy": _ratio(len(inter), len(exact)),
                "repaired_proxy_precision_against_exact": _ratio(len(inter), len(repaired)),
                "repair_confidence": confidence if repaired else "none",
                "comparison_status": "repaired_comparison_valid" if valid else "repaired_comparison_partial" if repaired else "repaired_comparison_unavailable",
                "interpretation": "preview_only_does_not_replace_pr212",
            }
        )
    return out


def _semantics_diagnosis(profile: list[dict[str, Any]], mapping_rows: list[dict[str, Any]], train013_found: bool) -> list[dict[str, Any]]:
    any_out = any(row.get("final_checkpoint_range_status") == "out_of_final_checkpoint_range" for row in profile)
    any_train013 = any("train013_100000_range" in str(row.get("suspicious_id_pattern")) for row in profile)
    mapping_available = any(row.get("mapping_status") == "verified_mapping_candidate" for row in mapping_rows)
    overall_sem = "mixed_namespace" if any_out else "final_checkpoint_compact_index"
    if any_train013 and not train013_found:
        train_sem = "synthetic_placeholder_id"
    elif any_train013:
        train_sem = "persistent_lifecycle_id"
    else:
        train_sem = "not_observed"
    rows = []
    for target, sem in [
        ("pr20_gaussian_id", overall_sem),
        ("pr20_root_gaussian_id", "unknown"),
        ("pr20_parent_gaussian_id", "unknown"),
        ("train013_gaussian_id_100000_range", train_sem),
        ("pr20_exact_available_proxy_ids", overall_sem),
        ("pr20_all_proxy_ids", overall_sem),
    ]:
        rows.append(
            {
                "diagnosis_target": target,
                "inferred_semantics": sem,
                "confidence": "medium" if sem != "unknown" else "low",
                "evidence": "PR20 profile, identity lookup, and mapping candidates",
                "blocking_issue": "" if mapping_available else "no_explicit_final_checkpoint_index_mapping",
                "repair_possible": _bool_text(mapping_available),
                "repair_strategy": "map_proxy_ids_to_verified_final_gaussian_index" if mapping_available else "re_export_proxy_with_final_checkpoint_indices",
                "paper_safe_wording": "PR20 proxy IDs require explicit mapping before numeric comparison to PR21 exact IDs.",
                "unsafe_wording_to_avoid": "PR20 proxy IDs are exact final checkpoint contributors.",
            }
        )
    return rows


def _code_audit(project_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
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
                if not any(term.lower() in lowered for term in CODE_TERMS):
                    continue
                if "pr200_pixel_gaussian_contributions.csv" in text:
                    role, confidence = "pr20_proxy_writer", "high"
                elif "gaussian_identity_table_grouped" in text or "gaussian_identity_table" in text:
                    role, confidence = "identity_table_source", "medium"
                elif "100000" in text:
                    role, confidence = "train013_100000_source_hint", "low"
                elif "contribution_rank" in text:
                    role, confidence = "proxy_rank_field", "medium"
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
    summary = {
        "pr20_proxy_writer_file": next((row["matched_file"] for row in rows if row["inferred_role"] == "pr20_proxy_writer"), "unknown"),
        "pr20_proxy_writer_function": "build_pr200_sparse_render_attribution" if any("build_pr200_sparse_render_attribution" in row["matched_text"] for row in rows) else "unknown",
        "pr20_candidate_source_file": next((row["matched_file"] for row in rows if "gaussian_identity_table_grouped" in row["matched_text"]), "unknown"),
        "identity_table_source_file": next((row["matched_file"] for row in rows if row["inferred_role"] == "identity_table_source"), "unknown"),
        "lifecycle_id_assignment_file": next((row["matched_file"] for row in rows if "lifecycle" in row["matched_text"].lower()), "unknown"),
        "train013_100000_source_hint": next((row["matched_text"] for row in rows if row["inferred_role"] == "train013_100000_source_hint"), "unknown"),
        "pr20_id_semantics_from_code": "upstream_identity_table_gaussian_id",
        "code_confidence": "medium",
        "caveats": "Code provenance does not prove final checkpoint compact index mapping.",
    }
    return rows, summary


def _write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# PR21.2b PR20 Proxy ID Namespace Source Audit",
        "",
        "## Purpose",
        "PR21.2a blocked zero-overlap paper claims because PR20 proxy IDs were not validated against the final checkpoint namespace. PR21.2b audits their source and repair feasibility.",
        "",
        "## PR20 Proxy ID Profile",
        f"PR20 proxy ID semantics: `{summary.get('pr20_proxy_id_semantics')}`. Train013 100000 IDs explained: `{summary.get('train013_100000_ids_explained')}`.",
        "",
        "## Identity and Lifecycle Sources",
        f"Explicit mapping available: `{summary.get('explicit_mapping_available')}` from `{summary.get('mapping_source')}`.",
        "",
        "## Suspicious ID Lookup",
        "The lookup table records whether suspicious PR20 IDs appear in candidate identity/lifecycle sources.",
        "",
        "## Mapping Candidate Results",
        f"Mapping confidence: `{summary.get('mapping_confidence')}`.",
        "",
        "## Repair Feasibility",
        f"All rows feasible: `{summary.get('all_pr20_proxy_rows_repair_feasible')}`. Exact-available rows feasible: `{summary.get('exact_available_proxy_rows_repair_feasible')}`.",
        "",
        "## Repaired Preview",
        f"Preview available: `{summary.get('repaired_exact_vs_proxy_preview_available')}`. This preview does not replace PR21.2 unless coverage is complete and verified.",
        "",
        "## Conclusion",
        str(summary.get("recommended_next_step", "")),
        "",
        "## Safety Boundary",
        "- Observation-only.",
        "- Proxy IDs are not exact contributors.",
        "- Proxy IDs remain unsafe for intervention.",
        "- No magnitude evidence.",
        "- Drums is not used as exact evidence.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_wording(path: Path) -> None:
    lines = [
        "# PR21.2b Proxy Namespace Wording",
        "",
        "## A. If Repair Feasible",
        "",
        "PR20 proxy IDs were remapped into the final checkpoint Gaussian index namespace using an explicit identity mapping. Under this repaired namespace, exact-vs-proxy comparison can be recomputed on exact-available chair pixels.",
        "",
        "## B. If Repair Not Feasible",
        "",
        "PR20 proxy IDs could not be reliably mapped into the final checkpoint Gaussian index namespace. Therefore, PR20 proxy evidence should be treated as diagnostic-only, and numeric comparisons against PR21 exact contributor IDs should be deferred until the proxy export pipeline records verified final checkpoint indices.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_next_step(path: Path, repair_feasible: bool) -> None:
    if repair_feasible:
        text = "Recommend PR21.2c repaired exact-vs-proxy comparison."
    else:
        text = "Recommend PR21.2d / PR20 re-export: future proxy attribution should include final checkpoint compact index, persistent lifecycle ID, root ID, parent ID, and mapping provenance. Do not proceed to PR21.4 until namespace is repaired."
    path.write_text("# PR21.2b Next-Step Decision Memo\n\n" + text + "\n", encoding="utf-8")


def _manifest_rows(output_dir: Path, inputs: list[tuple[str, Path, bool]]) -> list[dict[str, Any]]:
    items = [(name, path, required, "input") for name, path, required in inputs]
    items.extend((name, output_dir / name, True, "output_pr212b") for name in OUTPUT_FILES)
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


def build_pr212b_pr20_proxy_id_source_audit(
    *,
    pr200_chair_dir: Path,
    pr211_chair_dir: Path,
    pr212_chair_dir: Path,
    pr212a_chair_dir: Path,
    pr213_chair_dir: Path,
    run_dir: Path,
    output_dir: Path,
    scene: str = "chair",
    condition: str = "corrupt_occluder",
    subset_name: str = "seed_20260710",
    sample_id_count: int = 40,
    write_markdown: bool = False,
) -> tuple[dict[str, Any], int]:
    del sample_id_count, write_markdown
    output_dir.mkdir(parents=True, exist_ok=True)
    project_root = Path(__file__).resolve().parents[2]
    warnings: list[str] = []
    pr212a = load_json(pr212a_chair_dir / "pr212a_chair_id_namespace_audit_summary.json")
    _warn_if = lambda condition, message: warnings.append(message) if condition else None
    _warn_if(_truth(pr212a.get("same_global_gaussian_id_namespace_supported")), "PR21.2a namespace was expected to be unsupported")
    _warn_if(_truth(pr212a.get("zero_overlap_claim_safe_within_exact_available_scope")), "PR21.2a zero-overlap claim was expected unsafe")
    _warn_if(_truth(pr212a.get("pr20_proxy_ids_in_checkpoint_range")), "PR21.2a PR20 proxy IDs unexpectedly in checkpoint range")
    _warn_if(not _truth(pr212a.get("pr211_exact_ids_in_checkpoint_range")), "PR21.2a PR21 exact IDs expected in range")
    _warn_if(_truth(pr212a.get("proxy_safe_for_intervention")), "proxy_safe_for_intervention must remain false")
    checkpoint_count = _safe_int(pr212a.get("checkpoint_gaussian_count"))
    checkpoint_source = str(pr212a.get("checkpoint_gaussian_count_source") or "")
    if checkpoint_count is None:
        checkpoint_count, checkpoint_source = _load_checkpoint_count(run_dir)

    proxy_rows = load_csv_rows(pr200_chair_dir / "pr200_pixel_gaussian_contributions.csv")
    exact_rows = load_csv_rows(pr211_chair_dir / "pr211_exact_pixel_gaussian_contributions.csv")
    exact_keys = {_pixel_key(row) for row in exact_rows}
    profile = _profile_proxy_rows(proxy_rows, checkpoint_count)
    roots = [
        ("pr200", pr200_chair_dir),
        ("pr211", pr211_chair_dir),
        ("pr212", pr212_chair_dir),
        ("pr212a", pr212a_chair_dir),
        ("pr213", pr213_chair_dir),
        ("run_dir", run_dir),
    ]
    inventory = _inventory_sources(roots)
    proxy_ids = set(_id_values(proxy_rows, "gaussian_id"))
    root_ids = set(_id_values(proxy_rows, "root_gaussian_id"))
    parent_ids = set(_id_values(proxy_rows, "parent_gaussian_id"))
    train013_ids = {value for row in proxy_rows if row.get("view_name") == "train_013" for value in [_safe_int(row.get("gaussian_id"))] if value is not None and 100000 <= value <= 100016}
    exact_available_proxy_ids = {value for row in proxy_rows if _pixel_key(row) in exact_keys for value in [_safe_int(row.get("gaussian_id"))] if value is not None}
    lookup = _lookup_ids(
        {
            "proxy_gaussian_id": proxy_ids,
            "proxy_root_gaussian_id": root_ids,
            "proxy_parent_gaussian_id": parent_ids,
            "train013_suspicious_proxy_id": train013_ids,
        },
        inventory,
    )
    train013_found = any(row["id_role"] == "train013_suspicious_proxy_id" and row["found"] == "true" for row in lookup)
    mapping_rows, mapping, mapping_meta = _build_mapping_candidates(inventory, proxy_ids, exact_available_proxy_ids, train013_ids, checkpoint_count)
    semantics = _semantics_diagnosis(profile, mapping_rows, train013_found)
    repair = _repair_feasibility(proxy_rows, exact_keys, mapping, checkpoint_count, mapping_meta.get("mapping_confidence", ""))
    repaired_preview = _repaired_preview(scene, condition, subset_name, proxy_rows, mapping, mapping_meta)
    repaired_compare = _repaired_compare(exact_rows, proxy_rows, mapping, mapping_meta.get("mapping_confidence", ""))
    code_rows, code_summary = _code_audit(project_root)

    all_repair = next(row for row in repair if row["repair_scope"] == "all_pr20_proxy_rows")
    exact_repair = next(row for row in repair if row["repair_scope"] == "exact_available_pr20_proxy_rows")
    explicit_mapping = mapping_meta.get("mapping_status") == "verified_mapping_candidate"
    repaired_preview_available = any(row.get("comparison_status") == "repaired_comparison_valid" for row in repaired_compare)
    repaired_zero_safe = bool(repaired_preview_available and exact_repair["repair_feasible"] == "true")
    proxy_sem = next((row["inferred_semantics"] for row in semantics if row["diagnosis_target"] == "pr20_gaussian_id"), "unknown")
    summary = {
        "schema_name": "viewtrust.pr212b.pr20_proxy_id_source_audit.summary",
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
        "pr212a_chair_input_dir": str(pr212a_chair_dir),
        "pr213_chair_input_dir": str(pr213_chair_dir),
        "run_dir": str(run_dir),
        "checkpoint_gaussian_count": checkpoint_count,
        "checkpoint_gaussian_count_source": checkpoint_source,
        "pr20_proxy_id_semantics": proxy_sem,
        "pr20_proxy_id_semantics_confidence": "medium" if proxy_sem != "unknown" else "low",
        "train013_100000_ids_explained": train013_found,
        "train013_100000_source": "identity_or_lifecycle_source" if train013_found else "",
        "explicit_mapping_available": explicit_mapping,
        "mapping_source": mapping_meta.get("source_path", ""),
        "mapping_confidence": mapping_meta.get("mapping_confidence", "none"),
        "all_pr20_proxy_rows_repair_feasible": all_repair["repair_feasible"] == "true",
        "exact_available_proxy_rows_repair_feasible": exact_repair["repair_feasible"] == "true",
        "exact_available_mapping_coverage_rate": exact_repair["mapping_coverage_rate"],
        "repaired_exact_vs_proxy_preview_available": repaired_preview_available,
        "repaired_zero_overlap_claim_safe": repaired_zero_safe,
        "proxy_safe_for_intervention": False,
        "exact_render_contribution_available": False,
        "exact_contribution_magnitude_available": False,
        "drums_used_as_exact_evidence": False,
        "pr212b_ready_for_pr212c": exact_repair["repair_feasible"] == "true",
        "pr212b_ready_for_pr214": False,
        "recommended_next_step": "Run PR21.2c repaired exact-vs-proxy comparison on exact-available chair pixels."
        if exact_repair["repair_feasible"] == "true"
        else "Re-export PR20 proxy attribution with final checkpoint compact indices from the source pipeline.",
        "warnings": warnings,
    }

    write_json(output_dir / "pr212b_pr20_proxy_id_source_audit_summary.json", summary)
    write_csv_rows(output_dir / "pr212b_pr20_proxy_id_profile.csv", profile, PROFILE_FIELDS)
    write_csv_rows(output_dir / "pr212b_identity_mapping_inventory.csv", inventory, INVENTORY_FIELDS)
    write_csv_rows(output_dir / "pr212b_proxy_id_lookup_across_identity_sources.csv", lookup, LOOKUP_FIELDS)
    write_csv_rows(output_dir / "pr212b_pr20_id_semantics_diagnosis.csv", semantics, SEMANTICS_FIELDS)
    write_csv_rows(output_dir / "pr212b_mapping_candidate_table.csv", mapping_rows, MAPPING_FIELDS)
    write_csv_rows(output_dir / "pr212b_repair_feasibility_summary.csv", repair, REPAIR_FIELDS)
    write_csv_rows(output_dir / "pr212b_pr20_proxy_repaired_preview.csv", repaired_preview, REPAIRED_PREVIEW_FIELDS)
    write_csv_rows(output_dir / "pr212b_repaired_exact_vs_proxy_preview.csv", repaired_compare, REPAIRED_COMPARE_FIELDS)
    write_csv_rows(output_dir / "pr212b_code_proxy_id_source_audit.csv", code_rows, CODE_FIELDS)
    write_json(output_dir / "pr212b_code_proxy_id_source_summary.json", code_summary)
    _write_report(output_dir / "pr212b_pr20_proxy_id_source_audit_report.md", summary)
    _write_wording(output_dir / "pr212b_proxy_namespace_wording.md")
    _write_next_step(output_dir / "pr212b_next_step_decision_memo.md", exact_repair["repair_feasible"] == "true")
    inputs = [
        ("pr200_chair_dir", pr200_chair_dir, True),
        ("pr211_chair_dir", pr211_chair_dir, True),
        ("pr212_chair_dir", pr212_chair_dir, True),
        ("pr212a_chair_dir", pr212a_chair_dir, True),
        ("pr213_chair_dir", pr213_chair_dir, True),
        ("run_dir", run_dir, True),
    ]
    manifest = output_dir / "artifact_manifest.csv"
    write_csv_rows(manifest, _manifest_rows(output_dir, inputs), MANIFEST_FIELDS)
    write_csv_rows(manifest, _manifest_rows(output_dir, inputs), MANIFEST_FIELDS)
    return summary, 0
