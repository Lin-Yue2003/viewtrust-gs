"""Validation helpers for PR19.1 exact Gaussian lifecycle logs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


REQUIRED_EXACT_FILES = [
    "gaussian_identity_table.csv",
    "gaussian_lifecycle_events.csv",
    "view_gaussian_event_attribution.csv",
    "gaussian_support_summary.csv",
    "exact_gaussian_logging_summary.json",
    "exact_gaussian_logging_validation.json",
    "artifact_manifest.csv",
]

VALIDATION_OUTPUT_FILES = [
    "pr191_exact_gaussian_logging_validation_summary.json",
    "pr191_identity_consistency.csv",
    "pr191_parent_child_consistency.csv",
    "pr191_support_summary.csv",
    "pr191_missing_outputs.csv",
    "pr191_report.md",
    "artifact_manifest.csv",
]


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def write_csv_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: "" if row.get(field) is None else row.get(field) for field in fieldnames})


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_exact_lifecycle_events(exact_log_dir: Path) -> list[dict[str, str]]:
    return load_csv_rows(exact_log_dir / "gaussian_lifecycle_events.csv")


def load_gaussian_identity_table(exact_log_dir: Path) -> list[dict[str, str]]:
    return load_csv_rows(exact_log_dir / "gaussian_identity_table.csv")


def load_view_gaussian_attribution(exact_log_dir: Path) -> list[dict[str, str]]:
    return load_csv_rows(exact_log_dir / "view_gaussian_event_attribution.csv")


def _truth(value: Any) -> bool:
    return value is True or str(value).strip().lower() in {"true", "1", "yes"}


def validate_exact_gaussian_logs(exact_log_dir: Path) -> dict[str, Any]:
    identity_rows = load_gaussian_identity_table(exact_log_dir)
    event_rows = load_exact_lifecycle_events(exact_log_dir)
    support_rows = load_csv_rows(exact_log_dir / "gaussian_support_summary.csv")
    summary = load_json(exact_log_dir / "exact_gaussian_logging_summary.json")
    missing = [name for name in REQUIRED_EXACT_FILES if not (exact_log_dir / name).is_file()]
    ids = [row.get("gaussian_id", "") for row in identity_rows if row.get("gaussian_id", "") != ""]
    id_set = set(ids)
    alive_ids = [row.get("gaussian_id", "") for row in identity_rows if _truth(row.get("is_alive_final"))]
    parent_ok = all(
        row.get("parent_gaussian_id", "") in {"", *id_set}
        for row in identity_rows
    )
    root_ok = all(row.get("root_gaussian_id", "") in id_set for row in identity_rows)
    prune_ok = all(
        row.get("gaussian_id", "") in id_set
        for row in event_rows
        if row.get("event_type") == "prune_death"
    )
    row_count_matches = len(alive_ids) == int(summary.get("total_final_gaussians") or len(alive_ids))
    errors = []
    if missing:
        errors.append("missing required exact log outputs")
    if len(alive_ids) != len(set(alive_ids)):
        errors.append("duplicate alive gaussian IDs")
    if not row_count_matches:
        errors.append("alive identity count does not match final Gaussian count")
    if not parent_ok:
        errors.append("parent IDs missing")
    if not root_ok:
        errors.append("root IDs missing")
    if not prune_ok:
        errors.append("prune events reference missing IDs")
    if summary.get("uses_row_index_as_stable_id") is not False:
        errors.append("summary does not prove row index is not stable ID")
    return {
        "schema_name": "viewtrust.pr191.exact_gaussian_lifecycle_logging.validation",
        "schema_version": 1,
        "identity_consistency_passed": not errors,
        "no_duplicate_alive_gaussian_ids": len(alive_ids) == len(set(alive_ids)),
        "no_missing_alive_gaussian_ids": all(gaussian_id in id_set for gaussian_id in alive_ids),
        "row_count_matches_current_gaussian_count": row_count_matches,
        "parent_ids_exist_or_empty": parent_ok,
        "root_ids_exist": root_ok,
        "prune_events_reference_existing_ids": prune_ok,
        "exported_files_exist": not missing,
        "validation_errors": errors,
        "validation_warnings": [],
        "identity_row_count": len(identity_rows),
        "event_row_count": len(event_rows),
        "support_row_count": len(support_rows),
        "missing_files": missing,
    }


def compute_exact_support_summary(exact_log_dir: Path) -> list[dict[str, Any]]:
    return [dict(row) for row in load_csv_rows(exact_log_dir / "gaussian_support_summary.csv")]


def compute_exact_view_group_overlap(exact_log_dir: Path) -> list[dict[str, Any]]:
    rows = load_csv_rows(exact_log_dir / "gaussian_support_summary.csv")
    output = []
    for row in rows:
        direct = int(float(row.get("direct_corrupted_support_count") or 0))
        collateral = int(float(row.get("collateral_support_count") or 0))
        clean_prior = int(float(row.get("clean_prior_support_count") or 0))
        output.append(
            {
                "gaussian_id": row.get("gaussian_id", ""),
                "has_direct_support": direct > 0,
                "has_collateral_support": collateral > 0,
                "has_clean_prior_support": clean_prior > 0,
                "direct_collateral_overlap_supported": direct > 0 and collateral > 0,
                "corrupted_plus_collateral_ratio": row.get("corrupted_plus_collateral_ratio", ""),
                "clean_prior_ratio": row.get("clean_prior_ratio", ""),
            }
        )
    return output


def compute_exact_train013_control(exact_log_dir: Path) -> dict[str, Any]:
    support_rows = load_csv_rows(exact_log_dir / "gaussian_support_summary.csv")
    train_rows = [row for row in support_rows if "train_013" in str(row.get("support_view_names", "")).split(";")]
    overlapping = [
        row for row in train_rows
        if float(row.get("corrupted_plus_collateral_ratio") or 0.0) > 0.0
    ]
    return {
        "train013_present": bool(train_rows),
        "train013_cluster_count": len(train_rows),
        "train013_low_overlap_with_direct_collateral": len(overlapping) == 0,
        "control_supported": bool(train_rows) and len(overlapping) == 0,
    }


def write_artifact_manifest(output_dir: Path, exact_log_dir: Path) -> None:
    fields = ["relative_path", "path", "exists", "file_type", "size_bytes", "required", "artifact_group"]

    def rows() -> list[dict[str, Any]]:
        items: list[tuple[str, Path, bool, str]] = [("exact_log_dir", exact_log_dir, True, "input")]
        items.extend((name, output_dir / name, True, "output_pr191_validation") for name in VALIDATION_OUTPUT_FILES)
        output = []
        for relative, path, required, group in items:
            output.append(
                {
                    "relative_path": relative,
                    "path": str(path),
                    "exists": str(path.exists()).lower(),
                    "file_type": "directory" if path.is_dir() else path.suffix.lstrip("."),
                    "size_bytes": path.stat().st_size if path.is_file() else "",
                    "required": str(required).lower(),
                    "artifact_group": group,
                }
            )
        return output

    manifest = output_dir / "artifact_manifest.csv"
    write_csv_rows(manifest, rows(), fields)
    write_csv_rows(manifest, rows(), fields)
