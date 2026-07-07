#!/usr/bin/env python3
"""Compare clean and corrupt PR12 view influence tables."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

SCHEMA_NAME = "viewtrust.view_influence.comparison.summary"
SCHEMA_VERSION = 1


def _csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _json_file(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _float_or_none(value: Any) -> float | None:
    if value in ("", None):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _delta(corrupt: float | None, clean: float | None) -> float | None:
    return corrupt - clean if clean is not None and corrupt is not None else None


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def compare_tables(clean_dir: Path, corrupt_dir: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    clean_summary = _json_file(clean_dir / "view_influence_summary.json")
    corrupt_summary = _json_file(corrupt_dir / "view_influence_summary.json")
    clean_rows = {
        row["view_name"]: row
        for row in _csv_rows(clean_dir / "view_influence.csv")
        if row.get("view_name")
    }
    corrupt_rows = {
        row["view_name"]: row
        for row in _csv_rows(corrupt_dir / "view_influence.csv")
        if row.get("view_name")
    }
    rows: list[dict[str, Any]] = []
    for view_name in sorted(set(clean_rows) | set(corrupt_rows)):
        clean = clean_rows.get(view_name, {})
        corrupt = corrupt_rows.get(view_name, {})
        clean_birth = _float_or_none(clean.get("birth_event_count_after_view"))
        corrupt_birth = _float_or_none(corrupt.get("birth_event_count_after_view"))
        clean_prune = _float_or_none(clean.get("prune_death_count_after_view"))
        corrupt_prune = _float_or_none(corrupt.get("prune_death_count_after_view"))
        clean_survival = _float_or_none(clean.get("birth_survival_ratio_after_view"))
        corrupt_survival = _float_or_none(corrupt.get("birth_survival_ratio_after_view"))
        clean_visibility = _float_or_none(clean.get("mean_visibility_ratio"))
        corrupt_visibility = _float_or_none(corrupt.get("mean_visibility_ratio"))
        rows.append(
            {
                "view_name": view_name,
                "was_corrupted": corrupt.get("was_corrupted", "false"),
                "clean_times_sampled": clean.get("times_sampled", ""),
                "corrupt_times_sampled": corrupt.get("times_sampled", ""),
                "clean_birth_event_count_after_view": clean_birth,
                "corrupt_birth_event_count_after_view": corrupt_birth,
                "birth_event_count_delta": _delta(corrupt_birth, clean_birth),
                "clean_prune_death_count_after_view": clean_prune,
                "corrupt_prune_death_count_after_view": corrupt_prune,
                "prune_death_count_delta": _delta(corrupt_prune, clean_prune),
                "clean_birth_survival_ratio_after_view": clean_survival,
                "corrupt_birth_survival_ratio_after_view": corrupt_survival,
                "birth_survival_ratio_delta": _delta(corrupt_survival, clean_survival),
                "clean_mean_visibility_ratio": clean_visibility,
                "corrupt_mean_visibility_ratio": corrupt_visibility,
                "visibility_ratio_delta": _delta(corrupt_visibility, clean_visibility),
            }
        )
    fields = [
        "view_name",
        "was_corrupted",
        "clean_times_sampled",
        "corrupt_times_sampled",
        "clean_birth_event_count_after_view",
        "corrupt_birth_event_count_after_view",
        "birth_event_count_delta",
        "clean_prune_death_count_after_view",
        "corrupt_prune_death_count_after_view",
        "prune_death_count_delta",
        "clean_birth_survival_ratio_after_view",
        "corrupt_birth_survival_ratio_after_view",
        "birth_survival_ratio_delta",
        "clean_mean_visibility_ratio",
        "corrupt_mean_visibility_ratio",
        "visibility_ratio_delta",
    ]
    _write_csv(output_dir / "view_influence_comparison.csv", rows, fields)
    top_prune = sorted(
        rows,
        key=lambda row: _float_or_none(row.get("prune_death_count_delta")) or 0.0,
        reverse=True,
    )[:10]
    summary = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "clean_view_influence_dir": str(clean_dir.resolve()),
        "corrupt_view_influence_dir": str(corrupt_dir.resolve()),
        "clean_run_id": clean_summary.get("run_id"),
        "corrupt_run_id": corrupt_summary.get("run_id"),
        "scene": corrupt_summary.get("scene") or clean_summary.get("scene"),
        "clean_condition": clean_summary.get("condition"),
        "corrupt_condition": corrupt_summary.get("condition"),
        "joined_view_count": len(rows),
        "corrupted_view_count": sum(1 for row in rows if row.get("was_corrupted") == "true"),
        "top_views_by_prune_death_delta": [
            {
                "view_name": row["view_name"],
                "was_corrupted": row["was_corrupted"],
                "prune_death_count_delta": row["prune_death_count_delta"],
            }
            for row in top_prune
        ],
        "warnings": [],
    }
    (output_dir / "view_influence_comparison_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _markdown(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# View Influence Comparison Report",
            "",
            "This report compares observation-only view influence tables. It is not a trust score or defense.",
            "",
            "## Inputs",
            f"- Clean: `{summary.get('clean_view_influence_dir')}`",
            f"- Corrupt: `{summary.get('corrupt_view_influence_dir')}`",
            f"- Scene: `{summary.get('scene')}`",
            f"- Conditions: `{summary.get('clean_condition')}` vs `{summary.get('corrupt_condition')}`",
            "",
            "## Coverage",
            f"- Joined views: `{summary.get('joined_view_count')}`",
            f"- Corrupted views: `{summary.get('corrupted_view_count')}`",
            "",
            "## Interpretation",
            "- Use this as exploratory evidence for later offline signal design.",
            "- Do not interpret deltas as detection or classification.",
            "",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-view-influence-dir", required=True, type=Path)
    parser.add_argument("--corrupt-view-influence-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--write-markdown", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = compare_tables(
        args.clean_view_influence_dir,
        args.corrupt_view_influence_dir,
        args.output_dir,
    )
    if args.write_markdown:
        (args.output_dir / "view_influence_comparison_report.md").write_text(
            _markdown(summary),
            encoding="utf-8",
        )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
