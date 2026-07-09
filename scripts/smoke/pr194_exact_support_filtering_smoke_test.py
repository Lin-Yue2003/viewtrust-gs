#!/usr/bin/env python3
"""LOCAL-SAFE smoke test for PR19.4 exact support filtering."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path


REQUIRED_OUTPUTS = [
    "pr194_exact_support_filter_summary.json",
    "pr194_support_mode_comparison.csv",
    "pr194_filtered_gaussian_support_by_mode.csv",
    "pr194_direct_collateral_overlap_by_mode.csv",
    "pr194_train013_control_by_mode.csv",
    "pr194_gaussian_mode_membership.csv",
    "pr194_view_group_event_concentration.csv",
    "pr194_nontrivial_overlap_candidates.csv",
    "pr194_missing_inputs.csv",
    "pr194_report.md",
    "artifact_manifest.csv",
]


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, object]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = fields or sorted({field for row in rows for field in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def _make_pr193(pr193_dir: Path) -> None:
    views = {
        "train_004": "direct_corrupted",
        "train_009": "direct_corrupted",
        "train_014": "co_visible_collateral",
        "train_013": "clean_prior_demoted",
        "train_002": "other_clean",
    }
    _write_csv(
        pr193_dir / "pr193_view_group_map.csv",
        [
            {
                "scene": "chair",
                "condition": "corrupt_occluder",
                "subset_name": "seed_20260710",
                "view_name": view,
                "view_group": group,
                "was_corrupted": str(group == "direct_corrupted").lower(),
            }
            for view, group in views.items()
        ],
    )
    _write_json(
        pr193_dir / "pr193_view_group_binding_summary.json",
        {
            "schema_name": "viewtrust.pr193.exact_view_group_binding.summary",
            "scene": "chair",
            "condition": "corrupt_occluder",
            "subset_name": "seed_20260710",
            "output_exact_evidence_quality": "exact",
            "observation_only": True,
            "training_intervention": False,
            "defense_enabled": False,
        },
    )
    identity_rows = []
    support_rows = []
    for gid in ["g1", "g2", "g3", "g4", "g5"]:
        row = {
            "scene": "chair",
            "condition": "corrupt_occluder",
            "subset_name": "seed_20260710",
            "gaussian_id": gid,
            "root_gaussian_id": gid,
            "parent_gaussian_id": "",
            "birth_event_type": "clone_birth",
            "death_event_type": "",
            "is_alive_final": "true",
            "support_view_names": ";".join(views),
        }
        identity_rows.append(row)
        support_rows.append({**row, "support_view_count": len(views)})
    _write_csv(pr193_dir / "gaussian_identity_table_grouped.csv", identity_rows)
    _write_csv(pr193_dir / "gaussian_support_summary_grouped.csv", support_rows)

    attribution_rows = []
    for gid in ["g1", "g2", "g3", "g4", "g5"]:
        for view, group in views.items():
            attribution_rows.append(
                {
                    "scene": "chair",
                    "condition": "corrupt_occluder",
                    "subset_name": "seed_20260710",
                    "view_name": view,
                    "view_group": group,
                    "gaussian_id": gid,
                    "event_type": "visibility_observation",
                    "attribution_type": "visibility",
                    "contribution_value": 1,
                }
            )
    high_events = [
        ("g1", "train_004", 10),
        ("g1", "train_014", 9),
        ("g2", "train_013", 10),
        ("g3", "train_002", 10),
        ("g4", "train_004", 8),
        ("g5", "train_014", 8),
    ]
    for gid, view, weight in high_events:
        attribution_rows.append(
            {
                "scene": "chair",
                "condition": "corrupt_occluder",
                "subset_name": "seed_20260710",
                "view_name": view,
                "view_group": views[view],
                "gaussian_id": gid,
                "event_type": "update_observation",
                "attribution_type": "update",
                "contribution_value": weight,
            }
        )
    lifecycle_rows = [
        {
            "scene": "chair",
            "condition": "corrupt_occluder",
            "subset_name": "seed_20260710",
            "view_name": "train_004",
            "view_group": "direct_corrupted",
            "gaussian_id": "g1",
            "parent_gaussian_id": "root",
            "root_gaussian_id": "g1",
            "event_type": "clone_birth",
        },
        {
            "scene": "chair",
            "condition": "corrupt_occluder",
            "subset_name": "seed_20260710",
            "view_name": "train_013",
            "view_group": "clean_prior_demoted",
            "gaussian_id": "g2",
            "parent_gaussian_id": "root",
            "root_gaussian_id": "g2",
            "event_type": "clone_birth",
        },
    ]
    _write_csv(pr193_dir / "view_gaussian_event_attribution_grouped.csv", attribution_rows)
    _write_csv(pr193_dir / "gaussian_lifecycle_events_grouped.csv", lifecycle_rows)
    _write_csv(
        pr193_dir / "pr193_direct_collateral_exact_overlap.csv",
        [{"scene": "chair", "condition": "corrupt_occluder", "subset_name": "seed_20260710", "exact_overlap_jaccard": 1.0}],
    )
    _write_csv(
        pr193_dir / "pr193_train013_exact_control.csv",
        [{"scene": "chair", "condition": "corrupt_occluder", "subset_name": "seed_20260710", "train013_exact_control_supported": "false"}],
    )


def main() -> int:
    project_root = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory(prefix="viewtrust-pr194-") as tmp:
        root = Path(tmp)
        pr193_dir = root / "pr193"
        output_dir = root / "pr194"
        _make_pr193(pr193_dir)
        result = _run(
            [
                sys.executable,
                str(project_root / "scripts" / "measure" / "analyze_pr194_exact_support_filters.py"),
                "--pr193-dir",
                str(pr193_dir),
                "--scene",
                "chair",
                "--condition",
                "corrupt_occluder",
                "--subset-name",
                "seed_20260710",
                "--output-dir",
                str(output_dir),
                "--support-modes",
                "broad",
                "birth",
                "prune",
                "high_event",
                "dominant_source",
                "low_entropy",
                "suspicious_alive",
                "--event-percentile",
                "75",
                "--dominant-source-threshold",
                "0.5",
                "--low-entropy-threshold",
                "0.35",
                "--min-event-count",
                "3",
                "--write-markdown",
            ]
        )
        if result.returncode != 0:
            print(result.stdout)
            print(result.stderr, file=sys.stderr)
            return result.returncode
        for name in REQUIRED_OUTPUTS:
            path = output_dir / name
            assert path.exists(), name
            assert path.stat().st_size > 0, name
        summary = json.loads((output_dir / "pr194_exact_support_filter_summary.json").read_text())
        assert summary["schema_name"] == "viewtrust.pr194.exact_support_filter.summary"
        assert summary["observation_only"] is True
        assert summary["training_intervention"] is False
        assert summary["defense_enabled"] is False
        assert summary["uses_corruption_labels_for_scoring"] is False
        assert summary["broad_overlap_degeneracy_detected"] is True
        rows = {row["support_mode"]: row for row in _read_csv(output_dir / "pr194_support_mode_comparison.csv")}
        assert float(rows["broad"]["direct_collateral_jaccard"]) >= 0.95
        assert rows["broad"]["broad_overlap_degeneracy_flag"] == "true"
        assert rows["broad"]["train013_control_supported"] == "false"
        assert any(
            rows[mode]["nontrivial_overlap_supported"] == "true"
            and float(rows[mode]["direct_collateral_jaccard"]) < 0.95
            for mode in ("high_event", "dominant_source")
        )
        train_rows = {row["support_mode"]: row for row in _read_csv(output_dir / "pr194_train013_control_by_mode.csv")}
        assert any(train_rows[mode]["train013_control_supported"] == "true" for mode in ("high_event", "dominant_source"))
    print("pr194 exact support filtering smoke test ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
