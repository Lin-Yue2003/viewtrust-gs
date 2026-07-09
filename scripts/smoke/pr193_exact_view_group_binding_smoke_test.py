#!/usr/bin/env python3
"""LOCAL-SAFE smoke test for PR19.3 exact view-group binding."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path


REQUIRED_OUTPUTS = [
    "pr193_view_group_map.csv",
    "pr193_view_group_binding_summary.json",
    "gaussian_identity_table_grouped.csv",
    "gaussian_lifecycle_events_grouped.csv",
    "view_gaussian_event_attribution_grouped.csv",
    "gaussian_support_summary_grouped.csv",
    "pr193_exact_group_overlap_summary.csv",
    "pr193_train013_exact_control.csv",
    "pr193_direct_collateral_exact_overlap.csv",
    "pr193_pr19_exact_input_bundle_manifest.csv",
    "pr193_missing_inputs.csv",
    "pr193_report.md",
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


def _make_pr17(pr17_dir: Path) -> None:
    rows = [
        {
            "scene": "chair",
            "condition": "corrupt_occluder",
            "subset_name": "seed_20260710",
            "view_name": "train_004",
            "was_corrupted": "true",
            "raw_rank": 1,
            "normalized_rank": 1,
            "raw_false_positive": "false",
            "normalized_false_positive": "false",
        },
        {
            "scene": "chair",
            "condition": "corrupt_occluder",
            "subset_name": "seed_20260710",
            "view_name": "train_009",
            "was_corrupted": "true",
            "raw_rank": 2,
            "normalized_rank": 2,
            "raw_false_positive": "false",
            "normalized_false_positive": "false",
        },
        {
            "scene": "chair",
            "condition": "corrupt_occluder",
            "subset_name": "seed_20260710",
            "view_name": "train_014",
            "was_corrupted": "false",
            "raw_rank": 5,
            "normalized_rank": 3,
            "raw_false_positive": "true",
            "normalized_false_positive": "true",
        },
        {
            "scene": "chair",
            "condition": "corrupt_occluder",
            "subset_name": "seed_20260710",
            "view_name": "train_013",
            "was_corrupted": "false",
            "raw_rank": 3,
            "normalized_rank": 20,
            "raw_false_positive": "true",
            "normalized_false_positive": "false",
        },
    ]
    _write_csv(pr17_dir / "clean_prior_normalized_rows.csv", rows)
    _write_csv(pr17_dir / "clean_prior_normalized_rankings.csv", rows)
    _write_json(
        pr17_dir / "clean_prior_normalized_summary.json",
        {
            "schema_name": "viewtrust.pr17.clean_prior_normalized.summary",
            "observation_only": True,
            "training_intervention": False,
            "defense_enabled": False,
        },
    )


def _make_pr18(pr18_dir: Path) -> None:
    rows = [
        {
            "scene": "chair",
            "condition": "corrupt_occluder",
            "subset_name": "seed_20260710",
            "view_name": "train_014",
            "spillover_class": "co_visible_collateral",
            "spillover_confidence": "high",
            "camera_neighbor_evidence": "true",
            "index_neighbor_evidence": "true",
            "gaussian_overlap_evidence": "true",
            "collateral_lift_pattern": "true",
        },
        {
            "scene": "chair",
            "condition": "corrupt_occluder",
            "subset_name": "seed_20260710",
            "view_name": "train_013",
            "spillover_class": "clean_prior_false_positive",
            "spillover_confidence": "high",
            "camera_neighbor_evidence": "false",
            "index_neighbor_evidence": "false",
            "gaussian_overlap_evidence": "false",
            "collateral_lift_pattern": "false",
        },
    ]
    _write_csv(pr18_dir / "pr18_spillover_classification.csv", rows)
    _write_csv(
        pr18_dir / "pr18_condition_summary.csv",
        [{"scene": "chair", "condition": "corrupt_occluder", "subset_name": "seed_20260710"}],
    )
    _write_csv(
        pr18_dir / "pr18_view_identity_transition.csv",
        [{"scene": "chair", "condition": "corrupt_occluder", "subset_name": "seed_20260710", "view_name": "train_013", "transition": "clean_prior_demoted"}],
    )
    _write_json(
        pr18_dir / "pr18_covisibility_spillover_summary.json",
        {
            "schema_name": "viewtrust.pr18.covisibility_spillover.summary",
            "observation_only": True,
            "training_intervention": False,
            "defense_enabled": False,
        },
    )


def _make_exact_logs(exact_dir: Path) -> None:
    identity_rows = [
        {"scene": "chair", "condition": "corrupt_occluder", "subset_name": "seed_20260710", "run_id": "smoke", "gaussian_id": gid, "parent_gaussian_id": "", "root_gaussian_id": gid, "is_alive_final": "true", "support_view_names": ""}
        for gid in ["1", "2", "3", "4"]
    ]
    attribution_rows = [
        {"scene": "chair", "condition": "corrupt_occluder", "subset_name": "seed_20260710", "run_id": "smoke", "view_name": "train_004", "view_index": 4, "iteration": 100, "gaussian_id": "1", "event_type": "visibility_observation", "attribution_type": "visibility", "contribution_value": 1},
        {"scene": "chair", "condition": "corrupt_occluder", "subset_name": "seed_20260710", "run_id": "smoke", "view_name": "train_014", "view_index": 14, "iteration": 100, "gaussian_id": "1", "event_type": "visibility_observation", "attribution_type": "visibility", "contribution_value": 1},
        {"scene": "chair", "condition": "corrupt_occluder", "subset_name": "seed_20260710", "run_id": "smoke", "view_name": "train_009", "view_index": 9, "iteration": 110, "gaussian_id": "2", "event_type": "visibility_observation", "attribution_type": "visibility", "contribution_value": 1},
        {"scene": "chair", "condition": "corrupt_occluder", "subset_name": "seed_20260710", "run_id": "smoke", "view_name": "train_014", "view_index": 14, "iteration": 110, "gaussian_id": "2", "event_type": "visibility_observation", "attribution_type": "visibility", "contribution_value": 1},
        {"scene": "chair", "condition": "corrupt_occluder", "subset_name": "seed_20260710", "run_id": "smoke", "view_name": "train_013", "view_index": 13, "iteration": 120, "gaussian_id": "3", "event_type": "visibility_observation", "attribution_type": "visibility", "contribution_value": 3},
        {"scene": "chair", "condition": "corrupt_occluder", "subset_name": "seed_20260710", "run_id": "smoke", "view_name": "train_004", "view_index": 4, "iteration": 130, "gaussian_id": "4", "event_type": "update_observation", "attribution_type": "update", "contribution_value": 1},
        {"scene": "chair", "condition": "corrupt_occluder", "subset_name": "seed_20260710", "run_id": "smoke", "view_name": "train_009", "view_index": 9, "iteration": 130, "gaussian_id": "4", "event_type": "update_observation", "attribution_type": "update", "contribution_value": 1},
        {"scene": "chair", "condition": "corrupt_occluder", "subset_name": "seed_20260710", "run_id": "smoke", "view_name": "train_014", "view_index": 14, "iteration": 130, "gaussian_id": "4", "event_type": "update_observation", "attribution_type": "update", "contribution_value": 1},
    ]
    support_by_gid = {
        "1": "train_004;train_014",
        "2": "train_009;train_014",
        "3": "train_013",
        "4": "train_004;train_009;train_014",
    }
    support_rows = [
        {
            "scene": "chair",
            "condition": "corrupt_occluder",
            "subset_name": "seed_20260710",
            "run_id": "smoke",
            "gaussian_id": gid,
            "parent_gaussian_id": "",
            "root_gaussian_id": gid,
            "support_view_count": len(views.split(";")),
            "support_view_names": views,
            "is_alive_final": "true",
            "evidence_quality": "exact",
        }
        for gid, views in support_by_gid.items()
    ]
    event_rows = [
        {
            "scene": row["scene"],
            "condition": row["condition"],
            "subset_name": row["subset_name"],
            "run_id": row["run_id"],
            "iteration": row["iteration"],
            "view_name": row["view_name"],
            "view_index": row["view_index"],
            "event_type": row["event_type"],
            "gaussian_id": row["gaussian_id"],
            "parent_gaussian_id": "",
            "root_gaussian_id": row["gaussian_id"],
            "is_alive_after_event": "true",
            "event_source": "smoke",
            "evidence_quality": "exact",
        }
        for row in attribution_rows
    ]
    _write_csv(exact_dir / "gaussian_identity_table.csv", identity_rows)
    _write_csv(exact_dir / "gaussian_lifecycle_events.csv", event_rows)
    _write_csv(exact_dir / "view_gaussian_event_attribution.csv", attribution_rows)
    _write_csv(exact_dir / "gaussian_support_summary.csv", support_rows)
    _write_json(
        exact_dir / "exact_gaussian_logging_summary.json",
        {
            "schema_name": "viewtrust.pr191.exact_gaussian_lifecycle_logging.summary",
            "evidence_quality": "exact",
            "integration_source": "real_view_influence_runner",
            "parent_mapping_source": "exact_clone_split_masks",
            "observation_only": True,
            "training_intervention": False,
            "defense_enabled": False,
        },
    )
    _write_json(
        exact_dir / "exact_gaussian_logging_validation.json",
        {"identity_consistency_passed": True, "parent_child_consistency_passed": True},
    )
    _write_csv(
        exact_dir / "artifact_manifest.csv",
        [{"relative_path": "gaussian_identity_table.csv", "path": str(exact_dir / "gaussian_identity_table.csv"), "exists": "true", "required": "true"}],
    )


def main() -> int:
    project_root = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory(prefix="viewtrust-pr193-") as tmp:
        root = Path(tmp)
        exact_dir = root / "exact"
        pr17_dir = root / "pr17"
        pr18_dir = root / "pr18"
        output_dir = root / "pr193"
        _make_exact_logs(exact_dir)
        _make_pr17(pr17_dir)
        _make_pr18(pr18_dir)
        result = _run(
            [
                sys.executable,
                str(project_root / "scripts" / "measure" / "bind_pr193_exact_view_groups.py"),
                "--exact-log-dir",
                str(exact_dir),
                "--pr17-dir",
                str(pr17_dir),
                "--pr18-dir",
                str(pr18_dir),
                "--scene",
                "chair",
                "--condition",
                "corrupt_occluder",
                "--subset-name",
                "seed_20260710",
                "--output-dir",
                str(output_dir),
                "--top-k",
                "20",
                "--copy-pr19-ready-bundle",
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
        view_map = {row["view_name"]: row for row in _read_csv(output_dir / "pr193_view_group_map.csv")}
        assert view_map["train_004"]["view_group"] == "direct_corrupted"
        assert view_map["train_009"]["view_group"] == "direct_corrupted"
        assert view_map["train_014"]["view_group"] == "co_visible_collateral"
        assert view_map["train_013"]["view_group"] == "clean_prior_demoted"
        support_rows = _read_csv(output_dir / "gaussian_support_summary_grouped.csv")
        assert any(float(row["collateral_unique_view_count"]) > 0 for row in support_rows)
        assert any(float(row["clean_prior_unique_view_count"]) > 0 for row in support_rows)
        overlap = _read_csv(output_dir / "pr193_direct_collateral_exact_overlap.csv")[0]
        assert overlap["direct_collateral_exact_overlap_supported"] == "true"
        train013 = _read_csv(output_dir / "pr193_train013_exact_control.csv")[0]
        assert train013["train013_exact_control_supported"] == "true"
        summary = json.loads((output_dir / "pr193_view_group_binding_summary.json").read_text())
        assert summary["observation_only"] is True
        assert summary["training_intervention"] is False
        assert summary["defense_enabled"] is False
        assert summary["uses_corruption_labels_for_scoring"] is False
        assert summary["pr19_ready_bundle_written"] is True
        bundle = output_dir / "pr19_exact_input_bundle" / "exact_gaussian_logging"
        for name in [
            "gaussian_identity_table.csv",
            "gaussian_lifecycle_events.csv",
            "view_gaussian_event_attribution.csv",
            "gaussian_support_summary.csv",
            "exact_gaussian_logging_summary.json",
            "exact_gaussian_logging_validation.json",
            "artifact_manifest.csv",
        ]:
            assert (bundle / name).exists(), name
    print("pr193 exact view group binding smoke test ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
