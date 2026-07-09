#!/usr/bin/env python3
"""LOCAL-SAFE smoke test for PR19.1 exact Gaussian identity tracking."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path


REQUIRED_EXACT_OUTPUTS = [
    "gaussian_identity_table.csv",
    "gaussian_lifecycle_events.csv",
    "view_gaussian_event_attribution.csv",
    "gaussian_support_summary.csv",
    "exact_gaussian_logging_summary.json",
    "exact_gaussian_logging_validation.json",
    "artifact_manifest.csv",
]

REQUIRED_VALIDATION_OUTPUTS = [
    "pr191_exact_gaussian_logging_validation_summary.json",
    "pr191_identity_consistency.csv",
    "pr191_parent_child_consistency.csv",
    "pr191_support_summary.csv",
    "pr191_missing_outputs.csv",
    "pr191_report.md",
    "artifact_manifest.csv",
]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def main() -> int:
    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root))
    from viewtrust.instrumentation.gaussian_identity_tracker import GaussianIdentityTracker

    with tempfile.TemporaryDirectory(prefix="viewtrust-pr191-") as tmp:
        root = Path(tmp)
        exact_dir = root / "exact"
        validate_dir = root / "validate"
        tracker = GaussianIdentityTracker(
            output_dir=exact_dir,
            view_group_map={
                "train_004": "direct_corrupted",
                "train_009": "direct_corrupted",
                "train_014": "co_visible_collateral",
                "train_013": "clean_prior_demoted",
            },
        )
        tracker.initialize(
            num_gaussians=3,
            scene="chair",
            condition="corrupt_occluder",
            subset_name="seed_20260710",
            run_id="pr191-smoke",
        )
        tracker.before_view(100, "train_004", view_index=4)
        clone_ids = tracker.record_clone_birth([1], 100, "train_004")
        split_ids = tracker.record_split_birth([2], [2], 100, "train_004")
        assert clone_ids == [3]
        assert split_ids == [4, 5]
        pruned = tracker.record_prune([True, False, False, False, False, False], 120, "train_004")
        assert pruned == [0]
        tracker.record_visibility_observation([0, 2, 3], 130, "train_014")
        tracker.record_update_observation([2, 3, 4], 140, "train_014", metadata={"gradient_norm": 0.5})
        tracker.compact_with_alive_mask([True, True, True, True, True])
        summary = tracker.write_outputs(exact_dir)
        assert summary["observation_only"] is True
        assert summary["training_intervention"] is False
        assert summary["defense_enabled"] is False
        assert summary["stable_gaussian_ids_enabled"] is True
        assert summary["uses_row_index_as_stable_id"] is False
        for name in REQUIRED_EXACT_OUTPUTS:
            path = exact_dir / name
            assert path.exists(), name
            assert path.stat().st_size > 0, name

        identity = {int(row["gaussian_id"]): row for row in _read_csv(exact_dir / "gaussian_identity_table.csv")}
        assert sorted(identity) == [0, 1, 2, 3, 4, 5]
        assert identity[3]["parent_gaussian_id"] == "1"
        assert identity[4]["parent_gaussian_id"] == "2"
        assert identity[5]["parent_gaussian_id"] == "2"
        assert identity[3]["root_gaussian_id"] == "1"
        assert identity[4]["root_gaussian_id"] == "2"
        assert identity[5]["root_gaussian_id"] == "2"
        assert identity[0]["is_alive_final"] == "False"
        alive = {gaussian_id for gaussian_id, row in identity.items() if row["is_alive_final"] == "True"}
        assert alive == {1, 2, 3, 4, 5}
        final_indices = {int(row["final_row_index"]) for row in identity.values() if row["final_row_index"]}
        assert final_indices == {0, 1, 2, 3, 4}
        assert int(identity[3]["gaussian_id"]) != int(identity[3]["final_row_index"])

        events = _read_csv(exact_dir / "gaussian_lifecycle_events.csv")
        event_types = {row["event_type"] for row in events}
        assert {"initial_seed", "clone_birth", "split_birth", "prune_death", "visibility_observation", "update_observation"} <= event_types
        train_014_observations = [
            row for row in events
            if row["view_name"] == "train_014" and row["event_type"] in {"visibility_observation", "update_observation"}
        ]
        assert train_014_observations
        assert {row["view_index"] for row in train_014_observations} == {"14"}
        required_event_columns = {
            "scene", "condition", "subset_name", "run_id", "iteration", "view_name",
            "gaussian_id", "parent_gaussian_id", "root_gaussian_id", "row_index_before",
            "row_index_after", "event_type", "evidence_quality",
        }
        assert required_event_columns <= set(events[0])

        result = _run(
            [
                sys.executable,
                str(project_root / "scripts" / "measure" / "validate_pr191_exact_gaussian_logging.py"),
                "--exact-log-dir",
                str(exact_dir),
                "--output-dir",
                str(validate_dir),
                "--write-markdown",
            ]
        )
        if result.returncode != 0:
            print(result.stdout)
            print(result.stderr, file=sys.stderr)
            return result.returncode
        for name in REQUIRED_VALIDATION_OUTPUTS:
            path = validate_dir / name
            assert path.exists(), name
            assert path.stat().st_size > 0, name
        validation = json.loads((validate_dir / "pr191_exact_gaussian_logging_validation_summary.json").read_text())
        assert validation["schema_name"] == "viewtrust.pr191.exact_gaussian_lifecycle_logging.validation_summary"
        assert validation["observation_only"] is True
        assert validation["training_intervention"] is False
        assert validation["defense_enabled"] is False
        assert validation["uses_row_index_as_stable_id"] is False
        assert validation["identity_consistency_passed"] is True
    print("pr191 exact gaussian logging smoke test ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
