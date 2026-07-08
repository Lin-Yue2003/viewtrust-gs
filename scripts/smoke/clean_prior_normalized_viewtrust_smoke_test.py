#!/usr/bin/env python3
"""LOCAL-SAFE smoke test for PR17 clean-prior normalization."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path


REQUIRED_OUTPUTS = [
    "clean_prior_normalized_summary.json",
    "clean_prior_normalized_rows.csv",
    "clean_prior_normalized_rankings.csv",
    "clean_prior_normalized_group_metrics.csv",
    "clean_prior_normalized_ablation.csv",
    "clean_prior_false_positive_reduction.csv",
    "clean_prior_view_identity_diagnosis.csv",
    "clean_prior_component_comparison.csv",
    "clean_prior_missing_outputs.csv",
    "clean_prior_report.md",
    "artifact_manifest.csv",
]


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, object]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = fields or list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _make_signal_dir(root: Path) -> None:
    signal_dir = root / "offline_viewtrust_chair_corrupt_occluder_seed_20260710_pr16_input"
    clean_dir = root / "view_influence_chair_clean_pr16"
    rows = [
        {
            "scene": "chair",
            "clean_condition": "clean",
            "corrupt_condition": "corrupt_occluder",
            "view_name": "train_013",
            "view_split": "train",
            "was_corrupted": "false",
            "offline_viewtrust_risk": 1.2,
            "offline_viewtrust_consistency": 0.45,
        },
        {
            "scene": "chair",
            "clean_condition": "clean",
            "corrupt_condition": "corrupt_occluder",
            "view_name": "train_009",
            "view_split": "train",
            "was_corrupted": "true",
            "offline_viewtrust_risk": 1.0,
            "offline_viewtrust_consistency": 0.50,
        },
        {
            "scene": "chair",
            "clean_condition": "clean",
            "corrupt_condition": "corrupt_occluder",
            "view_name": "train_004",
            "view_split": "train",
            "was_corrupted": "false",
            "offline_viewtrust_risk": 0.1,
            "offline_viewtrust_consistency": 0.91,
        },
    ]
    clean_rows = [
        {
            "view_name": "train_013",
            "mean_total_loss": 100.0,
            "mean_loss": 100.0,
            "mean_visibility_ratio": 10.0,
            "birth_event_count_after_view": 100.0,
            "prune_death_count_after_view": 100.0,
            "birth_survival_ratio_after_view": 0.0,
        },
        {
            "view_name": "train_009",
            "mean_total_loss": 1.0,
            "mean_loss": 1.0,
            "mean_visibility_ratio": 1.0,
            "birth_event_count_after_view": 1.0,
            "prune_death_count_after_view": 1.0,
            "birth_survival_ratio_after_view": 1.0,
        },
        {
            "view_name": "train_004",
            "mean_total_loss": 1.0,
            "mean_loss": 1.0,
            "mean_visibility_ratio": 1.0,
            "birth_event_count_after_view": 1.0,
            "prune_death_count_after_view": 1.0,
            "birth_survival_ratio_after_view": 1.0,
        },
    ]
    _write_csv(clean_dir / "view_influence.csv", clean_rows)
    _write_json(clean_dir / "view_influence_summary.json", {"scene": "chair", "condition": "clean"})
    _write_json(
        signal_dir / "offline_viewtrust_summary.json",
        {
            "schema_name": "viewtrust.offline_signal.summary",
            "schema_version": 1,
            "scene": "chair",
            "corrupt_condition": "corrupt_occluder",
            "view_count": 3,
            "corrupted_view_count": 1,
            "observation_only": True,
            "training_intervention": False,
            "defense_enabled": False,
            "uses_corruption_labels_for_scoring": False,
            "uses_corruption_labels_for_evaluation": True,
        },
    )
    _write_csv(signal_dir / "offline_viewtrust_signals.csv", rows)
    _write_csv(
        signal_dir / "offline_viewtrust_rankings.csv",
        [
            {"rank": 1, **rows[0]},
            {"rank": 2, **rows[1]},
            {"rank": 3, **rows[2]},
        ],
    )
    _write_csv(signal_dir / "offline_viewtrust_group_metrics.csv", [{"group": "all", "view_count": 3}])
    _write_csv(
        signal_dir / "offline_viewtrust_signal_ablation.csv",
        [{"signal_name": "full_signal", "precision_at_k": 0.0, "recall_at_k": 0.0}],
    )
    _write_json(signal_dir / "offline_viewtrust_config.json", {"schema_name": "mock"})
    (signal_dir / "offline_viewtrust_report.md").write_text("offline\nnot a defense\n", encoding="utf-8")
    manifest_rows = [
        {
            "relative_path": "input_clean/view_influence.csv",
            "path": str(clean_dir / "view_influence.csv"),
            "exists": "true",
            "file_type": "csv",
            "size_bytes": 1,
            "required": "true",
            "artifact_group": "input_clean",
        },
        {
            "relative_path": "offline_viewtrust_summary.json",
            "path": str(signal_dir / "offline_viewtrust_summary.json"),
            "exists": "true",
            "file_type": "json",
            "size_bytes": 1,
            "required": "true",
            "artifact_group": "output_pr13",
        },
    ]
    _write_csv(signal_dir / "offline_viewtrust_artifact_manifest.csv", manifest_rows)


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def main() -> int:
    project_root = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory(prefix="viewtrust-pr17-") as tmp:
        root = Path(tmp)
        input_root = root / "reports"
        plan_dir = root / "plan"
        output_dir = root / "out"
        _make_signal_dir(input_root)
        _write_csv(
            plan_dir / "pr16_subset_manifest.csv",
            [{"scene": "chair", "subset_name": "seed_20260710", "subset_seed": "20260710"}],
        )
        result = _run(
            [
                sys.executable,
                str(project_root / "scripts" / "measure" / "analyze_clean_prior_normalized_viewtrust.py"),
                "--input-root",
                str(input_root),
                "--plan-dir",
                str(plan_dir),
                "--output-dir",
                str(output_dir),
                "--scenes",
                "chair",
                "--conditions",
                "corrupt_occluder",
                "corrupt_noise",
                "--subset-names",
                "seed_20260710",
                "--top-k",
                "1",
                "--allow-missing",
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
        summary = json.loads((output_dir / "clean_prior_normalized_summary.json").read_text())
        assert summary["schema_name"] == "viewtrust.pr17.clean_prior_normalized.summary"
        assert summary["observation_only"] is True
        assert summary["training_intervention"] is False
        assert summary["defense_enabled"] is False
        assert summary["uses_corruption_labels_for_scoring"] is False
        assert summary["uses_corruption_labels_for_evaluation"] is True
        rows = _read_csv(output_dir / "clean_prior_normalized_rows.csv")
        train_013 = next(row for row in rows if row["view_name"] == "train_013")
        train_009 = next(row for row in rows if row["view_name"] == "train_009")
        assert train_013["raw_top_k"] == "True"
        assert train_013["normalized_top_k"] == "False"
        assert train_009["was_corrupted"] == "true"
        assert train_009["normalized_top_k"] == "True"
        missing = _read_csv(output_dir / "clean_prior_missing_outputs.csv")
        assert any(row["condition"] == "corrupt_noise" for row in missing)
        report = (output_dir / "clean_prior_report.md").read_text(encoding="utf-8").lower()
        assert "not a defense" in report
        assert "corruption labels are used only for evaluation" in report
    print("clean prior normalized viewtrust smoke test ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
