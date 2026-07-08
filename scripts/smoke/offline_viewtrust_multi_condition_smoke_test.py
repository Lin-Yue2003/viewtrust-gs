#!/usr/bin/env python3
"""LOCAL-SAFE smoke test for PR14 multi-condition offline aggregation."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path


REQUIRED_PR13_FILES = [
    "offline_viewtrust_summary.json",
    "offline_viewtrust_signals.csv",
    "offline_viewtrust_rankings.csv",
    "offline_viewtrust_group_metrics.csv",
    "offline_viewtrust_signal_ablation.csv",
    "offline_viewtrust_config.json",
    "offline_viewtrust_report.md",
    "offline_viewtrust_artifact_manifest.csv",
]

ABLATION_NAMES = [
    "loss_only",
    "visibility_only",
    "birth_only",
    "prune_only",
    "survival_only",
    "delta_only",
    "lifecycle_only",
    "full_signal",
]


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _make_pr13_dir(
    root: Path,
    *,
    condition: str,
    suffix: str,
    top1_corrupted: bool,
    risk_gap: float,
) -> Path:
    path = root / f"offline_viewtrust_{condition}_{suffix}"
    corrupted_rank = 1 if top1_corrupted else 2
    precision = 1.0 if top1_corrupted else 0.5
    _write_json(
        path / "offline_viewtrust_summary.json",
        {
            "schema_name": "viewtrust.offline_signal.summary",
            "schema_version": 1,
            "scene": "chair",
            "clean_condition": "clean",
            "corrupt_condition": condition,
            "clean_run_id": "clean",
            "corrupt_run_id": condition,
            "view_count": 3,
            "corrupted_view_count": 1,
            "uncorrupted_view_count": 2,
            "observation_only": True,
            "uses_corruption_labels_for_scoring": False,
            "uses_corruption_labels_for_evaluation": True,
            "training_intervention": False,
            "defense_enabled": False,
            "top_k": 2,
            "corrupted_in_top_k": 1,
            "precision_at_k": precision,
            "recall_at_k": 1.0,
            "mean_corrupted_risk": 3.0,
            "mean_uncorrupted_risk": 3.0 - risk_gap,
            "risk_gap_corrupted_minus_uncorrupted": risk_gap,
            "best_ablation_signal": "full_signal",
            "warnings": [],
        },
    )
    ranking_rows = [
        {
            "rank": 1,
            "view_name": "train_000" if top1_corrupted else "train_001",
            "was_corrupted": "true" if top1_corrupted else "false",
            "corruption_type": "mock",
            "offline_viewtrust_risk": 3.0,
            "offline_viewtrust_consistency": 0.25,
            "loss_component": 0.0,
            "visibility_component": 0.0,
            "birth_component": 0.0,
            "prune_component": 3.0,
            "survival_component": 0.0,
            "delta_component": 0.0,
            "lifecycle_component": 1.0,
            "main_reason": "prune_component",
        },
        {
            "rank": 2,
            "view_name": "train_001" if top1_corrupted else "train_000",
            "was_corrupted": "false" if top1_corrupted else "true",
            "corruption_type": "" if top1_corrupted else "mock",
            "offline_viewtrust_risk": 2.0,
            "offline_viewtrust_consistency": 0.333,
            "loss_component": 0.0,
            "visibility_component": 0.0,
            "birth_component": 0.0,
            "prune_component": 2.0,
            "survival_component": 0.0,
            "delta_component": 0.0,
            "lifecycle_component": 0.67,
            "main_reason": "prune_component",
        },
        {
            "rank": 3,
            "view_name": "train_002",
            "was_corrupted": "false",
            "corruption_type": "",
            "offline_viewtrust_risk": 1.0,
            "offline_viewtrust_consistency": 0.5,
            "loss_component": 0.0,
            "visibility_component": 0.0,
            "birth_component": 0.0,
            "prune_component": 1.0,
            "survival_component": 0.0,
            "delta_component": 0.0,
            "lifecycle_component": 0.33,
            "main_reason": "prune_component",
        },
    ]
    _write_csv(path / "offline_viewtrust_rankings.csv", ranking_rows)
    _write_csv(path / "offline_viewtrust_signals.csv", ranking_rows)
    _write_csv(
        path / "offline_viewtrust_group_metrics.csv",
        [
            {"group": "all", "view_count": 3, "mean_offline_viewtrust_risk": 2.0},
            {"group": "corrupted", "view_count": 1, "mean_offline_viewtrust_risk": 3.0},
            {"group": "uncorrupted", "view_count": 2, "mean_offline_viewtrust_risk": 3.0 - risk_gap},
        ],
    )
    _write_csv(
        path / "offline_viewtrust_signal_ablation.csv",
        [
            {
                "signal_name": name,
                "top1_view_name": ranking_rows[0]["view_name"],
                "top1_was_corrupted": ranking_rows[0]["was_corrupted"],
                "topk": 2,
                "corrupted_in_topk": 1,
                "precision_at_k": precision,
                "recall_at_k": 1.0,
                "mean_corrupted_rank": corrupted_rank,
                "median_corrupted_rank": corrupted_rank,
                "mean_corrupted_score": 3.0,
                "mean_uncorrupted_score": 3.0 - risk_gap,
                "score_gap": risk_gap,
            }
            for name in ABLATION_NAMES
        ],
    )
    _write_json(path / "offline_viewtrust_config.json", {"schema_name": "mock"})
    (path / "offline_viewtrust_report.md").write_text(
        "offline\nnot a trust score\nnot a defense\n",
        encoding="utf-8",
    )
    _write_csv(
        path / "offline_viewtrust_artifact_manifest.csv",
        [
            {
                "relative_path": name,
                "path": str(path / name),
                "exists": "true",
                "file_type": Path(name).suffix.lstrip("."),
                "size_bytes": 1,
                "required": "true",
                "artifact_group": "output_pr13",
            }
            for name in REQUIRED_PR13_FILES
        ],
    )
    return path


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def main() -> int:
    project_root = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory(prefix="viewtrust-multi-condition-") as tmp:
        root = Path(tmp)
        input_root = root / "reports"
        output_dir = root / "out"
        strict_output_dir = root / "strict-out"
        input_root.mkdir()
        _make_pr13_dir(
            input_root,
            condition="corrupt_occluder",
            suffix="pr13_20200101T000000",
            top1_corrupted=False,
            risk_gap=0.1,
        )
        newest = _make_pr13_dir(
            input_root,
            condition="corrupt_occluder",
            suffix="pr131_20200102T000000",
            top1_corrupted=True,
            risk_gap=1.0,
        )
        _make_pr13_dir(
            input_root,
            condition="corrupt_blur",
            suffix="pr13_20200101T000000",
            top1_corrupted=False,
            risk_gap=-0.5,
        )

        base_command = [
            sys.executable,
            str(project_root / "scripts" / "measure" / "aggregate_offline_viewtrust_results.py"),
            "--input-root",
            str(input_root),
            "--scene",
            "chair",
            "--clean-condition",
            "clean",
            "--conditions",
            "corrupt_occluder",
            "corrupt_blur",
            "corrupt_noise",
            "--top-k",
            "2",
            "--write-markdown",
            "--quiet",
        ]
        partial = _run([*base_command, "--output-dir", str(output_dir)])
        if partial.returncode != 0:
            raise RuntimeError(partial.stderr or partial.stdout)
        required = [
            "offline_viewtrust_multi_condition_summary.json",
            "offline_viewtrust_multi_condition_results.csv",
            "offline_viewtrust_multi_condition_ablation.csv",
            "offline_viewtrust_condition_ranking.csv",
            "offline_viewtrust_failure_cases.csv",
            "offline_viewtrust_multi_condition_report.md",
            "offline_viewtrust_multi_condition_artifact_manifest.csv",
        ]
        for name in required:
            path = output_dir / name
            if not path.exists() or path.stat().st_size == 0:
                raise FileNotFoundError(path)
        summary = json.loads((output_dir / "offline_viewtrust_multi_condition_summary.json").read_text(encoding="utf-8"))
        if summary["condition_count_valid"] != 2:
            raise ValueError("valid condition count mismatch")
        if summary["conditions_missing"] != ["corrupt_noise"]:
            raise ValueError("missing condition was not reported")
        if summary["uses_corruption_labels_for_scoring"] is not False:
            raise ValueError("corruption labels must not be used for scoring")
        if summary["uses_corruption_labels_for_evaluation"] is not True:
            raise ValueError("corruption labels should be used for evaluation")

        results = list(csv.DictReader((output_dir / "offline_viewtrust_multi_condition_results.csv").open(newline="", encoding="utf-8")))
        occluder = next(row for row in results if row["condition"] == "corrupt_occluder")
        if occluder["offline_signal_dir"] != str(newest):
            raise ValueError("newest valid condition dir was not selected")
        failures = list(csv.DictReader((output_dir / "offline_viewtrust_failure_cases.csv").open(newline="", encoding="utf-8")))
        failure_types = {row["failure_type"] for row in failures}
        if "missing_condition_output" not in failure_types:
            raise ValueError("missing condition failure was not recorded")
        if "top_ranked_uncorrupted_view" not in failure_types:
            raise ValueError("top-ranked uncorrupted failure was not recorded")
        if "low_risk_gap" not in failure_types:
            raise ValueError("low risk gap failure was not recorded")

        manifest_rows = list(
            csv.DictReader(
                (output_dir / "offline_viewtrust_multi_condition_artifact_manifest.csv").open(
                    newline="",
                    encoding="utf-8",
                )
            )
        )
        relative_paths = {row.get("relative_path") for row in manifest_rows}
        if "relative_path" not in manifest_rows[0]:
            raise ValueError("manifest missing relative_path column")
        if not set(required).issubset(relative_paths):
            raise ValueError("manifest missing required output artifacts")
        self_row = next(
            row
            for row in manifest_rows
            if row.get("relative_path") == "offline_viewtrust_multi_condition_artifact_manifest.csv"
        )
        if self_row.get("exists") != "true" or int(self_row.get("size_bytes") or 0) <= 0:
            raise ValueError("manifest self row was not refreshed")

        report = (output_dir / "offline_viewtrust_multi_condition_report.md").read_text(encoding="utf-8").lower()
        for phrase in ("offline", "not a trust score", "not a defense"):
            if phrase not in report:
                raise ValueError(f"report missing required phrase: {phrase}")
        for phrase in ("detected poison", "defense success", "rejected view", "attack blocked"):
            if phrase in report:
                raise ValueError(f"report contains overclaim phrase: {phrase}")

        strict = _run([*base_command, "--output-dir", str(strict_output_dir), "--require-all-conditions"])
        if strict.returncode == 0:
            raise ValueError("strict aggregation should fail when a condition is missing")

    print("offline ViewTrust multi-condition smoke test ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
