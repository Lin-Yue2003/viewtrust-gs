#!/usr/bin/env python3
"""LOCAL-SAFE smoke test for PR13 offline ViewTrust signal generation."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _make_view_influence_dir(root: Path, *, condition: str, corrupt: bool) -> None:
    _write_json(
        root / "view_influence_summary.json",
        {
            "run_id": condition,
            "scene": "chair",
            "condition": condition,
            "observation_only": True,
            "view_count": 3,
        },
    )
    _write_csv(
        root / "view_influence.csv",
        [
            {
                "scene": "chair",
                "condition": condition,
                "run_id": condition,
                "view_name": "train_000",
                "view_split": "train",
                "was_corrupted": "true" if corrupt else "false",
                "corruption_type": "occluder" if corrupt else "",
                "times_sampled": 5,
                "mean_loss": 0.5 if corrupt else 0.2,
                "mean_total_loss": 0.5 if corrupt else 0.2,
                "mean_l1_loss": 0.2,
                "mean_ssim_loss": 0.7,
                "mean_visibility_ratio": 0.55 if corrupt else 0.9,
                "birth_event_count_after_view": 20 if corrupt else 1,
                "clone_birth_count_after_view": 10 if corrupt else 1,
                "split_birth_count_after_view": 10 if corrupt else 0,
                "prune_death_count_after_view": 100 if corrupt else 1,
                "final_survivor_birth_count_after_view": 5 if corrupt else 1,
                "dead_birth_count_after_view": 15 if corrupt else 0,
                "birth_survival_ratio_after_view": 0.25 if corrupt else 1.0,
            },
            {
                "scene": "chair",
                "condition": condition,
                "run_id": condition,
                "view_name": "train_001",
                "view_split": "train",
                "was_corrupted": "false",
                "corruption_type": "",
                "times_sampled": 5,
                "mean_loss": 0.2,
                "mean_total_loss": 0.2,
                "mean_l1_loss": 0.1,
                "mean_ssim_loss": 0.85,
                "mean_visibility_ratio": 0.85,
                "birth_event_count_after_view": 2,
                "clone_birth_count_after_view": 1,
                "split_birth_count_after_view": 1,
                "prune_death_count_after_view": 2,
                "final_survivor_birth_count_after_view": 2,
                "dead_birth_count_after_view": 0,
                "birth_survival_ratio_after_view": 1.0,
            },
            {
                "scene": "chair",
                "condition": condition,
                "run_id": condition,
                "view_name": "train_002",
                "view_split": "train",
                "was_corrupted": "false",
                "corruption_type": "",
                "times_sampled": 5,
                "mean_loss": "" if corrupt else 0.2,
                "mean_total_loss": "" if corrupt else 0.2,
                "mean_l1_loss": "",
                "mean_ssim_loss": "",
                "mean_visibility_ratio": 0.86,
                "birth_event_count_after_view": 1,
                "clone_birth_count_after_view": 1,
                "split_birth_count_after_view": 0,
                "prune_death_count_after_view": 1,
                "final_survivor_birth_count_after_view": 1,
                "dead_birth_count_after_view": 0,
                "birth_survival_ratio_after_view": 1.0,
            },
        ],
    )


def main() -> int:
    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root))

    from viewtrust.analysis.offline_signals import robust_z_scores, safe_divide

    if robust_z_scores([3, 3, 3]) != [0.0, 0.0, 0.0]:
        raise ValueError("identical robust z scores should be zero")
    outlier_scores = robust_z_scores([1, 1, 10])
    if outlier_scores[-1] <= 0:
        raise ValueError("high outlier should have positive robust z")
    if safe_divide(1, 0, default=-1.0) != -1.0:
        raise ValueError("safe_divide did not handle zero denominator")

    with tempfile.TemporaryDirectory(prefix="viewtrust-offline-signals-") as tmp:
        root = Path(tmp)
        clean_dir = root / "clean"
        corrupt_dir = root / "corrupt"
        comparison_dir = root / "comparison"
        output_dir = root / "output"
        config_path = root / "prune_only_config.json"
        _make_view_influence_dir(clean_dir, condition="clean", corrupt=False)
        _make_view_influence_dir(corrupt_dir, condition="corrupt_occluder", corrupt=True)
        _write_json(
            comparison_dir / "view_influence_comparison_summary.json",
            {
                "clean_run_id": "clean",
                "corrupt_run_id": "corrupt_occluder",
                "scene": "chair",
                "clean_condition": "clean",
                "corrupt_condition": "corrupt_occluder",
                "joined_view_count": 3,
                "corrupted_view_count": 1,
            },
        )
        _write_csv(
            comparison_dir / "view_influence_comparison.csv",
            [
                {
                    "view_name": "train_000",
                    "was_corrupted": "true",
                    "clean_times_sampled": 5,
                    "corrupt_times_sampled": 5,
                    "birth_event_count_delta": 19,
                    "prune_death_count_delta": 99,
                    "birth_survival_ratio_delta": -0.75,
                    "clean_mean_visibility_ratio": 0.9,
                    "corrupt_mean_visibility_ratio": 0.55,
                    "visibility_ratio_delta": -0.35,
                },
                {
                    "view_name": "train_001",
                    "was_corrupted": "false",
                    "clean_times_sampled": 5,
                    "corrupt_times_sampled": 5,
                    "birth_event_count_delta": 0,
                    "prune_death_count_delta": 0,
                    "birth_survival_ratio_delta": 0,
                    "clean_mean_visibility_ratio": 0.85,
                    "corrupt_mean_visibility_ratio": 0.85,
                    "visibility_ratio_delta": 0,
                },
                {
                    "view_name": "train_002",
                    "was_corrupted": "false",
                    "clean_times_sampled": 5,
                    "corrupt_times_sampled": 5,
                    "birth_event_count_delta": 0,
                    "prune_death_count_delta": 0,
                    "birth_survival_ratio_delta": 0,
                    "clean_mean_visibility_ratio": 0.86,
                    "corrupt_mean_visibility_ratio": 0.86,
                    "visibility_ratio_delta": 0,
                },
            ],
        )
        _write_json(
            config_path,
            {
                "schema_name": "viewtrust.offline_signal.config",
                "schema_version": 1,
                "normalization": {
                    "method": "robust_z",
                    "mad_scale": 1.4826,
                    "eps": 1e-8,
                    "positive_only": True,
                },
                "weights": {
                    "loss_component": 0.0,
                    "visibility_component": 0.0,
                    "birth_component": 0.0,
                    "prune_component": 1.0,
                    "survival_component": 0.0,
                    "delta_component": 0.0,
                },
                "ranking": {"higher_risk_is_more_suspicious": True},
                "labels": {
                    "use_corruption_labels_for_scoring": False,
                    "use_corruption_labels_for_evaluation": True,
                },
            },
        )
        completed = subprocess.run(
            [
                sys.executable,
                str(project_root / "scripts" / "measure" / "build_offline_viewtrust_signals.py"),
                "--clean-view-influence-dir",
                str(clean_dir),
                "--corrupt-view-influence-dir",
                str(corrupt_dir),
                "--view-influence-comparison-dir",
                str(comparison_dir),
                "--output-dir",
                str(output_dir),
                "--signal-config",
                str(config_path),
                "--write-markdown",
                "--top-k",
                "2",
                "--quiet",
            ],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if completed.returncode != 0:
            raise RuntimeError(completed.stderr or completed.stdout)
        for name in (
            "offline_viewtrust_summary.json",
            "offline_viewtrust_signals.csv",
            "offline_viewtrust_rankings.csv",
            "offline_viewtrust_group_metrics.csv",
            "offline_viewtrust_signal_ablation.csv",
            "offline_viewtrust_config.json",
            "offline_viewtrust_report.md",
            "offline_viewtrust_artifact_manifest.csv",
        ):
            path = output_dir / name
            if not path.exists() or path.stat().st_size == 0:
                raise FileNotFoundError(path)
        summary = json.loads((output_dir / "offline_viewtrust_summary.json").read_text(encoding="utf-8"))
        if summary["uses_corruption_labels_for_scoring"] is not False:
            raise ValueError("corruption labels must not be used for scoring")
        if summary["uses_corruption_labels_for_evaluation"] is not True:
            raise ValueError("corruption labels should be used for post-hoc evaluation")
        if summary["precision_at_k"] != 0.5:
            raise ValueError("precision@2 mismatch")
        if summary["recall_at_k"] != 1.0:
            raise ValueError("recall@2 mismatch")
        if "missing_loss_feature" not in summary["warnings"]:
            raise ValueError("missing optional loss feature should produce a warning")

        rankings = list(csv.DictReader((output_dir / "offline_viewtrust_rankings.csv").open(newline="", encoding="utf-8")))
        if rankings[0]["view_name"] != "train_000":
            raise ValueError("high prune-delta corrupted view should rank first")
        signals = list(csv.DictReader((output_dir / "offline_viewtrust_signals.csv").open(newline="", encoding="utf-8")))
        top_signal = signals[0]
        if abs(float(top_signal["offline_viewtrust_risk"]) - float(top_signal["prune_component"])) > 1e-9:
            raise ValueError("custom prune-only config weight was not honored")

    print("offline ViewTrust signals smoke test ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
