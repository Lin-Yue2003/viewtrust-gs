#!/usr/bin/env python3
"""LOCAL-SAFE smoke test for PR15 offline rank consistency diagnosis."""

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

REQUIRED_PR15_FILES = [
    "cross_condition_view_rank_table.csv",
    "cross_condition_view_rank_summary.json",
    "repeated_top_views.csv",
    "false_positive_topk_views.csv",
    "corrupted_view_rank_distribution.csv",
    "component_win_table.csv",
    "component_condition_summary.csv",
    "component_gap_table.csv",
    "rank_consistency_report.md",
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


def _make_pr14_dir(path: Path) -> None:
    _write_json(
        path / "offline_viewtrust_multi_condition_summary.json",
        {
            "schema_name": "viewtrust.offline_signal.multi_condition.summary",
            "schema_version": 1,
            "scene": "chair",
            "conditions_requested": ["corrupt_a", "corrupt_b"],
            "condition_count_requested": 2,
            "condition_count_valid": 2,
            "conditions_missing": [],
            "mean_precision_at_k": 0.5,
            "mean_recall_at_k": 1.0,
            "best_ablation_signal_counts": {"full_signal": 2},
            "observation_only": True,
            "training_intervention": False,
            "defense_enabled": False,
            "uses_corruption_labels_for_scoring": False,
            "uses_corruption_labels_for_evaluation": True,
        },
    )
    _write_csv(
        path / "offline_viewtrust_multi_condition_results.csv",
        [{"condition": "corrupt_a", "status": "ok"}, {"condition": "corrupt_b", "status": "ok"}],
    )
    _write_csv(
        path / "offline_viewtrust_multi_condition_ablation.csv",
        [{"condition": "corrupt_a", "signal_name": "full_signal"}],
    )
    _write_csv(
        path / "offline_viewtrust_condition_ranking.csv",
        [{"rank": 1, "condition": "corrupt_a"}, {"rank": 2, "condition": "corrupt_b"}],
    )
    _write_csv(path / "offline_viewtrust_failure_cases.csv", [], ["condition", "failure_type"])


def _make_condition_dir(root: Path, *, condition: str, corrupt_view: str) -> Path:
    path = root / f"offline_viewtrust_{condition}_pr14_input"
    rankings = [
        {
            "rank": 1,
            "view_name": "train_repeat",
            "was_corrupted": "false",
            "offline_viewtrust_risk": 5.0,
            "offline_viewtrust_consistency": 0.2,
            "loss_component": 5.0,
            "visibility_component": 0.1,
            "birth_component": 0.0,
            "prune_component": 0.0,
            "survival_component": 0.1,
            "delta_component": 0.0,
            "lifecycle_component": 0.2,
            "main_reason": "loss_component",
        },
        {
            "rank": 2,
            "view_name": corrupt_view,
            "was_corrupted": "true",
            "offline_viewtrust_risk": 4.0,
            "offline_viewtrust_consistency": 0.3,
            "loss_component": 3.5,
            "visibility_component": 0.2,
            "birth_component": 0.1,
            "prune_component": 0.4,
            "survival_component": 0.2,
            "delta_component": 0.1,
            "lifecycle_component": 1.0,
            "main_reason": "lifecycle_component",
        },
        {
            "rank": 3,
            "view_name": "train_other",
            "was_corrupted": "false",
            "offline_viewtrust_risk": 2.0,
            "offline_viewtrust_consistency": 0.5,
            "loss_component": 2.0,
            "visibility_component": 0.0,
            "birth_component": 0.0,
            "prune_component": 0.0,
            "survival_component": 0.0,
            "delta_component": 0.0,
            "lifecycle_component": 0.0,
            "main_reason": "loss_component",
        },
        {
            "rank": 4,
            "view_name": "train_tail",
            "was_corrupted": "false",
            "offline_viewtrust_risk": 1.0,
            "offline_viewtrust_consistency": 0.6,
            "loss_component": 1.0,
            "visibility_component": 0.0,
            "birth_component": 0.0,
            "prune_component": 0.0,
            "survival_component": 0.0,
            "delta_component": 0.0,
            "lifecycle_component": 0.0,
            "main_reason": "loss_component",
        },
    ]
    _write_json(
        path / "offline_viewtrust_summary.json",
        {
            "schema_name": "viewtrust.offline_signal.summary",
            "schema_version": 1,
            "scene": "chair",
            "clean_condition": "clean",
            "corrupt_condition": condition,
            "view_count": 4,
            "corrupted_view_count": 1,
            "uncorrupted_view_count": 3,
            "top_k": 2,
            "corrupted_in_top_k": 1,
            "precision_at_k": 0.5,
            "recall_at_k": 1.0,
            "risk_gap_corrupted_minus_uncorrupted": 1.0,
            "best_ablation_signal": "full_signal",
            "observation_only": True,
            "training_intervention": False,
            "defense_enabled": False,
            "uses_corruption_labels_for_scoring": False,
            "uses_corruption_labels_for_evaluation": True,
            "warnings": [],
        },
    )
    _write_csv(path / "offline_viewtrust_rankings.csv", rankings)
    _write_csv(path / "offline_viewtrust_signals.csv", rankings)
    _write_csv(
        path / "offline_viewtrust_group_metrics.csv",
        [
            {"group": "corrupted", "view_count": 1, "mean_offline_viewtrust_risk": 4.0},
            {"group": "uncorrupted", "view_count": 3, "mean_offline_viewtrust_risk": 2.67},
        ],
    )
    _write_csv(
        path / "offline_viewtrust_signal_ablation.csv",
        [
            {
                "signal_name": "loss_only",
                "top1_view_name": "train_repeat",
                "top1_was_corrupted": "false",
                "corrupted_in_topk": 1,
                "precision_at_k": 0.5,
                "recall_at_k": 1.0,
                "mean_corrupted_score": 3.5,
                "mean_uncorrupted_score": 2.8,
                "score_gap": 0.7,
            },
            {
                "signal_name": "lifecycle_only",
                "top1_view_name": corrupt_view,
                "top1_was_corrupted": "true",
                "corrupted_in_topk": 1,
                "precision_at_k": 0.5,
                "recall_at_k": 1.0,
                "mean_corrupted_score": 1.0,
                "mean_uncorrupted_score": 0.1,
                "score_gap": 0.9,
            },
            {
                "signal_name": "full_signal",
                "top1_view_name": "train_repeat",
                "top1_was_corrupted": "false",
                "corrupted_in_topk": 1,
                "precision_at_k": 0.5,
                "recall_at_k": 1.0,
                "mean_corrupted_score": 4.0,
                "mean_uncorrupted_score": 2.5,
                "score_gap": 1.5,
            },
        ],
    )
    _write_json(path / "offline_viewtrust_config.json", {"schema_name": "mock"})
    (path / "offline_viewtrust_report.md").write_text(
        "offline signal\nnot a defense\nnot a poison classifier\n",
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
    with tempfile.TemporaryDirectory(prefix="viewtrust-rank-consistency-") as tmp:
        root = Path(tmp)
        pr14_dir = root / "fake_pr14"
        reports_root = root / "fake_reports"
        output_dir = root / "fake_output"
        _make_pr14_dir(pr14_dir)
        _make_condition_dir(reports_root, condition="corrupt_a", corrupt_view="train_corrupt_a")
        _make_condition_dir(reports_root, condition="corrupt_b", corrupt_view="train_corrupt_b")

        result = _run(
            [
                sys.executable,
                str(project_root / "scripts" / "measure" / "analyze_offline_viewtrust_rank_consistency.py"),
                "--multi-condition-dir",
                str(pr14_dir),
                "--input-root",
                str(reports_root),
                "--scene",
                "chair",
                "--conditions",
                "corrupt_a",
                "corrupt_b",
                "--top-k",
                "2",
                "--output-dir",
                str(output_dir),
                "--write-markdown",
            ]
        )
        if result.returncode != 0:
            print(result.stdout)
            print(result.stderr, file=sys.stderr)
            return result.returncode

        for name in REQUIRED_PR15_FILES:
            path = output_dir / name
            assert path.exists(), name
            assert path.stat().st_size > 0, name

        repeated = _read_csv(output_dir / "repeated_top_views.csv")
        assert any(row["view_name"] == "train_repeat" for row in repeated)
        false_positive = _read_csv(output_dir / "false_positive_topk_views.csv")
        assert any(row["view_name"] == "train_repeat" for row in false_positive)
        corrupted = _read_csv(output_dir / "corrupted_view_rank_distribution.csv")
        assert {row["view_name"] for row in corrupted} == {"train_corrupt_a", "train_corrupt_b"}
        component_win = _read_csv(output_dir / "component_win_table.csv")
        assert {"loss_only", "lifecycle_only", "full_signal"}.issubset(
            {row["signal_name"] for row in component_win}
        )
        component_summary = _read_csv(output_dir / "component_condition_summary.csv")
        assert "full_minus_loss_recall" in component_summary[0]

        summary = json.loads((output_dir / "cross_condition_view_rank_summary.json").read_text(encoding="utf-8"))
        assert summary["schema_name"] == "viewtrust.offline_signal.rank_consistency.summary"
        assert summary["observation_only"] is True
        assert summary["training_intervention"] is False
        assert summary["defense_enabled"] is False
        assert summary["uses_corruption_labels_for_scoring"] is False
        assert summary["uses_corruption_labels_for_evaluation"] is True

        report = (output_dir / "rank_consistency_report.md").read_text(encoding="utf-8").lower()
        assert "not a defense" in report
        assert "not a poison classifier" in report
        assert "defense success" not in report
        assert "attack blocked" not in report

        manifest = _read_csv(output_dir / "artifact_manifest.csv")
        self_rows = [row for row in manifest if row["relative_path"] == "artifact_manifest.csv"]
        assert self_rows
        assert self_rows[0]["exists"] == "true"
        assert int(self_rows[0]["size_bytes"]) > 0

    print("offline viewtrust rank consistency smoke test ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
