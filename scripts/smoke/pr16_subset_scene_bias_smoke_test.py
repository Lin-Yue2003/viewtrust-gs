#!/usr/bin/env python3
"""LOCAL-SAFE smoke test for PR16 subset and scene bias probe."""

from __future__ import annotations

import csv
import json
import os
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

PLAN_FILES = [
    "pr16_condition_matrix.csv",
    "pr16_subset_manifest.csv",
    "pr16_seed_reproducibility_summary.json",
    "pr16_run_commands.sh",
    "pr16_plan_report.md",
    "artifact_manifest.csv",
]

ANALYSIS_FILES = [
    "pr16_bias_probe_summary.json",
    "pr16_scene_subset_condition_results.csv",
    "pr16_subset_bias_summary.csv",
    "pr16_scene_bias_summary.csv",
    "pr16_view_identity_bias_table.csv",
    "pr16_repeated_false_positive_table.csv",
    "pr16_component_comparison.csv",
    "pr16_missing_outputs.csv",
    "pr16_bias_probe_report.md",
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


def _make_scene(data_root: Path, scene: str) -> None:
    frames = [
        {"file_path": f"images/{name}.png", "transform_matrix": [[1, 0, 0, 0]]}
        for name in ["train_a", "train_b", "train_fp", "train_tail"]
    ]
    _write_json(
        data_root / "viewtrust-mini" / "nerf_synthetic" / scene / "clean" / "transforms_train.json",
        {"camera_angle_x": 0.7, "frames": frames},
    )


def _ranking_rows(corrupted_view: str) -> list[dict[str, object]]:
    views = ["train_fp", corrupted_view, "train_a" if corrupted_view != "train_a" else "train_b", "train_tail"]
    rows = []
    for index, view_name in enumerate(views, start=1):
        was_corrupted = view_name == corrupted_view
        rows.append(
            {
                "rank": index,
                "view_name": view_name,
                "was_corrupted": str(was_corrupted).lower(),
                "offline_viewtrust_risk": 6 - index,
                "offline_viewtrust_consistency": 0.1 * index,
                "loss_component": 6 - index,
                "visibility_component": 0.2 if was_corrupted else 0.0,
                "birth_component": 0.1 if was_corrupted else 0.0,
                "prune_component": 0.3 if was_corrupted else 0.0,
                "survival_component": 0.1 if was_corrupted else 0.0,
                "delta_component": 0.2 if was_corrupted else 0.0,
                "lifecycle_component": 1.0 if was_corrupted else 0.1,
                "main_reason": "loss_component" if view_name == "train_fp" else "lifecycle_component",
            }
        )
    return rows


def _make_offline_output(
    reports_root: Path,
    *,
    scene: str,
    subset_name: str,
    condition: str,
    corrupted_view: str,
) -> None:
    path = reports_root / f"offline_viewtrust_{scene}_{condition}_{subset_name}_pr16_input"
    rankings = _ranking_rows(corrupted_view)
    _write_json(
        path / "offline_viewtrust_summary.json",
        {
            "schema_name": "viewtrust.offline_signal.summary",
            "schema_version": 1,
            "scene": scene,
            "clean_condition": "clean",
            "corrupt_condition": condition,
            "view_count": 4,
            "corrupted_view_count": 1,
            "uncorrupted_view_count": 3,
            "top_k": 2,
            "corrupted_in_top_k": 1,
            "precision_at_k": 0.5,
            "recall_at_k": 1.0,
            "mean_corrupted_risk": 4.0,
            "mean_uncorrupted_risk": 2.5,
            "risk_gap_corrupted_minus_uncorrupted": 1.5,
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
            {"group": "uncorrupted", "view_count": 3, "mean_offline_viewtrust_risk": 2.5},
        ],
    )
    _write_csv(
        path / "offline_viewtrust_signal_ablation.csv",
        [
            {
                "signal_name": "loss_only",
                "corrupted_in_topk": 1,
                "precision_at_k": 0.5,
                "recall_at_k": 1.0,
                "score_gap": 0.8,
                "top1_view_name": "train_fp",
                "top1_was_corrupted": "false",
            },
            {
                "signal_name": "lifecycle_only",
                "corrupted_in_topk": 1,
                "precision_at_k": 0.5,
                "recall_at_k": 1.0,
                "score_gap": 1.0,
                "top1_view_name": corrupted_view,
                "top1_was_corrupted": "true",
            },
            {
                "signal_name": "full_signal",
                "corrupted_in_topk": 1,
                "precision_at_k": 0.5,
                "recall_at_k": 1.0,
                "score_gap": 1.5,
                "top1_view_name": "train_fp",
                "top1_was_corrupted": "false",
            },
        ],
    )
    _write_json(path / "offline_viewtrust_config.json", {"schema_name": "mock"})
    (path / "offline_viewtrust_report.md").write_text("offline\nnot a defense\n", encoding="utf-8")
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


def _run(command: list[str], env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    return subprocess.run(
        command,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=merged_env,
    )


def _assert_self_manifest(path: Path) -> None:
    rows = _read_csv(path)
    self_rows = [row for row in rows if row["relative_path"] == "artifact_manifest.csv"]
    assert self_rows
    assert self_rows[0]["exists"] == "true"
    assert int(self_rows[0]["size_bytes"]) > 0


def main() -> int:
    project_root = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory(prefix="viewtrust-pr16-") as tmp:
        root = Path(tmp)
        data_root = root / "data"
        reports_root = root / "reports"
        command_reports_root = root / "command_reports"
        plan_dir = root / "plan"
        analysis_dir = root / "analysis"
        for scene in ["chair", "drum"]:
            _make_scene(data_root, scene)
            for subset_name, corrupt_view in [("original", "train_a"), ("seed_20260708", "train_b")]:
                for condition in ["corrupt_occluder", "corrupt_noise"]:
                    _make_offline_output(
                        reports_root,
                        scene=scene,
                        subset_name=subset_name,
                        condition=condition,
                        corrupted_view=corrupt_view,
                    )

        planner = _run(
            [
                sys.executable,
                str(project_root / "scripts" / "experiments" / "plan_pr16_subset_scene_bias_probe.py"),
                "--data-root",
                str(data_root),
                "--input-root",
                str(reports_root),
                "--output-dir",
                str(plan_dir),
                "--scenes",
                "chair",
                "drum",
                "--conditions",
                "corrupt_occluder",
                "corrupt_noise",
                "--subset-names",
                "original",
                "seed_20260708",
                "seed_20260709",
                "--subset-seeds",
                "20260708",
                "20260709",
                "--corrupted-view-count",
                "1",
                "--top-k",
                "2",
                "--write-commands",
            ]
        )
        if planner.returncode != 0:
            print(planner.stdout)
            print(planner.stderr, file=sys.stderr)
            return planner.returncode
        for name in PLAN_FILES:
            path = plan_dir / name
            assert path.exists(), name
            assert path.stat().st_size > 0, name
        seed_summary = json.loads((plan_dir / "pr16_seed_reproducibility_summary.json").read_text())
        assert seed_summary["same_seed_reproducible"] is True
        manifest = _read_csv(plan_dir / "pr16_subset_manifest.csv")
        assert {"chair", "drum"}.issubset({row["scene"] for row in manifest})
        assert {"20260708", "20260709"}.issubset({row["subset_seed"] for row in manifest if row["subset_seed"]})
        _assert_self_manifest(plan_dir / "artifact_manifest.csv")
        run_commands = (plan_dir / "pr16_run_commands.sh").read_text(encoding="utf-8")
        assert '"$CLEAN_VIEW_INFLUENCE_DIR"' not in run_commands
        assert '"$CORRUPT_VIEW_INFLUENCE_DIR"' not in run_commands
        assert '"$VIEW_INFLUENCE_COMPARISON_DIR"' not in run_commands
        assert "Default mode is TODO-only" not in run_commands
        assert "full training, clean/corrupt view influence" not in run_commands
        command_result = _run(
            ["bash", str(plan_dir / "pr16_run_commands.sh")],
            env={
                "PR16_FAKE_MODE": "1",
                "VIEWTRUST_REPORT_ROOT": str(command_reports_root),
            },
        )
        if command_result.returncode != 0:
            print(command_result.stdout)
            print(command_result.stderr, file=sys.stderr)
            return command_result.returncode
        assert "TODO PR16" not in command_result.stdout
        for scene in ["chair", "drum"]:
            for subset_name in ["original", "seed_20260708", "seed_20260709"]:
                for condition in ["corrupt_occluder", "corrupt_noise"]:
                    expected = command_reports_root / f"offline_viewtrust_{scene}_{condition}_{subset_name}_pr16_input"
                    assert expected.is_dir(), expected
                    for required in [
                        "offline_viewtrust_summary.json",
                        "offline_viewtrust_rankings.csv",
                        "offline_viewtrust_signals.csv",
                        "offline_viewtrust_signal_ablation.csv",
                        "offline_viewtrust_group_metrics.csv",
                        "offline_viewtrust_config.json",
                        "offline_viewtrust_report.md",
                        "offline_viewtrust_artifact_manifest.csv",
                    ]:
                        assert (expected / required).is_file(), expected / required

        analyzer = _run(
            [
                sys.executable,
                str(project_root / "scripts" / "measure" / "analyze_pr16_subset_scene_bias.py"),
                "--input-root",
                str(reports_root),
                "--plan-dir",
                str(plan_dir),
                "--output-dir",
                str(analysis_dir),
                "--scenes",
                "chair",
                "drum",
                "--conditions",
                "corrupt_occluder",
                "corrupt_noise",
                "--subset-names",
                "original",
                "seed_20260708",
                "seed_20260709",
                "--top-k",
                "2",
                "--write-markdown",
            ]
        )
        if analyzer.returncode != 0:
            print(analyzer.stdout)
            print(analyzer.stderr, file=sys.stderr)
            return analyzer.returncode
        for name in ANALYSIS_FILES:
            path = analysis_dir / name
            assert path.exists(), name
            assert path.stat().st_size > 0, name
        summary = json.loads((analysis_dir / "pr16_bias_probe_summary.json").read_text())
        assert summary["schema_name"] == "viewtrust.pr16.subset_scene_bias.summary"
        assert summary["scenes"] == ["chair", "drum"]
        assert summary["observation_only"] is True
        assert summary["training_intervention"] is False
        assert summary["defense_enabled"] is False
        assert summary["uses_corruption_labels_for_scoring"] is False
        assert summary["uses_corruption_labels_for_evaluation"] is True

        identity = _read_csv(analysis_dir / "pr16_view_identity_bias_table.csv")
        assert any(
            row["view_name"] == "train_a"
            and int(row["corrupted_count"]) > 0
            and int(row["uncorrupted_count"]) > 0
            for row in identity
        )
        false_positive = _read_csv(analysis_dir / "pr16_repeated_false_positive_table.csv")
        assert any(row["view_name"] == "train_fp" for row in false_positive)
        components = _read_csv(analysis_dir / "pr16_component_comparison.csv")
        assert {"loss_only", "lifecycle_only", "full_signal"}.issubset(
            {row["signal_name"] for row in components}
        )
        report = (analysis_dir / "pr16_bias_probe_report.md").read_text(encoding="utf-8").lower()
        assert "not a defense" in report
        assert "not a poison classifier" in report
        assert "defense success" not in report
        assert "attack blocked" not in report
        _assert_self_manifest(analysis_dir / "artifact_manifest.csv")

    print("pr16 subset scene bias smoke test ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
