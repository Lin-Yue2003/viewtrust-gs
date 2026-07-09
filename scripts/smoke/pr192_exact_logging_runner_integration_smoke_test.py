#!/usr/bin/env python3
"""LOCAL-SAFE smoke test for PR19.2 exact logging runner integration."""

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


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({field for row in rows for field in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def _make_condition(data_root: Path) -> Path:
    root = data_root / "viewtrust-mini" / "nerf_synthetic" / "chair" / "corrupt_occluder"
    _write_json(root / "corruption_summary.json", {"scene": "chair", "output_condition": "corrupt_occluder"})
    _write_csv(
        root / "corruption_manifest.csv",
        [
            {"split": "train", "view_name": "train_004", "was_corrupted": "true", "corruption_type": "occluder"},
            {"split": "train", "view_name": "train_014", "was_corrupted": "false", "corruption_type": ""},
        ],
    )
    return root


def _make_run(run_dir: Path, condition_root: Path) -> None:
    _write_json(
        run_dir / "metadata.json",
        {
            "run_id": "pr192-smoke-run",
            "scene": "chair",
            "condition": "corrupt_occluder",
            "trainer": "gaussian-splatting",
            "prepared_scene_root": str(condition_root),
            "observation_only": True,
        },
    )
    _write_json(run_dir / "summary.json", {"returncode": 0, "observation_only": True})
    _write_json(run_dir / "training_events_summary.json", {"invalid_training_event_rows": 0, "observation_only": True})
    _write_json(
        run_dir / "gaussian_lifecycle_summary.json",
        {
            "invariant_violations": 0,
            "observation_only": True,
            "initial_gaussian_count": 3,
            "final_gaussian_count": 4,
        },
    )
    _write_csv(
        run_dir / "tables" / "training_events.csv",
        [
            {
                "iteration": 100,
                "event_type": "iteration_metrics",
                "view_name": "train_004",
                "view_split": "train",
                "loss": 0.3,
                "l1_loss": 0.2,
                "ssim": 0.8,
                "gaussian_count": 4,
                "visible_gaussian_count": 3,
                "visibility_ratio": 0.75,
                "radii_nonzero_count": 3,
                "densification_triggered": "true",
            },
            {
                "iteration": 110,
                "event_type": "iteration_metrics",
                "view_name": "train_014",
                "view_split": "train",
                "loss": 0.4,
                "l1_loss": 0.25,
                "ssim": 0.75,
                "gaussian_count": 4,
                "visible_gaussian_count": 2,
                "visibility_ratio": 0.5,
                "radii_nonzero_count": 2,
                "densification_triggered": "false",
            },
        ],
    )
    _write_csv(
        run_dir / "tables" / "densification_events.csv",
        [{"iteration": 100, "gaussian_count_before": 3, "gaussian_count_after": 4, "gaussian_count_delta": 1}],
    )
    _write_csv(
        run_dir / "tables" / "gaussian_lifecycle_events.csv",
        [
            {
                "iteration": 100,
                "source_iteration": 100,
                "source_view_name": "train_004",
                "source_view_split": "train",
                "event_type": "clone_birth",
                "gaussian_id": 3,
                "parent_gaussian_id": 1,
                "target_index": 3,
            },
            {
                "iteration": 100,
                "source_iteration": 100,
                "source_view_name": "train_004",
                "source_view_split": "train",
                "event_type": "prune_death",
                "gaussian_id": 0,
                "parent_gaussian_id": "",
                "source_index": 0,
            },
        ],
    )
    _write_csv(
        run_dir / "tables" / "gaussian_lifecycle_final.csv",
        [
            {"gaussian_id": 0, "parent_gaussian_id": "", "birth_type": "init", "alive": "false", "final_index": ""},
            {"gaussian_id": 1, "parent_gaussian_id": "", "birth_type": "init", "alive": "true", "final_index": 0},
            {"gaussian_id": 2, "parent_gaussian_id": "", "birth_type": "init", "alive": "true", "final_index": 1},
            {"gaussian_id": 3, "parent_gaussian_id": 1, "birth_type": "clone", "alive": "true", "final_index": 2},
        ],
    )
    _write_json(run_dir / "view_metrics_summary.json", {"view_count_total": 2})
    _write_csv(
        run_dir / "tables" / "view_metrics.csv",
        [
            {"split": "train", "image_name": "train_004.png", "psnr": 20.0, "ssim": 0.8, "l1_mean": 0.1},
            {"split": "train", "image_name": "train_014.png", "psnr": 22.0, "ssim": 0.85, "l1_mean": 0.08},
        ],
    )


def main() -> int:
    project_root = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory(prefix="viewtrust-pr192-") as tmp:
        root = Path(tmp)
        data_root = root / "data"
        condition_root = _make_condition(data_root)
        run_dir = root / "run"
        output_dir = root / "view_influence"
        exact_dir = root / "exact"
        validate_dir = root / "validate"
        _make_run(run_dir, condition_root)
        result = _run(
            [
                sys.executable,
                str(project_root / "scripts" / "measure" / "build_view_influence_table.py"),
                "--run-dir",
                str(run_dir),
                "--data-root",
                str(data_root),
                "--scene",
                "chair",
                "--condition",
                "corrupt_occluder",
                "--subset-name",
                "seed_20260710",
                "--output-dir",
                str(output_dir),
                "--enable-exact-gaussian-logging",
                "--exact-gaussian-log-dir",
                str(exact_dir),
                "--exact-gaussian-run-id",
                "pr192-smoke-exact",
                "--require-view-identity",
                "--require-source-view",
            ]
        )
        if result.returncode != 0:
            print(result.stdout)
            print(result.stderr, file=sys.stderr)
            return result.returncode
        for name in REQUIRED_EXACT_OUTPUTS:
            path = exact_dir / name
            assert path.exists(), name
            assert path.stat().st_size > 0, name
        summary = json.loads((exact_dir / "exact_gaussian_logging_summary.json").read_text())
        assert summary["integration_source"] == "real_view_influence_runner"
        assert summary["exact_gaussian_logging_enabled"] is True
        assert summary["stable_gaussian_ids_enabled"] is True
        assert summary["uses_row_index_as_stable_id"] is False
        assert summary["training_intervention"] is False
        assert summary["defense_enabled"] is False
        assert summary["parent_mapping_source"] in {"exact_clone_split_masks", "partial"}
        events = _read_csv(exact_dir / "gaussian_lifecycle_events.csv")
        train_014 = [row for row in events if row["view_name"] == "train_014"]
        assert train_014
        assert {row["view_index"] for row in train_014} == {"14"}
        view_summary = json.loads((output_dir / "view_influence_summary.json").read_text())
        assert view_summary["exact_gaussian_logging"]["enabled"] is True
        assert view_summary["exact_gaussian_logging"]["integration_source"] == "real_view_influence_runner"

        validation = _run(
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
        if validation.returncode != 0:
            print(validation.stdout)
            print(validation.stderr, file=sys.stderr)
            return validation.returncode
        validation_summary = json.loads((validate_dir / "pr191_exact_gaussian_logging_validation_summary.json").read_text())
        assert validation_summary["identity_consistency_passed"] is True
        assert validation_summary["uses_row_index_as_stable_id"] is False
    print("pr192 exact logging runner integration smoke test ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
