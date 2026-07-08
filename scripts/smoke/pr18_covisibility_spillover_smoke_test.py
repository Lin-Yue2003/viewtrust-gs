#!/usr/bin/env python3
"""LOCAL-SAFE smoke test for PR18 co-visibility spillover diagnosis."""

from __future__ import annotations

import csv
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path


REQUIRED_OUTPUTS = [
    "pr18_covisibility_spillover_summary.json",
    "pr18_candidate_false_positive_diagnosis.csv",
    "pr18_camera_neighbor_table.csv",
    "pr18_view_pair_distance_table.csv",
    "pr18_gaussian_support_overlap.csv",
    "pr18_spillover_classification.csv",
    "pr18_condition_summary.csv",
    "pr18_view_identity_transition.csv",
    "pr18_missing_outputs.csv",
    "pr18_report.md",
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


def _matrix(x: float, angle_deg: float = 0.0) -> list[list[float]]:
    angle = math.radians(angle_deg)
    c = math.cos(angle)
    s = math.sin(angle)
    return [
        [c, -s, 0.0, x],
        [s, c, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _make_transforms(data_root: Path) -> None:
    positions = {f"train_{index:03d}": 100.0 + index * 10.0 for index in range(20)}
    positions.update(
        {
            "train_004": 0.0,
            "train_007": 1.0,
            "train_009": 2.0,
            "train_012": 10.0,
            "train_014": 11.0,
            "train_017": 12.0,
            "train_013": 60.0,
            "train_001": 250.0,
        }
    )
    frames = [
        {"file_path": f"images/{view}.png", "transform_matrix": _matrix(x)}
        for view, x in sorted(positions.items())
    ]
    _write_json(
        data_root / "viewtrust-mini" / "nerf_synthetic" / "chair" / "clean" / "transforms_train.json",
        {"camera_angle_x": 0.5, "frames": frames},
    )


def _make_pr17_rows(pr17_dir: Path) -> None:
    corrupted = {"train_004", "train_009", "train_012", "train_017"}
    normalized_ranks = {
        "train_004": 1,
        "train_009": 2,
        "train_012": 3,
        "train_017": 4,
        "train_014": 5,
        "train_007": 6,
        "train_001": 7,
        "train_013": 12,
    }
    raw_ranks = {
        "train_013": 1,
        "train_004": 2,
        "train_009": 3,
        "train_012": 4,
        "train_017": 5,
        "train_014": 10,
        "train_007": 11,
        "train_001": 12,
    }
    rows: list[dict[str, object]] = []
    for index in range(20):
        view = f"train_{index:03d}"
        clean_prior = 0.1
        positive_delta = 0.2
        rank_lift = 0.0
        normalized_risk = 0.2
        raw_risk = 0.3
        if view in corrupted:
            positive_delta = 5.0
            rank_lift = 8.0
            normalized_risk = 10.0 - normalized_ranks[view]
            raw_risk = 6.0
        if view == "train_013":
            clean_prior = 10.0
            positive_delta = 0.1
            rank_lift = -8.0
            normalized_risk = 0.2
            raw_risk = 10.1
        if view == "train_014":
            positive_delta = 9.0
            rank_lift = 12.0
            normalized_risk = 5.5
            raw_risk = 9.1
        if view == "train_007":
            positive_delta = 8.0
            rank_lift = 11.0
            normalized_risk = 5.0
            raw_risk = 8.1
        if view == "train_001":
            positive_delta = 7.0
            rank_lift = 10.0
            normalized_risk = 4.8
            raw_risk = 7.1
        raw_rank = raw_ranks.get(view, 20 + index)
        normalized_rank = normalized_ranks.get(view, 20 + index)
        rows.append(
            {
                "scene": "chair",
                "subset_name": "seed_20260710",
                "subset_seed": "20260710",
                "condition": "corrupt_occluder",
                "view_name": view,
                "view_split": "train",
                "was_corrupted": str(view in corrupted).lower(),
                "raw_risk": raw_risk,
                "raw_rank": raw_rank,
                "clean_prior_risk": clean_prior,
                "clean_prior_rank": 1 if view == "train_013" else 15,
                "delta_risk": positive_delta if view != "train_013" else 0.1,
                "positive_delta_risk": positive_delta,
                "prior_suppressed_risk": raw_risk / (1.0 + clean_prior),
                "rank_lift_score": rank_lift,
                "normalized_viewtrust_risk": normalized_risk,
                "normalized_rank": normalized_rank,
                "normalized_consistency": 1.0 / (1.0 + normalized_risk),
                "raw_top_k": str(raw_rank <= 7),
                "normalized_top_k": str(normalized_rank <= 7),
                "raw_false_positive": str(raw_rank <= 7 and view not in corrupted),
                "normalized_false_positive": str(normalized_rank <= 7 and view not in corrupted),
                "prior_source": "mock",
                "component_warnings": "",
            }
        )
    fields = list(rows[0])
    _write_csv(pr17_dir / "clean_prior_normalized_rows.csv", rows, fields)
    _write_csv(
        pr17_dir / "clean_prior_normalized_rankings.csv",
        [
            {
                "scene": row["scene"],
                "subset_name": row["subset_name"],
                "subset_seed": row["subset_seed"],
                "condition": row["condition"],
                "score_name": "normalized_viewtrust_risk",
                "rank": row["normalized_rank"],
                "view_name": row["view_name"],
                "was_corrupted": row["was_corrupted"],
                "score": row["normalized_viewtrust_risk"],
            }
            for row in rows
        ],
    )
    _write_json(
        pr17_dir / "clean_prior_normalized_summary.json",
        {
            "schema_name": "viewtrust.pr17.clean_prior_normalized.summary",
            "observation_only": True,
            "training_intervention": False,
            "defense_enabled": False,
            "uses_corruption_labels_for_scoring": False,
        },
    )


def _make_pr16_plan(plan_dir: Path) -> None:
    _write_csv(
        plan_dir / "pr16_subset_manifest.csv",
        [
            {
                "scene": "chair",
                "subset_name": "seed_20260710",
                "subset_seed": "20260710",
                "train_view_count": 20,
                "corrupted_view_count": 4,
                "corrupted_view_names": "train_004;train_009;train_012;train_017",
                "corrupted_view_hash": "mock",
                "source": "mock",
                "status": "ok",
                "warnings": "",
            }
        ],
    )


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def main() -> int:
    project_root = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory(prefix="viewtrust-pr18-") as tmp:
        root = Path(tmp)
        data_root = root / "data"
        input_root = root / "reports"
        plan_dir = root / "plan"
        pr17_dir = root / "pr17"
        output_dir = root / "pr18"
        _make_transforms(data_root)
        _make_pr16_plan(plan_dir)
        _make_pr17_rows(pr17_dir)
        result = _run(
            [
                sys.executable,
                str(project_root / "scripts" / "measure" / "analyze_pr18_covisibility_spillover.py"),
                "--data-root",
                str(data_root),
                "--input-root",
                str(input_root),
                "--plan-dir",
                str(plan_dir),
                "--pr17-dir",
                str(pr17_dir),
                "--output-dir",
                str(output_dir),
                "--scenes",
                "chair",
                "--conditions",
                "corrupt_occluder",
                "--subset-names",
                "seed_20260710",
                "--top-k",
                "7",
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
        summary = json.loads((output_dir / "pr18_covisibility_spillover_summary.json").read_text())
        assert summary["schema_name"] == "viewtrust.pr18.covisibility_spillover.summary"
        assert summary["observation_only"] is True
        assert summary["training_intervention"] is False
        assert summary["defense_enabled"] is False
        assert summary["uses_corruption_labels_for_scoring"] is False
        assert summary["uses_corruption_labels_for_evaluation"] is True
        classes = {
            (row["view_name"], row["spillover_class"]): row
            for row in _read_csv(output_dir / "pr18_spillover_classification.csv")
        }
        assert ("train_014", "co_visible_collateral") in classes
        assert ("train_007", "co_visible_collateral") in classes
        assert ("train_001", "unexplained_false_positive") in classes
        assert ("train_013", "clean_prior_false_positive") in classes
        train_014 = classes[("train_014", "co_visible_collateral")]
        assert train_014["camera_neighbor_evidence"] == "True"
        train_001 = classes[("train_001", "unexplained_false_positive")]
        assert train_001["camera_neighbor_evidence"] == "False"
        manifest = _read_csv(output_dir / "artifact_manifest.csv")
        assert any(row["relative_path"] == "artifact_manifest.csv" and row["exists"] == "true" for row in manifest)
        report = (output_dir / "pr18_report.md").read_text(encoding="utf-8").lower()
        assert "not a defense" in report
    print("pr18 covisibility spillover smoke test ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
