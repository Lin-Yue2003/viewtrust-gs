#!/usr/bin/env python3
"""Plan PR16 subset/scene bias probe runs without executing heavy stages."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


def _bootstrap_project_imports() -> Path:
    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root))
    return project_root


def _resolve_path(project_root: Path, raw_path: str | Path) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else project_root / path


def _condition_to_corruption_type(condition: str) -> str:
    return condition.removeprefix("corrupt_")


def _write_run_commands(
    path: Path,
    *,
    matrix_rows: list[dict[str, Any]],
    subset_rows: list[dict[str, Any]],
    data_root: Path,
    top_k: int,
) -> None:
    subset_lookup = {
        (str(row.get("scene")), str(row.get("subset_name"))): row
        for row in subset_rows
    }
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "# PR16 command guide. This file is intentionally not executed by the planner.",
        "# Review paths and run IDs before using any command below.",
        "",
        f"export VIEWTRUST_DATA_ROOT=${{VIEWTRUST_DATA_ROOT:-{data_root}}}",
        "export VIEWTRUST_OUTPUT_ROOT=${VIEWTRUST_OUTPUT_ROOT:-./outputs}",
        "export VIEWTRUST_REPORT_ROOT=${VIEWTRUST_REPORT_ROOT:-./outputs/reports}",
        "",
    ]
    for row in matrix_rows:
        scene = str(row.get("scene"))
        subset_name = str(row.get("subset_name"))
        condition = str(row.get("condition"))
        subset = subset_lookup.get((scene, subset_name), {})
        corrupt_names = " ".join(str(subset.get("corrupted_view_names", "")).split(";"))
        corruption_type = _condition_to_corruption_type(condition)
        output_condition = f"{condition}_{subset_name}" if subset_name != "original" else condition
        offline_dir = f"${{VIEWTRUST_REPORT_ROOT}}/offline_viewtrust_{scene}_{condition}_{subset_name}_pr16_input"
        lines.extend(
            [
                f"# === {scene} / {subset_name} / {condition} ===",
                "# 1. Generate or verify natural corruption condition for this subset.",
                "python scripts/data/generate_natural_corruptions.py \\",
                "  --data-root \"$VIEWTRUST_DATA_ROOT\" \\",
                f"  --scene {scene} \\",
                "  --source-condition clean \\",
                f"  --output-condition {output_condition} \\",
                f"  --corruption-type {corruption_type} \\",
                f"  --corrupt-view-names {corrupt_names} \\",
                "  --copy-mode symlink \\",
                "  --overwrite",
                "",
                "# 2. TODO: run observed clean/corrupt training if the required runs do not exist.",
                "# python scripts/train/run_clean_chair_baseline.py ...",
                "",
                "# 3. TODO: build view influence and clean-vs-corrupt comparison artifacts.",
                "# python scripts/measure/build_view_influence_table.py ...",
                "# python scripts/measure/compare_view_influence_tables.py ...",
                "",
                "# 4. Build PR13-style offline signal output for this PR16 cell.",
                "# Fill CLEAN_VIEW_INFLUENCE_DIR, CORRUPT_VIEW_INFLUENCE_DIR, and VIEW_INFLUENCE_COMPARISON_DIR first.",
                "python scripts/measure/build_offline_viewtrust_signals.py \\",
                "  --clean-view-influence-dir \"$CLEAN_VIEW_INFLUENCE_DIR\" \\",
                "  --corrupt-view-influence-dir \"$CORRUPT_VIEW_INFLUENCE_DIR\" \\",
                "  --view-influence-comparison-dir \"$VIEW_INFLUENCE_COMPARISON_DIR\" \\",
                f"  --output-dir \"{offline_dir}\" \\",
                f"  --top-k {top_k} \\",
                "  --write-markdown",
                "",
            ]
        )
    lines.extend(
        [
            "# 5. After per-condition outputs exist, run PR14, PR15, then PR16 analysis.",
            "# python scripts/measure/aggregate_offline_viewtrust_results.py ...",
            "# python scripts/measure/analyze_offline_viewtrust_rank_consistency.py ...",
            "# python scripts/measure/analyze_pr16_subset_scene_bias.py ...",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
    path.chmod(0o755)


def _write_plan_report(
    path: Path,
    *,
    scenes: list[str],
    conditions: list[str],
    subset_names: list[str],
    summary: dict[str, Any],
) -> None:
    report = "\n".join(
        [
            "# PR16 Subset and Scene Bias Probe Plan",
            "",
            "This plan is offline preparation only. It does not run training, rendering, scoring, defense, or update gating.",
            "",
            "## Matrix",
            f"- Scenes: `{', '.join(scenes)}`",
            f"- Conditions: `{', '.join(conditions)}`",
            f"- Subsets: `{', '.join(subset_names)}`",
            "",
            "## Seed Reproducibility",
            f"- Same seed reproducible: `{summary.get('same_seed_reproducible')}`",
            f"- Different seed collision count: `{summary.get('different_seed_collision_count')}`",
            "",
            "## Notes",
            "- Random subset manifests are generated deterministically from discovered train views.",
            "- The original subset is inferred from existing outputs when possible.",
            "- Generated commands are a guide and must be reviewed before heavy server execution.",
            "",
            "## Warnings",
            *(f"- {warning}" for warning in summary.get("warnings", [])),
            "" if summary.get("warnings") else "- None",
            "",
        ]
    )
    path.write_text(report, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    _bootstrap_project_imports()
    from viewtrust.analysis.subset_scene_bias import (
        DEFAULT_CONDITIONS,
        DEFAULT_SCENES,
        DEFAULT_SUBSET_NAMES,
        DEFAULT_SUBSET_SEEDS,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=os.environ.get("VIEWTRUST_DATA_ROOT", "./data"))
    parser.add_argument("--input-root", default="outputs/reports")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--scenes", nargs="+", default=DEFAULT_SCENES)
    parser.add_argument("--conditions", nargs="+", default=DEFAULT_CONDITIONS)
    parser.add_argument("--subset-names", nargs="+", default=DEFAULT_SUBSET_NAMES)
    parser.add_argument("--subset-seeds", nargs="+", type=int, default=DEFAULT_SUBSET_SEEDS)
    parser.add_argument("--corrupted-view-count", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--write-commands", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> int:
    project_root = _bootstrap_project_imports()
    from viewtrust.analysis.subset_scene_bias import (
        PLAN_OUTPUT_FILES,
        build_condition_matrix,
        build_subset_manifest,
        write_artifact_manifest,
        write_csv_rows,
        write_json,
    )

    args = parse_args()
    data_root = _resolve_path(project_root, args.data_root)
    input_root = _resolve_path(project_root, args.input_root)
    output_dir = _resolve_path(project_root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    subset_rows, seed_summary = build_subset_manifest(
        data_root=data_root,
        input_root=input_root,
        scenes=args.scenes,
        subset_names=args.subset_names,
        subset_seeds=args.subset_seeds,
        conditions=args.conditions,
        corrupted_view_count=args.corrupted_view_count,
    )
    if seed_summary.get("same_seed_reproducible") is not True:
        raise SystemExit("ERROR: same seed did not reproduce the same subset")
    invalid_random = [
        row for row in subset_rows
        if row.get("subset_name") != "original" and row.get("status") != "ok"
    ]
    if invalid_random:
        raise SystemExit("ERROR: one or more seeded random subsets are invalid")

    matrix_rows = build_condition_matrix(
        scenes=args.scenes,
        subset_manifest_rows=subset_rows,
        conditions=args.conditions,
        top_k=args.top_k,
    )

    subset_fields = [
        "scene",
        "subset_name",
        "subset_seed",
        "train_view_count",
        "corrupted_view_count",
        "corrupted_view_names",
        "corrupted_view_hash",
        "source",
        "status",
        "warnings",
    ]
    matrix_fields = [
        "scene",
        "subset_name",
        "subset_seed",
        "condition",
        "corrupted_view_count",
        "top_k",
        "expected_output_key",
        "expected_offline_signal_dir",
        "expected_pr14_dir",
        "expected_pr15_dir",
        "status",
        "warnings",
    ]
    write_csv_rows(output_dir / "pr16_subset_manifest.csv", subset_rows, subset_fields)
    write_csv_rows(output_dir / "pr16_condition_matrix.csv", matrix_rows, matrix_fields)
    write_json(output_dir / "pr16_seed_reproducibility_summary.json", seed_summary)
    if args.write_commands:
        _write_run_commands(
            output_dir / "pr16_run_commands.sh",
            matrix_rows=matrix_rows,
            subset_rows=subset_rows,
            data_root=data_root,
            top_k=args.top_k,
        )
    else:
        (output_dir / "pr16_run_commands.sh").write_text(
            "# Re-run planner with --write-commands to generate PR16 command guide.\n",
            encoding="utf-8",
        )
    _write_plan_report(
        output_dir / "pr16_plan_report.md",
        scenes=args.scenes,
        conditions=args.conditions,
        subset_names=args.subset_names,
        summary=seed_summary,
    )
    write_artifact_manifest(
        output_dir / "artifact_manifest.csv",
        output_dir=output_dir,
        output_files=PLAN_OUTPUT_FILES,
        inputs=[
            ("data_root", data_root, True, "input", "Data root used for train-view discovery"),
            ("input_root", input_root, False, "input", "Report root used for original subset inference"),
        ],
    )
    if not args.quiet:
        print(f"PR16 planned matrix rows: {len(matrix_rows)}", file=sys.stderr)
    print(json.dumps(seed_summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
