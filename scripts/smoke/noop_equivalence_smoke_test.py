#!/usr/bin/env python3
"""LOCAL-SAFE smoke test for PR9 no-op equivalence reporting."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_run(
    run_dir: Path,
    *,
    observed: bool,
    invalid_events: bool = False,
    write_point_cloud: bool = True,
) -> None:
    run_id = run_dir.name
    _write_json(
        run_dir / "summary.json",
        {
            "summary": {
                "label": "chair_clean_gaussian_splatting",
                "returncode": 0,
                "elapsed_s": 20.0 if not observed else 36.0,
                "gpu_sample_count": 2,
                "observation_only": True,
            }
        },
    )
    _write_json(
        run_dir / "metadata.json",
        {
            "run_id": run_id,
            "trainer": "gaussian-splatting",
            "scene": "chair",
            "condition": "clean",
            "iterations": 700,
            "command": ["python", "train.py", "--iterations", "700"],
            "training_behavior_modified": False,
        },
    )
    _write_json(run_dir / "stats.json", {"stats": {}})
    _write_json(run_dir / "config_snapshot.json", {"observation_only": True})
    (run_dir / "stdout.log").write_text("iteration 700 complete\n", encoding="utf-8")
    (run_dir / "stderr.log").write_text("", encoding="utf-8")
    (run_dir / "tables").mkdir(parents=True, exist_ok=True)
    (run_dir / "tables" / "command_summary.csv").write_text(
        "label,returncode,elapsed_s\nchair_clean_gaussian_splatting,0,20.0\n",
        encoding="utf-8",
    )
    (run_dir / "tables" / "gpu_memory_samples.csv").write_text(
        "elapsed_s,gpu_index,gpu_name,memory_used_mb,memory_total_mb,utilization_gpu_percent\n"
        "0.0,0,Mock GPU,100,1000,10\n"
        "1.0,0,Mock GPU,250,1000,20\n",
        encoding="utf-8",
    )
    if write_point_cloud:
        point_cloud = (
            run_dir / "trainer_output" / "point_cloud" / "iteration_700" / "point_cloud.ply"
        )
        point_cloud.parent.mkdir(parents=True, exist_ok=True)
        point_cloud.write_text(
            "ply\n"
            "format binary_little_endian 1.0\n"
            "element vertex 123\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "end_header\n",
            encoding="utf-8",
        )
    if not observed:
        return

    _write_json(
        run_dir / "training_events_summary.json",
        {
            "requested_iterations": 700,
            "scene": "chair",
            "condition": "clean",
            "trainer": "gaussian-splatting",
            "final_gaussian_count": 92955,
            "invalid_training_event_rows": 1 if invalid_events else 0,
        },
    )
    _write_json(
        run_dir / "gaussian_lifecycle_summary.json",
        {
            "scene": "chair",
            "condition": "clean",
            "trainer": "gaussian-splatting",
            "final_gaussian_count": 92955,
            "known_gaussian_count": 101437,
            "alive_final_count": 92955,
            "dead_final_count": 8482,
            "invariant_violations": 0,
        },
    )


def _run_compare(project_root: Path, baseline: Path, observed: Path, output: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts" / "measure" / "compare_noop_runs.py"),
            "--baseline-run-dir",
            str(baseline),
            "--observed-run-dir",
            str(observed),
            "--output-dir",
            str(output),
            "--require-success",
            "--require-observation-invariants",
            "--write-markdown",
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def main() -> int:
    project_root = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory(prefix="viewtrust-noop-") as tmp:
        tmp_root = Path(tmp)
        baseline = tmp_root / "baseline"
        observed = tmp_root / "observed"
        invalid = tmp_root / "invalid"
        no_count = tmp_root / "no-count"
        _write_run(baseline, observed=False)
        _write_run(observed, observed=True)
        _write_run(invalid, observed=True, invalid_events=True)
        _write_run(no_count, observed=False, write_point_cloud=False)

        output = tmp_root / "report"
        completed = _run_compare(project_root, baseline, observed, output)
        if completed.returncode != 0:
            raise RuntimeError(completed.stderr or completed.stdout)
        for name in (
            "noop_equivalence_summary.json",
            "noop_equivalence_report.md",
            "noop_equivalence_metrics.csv",
        ):
            if not (output / name).exists():
                raise FileNotFoundError(output / name)
        summary = json.loads((output / "noop_equivalence_summary.json").read_text())
        if summary["baseline_success"] is not True or summary["observed_success"] is not True:
            raise ValueError("success flags mismatch")
        if summary["training_event_invariants_ok"] is not True:
            raise ValueError("training event invariants should pass")
        if summary["gaussian_lifecycle_invariants_ok"] is not True:
            raise ValueError("lifecycle invariants should pass")
        if summary["baseline_final_gaussian_count"] != 123:
            raise ValueError("baseline final Gaussian count did not use PLY fallback")
        if summary["final_gaussian_count_delta"] != 92955 - 123:
            raise ValueError("final Gaussian count delta mismatch")
        if "baseline run final Gaussian count unavailable" in summary["warnings"]:
            raise ValueError("PLY fallback should suppress missing count warning")

        invalid_completed = _run_compare(project_root, baseline, invalid, tmp_root / "invalid-report")
        if invalid_completed.returncode == 0:
            raise ValueError("invalid observed run should fail invariant requirement")

        no_count_completed = _run_compare(project_root, no_count, observed, tmp_root / "no-count-report")
        if no_count_completed.returncode == 0:
            raise ValueError("missing point cloud should fail --require-success")
        no_count_summary_path = tmp_root / "no-count-report" / "noop_equivalence_summary.json"
        no_count_summary = json.loads(no_count_summary_path.read_text())
        if "baseline run final Gaussian count unavailable" not in no_count_summary["warnings"]:
            raise ValueError("missing PLY/count warning should remain")

    print("noop equivalence smoke test ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
