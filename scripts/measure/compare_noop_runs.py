#!/usr/bin/env python3
"""Compare uninstrumented and observed clean baseline runs."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

SCHEMA_NAME = "viewtrust.noop_equivalence.summary"
SCHEMA_VERSION = 1


def _json_file(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _summary(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "summary.json"
    if not path.exists():
        return {}
    payload = _json_file(path)
    return payload.get("summary", payload)


def _metadata(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "metadata.json"
    return _json_file(path) if path.exists() else {}


def _nested(path: Path) -> dict[str, Any]:
    return _json_file(path) if path.exists() else {}


def _int_or_none(value: Any) -> int | None:
    if value in ("", None):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _float_or_none(value: Any) -> float | None:
    if value in ("", None):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _requested_iterations(metadata: dict[str, Any], training_events: dict[str, Any]) -> int | None:
    explicit = _int_or_none(metadata.get("iterations")) or _int_or_none(
        training_events.get("requested_iterations")
    )
    if explicit is not None:
        return explicit
    command = metadata.get("command")
    if isinstance(command, list):
        for index, token in enumerate(command[:-1]):
            if token == "--iterations":
                return _int_or_none(command[index + 1])
    return None


def _point_cloud_exists(run_dir: Path, requested_iterations: int | None) -> bool:
    point_root = run_dir / "trainer_output" / "point_cloud"
    if not point_root.exists():
        return False
    candidates: list[Path] = []
    if requested_iterations is not None:
        candidates.extend(
            [
                point_root / f"iteration_{requested_iterations}" / "point_cloud.ply",
                point_root / f"iteration_{requested_iterations}.ply",
            ]
        )
    candidates.extend(point_root.glob("iteration_*/point_cloud.ply"))
    candidates.extend(point_root.glob("iteration_*.ply"))
    return any(path.is_file() for path in candidates)


def _final_gaussian_count(run_dir: Path, training_events: dict[str, Any], lifecycle: dict[str, Any]) -> int | None:
    for value in (
        lifecycle.get("final_gaussian_count"),
        training_events.get("final_gaussian_count"),
        _nested(run_dir / "training_dynamics_summary.json").get("final_gaussian_count"),
    ):
        parsed = _int_or_none(value)
        if parsed is not None:
            return parsed
    final_summary = run_dir / "tables" / "final_gaussian_summary.csv"
    if final_summary.exists():
        with final_summary.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if rows:
            return _int_or_none(rows[0].get("gaussian_count"))
    return None


def _gpu_peak_mb(run_dir: Path) -> float | None:
    path = run_dir / "tables" / "gpu_memory_samples.csv"
    if not path.exists():
        return None
    peaks: list[float] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            value = _float_or_none(row.get("memory_used_mb"))
            if value is not None:
                peaks.append(value)
    return max(peaks) if peaks else None


def _run_info(run_dir: Path) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    metadata = _metadata(run_dir)
    summary = _summary(run_dir)
    training_events = _nested(run_dir / "training_events_summary.json")
    lifecycle = _nested(run_dir / "gaussian_lifecycle_summary.json")
    requested_iterations = _requested_iterations(metadata, training_events)
    return {
        "run_dir": str(run_dir),
        "run_id": metadata.get("run_id") or run_dir.name,
        "metadata": metadata,
        "summary": summary,
        "training_events": training_events,
        "lifecycle": lifecycle,
        "scene": metadata.get("scene") or training_events.get("scene") or lifecycle.get("scene"),
        "condition": metadata.get("condition")
        or training_events.get("condition")
        or lifecycle.get("condition"),
        "trainer": metadata.get("trainer") or training_events.get("trainer") or lifecycle.get("trainer"),
        "requested_iterations": requested_iterations,
        "success": summary.get("returncode") == 0,
        "returncode": summary.get("returncode"),
        "elapsed_s": _float_or_none(summary.get("elapsed_s")),
        "observation_only": summary.get("observation_only"),
        "point_cloud_exists": _point_cloud_exists(run_dir, requested_iterations),
        "final_gaussian_count": _final_gaussian_count(run_dir, training_events, lifecycle),
        "gpu_peak_mb": _gpu_peak_mb(run_dir),
        "has_summary": (run_dir / "summary.json").exists(),
        "has_metadata": (run_dir / "metadata.json").exists(),
    }


def compare_runs(baseline_run_dir: Path, observed_run_dir: Path) -> dict[str, Any]:
    baseline = _run_info(baseline_run_dir)
    observed = _run_info(observed_run_dir)
    warnings: list[str] = []

    training_events = observed["training_events"]
    lifecycle = observed["lifecycle"]
    training_event_invariants_ok = (
        bool(training_events)
        and _int_or_none(training_events.get("invalid_training_event_rows")) == 0
    )
    alive_final = _int_or_none(lifecycle.get("alive_final_count"))
    dead_final = _int_or_none(lifecycle.get("dead_final_count"))
    known_count = _int_or_none(lifecycle.get("known_gaussian_count"))
    lifecycle_final = _int_or_none(lifecycle.get("final_gaussian_count"))
    lifecycle_violations = _int_or_none(lifecycle.get("invariant_violations")) or 0
    gaussian_lifecycle_invariants_ok = (
        bool(lifecycle)
        and lifecycle_violations == 0
        and alive_final == lifecycle_final
        and (
            alive_final is None
            or dead_final is None
            or known_count is None
            or alive_final + dead_final == known_count
        )
    )

    baseline_count = baseline["final_gaussian_count"]
    observed_count = observed["final_gaussian_count"]
    count_delta = (
        observed_count - baseline_count
        if baseline_count is not None and observed_count is not None
        else None
    )
    count_delta_ratio = (
        count_delta / baseline_count
        if count_delta is not None and baseline_count not in (None, 0)
        else None
    )
    baseline_elapsed = baseline["elapsed_s"]
    observed_elapsed = observed["elapsed_s"]
    overhead_s = (
        observed_elapsed - baseline_elapsed
        if baseline_elapsed is not None and observed_elapsed is not None
        else None
    )
    overhead_ratio = (
        observed_elapsed / baseline_elapsed
        if baseline_elapsed not in (None, 0) and observed_elapsed is not None
        else None
    )
    baseline_gpu_peak = baseline["gpu_peak_mb"]
    observed_gpu_peak = observed["gpu_peak_mb"]
    gpu_peak_delta = (
        observed_gpu_peak - baseline_gpu_peak
        if baseline_gpu_peak is not None and observed_gpu_peak is not None
        else None
    )

    for label, info in (("baseline", baseline), ("observed", observed)):
        if not info["has_summary"]:
            warnings.append(f"{label} run missing summary.json")
        if not info["has_metadata"]:
            warnings.append(f"{label} run missing metadata.json")
        if not info["point_cloud_exists"]:
            warnings.append(f"{label} run missing final point_cloud.ply")
        if info["final_gaussian_count"] is None:
            warnings.append(f"{label} run final Gaussian count unavailable")

    return {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "baseline_run_dir": baseline["run_dir"],
        "observed_run_dir": observed["run_dir"],
        "baseline_run_id": baseline["run_id"],
        "observed_run_id": observed["run_id"],
        "scene": observed["scene"] or baseline["scene"],
        "condition": observed["condition"] or baseline["condition"],
        "trainer": observed["trainer"] or baseline["trainer"],
        "requested_iterations": observed["requested_iterations"] or baseline["requested_iterations"],
        "baseline_success": baseline["success"],
        "observed_success": observed["success"],
        "observation_only": baseline["observation_only"] is True
        and observed["observation_only"] is True,
        "training_event_invariants_ok": training_event_invariants_ok,
        "gaussian_lifecycle_invariants_ok": gaussian_lifecycle_invariants_ok,
        "baseline_final_gaussian_count": baseline_count,
        "observed_final_gaussian_count": observed_count,
        "final_gaussian_count_delta": count_delta,
        "final_gaussian_count_delta_ratio": count_delta_ratio,
        "baseline_elapsed_s": baseline_elapsed,
        "observed_elapsed_s": observed_elapsed,
        "runtime_overhead_s": overhead_s,
        "runtime_overhead_ratio": overhead_ratio,
        "baseline_gpu_peak_mb": baseline_gpu_peak,
        "observed_gpu_peak_mb": observed_gpu_peak,
        "gpu_peak_delta_mb": gpu_peak_delta,
        "baseline_point_cloud_exists": baseline["point_cloud_exists"],
        "observed_point_cloud_exists": observed["point_cloud_exists"],
        "warnings": warnings,
    }


def _write_metrics_csv(path: Path, summary: dict[str, Any]) -> None:
    fields = [
        "metric",
        "baseline",
        "observed",
        "delta",
        "ratio",
    ]
    rows = [
        {
            "metric": "elapsed_s",
            "baseline": summary.get("baseline_elapsed_s"),
            "observed": summary.get("observed_elapsed_s"),
            "delta": summary.get("runtime_overhead_s"),
            "ratio": summary.get("runtime_overhead_ratio"),
        },
        {
            "metric": "final_gaussian_count",
            "baseline": summary.get("baseline_final_gaussian_count"),
            "observed": summary.get("observed_final_gaussian_count"),
            "delta": summary.get("final_gaussian_count_delta"),
            "ratio": summary.get("final_gaussian_count_delta_ratio"),
        },
        {
            "metric": "gpu_peak_mb",
            "baseline": summary.get("baseline_gpu_peak_mb"),
            "observed": summary.get("observed_gpu_peak_mb"),
            "delta": summary.get("gpu_peak_delta_mb"),
            "ratio": "",
        },
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _markdown(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# No-op Equivalence Report",
            "",
            "PR9 does not claim bitwise determinism. It checks observation-only design and gross deviations.",
            "",
            "## Runs",
            "",
            f"- Baseline: `{summary['baseline_run_dir']}`",
            f"- Observed: `{summary['observed_run_dir']}`",
            f"- Scene: `{summary.get('scene')}`",
            f"- Condition: `{summary.get('condition')}`",
            f"- Trainer: `{summary.get('trainer')}`",
            f"- Requested iterations: `{summary.get('requested_iterations')}`",
            "",
            "## Success",
            "",
            f"- Baseline success: `{summary['baseline_success']}`",
            f"- Observed success: `{summary['observed_success']}`",
            f"- Observation only: `{summary['observation_only']}`",
            f"- Training event invariants: `{summary['training_event_invariants_ok']}`",
            f"- Gaussian lifecycle invariants: `{summary['gaussian_lifecycle_invariants_ok']}`",
            "",
            "## Metrics",
            "",
            f"- Runtime overhead seconds: `{summary.get('runtime_overhead_s')}`",
            f"- Runtime overhead ratio: `{summary.get('runtime_overhead_ratio')}`",
            f"- Final Gaussian count delta: `{summary.get('final_gaussian_count_delta')}`",
            f"- Final Gaussian count delta ratio: `{summary.get('final_gaussian_count_delta_ratio')}`",
            "",
            "## Warnings",
            "",
            *[f"- {warning}" for warning in summary.get("warnings", [])],
            "" if summary.get("warnings") else "- None",
            "",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-run-dir", required=True, type=Path)
    parser.add_argument("--observed-run-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--require-success", action="store_true")
    parser.add_argument("--require-observation-invariants", action="store_true")
    parser.add_argument("--write-markdown", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = compare_runs(args.baseline_run_dir, args.observed_run_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "noop_equivalence_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_metrics_csv(args.output_dir / "noop_equivalence_metrics.csv", summary)
    if args.write_markdown:
        (args.output_dir / "noop_equivalence_report.md").write_text(
            _markdown(summary),
            encoding="utf-8",
        )
    print(json.dumps(summary, indent=2, sort_keys=True))

    if args.require_success:
        required_ok = (
            summary["baseline_success"]
            and summary["observed_success"]
            and summary["baseline_point_cloud_exists"]
            and summary["observed_point_cloud_exists"]
        )
        if not required_ok:
            print("ERROR: no-op comparison required successful runnable outputs.", file=sys.stderr)
            return 1
    if args.require_observation_invariants:
        if not summary["training_event_invariants_ok"] or not summary["gaussian_lifecycle_invariants_ok"]:
            print("ERROR: observation invariants failed.", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
