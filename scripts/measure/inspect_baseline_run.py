#!/usr/bin/env python3
"""Inspect a completed clean chair baseline run directory."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


REQUIRED_FILES = (
    "stdout.log",
    "stderr.log",
    "summary.json",
    "tables/command_summary.csv",
    "tables/gpu_memory_samples.csv",
)


def _csv_row_count(path: Path) -> int | None:
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        rows = list(reader)
    return max(0, len(rows) - 1)


def _json_file(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _trainer_output_file_count(path: Path) -> int:
    if not path.is_dir():
        return 0
    return sum(1 for child in path.rglob("*") if child.is_file())


def _detected_iterations(run_dir: Path, metadata: dict[str, Any]) -> list[int]:
    detected: set[int] = set()

    command = metadata.get("command", [])
    if isinstance(command, list):
        for index, token in enumerate(command[:-1]):
            if token == "--iterations":
                try:
                    detected.add(int(command[index + 1]))
                except (TypeError, ValueError):
                    pass

    for log_name in ("stdout.log", "stderr.log"):
        path = run_dir / log_name
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for match in re.finditer(r"\b(?:iter|iteration|iterations)\D+(\d+)\b", text, re.IGNORECASE):
            detected.add(int(match.group(1)))

    return sorted(detected)


def inspect_baseline_run(run_dir: Path) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    missing = [name for name in REQUIRED_FILES if not (run_dir / name).exists()]
    if not (run_dir / "trainer_output").is_dir():
        missing.append("trainer_output/")

    summary: dict[str, Any] = {}
    metadata: dict[str, Any] = {}
    if (run_dir / "summary.json").exists():
        summary = _json_file(run_dir / "summary.json").get("summary", {})
    if (run_dir / "metadata.json").exists():
        metadata = _json_file(run_dir / "metadata.json")

    stdout_path = run_dir / "stdout.log"
    stderr_path = run_dir / "stderr.log"
    events_path = run_dir / "events.jsonl"
    gpu_samples_path = run_dir / "tables" / "gpu_memory_samples.csv"
    gpu_samples_rows = _csv_row_count(gpu_samples_path)
    trainer_output = run_dir / "trainer_output"

    return {
        "run_dir": str(run_dir),
        "missing_required_paths": missing,
        "label": metadata.get("label") or summary.get("label"),
        "returncode": summary.get("returncode"),
        "elapsed_s": summary.get("elapsed_s"),
        "has_stdout": stdout_path.exists(),
        "has_stderr": stderr_path.exists(),
        "has_gpu_samples": gpu_samples_path.exists() and (gpu_samples_rows or 0) > 0,
        "gpu_sample_count": summary.get("gpu_sample_count", gpu_samples_rows),
        "trainer_output_exists": trainer_output.is_dir(),
        "trainer_output_file_count": _trainer_output_file_count(trainer_output),
        "detected_iterations": _detected_iterations(run_dir, metadata),
        "observation_only": summary.get("observation_only"),
        "command_summary_rows": _csv_row_count(run_dir / "tables" / "command_summary.csv"),
        "gpu_memory_samples_rows": gpu_samples_rows,
        "events_jsonl_lines": sum(1 for _ in events_path.open(encoding="utf-8"))
        if events_path.exists()
        else None,
        "stdout_log_bytes": stdout_path.stat().st_size if stdout_path.exists() else None,
        "stderr_log_bytes": stderr_path.stat().st_size if stderr_path.exists() else None,
        "training_behavior_modified": metadata.get("training_behavior_modified"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect a completed ViewTrust-GS clean chair baseline run."
    )
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument(
        "--require-success",
        action="store_true",
        help="Fail unless the observed command returncode is 0.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = inspect_baseline_run(args.run_dir)
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["missing_required_paths"]:
        return 1
    if args.require_success and report["returncode"] != 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
