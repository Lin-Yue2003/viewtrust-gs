"""Run a command while recording Priority 0 observation artifacts."""

from __future__ import annotations

import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from viewtrust.analysis.statistics import summarize_table
from viewtrust.logging.writer import Priority0Logger


@dataclass(frozen=True)
class ObservedCommandResult:
    run_dir: Path
    returncode: int
    elapsed_s: float
    stdout_path: Path
    stderr_path: Path


def _sample_gpu_memory() -> list[dict[str, object]]:
    if shutil.which("nvidia-smi") is None:
        return []

    query = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(
        query,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if completed.returncode != 0:
        return []

    rows: list[dict[str, object]] = []
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 5:
            continue
        rows.append(
            {
                "gpu_index": int(parts[0]),
                "gpu_name": parts[1],
                "memory_used_mb": float(parts[2]),
                "memory_total_mb": float(parts[3]),
                "utilization_gpu_percent": float(parts[4]),
            }
        )
    return rows


def run_observed_command(
    *,
    command: list[str],
    logger: Priority0Logger,
    config: dict[str, Any],
    sample_interval_s: float,
    label: str,
) -> ObservedCommandResult:
    """Run a command and write observation-only measurement artifacts."""

    if not command:
        raise ValueError("command must not be empty")
    if sample_interval_s <= 0:
        raise ValueError("sample_interval_s must be positive")

    stdout_path = logger.run_dir / "stdout.log"
    stderr_path = logger.run_dir / "stderr.log"
    logger.write_metadata(
        {
            "label": label,
            "command": command,
            "measurement_mode": "observed_subprocess",
            "training_behavior_modified": False,
        }
    )
    logger.write_config_snapshot(config)
    logger.write_run_start({"label": label, "command": command})
    logger.write_event("command_start", {"label": label, "command": command})

    gpu_rows: list[dict[str, object]] = []
    started = time.monotonic()
    next_sample = started

    with stdout_path.open("w", encoding="utf-8") as stdout_handle, stderr_path.open(
        "w",
        encoding="utf-8",
    ) as stderr_handle:
        process = subprocess.Popen(
            command,
            stdout=stdout_handle,
            stderr=stderr_handle,
            text=True,
        )
        while process.poll() is None:
            now = time.monotonic()
            if now >= next_sample:
                for row in _sample_gpu_memory():
                    row["elapsed_s"] = now - started
                    gpu_rows.append(row)
                    logger.write_event("gpu_memory_observation", row)
                next_sample = now + sample_interval_s
            time.sleep(min(0.1, sample_interval_s))

        returncode = process.wait()

    elapsed_s = time.monotonic() - started
    command_row = {
        "label": label,
        "returncode": returncode,
        "elapsed_s": elapsed_s,
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
    }
    logger.write_event("timing_observation", command_row)
    logger.write_event("command_end", command_row)

    logger.write_table(
        "command_summary",
        [command_row],
        ["label", "returncode", "elapsed_s", "stdout_path", "stderr_path"],
    )
    if gpu_rows:
        logger.write_table(
            "gpu_memory_samples",
            gpu_rows,
            [
                "elapsed_s",
                "gpu_index",
                "gpu_name",
                "memory_used_mb",
                "memory_total_mb",
                "utilization_gpu_percent",
            ],
        )

    stats: dict[str, Any] = {
        "command_summary": summarize_table([command_row], ["elapsed_s"]),
    }
    if gpu_rows:
        stats["gpu_memory_samples"] = summarize_table(
            gpu_rows,
            ["memory_used_mb", "memory_total_mb", "utilization_gpu_percent"],
        )
    logger.write_stats(stats)
    logger.write_summary(
        {
            "label": label,
            "returncode": returncode,
            "elapsed_s": elapsed_s,
            "gpu_sample_count": len(gpu_rows),
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
            "observation_only": True,
        }
    )
    logger.write_run_end({"status": "ok" if returncode == 0 else "failed"})

    return ObservedCommandResult(
        run_dir=logger.run_dir,
        returncode=returncode,
        elapsed_s=elapsed_s,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
    )
