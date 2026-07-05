"""Post-hoc training dynamics extraction for observed baseline runs.

This module is local-safe: it reads existing run artifacts and writes derived
tables, but it does not import CUDA libraries, run training, or edit trainer
outputs.
"""

from __future__ import annotations

import csv
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from viewtrust.analysis.tables import write_csv_table

TRAINING_DYNAMICS_SCHEMA = "viewtrust.training_dynamics.summary"
TRAINING_DYNAMICS_SCHEMA_VERSION = 1

TRAINING_DYNAMICS_FIELDS = [
    "run_id",
    "source",
    "iteration",
    "tag",
    "value",
    "wall_time",
]
TRAINING_ARTIFACTS_FIELDS = [
    "run_id",
    "relative_path",
    "file_type",
    "size_bytes",
    "modified_time",
]
FINAL_GAUSSIAN_FIELDS = [
    "run_id",
    "iteration",
    "point_cloud_path",
    "exists",
    "gaussian_count",
    "size_bytes",
    "parse_status",
]
REQUIRED_OBSERVED_FILES = [
    "summary.json",
    "metadata.json",
    "config_snapshot.json",
    "stdout.log",
    "stderr.log",
    "tables/command_summary.csv",
    "tables/gpu_memory_samples.csv",
    "trainer_output",
]
PREFERRED_SCALAR_TAGS = {
    "train_loss_patches/l1_loss",
    "train_loss_patches/total_loss",
    "iter_time",
    "test/loss_viewpoint - l1_loss",
    "test/loss_viewpoint - psnr",
    "train/loss_viewpoint - l1_loss",
    "train/loss_viewpoint - psnr",
    "total_points",
}


@dataclass(frozen=True)
class TrainingDynamicsExtractionConfig:
    run_dir: Path
    require_success: bool = False
    allow_missing_tensorboard: bool = True


def load_observed_summary(run_dir: Path) -> dict[str, Any]:
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"observed run is incomplete: missing summary.json")
    document = json.loads(summary_path.read_text(encoding="utf-8"))
    summary = document.get("summary", document)
    if not isinstance(summary, dict):
        raise ValueError("summary.json does not contain a summary object")
    return summary


def validate_observed_run_dir(
    run_dir: Path,
    require_success: bool = False,
) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"run_dir does not exist: {run_dir}")
    if not run_dir.is_dir():
        raise NotADirectoryError(f"run_dir is not a directory: {run_dir}")

    missing = [
        name
        for name in REQUIRED_OBSERVED_FILES
        if not (run_dir / name).exists()
    ]
    if missing:
        raise FileNotFoundError(
            "observed run is incomplete: missing " + ", ".join(missing)
        )

    summary = load_observed_summary(run_dir)
    if require_success and summary.get("returncode") != 0:
        raise RuntimeError(
            f"observed run returncode is not 0: {summary.get('returncode')}"
        )
    return summary


def find_tensorboard_event_files(run_dir: Path) -> list[Path]:
    run_dir = run_dir.resolve()
    trainer_output = run_dir / "trainer_output"
    if not trainer_output.is_dir():
        return []

    event_files: list[Path] = []
    for current_root, dir_names, file_names in os.walk(trainer_output, followlinks=False):
        current_path = Path(current_root)
        dir_names[:] = [
            name
            for name in dir_names
            if not (current_path / name).is_symlink()
            or _is_within(current_path / name, run_dir)
        ]
        for file_name in file_names:
            path = current_path / file_name
            if "tfevents" not in path.name:
                continue
            if path.is_symlink() and not _is_within(path, run_dir):
                continue
            if path.is_file():
                event_files.append(path)
    return sorted(event_files)


def extract_tensorboard_scalars(event_files: list[Path]) -> list[dict[str, Any]]:
    if not event_files:
        return []

    try:
        from tensorboard.backend.event_processing.event_accumulator import (  # type: ignore[import-not-found]
            EventAccumulator,
        )
    except ImportError:
        return []

    rows: list[dict[str, Any]] = []
    for event_file in event_files:
        accumulator = EventAccumulator(str(event_file))
        accumulator.Reload()
        for tag in accumulator.Tags().get("scalars", []):
            for scalar in accumulator.Scalars(tag):
                rows.append(
                    {
                        "source": "tensorboard",
                        "iteration": scalar.step,
                        "tag": tag,
                        "value": scalar.value,
                        "wall_time": scalar.wall_time,
                    }
                )
    return rows


def _tensorboard_available() -> bool:
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator  # noqa: F401
    except ImportError:
        return False
    return True


def parse_gpu_memory_samples(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "tables" / "gpu_memory_samples.csv"
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _classify_artifact(path: Path) -> str:
    name = path.name.lower()
    suffix = path.suffix.lower()
    if "tfevents" in name:
        return "tensorboard_event"
    if suffix == ".ply":
        return "ply"
    if suffix in {".pt", ".pth", ".ckpt"}:
        return "checkpoint"
    if name == "cfg_args" or suffix in {".json", ".yaml", ".yml", ".toml"}:
        return "config"
    if suffix in {".png", ".jpg", ".jpeg", ".webp"}:
        return "image"
    if suffix in {".log", ".txt"}:
        return "log"
    return "other"


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def inspect_trainer_artifacts(run_dir: Path) -> list[dict[str, Any]]:
    run_dir = run_dir.resolve()
    trainer_output = run_dir / "trainer_output"
    if not trainer_output.is_dir():
        return []

    rows: list[dict[str, Any]] = []
    for current_root, dir_names, file_names in os.walk(trainer_output, followlinks=False):
        current_path = Path(current_root)
        dir_names[:] = [
            name
            for name in dir_names
            if not (current_path / name).is_symlink()
            or _is_within(current_path / name, run_dir)
        ]
        for file_name in sorted(file_names):
            path = current_path / file_name
            if path.is_symlink() and not _is_within(path, run_dir):
                continue
            stat = path.stat()
            rows.append(
                {
                    "relative_path": path.relative_to(run_dir).as_posix(),
                    "file_type": _classify_artifact(path),
                    "size_bytes": stat.st_size,
                    "modified_time": datetime.fromtimestamp(
                        stat.st_mtime,
                        tz=timezone.utc,
                    ).isoformat(),
                }
            )
    return sorted(rows, key=lambda row: str(row["relative_path"]))


def count_gaussians_from_ply(ply_path: Path) -> int | None:
    try:
        with ply_path.open("rb") as handle:
            for raw_line in handle:
                line = raw_line.decode("ascii", errors="replace").strip()
                match = re.match(r"^element\s+vertex\s+(\d+)\s*$", line)
                if match:
                    return int(match.group(1))
                if line == "end_header":
                    return None
    except OSError:
        return None
    return None


def _final_point_cloud_row(run_dir: Path) -> dict[str, Any]:
    candidates: list[tuple[int, Path]] = []
    point_cloud_root = run_dir / "trainer_output" / "point_cloud"
    for path in point_cloud_root.glob("iteration_*/point_cloud.ply"):
        match = re.match(r"iteration_(\d+)$", path.parent.name)
        if match:
            candidates.append((int(match.group(1)), path))

    if not candidates:
        return {
            "iteration": "",
            "point_cloud_path": "",
            "exists": False,
            "gaussian_count": "",
            "size_bytes": "",
            "parse_status": "not_found",
        }

    iteration, path = max(candidates, key=lambda item: item[0])
    gaussian_count = count_gaussians_from_ply(path)
    return {
        "iteration": iteration,
        "point_cloud_path": path.relative_to(run_dir).as_posix(),
        "exists": path.exists(),
        "gaussian_count": gaussian_count if gaussian_count is not None else "",
        "size_bytes": path.stat().st_size if path.exists() else "",
        "parse_status": "ok" if gaussian_count is not None else "parse_failed",
    }


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _with_run_id(rows: list[dict[str, Any]], run_id: str) -> list[dict[str, Any]]:
    return [{"run_id": run_id, **row} for row in rows]


def extract_training_dynamics(
    config: TrainingDynamicsExtractionConfig,
) -> dict[str, Any]:
    run_dir = config.run_dir.resolve()
    summary = validate_observed_run_dir(run_dir, config.require_success)
    run_id = run_dir.name
    warnings: list[str] = []
    missing_fields: list[str] = []

    tensorboard_available = _tensorboard_available()
    event_files = find_tensorboard_event_files(run_dir)
    scalar_rows: list[dict[str, Any]] = []
    tensorboard_missing_reason = ""

    if event_files and tensorboard_available:
        scalar_rows = extract_tensorboard_scalars(event_files)
    elif event_files and not tensorboard_available:
        tensorboard_missing_reason = "tensorboard package is not installed"
        warnings.append(
            "TensorBoard event files were found, but tensorboard is not installed; loss curves were not extracted."
        )
    else:
        tensorboard_missing_reason = "no TensorBoard event files found"
        warnings.append("No TensorBoard event files found; loss curves were not extracted.")

    if not scalar_rows:
        missing_fields.append("training_scalars")
        if event_files and tensorboard_available:
            warnings.append("TensorBoard event files were found, but no scalar data was extracted.")

    if not config.allow_missing_tensorboard and not scalar_rows:
        raise RuntimeError(tensorboard_missing_reason or "TensorBoard scalar data is missing")

    gpu_rows = parse_gpu_memory_samples(run_dir)
    artifact_rows = inspect_trainer_artifacts(run_dir)
    final_gaussian_row = _final_point_cloud_row(run_dir)
    final_gaussian_count = final_gaussian_row["gaussian_count"]
    final_point_cloud_path = final_gaussian_row["point_cloud_path"]
    stdout_path = run_dir / "stdout.log"
    stderr_path = run_dir / "stderr.log"

    tables_dir = run_dir / "tables"
    write_csv_table(
        tables_dir / "training_dynamics.csv",
        _with_run_id(scalar_rows, run_id),
        TRAINING_DYNAMICS_FIELDS,
    )
    write_csv_table(
        tables_dir / "training_artifacts.csv",
        _with_run_id(artifact_rows, run_id),
        TRAINING_ARTIFACTS_FIELDS,
    )
    write_csv_table(
        tables_dir / "final_gaussian_summary.csv",
        _with_run_id([final_gaussian_row], run_id),
        FINAL_GAUSSIAN_FIELDS,
    )

    if final_gaussian_row["parse_status"] != "ok":
        missing_fields.append("final_gaussian_count")
        warnings.append("Final point cloud Gaussian count was not parsed.")

    summary_document = {
        "schema_name": TRAINING_DYNAMICS_SCHEMA,
        "schema_version": TRAINING_DYNAMICS_SCHEMA_VERSION,
        "run_dir": str(run_dir),
        "run_id": run_id,
        "observation_only": True,
        "returncode": summary.get("returncode"),
        "elapsed_s": summary.get("elapsed_s"),
        "stdout_log_bytes": stdout_path.stat().st_size if stdout_path.exists() else None,
        "stderr_log_bytes": stderr_path.stat().st_size if stderr_path.exists() else None,
        "tensorboard_available": tensorboard_available,
        "tensorboard_event_file_count": len(event_files),
        "tensorboard_event_files": [
            path.relative_to(run_dir).as_posix() for path in event_files
        ],
        "tensorboard_missing_reason": tensorboard_missing_reason,
        "preferred_scalar_tags": sorted(PREFERRED_SCALAR_TAGS),
        "training_scalar_count": len(scalar_rows),
        "gpu_sample_count": len(gpu_rows),
        "trainer_artifact_count": len(artifact_rows),
        "final_gaussian_count": final_gaussian_count if final_gaussian_count != "" else None,
        "final_point_cloud_path": final_point_cloud_path,
        "missing_fields": sorted(set(missing_fields)),
        "warnings": warnings,
    }
    _write_json(run_dir / "training_dynamics_summary.json", summary_document)
    return summary_document
