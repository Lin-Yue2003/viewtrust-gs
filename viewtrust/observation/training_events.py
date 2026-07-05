"""Observation-only training event CSV writer.

This module is intentionally CPU-only. It stores Python scalars only and never
imports CUDA libraries or trainer internals.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

TRAINING_EVENTS_SCHEMA = "viewtrust.training_events.summary"
TRAINING_EVENTS_SCHEMA_VERSION = 1

TRAINING_EVENT_FIELDS = [
    "run_id",
    "iteration",
    "event_type",
    "camera_index",
    "camera_image_name",
    "loss",
    "l1_loss",
    "ssim",
    "depth_l1",
    "iter_time_ms",
    "gaussian_count",
    "visible_gaussian_count",
    "visibility_ratio",
    "radii_min",
    "radii_mean",
    "radii_max",
    "radii_nonzero_count",
    "position_grad_mean",
    "position_grad_max",
    "densification_eligible",
    "densification_triggered",
    "opacity_reset_triggered",
    "optimizer_step",
    "timestamp_utc",
    "status",
    "warning",
]

DENSIFICATION_EVENT_FIELDS = [
    "run_id",
    "iteration",
    "densification_eligible",
    "densification_triggered",
    "densify_from_iter",
    "densify_until_iter",
    "densification_interval",
    "densify_grad_threshold",
    "size_threshold",
    "gaussian_count_before",
    "gaussian_count_after",
    "gaussian_count_delta",
    "opacity_reset_triggered",
    "timestamp_utc",
    "status",
    "warning",
]

GAUSSIAN_COUNT_FIELDS = [
    "run_id",
    "iteration",
    "stage",
    "gaussian_count",
    "timestamp_utc",
]


@dataclass(frozen=True)
class TrainingEventObserverConfig:
    output_dir: Path
    run_id: str
    scene: str
    condition: str
    trainer: str
    observation_only: bool = True
    flush_every: int = 10
    strict: bool = False


class TrainingEventObserver:
    """Append-only observer for global training events."""

    def __init__(self, config: TrainingEventObserverConfig):
        self.config = config
        self.output_dir = config.output_dir.resolve()
        self.run_dir = self.output_dir.parent
        self.tables_dir = self.run_dir / "tables"
        self.enabled = True
        self.warning_messages: list[str] = []
        self.training_event_rows = 0
        self.densification_event_rows = 0
        self.densification_trigger_count = 0
        self.opacity_reset_count = 0
        self.gaussian_count_rows = 0
        self.iterations_seen: set[int] = set()
        self.initial_gaussian_count: int | None = None
        self.final_gaussian_count: int | None = None

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.tables_dir.mkdir(parents=True, exist_ok=True)
        self.training_events_path = self.output_dir / "training_events.csv"
        self.densification_events_path = self.output_dir / "densification_events.csv"
        self.gaussian_count_path = self.output_dir / "gaussian_count_timeseries.csv"
        self.warnings_path = self.output_dir / "observer_warnings.jsonl"
        self.summary_path = self.output_dir / "training_events_summary.json"

        self._ensure_header(self.training_events_path, TRAINING_EVENT_FIELDS)
        self._ensure_header(self.densification_events_path, DENSIFICATION_EVENT_FIELDS)
        self._ensure_header(self.gaussian_count_path, GAUSSIAN_COUNT_FIELDS)

    @classmethod
    def from_environment(cls) -> "TrainingEventObserver | None":
        if os.environ.get("VIEWTRUST_ENABLE_TRAINING_EVENTS") != "1":
            return None
        output_dir = os.environ.get("VIEWTRUST_TRAINING_EVENTS_DIR")
        if not output_dir:
            return None
        return cls(
            TrainingEventObserverConfig(
                output_dir=Path(output_dir),
                run_id=os.environ.get("VIEWTRUST_RUN_ID", "unknown"),
                scene=os.environ.get("VIEWTRUST_SCENE", "unknown"),
                condition=os.environ.get("VIEWTRUST_CONDITION", "unknown"),
                trainer=os.environ.get("VIEWTRUST_TRAINER", "unknown"),
                observation_only=os.environ.get("VIEWTRUST_OBSERVATION_ONLY", "1") == "1",
                flush_every=max(
                    1,
                    int(os.environ.get("VIEWTRUST_TRAINING_EVENT_LOG_INTERVAL", "10")),
                ),
                strict=os.environ.get("VIEWTRUST_OBSERVER_STRICT") == "1",
            )
        )

    def log_iteration_start(self, **kwargs: Any) -> None:
        self._safe(lambda: None)

    def log_iteration_metrics(self, **kwargs: Any) -> None:
        def op() -> None:
            iteration = self._int_or_none(kwargs.get("iteration"))
            event_type = str(kwargs.get("event_type", "iteration_metrics"))
            always_log = event_type != "iteration_metrics"
            if not self._should_log_iteration(iteration) and not always_log:
                return

            self.iterations_seen.add(iteration) if iteration is not None else None
            row = self._row(TRAINING_EVENT_FIELDS, kwargs)
            row["run_id"] = self.config.run_id
            row["event_type"] = event_type
            row["timestamp_utc"] = row.get("timestamp_utc") or self._now()
            row["status"] = row.get("status") or "ok"
            self._append(self.training_events_path, TRAINING_EVENT_FIELDS, row)
            self.training_event_rows += 1
            if self._truthy(row.get("opacity_reset_triggered")):
                self.opacity_reset_count += 1

        self._safe(op)

    def log_densification_event(self, **kwargs: Any) -> None:
        def op() -> None:
            row = self._row(DENSIFICATION_EVENT_FIELDS, kwargs)
            row["run_id"] = self.config.run_id
            row["timestamp_utc"] = row.get("timestamp_utc") or self._now()
            row["status"] = row.get("status") or "ok"
            before = self._int_or_none(row.get("gaussian_count_before"))
            after = self._int_or_none(row.get("gaussian_count_after"))
            if before is not None and after is not None:
                row["gaussian_count_delta"] = after - before
            self._append(self.densification_events_path, DENSIFICATION_EVENT_FIELDS, row)
            self.densification_event_rows += 1
            if self._truthy(row.get("densification_triggered")):
                self.densification_trigger_count += 1
            if self._truthy(row.get("opacity_reset_triggered")):
                self.opacity_reset_count += 1

        self._safe(op)

    def log_gaussian_count(self, **kwargs: Any) -> None:
        def op() -> None:
            row = self._row(GAUSSIAN_COUNT_FIELDS, kwargs)
            row["run_id"] = self.config.run_id
            row["timestamp_utc"] = row.get("timestamp_utc") or self._now()
            count = self._int_or_none(row.get("gaussian_count"))
            stage = str(row.get("stage", ""))
            if stage == "after_scene_init" and count is not None:
                self.initial_gaussian_count = count
            if stage == "final" and count is not None:
                self.final_gaussian_count = count
            self._append(self.gaussian_count_path, GAUSSIAN_COUNT_FIELDS, row)
            self.gaussian_count_rows += 1

        self._safe(op)

    def log_optimizer_step(self, **kwargs: Any) -> None:
        def op() -> None:
            iteration = self._int_or_none(kwargs.get("iteration"))
            if not self._should_log_iteration(iteration):
                return
            self.log_gaussian_count(
                iteration=iteration,
                stage=kwargs.get("stage", "after_optimizer_step"),
                gaussian_count=kwargs.get("gaussian_count"),
            )

        self._safe(op)

    def finalize(self, **kwargs: Any) -> dict[str, Any]:
        def op() -> dict[str, Any]:
            final_count = self._int_or_none(kwargs.get("final_gaussian_count"))
            if final_count is not None:
                self.log_gaussian_count(
                    iteration=kwargs.get("iteration", ""),
                    stage="final",
                    gaussian_count=final_count,
                )
                self.final_gaussian_count = final_count

            warnings = list(self.warning_messages)
            if self.densification_trigger_count == 0:
                warnings.append(
                    "No densification event was triggered; this is expected for short runs when iteration count does not exceed densify_from_iter."
                )

            summary = {
                "schema_name": TRAINING_EVENTS_SCHEMA,
                "schema_version": TRAINING_EVENTS_SCHEMA_VERSION,
                "run_id": self.config.run_id,
                "scene": self.config.scene,
                "condition": self.config.condition,
                "trainer": self.config.trainer,
                "observation_only": self.config.observation_only,
                "enabled": True,
                "iteration_count": len(self.iterations_seen),
                "training_event_rows": self.training_event_rows,
                "densification_event_rows": self.densification_event_rows,
                "densification_trigger_count": self.densification_trigger_count,
                "opacity_reset_count": self.opacity_reset_count,
                "gaussian_count_rows": self.gaussian_count_rows,
                "initial_gaussian_count": self.initial_gaussian_count,
                "final_gaussian_count": self.final_gaussian_count,
                "final_gaussian_count_delta": self._delta(
                    self.initial_gaussian_count,
                    self.final_gaussian_count,
                ),
                "warnings": warnings,
            }
            self.summary_path.write_text(
                json.dumps(summary, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            self._mirror_outputs()
            return summary

        result = self._safe(op, allow_disabled=True)
        return result if isinstance(result, dict) else {}

    def _safe(self, op: Any, *, allow_disabled: bool = False) -> Any:
        if not self.enabled and not self.config.strict and not allow_disabled:
            return None
        try:
            return op()
        except Exception as exc:
            self._record_warning(str(exc))
            if self.config.strict:
                raise
            self.enabled = False
            return None

    def _record_warning(self, message: str) -> None:
        self.warning_messages.append(message)
        warning = {"timestamp_utc": self._now(), "warning": message}
        with self.warnings_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(warning, sort_keys=True) + "\n")

    def _mirror_outputs(self) -> None:
        mirror_pairs = [
            (self.training_events_path, self.tables_dir / "training_events.csv"),
            (self.densification_events_path, self.tables_dir / "densification_events.csv"),
            (self.gaussian_count_path, self.tables_dir / "gaussian_count_timeseries.csv"),
            (self.summary_path, self.run_dir / "training_events_summary.json"),
        ]
        for source, target in mirror_pairs:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)

    def _should_log_iteration(self, iteration: int | None) -> bool:
        if iteration is None:
            return True
        return iteration == 1 or iteration % self.config.flush_every == 0

    @staticmethod
    def _ensure_header(path: Path, fields: list[str]) -> None:
        if path.exists():
            return
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()

    @staticmethod
    def _append(path: Path, fields: list[str], row: dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writerow({field: row.get(field, "") for field in fields})

    @staticmethod
    def _row(fields: list[str], values: dict[str, Any]) -> dict[str, Any]:
        return {field: values.get(field, "") for field in fields}

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _int_or_none(value: Any) -> int | None:
        if value in ("", None):
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _truthy(value: Any) -> bool:
        return value in (True, "True", "true", "1", 1)

    @staticmethod
    def _delta(first: int | None, second: int | None) -> int | None:
        if first is None or second is None:
            return None
        return second - first
