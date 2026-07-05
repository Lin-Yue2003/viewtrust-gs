"""Observation-only Gaussian lifecycle CSV writer.

This module is intentionally CPU-only. It may receive tensor-like objects from
an instrumented trainer, but it immediately detaches and converts them to Python
scalars/lists. It never stores tensor references.
"""

from __future__ import annotations

import csv
import json
import math
import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

GAUSSIAN_LIFECYCLE_SCHEMA = "viewtrust.gaussian_lifecycle.summary"
GAUSSIAN_LIFECYCLE_SCHEMA_VERSION = 1

LIFECYCLE_EVENT_FIELDS = [
    "run_id",
    "iteration",
    "event_type",
    "stage",
    "gaussian_id",
    "parent_gaussian_id",
    "birth_iteration",
    "death_iteration",
    "birth_type",
    "death_type",
    "source_index",
    "target_index",
    "gaussian_count_before",
    "gaussian_count_after",
    "gaussian_count_delta",
    "alive_after_event",
    "position_x",
    "position_y",
    "position_z",
    "opacity",
    "scale_min",
    "scale_mean",
    "scale_max",
    "rotation_norm",
    "timestamp_utc",
    "status",
    "warning",
]

LIFECYCLE_FINAL_FIELDS = [
    "run_id",
    "gaussian_id",
    "parent_gaussian_id",
    "birth_iteration",
    "death_iteration",
    "birth_type",
    "death_type",
    "alive",
    "final_index",
    "lifetime_iterations",
    "position_x",
    "position_y",
    "position_z",
    "opacity",
    "scale_min",
    "scale_mean",
    "scale_max",
    "rotation_norm",
    "created_by_iteration",
    "pruned_by_iteration",
    "status",
    "warning",
]


@dataclass(frozen=True)
class GaussianLifecycleConfig:
    run_id: str
    output_dir: Path
    scene: str
    condition: str
    trainer: str
    observation_only: bool = True
    enabled: bool = True
    strict: bool = False
    log_snapshot_stats: bool = True


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_float(value: Any) -> float | None:
    if value is None or (isinstance(value, str) and value == ""):
        return None
    try:
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "numel") and int(value.numel()) != 1:
            return None
        if hasattr(value, "item"):
            value = value.item()
        result = float(value)
    except (TypeError, ValueError, RuntimeError):
        return None
    return result if math.isfinite(result) else None


def safe_int(value: Any) -> int | None:
    number = safe_float(value)
    if number is None:
        return None
    return int(number)


def detach_cpu(value: Any) -> Any:
    try:
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
    except RuntimeError:
        return None
    return value


def tensor_scalar(value: Any) -> float | None:
    return safe_float(value)


def _as_list(value: Any) -> list[Any]:
    value = detach_cpu(value)
    if value is None:
        return []
    if hasattr(value, "reshape"):
        try:
            value = value.reshape(-1)
        except RuntimeError:
            pass
    if hasattr(value, "tolist"):
        result = value.tolist()
        return result if isinstance(result, list) else [result]
    try:
        return list(value)
    except TypeError:
        return [value]


def _mask_to_bools(mask: Any) -> list[bool]:
    return [bool(item) for item in _as_list(mask)]


def tensor_vector_stats(value: Any) -> dict[str, float | None]:
    items = [safe_float(item) for item in _as_list(value)]
    numbers = [item for item in items if item is not None]
    if not numbers:
        return {"min": None, "mean": None, "max": None, "norm": None}
    norm = math.sqrt(sum(item * item for item in numbers))
    return {
        "min": min(numbers),
        "mean": sum(numbers) / len(numbers),
        "max": max(numbers),
        "norm": norm,
    }


def _count_gaussians(gaussians: Any = None, gaussian_count: Any = None) -> int | None:
    explicit = safe_int(gaussian_count)
    if explicit is not None:
        return explicit
    if gaussians is None:
        return None
    try:
        xyz = gaussians.get_xyz
        if hasattr(xyz, "shape"):
            return int(xyz.shape[0])
        return len(xyz)
    except (AttributeError, TypeError, ValueError):
        return None


def _row_at(value: Any, index: int) -> Any:
    if value is None:
        return None
    try:
        return detach_cpu(value[index])
    except (IndexError, TypeError, RuntimeError):
        return None


class GaussianLifecycleObserver:
    """Track stable per-run Gaussian IDs through observation-only hooks."""

    def __init__(self, config: GaussianLifecycleConfig):
        self.config = config
        self.output_dir = config.output_dir.resolve()
        self.run_dir = self.output_dir.parent
        self.tables_dir = self.run_dir / "tables"
        self.enabled = config.enabled
        self.warning_messages: list[str] = []
        self.invariant_violations = 0
        self.current_ids: list[int] = []
        self.states: dict[int, dict[str, Any]] = {}
        self.next_id = 0
        self.initial_gaussian_count: int | None = None
        self.final_gaussian_count: int | None = None
        self.requested_iterations: int | None = None
        self.lifecycle_event_rows = 0
        self.birth_event_count = 0
        self.clone_birth_count = 0
        self.split_birth_count = 0
        self.densification_birth_count = 0
        self.prune_death_count = 0
        self._pending_prune: dict[str, Any] | None = None

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.tables_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.output_dir / "gaussian_lifecycle_events.csv"
        self.final_path = self.output_dir / "gaussian_lifecycle_final.csv"
        self.summary_path = self.output_dir / "gaussian_lifecycle_summary.json"
        self.warnings_path = self.output_dir / "gaussian_lifecycle_warnings.jsonl"
        self._ensure_header(self.events_path, LIFECYCLE_EVENT_FIELDS)
        self._ensure_header(self.final_path, LIFECYCLE_FINAL_FIELDS)

    @classmethod
    def from_environment(cls) -> "GaussianLifecycleObserver | None":
        if os.environ.get("VIEWTRUST_ENABLE_GAUSSIAN_LIFECYCLE") != "1":
            return None
        output_dir = os.environ.get("VIEWTRUST_GAUSSIAN_LIFECYCLE_DIR")
        if not output_dir:
            return None
        return cls(
            GaussianLifecycleConfig(
                output_dir=Path(output_dir),
                run_id=os.environ.get("VIEWTRUST_RUN_ID", "unknown"),
                scene=os.environ.get("VIEWTRUST_SCENE", "unknown"),
                condition=os.environ.get("VIEWTRUST_CONDITION", "unknown"),
                trainer=os.environ.get("VIEWTRUST_TRAINER", "unknown"),
                observation_only=os.environ.get("VIEWTRUST_OBSERVATION_ONLY", "1")
                == "1",
                enabled=True,
                strict=os.environ.get("VIEWTRUST_GAUSSIAN_LIFECYCLE_STRICT") == "1",
                log_snapshot_stats=os.environ.get(
                    "VIEWTRUST_GAUSSIAN_LIFECYCLE_LOG_SNAPSHOT_STATS",
                    "1",
                )
                == "1",
            )
        )

    def initialize(self, **kwargs: Any) -> None:
        self.on_after_scene_init(**kwargs)

    def on_after_scene_init(self, **kwargs: Any) -> None:
        self._safe(lambda: self._initialize_impl(**kwargs))

    def on_before_clone(self, **kwargs: Any) -> None:
        self._safe(lambda: None)

    def on_after_clone(self, **kwargs: Any) -> None:
        self._safe(lambda: self._append_births(birth_type="clone", stage="after_clone", **kwargs))

    def on_before_split(self, **kwargs: Any) -> None:
        self._safe(lambda: None)

    def on_after_split(self, **kwargs: Any) -> None:
        self._safe(lambda: self._append_births(birth_type="split", stage="after_split", **kwargs))

    def on_before_prune(self, **kwargs: Any) -> None:
        self._safe(lambda: self._before_prune_impl(**kwargs))

    def on_after_prune(self, **kwargs: Any) -> None:
        self._safe(lambda: self._after_prune_impl(**kwargs))

    def on_after_densification(self, **kwargs: Any) -> None:
        self._safe(lambda: self._reconcile_count(stage="after_densification", **kwargs))

    def finalize(self, **kwargs: Any) -> dict[str, Any]:
        result = self._safe(lambda: self._finalize_impl(**kwargs), allow_disabled=True)
        return result if isinstance(result, dict) else {}

    def _initialize_impl(self, **kwargs: Any) -> None:
        count = _count_gaussians(kwargs.get("gaussians"), kwargs.get("gaussian_count"))
        if count is None:
            self._warning("cannot initialize lifecycle IDs without Gaussian count")
            return
        self.current_ids = list(range(count))
        self.next_id = count
        self.initial_gaussian_count = count
        iteration = safe_int(kwargs.get("iteration")) or 0
        for gaussian_id in self.current_ids:
            self.states[gaussian_id] = {
                "gaussian_id": gaussian_id,
                "parent_gaussian_id": "",
                "birth_iteration": iteration,
                "death_iteration": "",
                "birth_type": "init",
                "death_type": "none",
                "alive": True,
                "created_by_iteration": iteration,
                "pruned_by_iteration": "",
                "last_snapshot": {},
                "warning": "",
            }
        self._check_invariants(count, "after_scene_init")

    def _append_births(self, *, birth_type: str, stage: str, **kwargs: Any) -> None:
        before_count = len(self.current_ids)
        after_count = _count_gaussians(kwargs.get("gaussians"), kwargs.get("gaussian_count"))
        if after_count is None:
            return
        appended_count = after_count - before_count
        if appended_count <= 0:
            self._check_invariants(after_count, stage)
            return

        iteration = safe_int(kwargs.get("iteration")) or 0
        source_mask = _mask_to_bools(kwargs.get("source_mask"))
        n_children = max(1, safe_int(kwargs.get("children_per_source")) or 1)
        selected_parent_ids = [
            gaussian_id
            for gaussian_id, selected in zip(self.current_ids, source_mask)
            if selected
        ]
        parent_ids: list[int | str] = []
        if selected_parent_ids:
            for parent_id in selected_parent_ids:
                parent_ids.extend([parent_id] * n_children)
        warning = ""
        if len(parent_ids) != appended_count:
            parent_ids = [""] * appended_count
            birth_type = "densification_unknown"
            warning = "parent lineage unavailable in PR8"

        new_ids = list(range(self.next_id, self.next_id + appended_count))
        self.next_id += appended_count
        self.current_ids.extend(new_ids)
        for offset, gaussian_id in enumerate(new_ids):
            target_index = before_count + offset
            snapshot = self._snapshot(kwargs.get("gaussians"), target_index)
            parent_id = parent_ids[offset] if offset < len(parent_ids) else ""
            self.states[gaussian_id] = {
                "gaussian_id": gaussian_id,
                "parent_gaussian_id": parent_id,
                "birth_iteration": iteration,
                "death_iteration": "",
                "birth_type": birth_type,
                "death_type": "none",
                "alive": True,
                "created_by_iteration": iteration,
                "pruned_by_iteration": "",
                "last_snapshot": snapshot,
                "warning": warning,
            }
            self._write_event(
                {
                    "iteration": iteration,
                    "event_type": f"{birth_type}_birth"
                    if birth_type in ("clone", "split")
                    else "densification_birth",
                    "stage": stage,
                    "gaussian_id": gaussian_id,
                    "parent_gaussian_id": parent_id,
                    "birth_iteration": iteration,
                    "birth_type": birth_type,
                    "source_index": "",
                    "target_index": target_index,
                    "gaussian_count_before": before_count,
                    "gaussian_count_after": after_count,
                    "gaussian_count_delta": appended_count,
                    "alive_after_event": True,
                    "status": "ok",
                    "warning": warning,
                    **snapshot,
                }
            )
            self.birth_event_count += 1
            if birth_type == "clone":
                self.clone_birth_count += 1
            elif birth_type == "split":
                self.split_birth_count += 1
            else:
                self.densification_birth_count += 1
        self._check_invariants(after_count, stage)

    def _before_prune_impl(self, **kwargs: Any) -> None:
        prune_mask = _mask_to_bools(kwargs.get("prune_mask"))
        if not prune_mask:
            return
        before_count = len(self.current_ids)
        if len(prune_mask) != before_count:
            self._violation(
                "before_prune",
                f"prune mask length {len(prune_mask)} != lifecycle length {before_count}",
            )
            return
        snapshots = {}
        gaussians = kwargs.get("gaussians")
        for index, should_prune in enumerate(prune_mask):
            if should_prune:
                snapshots[self.current_ids[index]] = self._snapshot(gaussians, index)
        self._pending_prune = {
            "iteration": safe_int(kwargs.get("iteration")) or 0,
            "stage": kwargs.get("stage", "before_prune"),
            "ids_before": list(self.current_ids),
            "prune_mask": prune_mask,
            "snapshots": snapshots,
        }

    def _after_prune_impl(self, **kwargs: Any) -> None:
        if self._pending_prune is None:
            return
        pending = self._pending_prune
        self._pending_prune = None
        ids_before = pending["ids_before"]
        prune_mask = pending["prune_mask"]
        before_count = len(ids_before)
        kept_ids = [
            gaussian_id
            for gaussian_id, should_prune in zip(ids_before, prune_mask)
            if not should_prune
        ]
        pruned_ids = [
            gaussian_id
            for gaussian_id, should_prune in zip(ids_before, prune_mask)
            if should_prune
        ]
        after_count = _count_gaussians(kwargs.get("gaussians"), kwargs.get("gaussian_count"))
        if after_count is not None and len(kept_ids) != after_count:
            self._violation(
                "after_prune",
                f"kept lifecycle IDs {len(kept_ids)} != Gaussian count {after_count}",
            )
        self.current_ids = kept_ids
        iteration = safe_int(kwargs.get("iteration")) or pending["iteration"]
        for gaussian_id in pruned_ids:
            state = self.states.get(gaussian_id)
            if state is None:
                self._violation("after_prune", f"missing state for pruned id {gaussian_id}")
                continue
            snapshot = pending["snapshots"].get(gaussian_id, {})
            state.update(
                {
                    "death_iteration": iteration,
                    "death_type": "prune",
                    "alive": False,
                    "pruned_by_iteration": iteration,
                    "last_snapshot": snapshot,
                }
            )
            self._write_event(
                {
                    "iteration": iteration,
                    "event_type": "prune_death",
                    "stage": "after_prune",
                    "gaussian_id": gaussian_id,
                    "parent_gaussian_id": state.get("parent_gaussian_id", ""),
                    "birth_iteration": state.get("birth_iteration", ""),
                    "death_iteration": iteration,
                    "birth_type": state.get("birth_type", ""),
                    "death_type": "prune",
                    "gaussian_count_before": before_count,
                    "gaussian_count_after": len(kept_ids),
                    "gaussian_count_delta": len(kept_ids) - before_count,
                    "alive_after_event": False,
                    "status": "ok",
                    **snapshot,
                }
            )
            self.prune_death_count += 1
        self._check_invariants(after_count or len(self.current_ids), "after_prune")

    def _reconcile_count(self, *, stage: str, **kwargs: Any) -> None:
        count = _count_gaussians(kwargs.get("gaussians"), kwargs.get("gaussian_count"))
        if count is None:
            return
        if count > len(self.current_ids):
            self._append_births(
                birth_type="densification_unknown",
                stage=stage,
                iteration=kwargs.get("iteration"),
                gaussians=kwargs.get("gaussians"),
                gaussian_count=count,
                source_mask=[],
            )
        elif count < len(self.current_ids):
            self._violation(stage, "Gaussian count decreased without prune hook")
        self._check_invariants(count, stage)

    def _finalize_impl(self, **kwargs: Any) -> dict[str, Any]:
        self.requested_iterations = safe_int(kwargs.get("requested_iterations"))
        final_iteration = safe_int(kwargs.get("iteration")) or self.requested_iterations or 0
        self.final_gaussian_count = _count_gaussians(
            kwargs.get("gaussians"),
            kwargs.get("final_gaussian_count"),
        )
        if self.final_gaussian_count is not None:
            self._reconcile_count(
                stage="final",
                iteration=final_iteration,
                gaussians=kwargs.get("gaussians"),
                gaussian_count=self.final_gaussian_count,
            )
        self._write_final_rows(final_iteration, kwargs.get("gaussians"))
        alive_final_count = sum(1 for state in self.states.values() if state["alive"])
        dead_final_count = len(self.states) - alive_final_count
        if self.final_gaussian_count is None:
            self.final_gaussian_count = alive_final_count
        if alive_final_count != self.final_gaussian_count:
            self._violation(
                "final",
                "alive_final_count does not match final_gaussian_count",
            )
        if self.birth_event_count == 0:
            self._warning("birth lineage not yet observed or no net births occurred")
        summary = {
            "schema_name": GAUSSIAN_LIFECYCLE_SCHEMA,
            "schema_version": GAUSSIAN_LIFECYCLE_SCHEMA_VERSION,
            "run_id": self.config.run_id,
            "scene": self.config.scene,
            "condition": self.config.condition,
            "trainer": self.config.trainer,
            "observation_only": self.config.observation_only,
            "enabled": self.enabled,
            "requested_iterations": self.requested_iterations,
            "initial_gaussian_count": self.initial_gaussian_count,
            "final_gaussian_count": self.final_gaussian_count,
            "known_gaussian_count": len(self.states),
            "birth_event_count": self.birth_event_count,
            "clone_birth_count": self.clone_birth_count,
            "split_birth_count": self.split_birth_count,
            "densification_birth_count": self.densification_birth_count,
            "prune_death_count": self.prune_death_count,
            "alive_final_count": alive_final_count,
            "dead_final_count": dead_final_count,
            "lifecycle_event_rows": self.lifecycle_event_rows,
            "final_lifecycle_rows": len(self.states),
            "invariant_violations": self.invariant_violations,
            "warnings": self.warning_messages,
        }
        self.summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        self._mirror_outputs()
        return summary

    def _write_final_rows(self, final_iteration: int, gaussians: Any) -> None:
        alive_index_by_id = {gaussian_id: index for index, gaussian_id in enumerate(self.current_ids)}
        with self.final_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=LIFECYCLE_FINAL_FIELDS)
            writer.writeheader()
            for gaussian_id in sorted(self.states):
                state = self.states[gaussian_id]
                alive = bool(state["alive"])
                final_index = alive_index_by_id.get(gaussian_id, "")
                snapshot = (
                    self._snapshot(gaussians, final_index)
                    if alive and final_index != ""
                    else state.get("last_snapshot", {})
                )
                birth_iteration = safe_int(state.get("birth_iteration")) or 0
                death_iteration = safe_int(state.get("death_iteration"))
                lifetime_end = final_iteration if alive else death_iteration
                lifetime_iterations = (
                    lifetime_end - birth_iteration
                    if lifetime_end is not None and lifetime_end >= birth_iteration
                    else ""
                )
                writer.writerow(
                    {
                        "run_id": self.config.run_id,
                        "gaussian_id": gaussian_id,
                        "parent_gaussian_id": state.get("parent_gaussian_id", ""),
                        "birth_iteration": birth_iteration,
                        "death_iteration": "" if alive else state.get("death_iteration", ""),
                        "birth_type": state.get("birth_type", ""),
                        "death_type": "none" if alive else state.get("death_type", ""),
                        "alive": str(alive).lower(),
                        "final_index": final_index,
                        "lifetime_iterations": lifetime_iterations,
                        "created_by_iteration": state.get("created_by_iteration", ""),
                        "pruned_by_iteration": state.get("pruned_by_iteration", ""),
                        "status": "ok",
                        "warning": state.get("warning", ""),
                        **snapshot,
                    }
                )

    def _snapshot(self, gaussians: Any, index: Any) -> dict[str, Any]:
        empty = {
            "position_x": "",
            "position_y": "",
            "position_z": "",
            "opacity": "",
            "scale_min": "",
            "scale_mean": "",
            "scale_max": "",
            "rotation_norm": "",
        }
        if not self.config.log_snapshot_stats or gaussians is None or index == "":
            return empty
        int_index = safe_int(index)
        if int_index is None:
            return empty
        position = _as_list(_row_at(getattr(gaussians, "get_xyz", None), int_index))
        scale_stats = tensor_vector_stats(_row_at(getattr(gaussians, "get_scaling", None), int_index))
        rotation_stats = tensor_vector_stats(_row_at(getattr(gaussians, "get_rotation", None), int_index))
        opacity_values = _as_list(_row_at(getattr(gaussians, "get_opacity", None), int_index))
        return {
            "position_x": safe_float(position[0]) if len(position) > 0 else "",
            "position_y": safe_float(position[1]) if len(position) > 1 else "",
            "position_z": safe_float(position[2]) if len(position) > 2 else "",
            "opacity": safe_float(opacity_values[0]) if opacity_values else "",
            "scale_min": scale_stats["min"] if scale_stats["min"] is not None else "",
            "scale_mean": scale_stats["mean"] if scale_stats["mean"] is not None else "",
            "scale_max": scale_stats["max"] if scale_stats["max"] is not None else "",
            "rotation_norm": rotation_stats["norm"] if rotation_stats["norm"] is not None else "",
        }

    def _write_event(self, values: dict[str, Any]) -> None:
        row = {field: values.get(field, "") for field in LIFECYCLE_EVENT_FIELDS}
        row["run_id"] = self.config.run_id
        row["timestamp_utc"] = row.get("timestamp_utc") or now_utc_iso()
        with self.events_path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=LIFECYCLE_EVENT_FIELDS)
            writer.writerow(row)
        self.lifecycle_event_rows += 1

    def _check_invariants(self, expected_count: int | None, stage: str) -> None:
        if expected_count is not None and len(self.current_ids) != expected_count:
            self._violation(stage, "lifecycle ID length does not match Gaussian count")
        if len(set(self.current_ids)) != len(self.current_ids):
            self._violation(stage, "duplicate alive lifecycle IDs")
        for gaussian_id in self.current_ids:
            state = self.states.get(gaussian_id)
            if state is None:
                self._violation(stage, f"alive Gaussian ID missing state: {gaussian_id}")
            elif not state.get("alive"):
                self._violation(stage, f"dead Gaussian ID remains alive: {gaussian_id}")

    def _violation(self, stage: str, message: str) -> None:
        self.invariant_violations += 1
        self._warning(f"invariant violation at {stage}: {message}")
        if self.config.strict:
            raise RuntimeError(message)

    def _warning(self, message: str) -> None:
        self.warning_messages.append(message)
        warning = {"timestamp_utc": now_utc_iso(), "warning": message}
        with self.warnings_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(warning, sort_keys=True) + "\n")

    def _safe(self, op: Any, *, allow_disabled: bool = False) -> Any:
        if not self.enabled and not self.config.strict and not allow_disabled:
            return None
        try:
            return op()
        except Exception as exc:
            self._warning(str(exc))
            if self.config.strict:
                raise
            self.enabled = False
            return None

    def _mirror_outputs(self) -> None:
        mirror_pairs = [
            (self.events_path, self.tables_dir / "gaussian_lifecycle_events.csv"),
            (self.final_path, self.tables_dir / "gaussian_lifecycle_final.csv"),
            (self.summary_path, self.run_dir / "gaussian_lifecycle_summary.json"),
        ]
        for source, target in mirror_pairs:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)

    @staticmethod
    def _ensure_header(path: Path, fields: list[str]) -> None:
        if path.exists():
            return
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
