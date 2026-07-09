"""Observation-only stable Gaussian identity sidecar tracker.

The tracker mirrors Gaussian lifecycle operations in plain Python state. It
does not store tensor references, does not participate in autograd, and never
uses the mutable tensor row index as a stable Gaussian identity.
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


IDENTITY_FIELDS = [
    "scene", "condition", "subset_name", "run_id", "gaussian_id",
    "parent_gaussian_id", "root_gaussian_id", "birth_iteration",
    "birth_view_name", "birth_event_type", "prune_iteration",
    "prune_view_name", "death_event_type", "is_alive_final",
    "final_row_index", "total_visible_views", "total_update_views",
    "direct_corrupted_support_count", "collateral_support_count",
    "clean_prior_support_count", "other_clean_support_count",
    "support_view_names", "evidence_quality", "warnings",
]

EVENT_FIELDS = [
    "scene", "condition", "subset_name", "run_id", "iteration",
    "view_name", "view_index", "event_type", "gaussian_id",
    "parent_gaussian_id", "root_gaussian_id", "row_index_before",
    "row_index_after", "opacity_before", "opacity_after", "scale_before",
    "scale_after", "visibility_count_before", "visibility_count_after",
    "gradient_norm_before", "gradient_norm_after", "is_alive_after_event",
    "event_source", "evidence_quality", "notes",
]

ATTRIBUTION_FIELDS = [
    "scene", "condition", "subset_name", "run_id", "view_name",
    "view_index", "iteration", "gaussian_id", "event_type",
    "attribution_type", "contribution_value", "parent_gaussian_id",
    "root_gaussian_id", "is_alive_after_event", "evidence_quality", "notes",
]

SUPPORT_FIELDS = [
    "scene", "condition", "subset_name", "run_id", "gaussian_id",
    "parent_gaussian_id", "root_gaussian_id", "support_view_count",
    "support_view_names", "direct_corrupted_support_count",
    "collateral_support_count", "clean_prior_support_count",
    "other_clean_support_count", "source_entropy", "source_concentration",
    "corrupted_plus_collateral_ratio", "clean_prior_ratio", "birth_event_type",
    "death_event_type", "is_alive_final", "total_lifecycle_event_count",
    "total_visibility_event_count", "total_update_event_count",
    "evidence_quality", "warnings",
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    try:
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "reshape"):
            value = value.reshape(-1)
        if hasattr(value, "tolist"):
            result = value.tolist()
            return result if isinstance(result, list) else [result]
    except RuntimeError:
        return []
    try:
        return list(value)
    except TypeError:
        return [value]


def _mask_to_bools(mask: Any) -> list[bool]:
    return [bool(item) for item in _as_list(mask)]


def _indices_or_mask(value: Any, length: int) -> list[int]:
    items = _as_list(value)
    if not items:
        return []
    if len(items) == length and all(isinstance(item, bool) for item in items):
        return [index for index, enabled in enumerate(items) if enabled]
    if len(items) == length and all(str(item).lower() in {"true", "false", "0", "1"} for item in items):
        bools = [str(item).lower() in {"true", "1"} for item in items]
        return [index for index, enabled in enumerate(bools) if enabled]
    indices = []
    for item in items:
        try:
            index = int(item)
        except (TypeError, ValueError):
            continue
        if 0 <= index < length:
            indices.append(index)
    return indices


def _safe_float(value: Any) -> float | None:
    if value in ("", None):
        return None
    try:
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "item"):
            value = value.item()
        number = float(value)
    except (TypeError, ValueError, RuntimeError):
        return None
    return number if math.isfinite(number) else None


def _infer_view_index(view_name: str) -> int | None:
    token = str(view_name or "").rsplit("_", 1)[-1]
    try:
        return int(token)
    except ValueError:
        return None


def _normalize_event_type(event_type: str) -> str:
    if event_type == "clone":
        return "clone_birth"
    if event_type == "split":
        return "split_birth"
    if event_type == "densification_birth":
        return "densify_birth_unknown"
    if event_type == "init":
        return "initial_seed"
    return event_type or "unknown"


def _int_or_empty(value: Any) -> int | str:
    if value in ("", None):
        return ""
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return ""


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: "" if row.get(field) is None else row.get(field) for field in fields})


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


class GaussianIdentityTracker:
    """Stable ID sidecar for clone/split/prune/update observations."""

    def __init__(
        self,
        *,
        output_dir: Path | None = None,
        view_group_map: dict[str, str] | None = None,
        evidence_quality: str = "exact",
        integration_source: str = "fake_smoke_only",
        parent_mapping_source: str = "exact_parent_indices",
    ) -> None:
        self.output_dir = output_dir
        self.view_group_map = dict(view_group_map or {})
        self.default_evidence_quality = evidence_quality
        self.integration_source = integration_source
        self.parent_mapping_source = parent_mapping_source
        self.scene = ""
        self.condition = ""
        self.subset_name = ""
        self.run_id = ""
        self.active_ids: list[int] = []
        self.states: dict[int, dict[str, Any]] = {}
        self.next_id = 0
        self.current_view_name = ""
        self.current_view_index: int | str = ""
        self.warnings: list[str] = []
        self.events: list[dict[str, Any]] = []
        self.attribution_rows: list[dict[str, Any]] = []
        self.initial_gaussian_count = 0
        self.unknown_birth_count = 0

    def initialize(
        self,
        num_gaussians: int,
        scene: str,
        condition: str,
        subset_name: str,
        run_id: str,
    ) -> None:
        self.scene = scene
        self.condition = condition
        self.subset_name = subset_name
        self.run_id = run_id
        self.initial_gaussian_count = int(num_gaussians)
        self.active_ids = list(range(self.initial_gaussian_count))
        self.next_id = self.initial_gaussian_count
        for row_index, gaussian_id in enumerate(self.active_ids):
            self.states[gaussian_id] = self._new_state(
                gaussian_id=gaussian_id,
                parent_gaussian_id="",
                root_gaussian_id=gaussian_id,
                birth_iteration=0,
                birth_view_name="",
                birth_event_type="initial_seed",
                final_row_index=row_index,
                evidence_quality="exact",
            )
            self._record_event(
                iteration=0,
                view_name="",
                view_index="",
                event_type="initial_seed",
                gaussian_id=gaussian_id,
                parent_gaussian_id="",
                root_gaussian_id=gaussian_id,
                row_index_before="",
                row_index_after=row_index,
                is_alive_after_event=True,
                event_source="initialize",
                evidence_quality="exact",
                notes="initial stable Gaussian ID assignment",
            )

    def before_view(self, iteration: int, view_name: str, view_index: int | None = None) -> None:
        self.current_view_name = view_name
        self.current_view_index = "" if view_index is None else view_index
        for gaussian_id in self.active_ids:
            state = self.states[gaussian_id]
            state["last_seen_iteration"] = iteration
            state["last_seen_view_name"] = view_name

    def after_view(self, iteration: int, view_name: str, view_index: int | None = None) -> None:
        self.before_view(iteration, view_name, view_index)

    def record_clone_birth(
        self,
        parent_indices: Any,
        iteration: int,
        view_name: str,
        row_indices_after: Any = None,
    ) -> list[int]:
        parent_ids = [self.active_ids[index] for index in _indices_or_mask(parent_indices, len(self.active_ids))]
        root_ids = [self.states[parent_id]["root_gaussian_id"] for parent_id in parent_ids]
        return self.append_new_gaussians(
            parent_ids,
            root_ids,
            "clone_birth",
            iteration,
            view_name,
            row_indices_after=row_indices_after,
        )

    def record_split_birth(
        self,
        parent_indices: Any,
        child_count_per_parent: Any,
        iteration: int,
        view_name: str,
        row_indices_after: Any = None,
    ) -> list[int]:
        parent_row_indices = _indices_or_mask(parent_indices, len(self.active_ids))
        counts = _as_list(child_count_per_parent)
        if not counts:
            counts = [2 for _ in parent_row_indices]
        if len(counts) == 1 and len(parent_row_indices) > 1:
            counts = counts * len(parent_row_indices)
        parent_ids: list[int] = []
        root_ids: list[int] = []
        for offset, row_index in enumerate(parent_row_indices):
            parent_id = self.active_ids[row_index]
            count = int(counts[offset]) if offset < len(counts) else 2
            parent_ids.extend([parent_id] * count)
            root_ids.extend([self.states[parent_id]["root_gaussian_id"]] * count)
        return self.append_new_gaussians(
            parent_ids,
            root_ids,
            "split_birth",
            iteration,
            view_name,
            row_indices_after=row_indices_after,
        )

    def record_densify_birth_unknown(
        self,
        new_count: int,
        iteration: int,
        view_name: str,
        parent_indices: Any = None,
    ) -> list[int]:
        parent_row_indices = _indices_or_mask(parent_indices, len(self.active_ids))
        parent_ids = [self.active_ids[index] for index in parent_row_indices]
        if len(parent_ids) != new_count:
            parent_ids = [""] * int(new_count)
            root_ids: list[int | str] = [""] * int(new_count)
        else:
            root_ids = [self.states[int(parent_id)]["root_gaussian_id"] for parent_id in parent_ids]
        self.unknown_birth_count += int(new_count)
        return self.append_new_gaussians(
            parent_ids,
            root_ids,
            "densify_birth_unknown",
            iteration,
            view_name,
            evidence_quality="partial",
        )

    def record_prune(self, prune_mask: Any, iteration: int, view_name: str) -> list[int]:
        bools = _mask_to_bools(prune_mask)
        if len(bools) != len(self.active_ids):
            self.warnings.append(
                f"prune mask length {len(bools)} does not match active Gaussian count {len(self.active_ids)}"
            )
            return []
        pruned_ids = [gaussian_id for gaussian_id, should_prune in zip(self.active_ids, bools) if should_prune]
        kept_ids = [gaussian_id for gaussian_id, should_prune in zip(self.active_ids, bools) if not should_prune]
        for row_index, gaussian_id in enumerate(self.active_ids):
            if gaussian_id not in pruned_ids:
                continue
            state = self.states[gaussian_id]
            state.update(
                {
                    "prune_iteration": iteration,
                    "prune_view_name": view_name,
                    "death_event_type": "prune_death",
                    "is_alive_final": False,
                    "final_row_index": "",
                }
            )
            self._record_event(
                iteration=iteration,
                view_name=view_name,
                view_index=self.current_view_index,
                event_type="prune_death",
                gaussian_id=gaussian_id,
                parent_gaussian_id=state["parent_gaussian_id"],
                root_gaussian_id=state["root_gaussian_id"],
                row_index_before=row_index,
                row_index_after="",
                is_alive_after_event=False,
                event_source="record_prune",
                evidence_quality=state["evidence_quality"],
                notes="sidecar ID pruned with alive mask",
            )
        self.active_ids = kept_ids
        self._refresh_final_row_indices()
        return pruned_ids

    def record_visibility_observation(
        self,
        visible_mask_or_indices: Any,
        iteration: int,
        view_name: str,
        view_index: int | None = None,
    ) -> None:
        effective_view_index = self._effective_view_index(view_name, view_index)
        for row_index in _indices_or_mask(visible_mask_or_indices, len(self.active_ids)):
            gaussian_id = self.active_ids[row_index]
            self._record_support(gaussian_id, view_name, "visibility")
            state = self.states[gaussian_id]
            state["total_visible_views"] = len(state["visible_views"])
            self._record_event(
                iteration=iteration,
                view_name=view_name,
                view_index=effective_view_index,
                event_type="visibility_observation",
                gaussian_id=gaussian_id,
                parent_gaussian_id=state["parent_gaussian_id"],
                root_gaussian_id=state["root_gaussian_id"],
                row_index_before=row_index,
                row_index_after=row_index,
                visibility_count_after=state["total_visible_views"],
                is_alive_after_event=True,
                event_source="record_visibility_observation",
                evidence_quality=state["evidence_quality"],
                notes="visibility observation only",
            )
            self._record_attribution(iteration, view_name, gaussian_id, "visibility_observation", "visibility", 1.0, view_index=effective_view_index)

    def record_update_observation(
        self,
        indices: Any,
        iteration: int,
        view_name: str,
        view_index: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        metadata = metadata or {}
        effective_view_index = self._effective_view_index(view_name, view_index)
        for row_index in _indices_or_mask(indices, len(self.active_ids)):
            gaussian_id = self.active_ids[row_index]
            self._record_support(gaussian_id, view_name, "update")
            state = self.states[gaussian_id]
            state["total_update_views"] = len(state["update_views"])
            self._record_event(
                iteration=iteration,
                view_name=view_name,
                view_index=effective_view_index,
                event_type="update_observation",
                gaussian_id=gaussian_id,
                parent_gaussian_id=state["parent_gaussian_id"],
                root_gaussian_id=state["root_gaussian_id"],
                row_index_before=row_index,
                row_index_after=row_index,
                gradient_norm_after=_safe_float(metadata.get("gradient_norm")),
                is_alive_after_event=True,
                event_source="record_update_observation",
                evidence_quality=state["evidence_quality"],
                notes=str(metadata.get("notes", "update observation only")),
            )
            self._record_attribution(iteration, view_name, gaussian_id, "update_observation", "update", 1.0, view_index=effective_view_index)

    def compact_with_alive_mask(self, alive_mask: Any) -> None:
        bools = _mask_to_bools(alive_mask)
        if len(bools) != len(self.active_ids):
            self.warnings.append(
                f"alive mask length {len(bools)} does not match active Gaussian count {len(self.active_ids)}"
            )
            return
        self.active_ids = [gaussian_id for gaussian_id, alive in zip(self.active_ids, bools) if alive]
        self._refresh_final_row_indices()

    def append_new_gaussians(
        self,
        parent_ids: list[int | str],
        root_ids: list[int | str],
        birth_event_type: str,
        iteration: int,
        view_name: str,
        row_indices_after: Any = None,
        evidence_quality: str | None = None,
    ) -> list[int]:
        evidence_quality = evidence_quality or self.default_evidence_quality
        new_count = len(parent_ids)
        row_indices = _as_list(row_indices_after)
        if len(row_indices) != new_count:
            row_indices = list(range(len(self.active_ids), len(self.active_ids) + new_count))
        new_ids = list(range(self.next_id, self.next_id + new_count))
        self.next_id += new_count
        for offset, gaussian_id in enumerate(new_ids):
            parent_id = parent_ids[offset] if offset < len(parent_ids) else ""
            root_id = root_ids[offset] if offset < len(root_ids) and root_ids[offset] != "" else gaussian_id
            state = self._new_state(
                gaussian_id=gaussian_id,
                parent_gaussian_id=parent_id,
                root_gaussian_id=root_id,
                birth_iteration=iteration,
                birth_view_name=view_name,
                birth_event_type=birth_event_type,
                final_row_index=row_indices[offset],
                evidence_quality=evidence_quality,
            )
            self.states[gaussian_id] = state
            self.active_ids.append(gaussian_id)
            self._record_support(gaussian_id, view_name, "birth")
            self._record_event(
                iteration=iteration,
                view_name=view_name,
                view_index=self.current_view_index,
                event_type=birth_event_type,
                gaussian_id=gaussian_id,
                parent_gaussian_id=parent_id,
                root_gaussian_id=root_id,
                row_index_before=self._row_index_for_id(parent_id),
                row_index_after=row_indices[offset],
                is_alive_after_event=True,
                event_source="append_new_gaussians",
                evidence_quality=evidence_quality,
                notes="" if parent_id != "" else "parent mapping unavailable",
            )
            self._record_attribution(iteration, view_name, gaussian_id, birth_event_type, "birth", 1.0)
        self._refresh_final_row_indices()
        return new_ids

    @classmethod
    def from_existing_lifecycle_tables(
        cls,
        *,
        event_rows: list[dict[str, Any]],
        final_rows: list[dict[str, Any]],
        scene: str,
        condition: str,
        subset_name: str,
        run_id: str,
        view_group_map: dict[str, str] | None = None,
        output_dir: Path | None = None,
        integration_source: str = "real_view_influence_runner",
    ) -> "GaussianIdentityTracker":
        tracker = cls(
            output_dir=output_dir,
            view_group_map=view_group_map,
            evidence_quality="exact",
            integration_source=integration_source,
            parent_mapping_source="exact_clone_split_masks",
        )
        tracker.scene = scene
        tracker.condition = condition
        tracker.subset_name = subset_name
        tracker.run_id = run_id
        ids = sorted(
            {
                int(str(row.get("gaussian_id")))
                for row in [*event_rows, *final_rows]
                if str(row.get("gaussian_id", "")).strip().lstrip("-").isdigit()
            }
        )
        if not ids:
            tracker.warnings.append("no gaussian_id rows available in existing lifecycle tables")
            tracker.default_evidence_quality = "partial"
            tracker.parent_mapping_source = "unavailable"
            return tracker
        tracker.next_id = max(ids) + 1
        final_by_id = {str(row.get("gaussian_id", "")): row for row in final_rows}
        birth_by_id: dict[str, dict[str, Any]] = {}
        death_by_id: dict[str, dict[str, Any]] = {}
        for row in event_rows:
            gid = str(row.get("gaussian_id", "") or "")
            event_type = str(row.get("event_type", "") or "")
            if not gid:
                continue
            if "birth" in event_type and gid not in birth_by_id:
                birth_by_id[gid] = row
            if event_type == "prune_death":
                death_by_id[gid] = row
        for gaussian_id in ids:
            gid = str(gaussian_id)
            birth = birth_by_id.get(gid, {})
            final = final_by_id.get(gid, {})
            death = death_by_id.get(gid, {})
            parent = birth.get("parent_gaussian_id", "") or final.get("parent_gaussian_id", "")
            birth_type = _normalize_event_type(str(birth.get("event_type", "") or final.get("birth_type", "") or "initial_seed"))
            if birth_type == "densify_birth_unknown" or (birth_type in {"clone_birth", "split_birth"} and parent in ("", None)):
                tracker.default_evidence_quality = "partial"
                tracker.parent_mapping_source = "partial"
            tracker.states[gaussian_id] = tracker._new_state(
                gaussian_id=gaussian_id,
                parent_gaussian_id=_int_or_empty(parent),
                root_gaussian_id=gaussian_id,
                birth_iteration=int(float(birth.get("birth_iteration") or birth.get("iteration") or final.get("birth_iteration") or 0)),
                birth_view_name=str(birth.get("source_view_name", "") or ""),
                birth_event_type=birth_type,
                final_row_index=final.get("final_index", ""),
                evidence_quality=tracker.default_evidence_quality,
            )
            if death:
                tracker.states[gaussian_id].update(
                    {
                        "prune_iteration": death.get("iteration", ""),
                        "prune_view_name": death.get("source_view_name", ""),
                        "death_event_type": "prune_death",
                        "is_alive_final": False,
                        "final_row_index": "",
                    }
                )
        for gaussian_id in ids:
            tracker.states[gaussian_id]["root_gaussian_id"] = tracker._root_for(gaussian_id)
        tracker.active_ids = [
            int(row.get("gaussian_id"))
            for row in final_rows
            if str(row.get("gaussian_id", "")).strip().lstrip("-").isdigit()
            and str(row.get("alive", "")).lower() == "true"
        ]
        if not tracker.active_ids:
            tracker.active_ids = [gaussian_id for gaussian_id in ids if tracker.states[gaussian_id]["is_alive_final"]]
        tracker.initial_gaussian_count = sum(1 for state in tracker.states.values() if state["birth_event_type"] == "initial_seed")
        if tracker.initial_gaussian_count == 0:
            tracker.initial_gaussian_count = len(ids) - sum(1 for row in event_rows if "birth" in str(row.get("event_type", "")))
        tracker._replay_existing_events(event_rows)
        tracker._refresh_final_row_indices()
        return tracker

    def export_identity_table(self) -> list[dict[str, Any]]:
        self._refresh_support_counts()
        return [
            self._identity_row(self.states[gaussian_id])
            for gaussian_id in sorted(self.states)
        ]

    def export_lifecycle_events(self) -> list[dict[str, Any]]:
        return list(self.events)

    def export_view_gaussian_attribution(self) -> list[dict[str, Any]]:
        return list(self.attribution_rows)

    def export_support_summary(self) -> list[dict[str, Any]]:
        self._refresh_support_counts()
        return [
            self._support_row(self.states[gaussian_id])
            for gaussian_id in sorted(self.states)
        ]

    def validate_consistency(self, num_current_gaussians: int) -> dict[str, Any]:
        alive_ids = [gaussian_id for gaussian_id in self.active_ids if self.states.get(gaussian_id, {}).get("is_alive_final")]
        duplicate_alive = len(alive_ids) != len(set(alive_ids))
        missing_alive = any(gaussian_id not in self.states for gaussian_id in self.active_ids)
        parent_ok = all(
            state["parent_gaussian_id"] == "" or state["parent_gaussian_id"] in self.states
            for state in self.states.values()
        )
        root_ok = all(state["root_gaussian_id"] in self.states for state in self.states.values())
        prune_ok = all(
            row["gaussian_id"] in self.states
            for row in self.events
            if row.get("event_type") == "prune_death"
        )
        validation_errors = []
        if duplicate_alive:
            validation_errors.append("duplicate alive gaussian IDs")
        if missing_alive:
            validation_errors.append("missing alive gaussian IDs")
        if len(self.active_ids) != int(num_current_gaussians):
            validation_errors.append("row count does not match current gaussian count")
        if not parent_ok:
            validation_errors.append("parent IDs missing")
        if not root_ok:
            validation_errors.append("root IDs missing")
        return {
            "schema_name": "viewtrust.pr191.exact_gaussian_lifecycle_logging.validation",
            "schema_version": 1,
            "identity_consistency_passed": not validation_errors,
            "no_duplicate_alive_gaussian_ids": not duplicate_alive,
            "no_missing_alive_gaussian_ids": not missing_alive,
            "row_count_matches_current_gaussian_count": len(self.active_ids) == int(num_current_gaussians),
            "parent_ids_exist_or_empty": parent_ok,
            "root_ids_exist": root_ok,
            "prune_events_reference_existing_ids": prune_ok,
            "exported_files_exist": False,
            "validation_errors": validation_errors,
            "validation_warnings": self.warnings,
        }

    def write_outputs(self, output_dir: Path | None = None) -> dict[str, Any]:
        output_dir = (output_dir or self.output_dir)
        if output_dir is None:
            raise ValueError("output_dir is required")
        output_dir.mkdir(parents=True, exist_ok=True)
        identity_rows = self.export_identity_table()
        event_rows = self.export_lifecycle_events()
        attribution_rows = self.export_view_gaussian_attribution()
        support_rows = self.export_support_summary()
        _write_csv(output_dir / "gaussian_identity_table.csv", identity_rows, IDENTITY_FIELDS)
        _write_csv(output_dir / "gaussian_lifecycle_events.csv", event_rows, EVENT_FIELDS)
        _write_csv(output_dir / "view_gaussian_event_attribution.csv", attribution_rows, ATTRIBUTION_FIELDS)
        _write_csv(output_dir / "gaussian_support_summary.csv", support_rows, SUPPORT_FIELDS)
        validation = self.validate_consistency(len(self.active_ids))
        validation["exported_files_exist"] = all(
            (output_dir / name).is_file()
            for name in [
                "gaussian_identity_table.csv",
                "gaussian_lifecycle_events.csv",
                "view_gaussian_event_attribution.csv",
                "gaussian_support_summary.csv",
            ]
        )
        summary = self._summary(validation)
        _write_json(output_dir / "exact_gaussian_logging_summary.json", summary)
        _write_json(output_dir / "exact_gaussian_logging_validation.json", validation)
        self._write_artifact_manifest(output_dir)
        return summary

    def _new_state(
        self,
        *,
        gaussian_id: int,
        parent_gaussian_id: int | str,
        root_gaussian_id: int | str,
        birth_iteration: int,
        birth_view_name: str,
        birth_event_type: str,
        final_row_index: int | str,
        evidence_quality: str,
    ) -> dict[str, Any]:
        return {
            "gaussian_id": gaussian_id,
            "parent_gaussian_id": parent_gaussian_id,
            "root_gaussian_id": root_gaussian_id,
            "birth_iteration": birth_iteration,
            "birth_view_name": birth_view_name,
            "birth_event_type": birth_event_type,
            "last_seen_iteration": birth_iteration,
            "last_seen_view_name": birth_view_name,
            "prune_iteration": "",
            "prune_view_name": "",
            "death_event_type": "",
            "is_alive_final": True,
            "final_row_index": final_row_index,
            "visible_views": set(),
            "update_views": set(),
            "support_views": set(),
            "support_by_group": Counter(),
            "lifecycle_event_count": 0,
            "visibility_event_count": 0,
            "update_event_count": 0,
            "evidence_quality": evidence_quality,
            "warnings": "",
        }

    def _record_event(self, **kwargs: Any) -> None:
        gaussian_id = kwargs.get("gaussian_id")
        if gaussian_id in self.states:
            state = self.states[gaussian_id]
            state["lifecycle_event_count"] += 1
            if kwargs.get("event_type") == "visibility_observation":
                state["visibility_event_count"] += 1
            if kwargs.get("event_type") == "update_observation":
                state["update_event_count"] += 1
        row = {
            "scene": self.scene,
            "condition": self.condition,
            "subset_name": self.subset_name,
            "run_id": self.run_id,
            "opacity_before": "",
            "opacity_after": "",
            "scale_before": "",
            "scale_after": "",
            "visibility_count_before": "",
            "visibility_count_after": "",
            "gradient_norm_before": "",
            "gradient_norm_after": "",
            **kwargs,
        }
        self.events.append(row)

    def _record_attribution(
        self,
        iteration: int,
        view_name: str,
        gaussian_id: int,
        event_type: str,
        attribution_type: str,
        contribution_value: float,
        view_index: int | str | None = None,
    ) -> None:
        state = self.states[gaussian_id]
        self.attribution_rows.append(
            {
                "scene": self.scene,
                "condition": self.condition,
                "subset_name": self.subset_name,
                "run_id": self.run_id,
                "view_name": view_name,
                "view_index": self._effective_view_index(view_name, view_index),
                "iteration": iteration,
                "gaussian_id": gaussian_id,
                "event_type": event_type,
                "attribution_type": attribution_type,
                "contribution_value": contribution_value,
                "parent_gaussian_id": state["parent_gaussian_id"],
                "root_gaussian_id": state["root_gaussian_id"],
                "is_alive_after_event": state["is_alive_final"],
                "evidence_quality": state["evidence_quality"],
                "notes": "",
            }
        )

    def _record_support(self, gaussian_id: int, view_name: str, kind: str) -> None:
        if not view_name:
            return
        state = self.states[gaussian_id]
        state["support_views"].add(view_name)
        if kind == "visibility":
            state["visible_views"].add(view_name)
        elif kind == "update":
            state["update_views"].add(view_name)
        group = self.view_group_map.get(view_name, "other_clean")
        state["support_by_group"][group] += 1

    def _refresh_final_row_indices(self) -> None:
        for row_index, gaussian_id in enumerate(self.active_ids):
            self.states[gaussian_id]["final_row_index"] = row_index
            self.states[gaussian_id]["is_alive_final"] = True
        dead_ids = set(self.states) - set(self.active_ids)
        for gaussian_id in dead_ids:
            self.states[gaussian_id]["is_alive_final"] = False
            self.states[gaussian_id]["final_row_index"] = ""

    def _refresh_support_counts(self) -> None:
        for state in self.states.values():
            state["total_visible_views"] = len(state["visible_views"])
            state["total_update_views"] = len(state["update_views"])

    def _row_index_for_id(self, gaussian_id: int | str) -> int | str:
        if gaussian_id == "":
            return ""
        try:
            return self.active_ids.index(int(gaussian_id))
        except (ValueError, TypeError):
            return ""

    def _identity_row(self, state: dict[str, Any]) -> dict[str, Any]:
        groups = state["support_by_group"]
        return {
            "scene": self.scene,
            "condition": self.condition,
            "subset_name": self.subset_name,
            "run_id": self.run_id,
            "gaussian_id": state["gaussian_id"],
            "parent_gaussian_id": state["parent_gaussian_id"],
            "root_gaussian_id": state["root_gaussian_id"],
            "birth_iteration": state["birth_iteration"],
            "birth_view_name": state["birth_view_name"],
            "birth_event_type": state["birth_event_type"],
            "prune_iteration": state["prune_iteration"],
            "prune_view_name": state["prune_view_name"],
            "death_event_type": state["death_event_type"],
            "is_alive_final": state["is_alive_final"],
            "final_row_index": state["final_row_index"],
            "total_visible_views": state.get("total_visible_views", 0),
            "total_update_views": state.get("total_update_views", 0),
            "direct_corrupted_support_count": groups.get("direct_corrupted", 0),
            "collateral_support_count": groups.get("co_visible_collateral", 0),
            "clean_prior_support_count": groups.get("clean_prior_demoted", 0),
            "other_clean_support_count": groups.get("other_clean", 0),
            "support_view_names": ";".join(sorted(state["support_views"])),
            "evidence_quality": state["evidence_quality"],
            "warnings": state["warnings"],
        }

    def _support_row(self, state: dict[str, Any]) -> dict[str, Any]:
        groups = state["support_by_group"]
        total = sum(groups.values()) or 0
        probabilities = [count / total for count in groups.values() if total and count > 0]
        entropy = -sum(p * math.log(p) for p in probabilities) if probabilities else 0.0
        max_entropy = math.log(len(probabilities)) if len(probabilities) > 1 else 1.0
        concentration = 1.0 - entropy / max_entropy if probabilities else 0.0
        corrupted_collateral = groups.get("direct_corrupted", 0) + groups.get("co_visible_collateral", 0)
        return {
            "scene": self.scene,
            "condition": self.condition,
            "subset_name": self.subset_name,
            "run_id": self.run_id,
            "gaussian_id": state["gaussian_id"],
            "parent_gaussian_id": state["parent_gaussian_id"],
            "root_gaussian_id": state["root_gaussian_id"],
            "support_view_count": len(state["support_views"]),
            "support_view_names": ";".join(sorted(state["support_views"])),
            "direct_corrupted_support_count": groups.get("direct_corrupted", 0),
            "collateral_support_count": groups.get("co_visible_collateral", 0),
            "clean_prior_support_count": groups.get("clean_prior_demoted", 0),
            "other_clean_support_count": groups.get("other_clean", 0),
            "source_entropy": entropy,
            "source_concentration": max(0.0, min(1.0, concentration)),
            "corrupted_plus_collateral_ratio": corrupted_collateral / total if total else 0.0,
            "clean_prior_ratio": groups.get("clean_prior_demoted", 0) / total if total else 0.0,
            "birth_event_type": state["birth_event_type"],
            "death_event_type": state["death_event_type"],
            "is_alive_final": state["is_alive_final"],
            "total_lifecycle_event_count": state["lifecycle_event_count"],
            "total_visibility_event_count": state["visibility_event_count"],
            "total_update_event_count": state["update_event_count"],
            "evidence_quality": state["evidence_quality"],
            "warnings": state["warnings"],
        }

    def _summary(self, validation: dict[str, Any]) -> dict[str, Any]:
        event_counts = Counter(row["event_type"] for row in self.events)
        qualities = {state["evidence_quality"] for state in self.states.values()}
        return {
            "schema_name": "viewtrust.pr191.exact_gaussian_lifecycle_logging.summary",
            "schema_version": 1,
            "created_at_utc": _utc_now(),
            "scene": self.scene,
            "condition": self.condition,
            "subset_name": self.subset_name,
            "run_id": self.run_id,
            "observation_only": True,
            "training_intervention": False,
            "defense_enabled": False,
            "exact_gaussian_logging_enabled": True,
            "stable_gaussian_ids_enabled": True,
            "uses_row_index_as_stable_id": False,
            "integration_source": self.integration_source,
            "parent_mapping_source": self.parent_mapping_source,
            "total_initial_gaussians": self.initial_gaussian_count,
            "total_final_gaussians": len(self.active_ids),
            "total_unique_gaussian_ids": len(self.states),
            "total_clone_birth_events": event_counts.get("clone_birth", 0),
            "total_split_birth_events": event_counts.get("split_birth", 0),
            "total_unknown_birth_events": event_counts.get("densify_birth_unknown", 0),
            "total_prune_death_events": event_counts.get("prune_death", 0),
            "total_visibility_observations": event_counts.get("visibility_observation", 0),
            "total_update_observations": event_counts.get("update_observation", 0),
            "identity_consistency_passed": validation["identity_consistency_passed"],
            "parent_child_consistency_passed": validation["parent_ids_exist_or_empty"] and validation["root_ids_exist"],
            "prune_consistency_passed": validation["prune_events_reference_existing_ids"],
            "evidence_quality": "partial" if "partial" in qualities else "exact",
            "warnings": self.warnings,
        }

    def _effective_view_index(self, view_name: str, view_index: int | str | None = None) -> int | str:
        if view_index not in (None, ""):
            return view_index
        inferred = _infer_view_index(view_name)
        if inferred is not None:
            return inferred
        if view_name == self.current_view_name:
            return self.current_view_index
        return ""

    def _root_for(self, gaussian_id: int) -> int:
        seen = set()
        current = gaussian_id
        while current not in seen:
            seen.add(current)
            parent = self.states.get(current, {}).get("parent_gaussian_id", "")
            if parent in ("", None):
                return current
            try:
                current = int(parent)
            except (TypeError, ValueError):
                return gaussian_id
            if current not in self.states:
                return gaussian_id
        return gaussian_id

    def _replay_existing_events(self, event_rows: list[dict[str, Any]]) -> None:
        for row in event_rows:
            gid_text = str(row.get("gaussian_id", "") or "")
            if not gid_text.lstrip("-").isdigit():
                continue
            gaussian_id = int(gid_text)
            if gaussian_id not in self.states:
                continue
            state = self.states[gaussian_id]
            event_type = _normalize_event_type(str(row.get("event_type", "") or "unknown"))
            view_name = str(row.get("source_view_name", "") or "")
            iteration = int(float(row.get("iteration") or row.get("source_iteration") or 0))
            if view_name:
                self._record_support(gaussian_id, view_name, "birth" if "birth" in event_type else "update")
            self._record_event(
                iteration=iteration,
                view_name=view_name,
                view_index=self._effective_view_index(view_name),
                event_type=event_type,
                gaussian_id=gaussian_id,
                parent_gaussian_id=state["parent_gaussian_id"],
                root_gaussian_id=state["root_gaussian_id"],
                row_index_before=row.get("source_index", ""),
                row_index_after=row.get("target_index", row.get("final_index", "")),
                opacity_after=row.get("opacity", ""),
                scale_after=row.get("scale_mean", ""),
                is_alive_after_event=state["is_alive_final"],
                event_source="existing_gaussian_lifecycle_events",
                evidence_quality=state["evidence_quality"],
                notes="replayed from real lifecycle runner output",
            )
            self._record_attribution(
                iteration,
                view_name,
                gaussian_id,
                event_type,
                "lifecycle",
                1.0,
                view_index=self._effective_view_index(view_name),
            )

    def _write_artifact_manifest(self, output_dir: Path) -> None:
        rows = []
        for name in [
            "gaussian_identity_table.csv",
            "gaussian_lifecycle_events.csv",
            "view_gaussian_event_attribution.csv",
            "gaussian_support_summary.csv",
            "exact_gaussian_logging_summary.json",
            "exact_gaussian_logging_validation.json",
            "artifact_manifest.csv",
        ]:
            path = output_dir / name
            rows.append(
                {
                    "relative_path": name,
                    "path": str(path),
                    "exists": str(path.exists()).lower(),
                    "file_type": path.suffix.lstrip("."),
                    "size_bytes": path.stat().st_size if path.is_file() else "",
                    "required": "true",
                    "artifact_group": "pr191_exact_gaussian_logging",
                }
            )
        _write_csv(
            output_dir / "artifact_manifest.csv",
            rows,
            ["relative_path", "path", "exists", "file_type", "size_bytes", "required", "artifact_group"],
        )
        rows[-1]["exists"] = "true"
        rows[-1]["size_bytes"] = (output_dir / "artifact_manifest.csv").stat().st_size
        _write_csv(
            output_dir / "artifact_manifest.csv",
            rows,
            ["relative_path", "path", "exists", "file_type", "size_bytes", "required", "artifact_group"],
        )
