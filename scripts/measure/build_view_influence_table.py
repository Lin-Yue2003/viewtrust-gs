#!/usr/bin/env python3
"""Build PR12 view-to-Gaussian influence attribution tables."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

SCHEMA_NAME = "viewtrust.view_influence.summary"
SCHEMA_VERSION = 1


def _json_file(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _csv_iter(path: Path):
    if not path.exists():
        return
    with path.open(newline="", encoding="utf-8") as handle:
        yield from csv.DictReader(handle)


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


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


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _ratio(numerator: int | float | None, denominator: int | float | None) -> float | None:
    return numerator / denominator if numerator is not None and denominator not in (None, 0) else None


def _truthy(value: Any) -> bool:
    return value in (True, "true", "True", "1", 1)


def _bool_or_none(value: Any) -> bool | None:
    if value in ("", None):
        return None
    if _truthy(value):
        return True
    if value in (False, "false", "False", "0", 0):
        return False
    return None


def _view_index_from_name(view_name: str) -> int | str:
    token = str(view_name or "").rsplit("_", 1)[-1]
    try:
        return int(token)
    except ValueError:
        return ""


def _wrapped_payload(payload: dict[str, Any], key: str) -> dict[str, Any]:
    nested = payload.get(key)
    return nested if isinstance(nested, dict) else payload


def _view_split(row: dict[str, str]) -> str:
    split = str(row.get("view_split", "") or "").strip()
    if split in {"train", "test", "target"}:
        return split
    name = str(row.get("view_name", "") or "")
    for candidate in ("train", "test", "target"):
        if name.startswith(f"{candidate}_"):
            return candidate
    return "unknown"


def _view_name_from_metric(row: dict[str, str]) -> str:
    return Path(str(row.get("image_name", ""))).stem


def _resolve_condition_root(
    *,
    run_dir: Path,
    data_root: Path | None,
    scene: str,
    condition: str,
    corruption_condition_root: Path | None,
) -> Path | None:
    metadata = _json_file(run_dir / "metadata.json")
    candidates: list[Path] = []
    if metadata.get("prepared_scene_root"):
        candidates.append(Path(str(metadata["prepared_scene_root"])))
    if corruption_condition_root is not None:
        candidates.append(corruption_condition_root)
    if data_root is not None:
        candidates.append(data_root / "viewtrust-mini" / "nerf_synthetic" / scene / condition)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def _corruption_labels(condition_root: Path | None) -> tuple[dict[str, dict[str, str]], dict[str, Any]]:
    if condition_root is None:
        return {}, {}
    rows = _csv_rows(condition_root / "corruption_manifest.csv")
    summary = _json_file(condition_root / "corruption_summary.json")
    labels = {
        str(row.get("view_name", "")): row
        for row in rows
        if row.get("split") == "train" and row.get("view_name")
    }
    return labels, summary


def _view_metric_by_name(run_dir: Path) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for row in _csv_rows(run_dir / "tables" / "view_metrics.csv"):
        name = _view_name_from_metric(row)
        if name:
            result[name] = row
    return result


def _artifact_rows(run_dir: Path) -> list[dict[str, Any]]:
    paths = [
        ("metadata.json", True, "run_metadata"),
        ("summary.json", True, "run_metadata"),
        ("training_events_summary.json", True, "training_events"),
        ("tables/training_events.csv", True, "training_events"),
        ("tables/densification_events.csv", True, "training_events"),
        ("gaussian_lifecycle_summary.json", True, "gaussian_lifecycle"),
        ("tables/gaussian_lifecycle_events.csv", True, "gaussian_lifecycle"),
        ("tables/gaussian_lifecycle_final.csv", True, "gaussian_lifecycle"),
        ("view_metrics_summary.json", False, "view_metrics"),
        ("tables/view_metrics.csv", False, "view_metrics"),
    ]
    rows = []
    for relative, required, group in paths:
        path = run_dir / relative
        rows.append(
            {
                "relative_path": relative,
                "exists": str(path.exists()).lower(),
                "file_type": path.suffix.lstrip(".") if path.is_file() else "directory" if path.is_dir() else "",
                "size_bytes": path.stat().st_size if path.is_file() else "",
                "required": str(required).lower(),
                "artifact_group": group,
            }
        )
    return rows


def build_view_influence(
    *,
    run_dir: Path,
    data_root: Path | None,
    scene: str,
    condition: str,
    output_dir: Path,
    corruption_condition_root: Path | None,
    require_view_identity: bool,
    require_source_view: bool,
    progress_interval_rows: int = 50000,
    quiet: bool = False,
    enable_exact_gaussian_logging: bool = False,
    exact_gaussian_log_dir: Path | None = None,
    exact_gaussian_logging_config: Path | None = None,
    exact_gaussian_run_id: str | None = None,
    subset_name: str = "",
) -> dict[str, Any]:
    started = time.perf_counter()
    timing: dict[str, float] = {}
    run_dir = run_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_metadata = _json_file(run_dir / "metadata.json")
    raw_command_summary = _json_file(run_dir / "summary.json")
    metadata = _wrapped_payload(raw_metadata, "metadata")
    command_summary = _wrapped_payload(raw_command_summary, "summary")
    training_summary = _json_file(run_dir / "training_events_summary.json")
    lifecycle_summary = _json_file(run_dir / "gaussian_lifecycle_summary.json")
    run_id = metadata.get("run_id") or run_dir.name

    section_started = time.perf_counter()
    all_training_rows = _csv_rows(run_dir / "tables" / "training_events.csv")
    timing["load_training_events_s"] = time.perf_counter() - section_started
    training_rows = [
        row
        for row in all_training_rows
        if row.get("event_type") == "iteration_metrics"
    ]
    missing_view_identity = sum(1 for row in training_rows if not row.get("view_name"))
    if require_view_identity and missing_view_identity:
        raise ValueError(f"missing view identity rows: {missing_view_identity}")

    section_started = time.perf_counter()
    densification_rows = _csv_rows(run_dir / "tables" / "densification_events.csv")
    timing["load_densification_events_s"] = time.perf_counter() - section_started
    densification_by_iteration = {
        str(row.get("iteration", "")): row
        for row in densification_rows
    }

    section_started = time.perf_counter()
    final_rows = _csv_rows(run_dir / "tables" / "gaussian_lifecycle_final.csv")
    timing["load_final_lifecycle_s"] = time.perf_counter() - section_started
    final_by_id = {str(row.get("gaussian_id", "")): row for row in final_rows}

    section_started = time.perf_counter()
    condition_root = _resolve_condition_root(
        run_dir=run_dir,
        data_root=data_root,
        scene=scene,
        condition=condition,
        corruption_condition_root=corruption_condition_root,
    )
    corruption_labels, corruption_summary = _corruption_labels(condition_root)
    timing["load_corruption_manifest_s"] = time.perf_counter() - section_started

    section_started = time.perf_counter()
    view_metrics = _view_metric_by_name(run_dir)
    view_metrics_summary = _json_file(run_dir / "view_metrics_summary.json")
    timing["load_view_metrics_s"] = time.perf_counter() - section_started

    source_event_types = {"clone_birth", "split_birth", "densification_birth", "prune_death"}
    source_lifecycle_rows_count = 0
    lifecycle_event_rows_count = 0
    missing_source_view = 0
    lifecycle_by_iteration: dict[str, Counter[str]] = defaultdict(Counter)
    lifecycle_by_view: dict[str, Counter[str]] = defaultdict(Counter)
    lifecycle_view_iterations: dict[str, set[str]] = defaultdict(set)
    alive_births_by_view: Counter[str] = Counter()
    dead_births_by_view: Counter[str] = Counter()
    attribution_groups: dict[tuple[str, str, str, str, str, str], dict[str, Any]] = {}

    section_started = time.perf_counter()
    lifecycle_path = run_dir / "tables" / "gaussian_lifecycle_events.csv"
    for row in _csv_iter(lifecycle_path) or ():
        lifecycle_event_rows_count += 1
        if (
            progress_interval_rows > 0
            and not quiet
            and lifecycle_event_rows_count % progress_interval_rows == 0
        ):
            print(
                f"processed lifecycle events: {lifecycle_event_rows_count} rows",
                file=sys.stderr,
            )
        event_type = str(row.get("event_type", ""))
        if event_type not in source_event_types:
            continue
        source_lifecycle_rows_count += 1
        source_view_name = str(row.get("source_view_name", "") or "")
        if not source_view_name:
            missing_source_view += 1
            continue
        source_view_split = str(row.get("source_view_split", "") or "")
        source_iteration = str(row.get("source_iteration") or row.get("iteration") or "")
        label = corruption_labels.get(source_view_name, {})
        was_corrupted = str(label.get("was_corrupted", "false"))
        corruption_type = str(label.get("corruption_type", ""))

        lifecycle_by_iteration[source_iteration][event_type] += 1
        if "birth" in event_type:
            lifecycle_by_iteration[source_iteration]["birth"] += 1

        lifecycle_by_view[source_view_name][event_type] += 1
        if "birth" in event_type:
            lifecycle_by_view[source_view_name]["birth"] += 1
        if source_iteration:
            lifecycle_view_iterations[source_view_name].add(source_iteration)

        gaussian_id = str(row.get("gaussian_id", "") or "")
        if "birth" in event_type and gaussian_id:
            final_row = final_by_id.get(gaussian_id, {})
            alive_value = final_row.get("alive")
            if _truthy(alive_value):
                alive_births_by_view[source_view_name] += 1
            elif alive_value not in ("", None):
                dead_births_by_view[source_view_name] += 1

        key = (
            source_view_name,
            source_view_split,
            was_corrupted,
            corruption_type,
            source_iteration,
            event_type,
        )
        group = attribution_groups.setdefault(
            key,
            {
                "event_count": 0,
                "clone_birth_count": 0,
                "split_birth_count": 0,
                "prune_death_count": 0,
                "gaussian_ids": set(),
            },
        )
        group["event_count"] += 1
        if event_type in {"clone_birth", "split_birth", "prune_death"}:
            group[f"{event_type}_count"] += 1
        if gaussian_id:
            group["gaussian_ids"].add(gaussian_id)
    timing["stream_lifecycle_events_s"] = time.perf_counter() - section_started

    if require_source_view and missing_source_view:
        raise ValueError(f"missing lifecycle source view rows: {missing_source_view}")

    iteration_rows: list[dict[str, Any]] = []
    samples_by_view: dict[str, list[dict[str, str]]] = defaultdict(list)
    sampled_view_splits: dict[str, str] = {}
    for row in training_rows:
        view_name = str(row.get("view_name", ""))
        if not view_name:
            continue
        samples_by_view[view_name].append(row)
        sampled_view_splits.setdefault(view_name, _view_split(row))
        iteration = str(row.get("iteration", ""))
        densification = densification_by_iteration.get(iteration, {})
        event_counts = lifecycle_by_iteration.get(iteration, Counter())
        label = corruption_labels.get(view_name, {})
        iteration_rows.append(
            {
                "run_id": run_id,
                "iteration": iteration,
                "view_name": view_name,
                "view_split": row.get("view_split", ""),
                "was_corrupted": label.get("was_corrupted", "false"),
                "corruption_type": label.get("corruption_type", ""),
                "gaussian_count": row.get("gaussian_count", ""),
                "visible_gaussian_count": row.get("visible_gaussian_count", ""),
                "visibility_ratio": row.get("visibility_ratio", ""),
                "radii_nonzero_count": row.get("radii_nonzero_count", ""),
                "l1_loss": row.get("l1_loss", ""),
                "ssim_loss": row.get("ssim", ""),
                "total_loss": row.get("loss", ""),
                "densification_triggered": row.get("densification_triggered", ""),
                "gaussian_count_before_densification": densification.get("gaussian_count_before", ""),
                "gaussian_count_after_densification": densification.get("gaussian_count_after", ""),
                "gaussian_count_delta_densification": densification.get("gaussian_count_delta", ""),
                "birth_event_count": event_counts.get("birth", 0),
                "clone_birth_count": event_counts.get("clone_birth", 0),
                "split_birth_count": event_counts.get("split_birth", 0),
                "prune_death_count": event_counts.get("prune_death", 0),
            }
        )

    split_counts = Counter(sampled_view_splits.values())
    unexpected_non_train_sampled_views = sorted(
        view_name
        for view_name, split in sampled_view_splits.items()
        if split != "train"
    )
    all_view_names = sorted(set(samples_by_view) | set(corruption_labels) | set(lifecycle_by_view))
    view_rows: list[dict[str, Any]] = []
    attribution_rows: list[dict[str, Any]] = []
    for view_name in all_view_names:
        samples = samples_by_view.get(view_name, [])
        label = corruption_labels.get(view_name, {})
        event_counts = lifecycle_by_view.get(view_name, Counter())
        iterations = [_int_or_none(row.get("iteration")) for row in samples]
        numeric_iterations = [value for value in iterations if value is not None]
        visibility_values = [
            value
            for value in (_float_or_none(row.get("visibility_ratio")) for row in samples)
            if value is not None
        ]
        visible_counts = [
            value
            for value in (_float_or_none(row.get("visible_gaussian_count")) for row in samples)
            if value is not None
        ]
        radii_counts = [
            value
            for value in (_float_or_none(row.get("radii_nonzero_count")) for row in samples)
            if value is not None
        ]
        metric = view_metrics.get(view_name, {})
        view_rows.append(
            {
                "scene": scene,
                "condition": condition,
                "run_id": run_id,
                "view_name": view_name,
                "view_split": samples[0].get("view_split", "train") if samples else "train",
                "was_corrupted": label.get("was_corrupted", "false"),
                "corruption_type": label.get("corruption_type", ""),
                "times_sampled": len(samples),
                "first_sample_iteration": min(numeric_iterations) if numeric_iterations else "",
                "last_sample_iteration": max(numeric_iterations) if numeric_iterations else "",
                "mean_loss": _mean([v for v in (_float_or_none(row.get("loss")) for row in samples) if v is not None]),
                "mean_l1_loss": _mean([v for v in (_float_or_none(row.get("l1_loss")) for row in samples) if v is not None]),
                "mean_ssim_loss": _mean([v for v in (_float_or_none(row.get("ssim")) for row in samples) if v is not None]),
                "mean_total_loss": _mean([v for v in (_float_or_none(row.get("loss")) for row in samples) if v is not None]),
                "mean_gaussian_count": _mean([v for v in (_float_or_none(row.get("gaussian_count")) for row in samples) if v is not None]),
                "mean_visible_gaussian_count": _mean(visible_counts),
                "mean_visibility_ratio": _mean(visibility_values),
                "max_visible_gaussian_count": max(visible_counts) if visible_counts else "",
                "max_visibility_ratio": max(visibility_values) if visibility_values else "",
                "mean_radii_nonzero_count": _mean(radii_counts),
                "max_radii_nonzero_count": max(radii_counts) if radii_counts else "",
                "densification_context_count": len(lifecycle_view_iterations.get(view_name, set())),
                "birth_event_count_after_view": event_counts.get("birth", 0),
                "clone_birth_count_after_view": event_counts.get("clone_birth", 0),
                "split_birth_count_after_view": event_counts.get("split_birth", 0),
                "prune_death_count_after_view": event_counts.get("prune_death", 0),
                "final_survivor_birth_count_after_view": alive_births_by_view.get(view_name, 0),
                "dead_birth_count_after_view": dead_births_by_view.get(view_name, 0),
                "birth_survival_ratio_after_view": _ratio(
                    alive_births_by_view.get(view_name, 0),
                    event_counts.get("birth", 0),
                ),
                "mean_view_psnr": _float_or_none(metric.get("psnr")),
                "mean_view_ssim": _float_or_none(metric.get("ssim")),
                "mean_view_l1": _float_or_none(metric.get("l1_mean")),
                "warnings": "",
            }
        )

    for key, group in sorted(attribution_groups.items()):
        view_name, source_view_split, was_corrupted, corruption_type, iteration, event_type = key
        gaussian_ids = group["gaussian_ids"]
        final_alive = sum(
            1
            for gid in gaussian_ids
            if _truthy(final_by_id.get(str(gid), {}).get("alive"))
        )
        final_dead = sum(
            1
            for gid in gaussian_ids
            if str(gid) in final_by_id and not _truthy(final_by_id[str(gid)].get("alive"))
        )
        attribution_rows.append(
            {
                "run_id": run_id,
                "scene": scene,
                "condition": condition,
                "source_view_name": view_name,
                "source_view_split": source_view_split,
                "was_corrupted": was_corrupted,
                "corruption_type": corruption_type,
                "source_iteration": iteration,
                "event_type": event_type,
                "lifecycle_action": event_type,
                "event_count": group["event_count"],
                "clone_birth_count": group["clone_birth_count"],
                "split_birth_count": group["split_birth_count"],
                "prune_death_count": group["prune_death_count"],
                "unique_gaussian_count": len(gaussian_ids),
                "final_alive_count": final_alive,
                "final_dead_count": final_dead,
            }
        )

    view_fields = [
        "scene", "condition", "run_id", "view_name", "view_split", "was_corrupted",
        "corruption_type", "times_sampled", "first_sample_iteration", "last_sample_iteration",
        "mean_loss", "mean_l1_loss", "mean_ssim_loss", "mean_total_loss",
        "mean_gaussian_count", "mean_visible_gaussian_count", "mean_visibility_ratio",
        "max_visible_gaussian_count", "max_visibility_ratio", "mean_radii_nonzero_count",
        "max_radii_nonzero_count", "densification_context_count",
        "birth_event_count_after_view", "clone_birth_count_after_view",
        "split_birth_count_after_view", "prune_death_count_after_view",
        "final_survivor_birth_count_after_view", "dead_birth_count_after_view",
        "birth_survival_ratio_after_view", "mean_view_psnr", "mean_view_ssim",
        "mean_view_l1", "warnings",
    ]
    attribution_fields = [
        "run_id", "scene", "condition", "source_view_name", "source_view_split",
        "was_corrupted", "corruption_type", "source_iteration", "event_type",
        "lifecycle_action", "event_count", "clone_birth_count", "split_birth_count",
        "prune_death_count", "unique_gaussian_count", "final_alive_count", "final_dead_count",
    ]
    iteration_fields = [
        "run_id", "iteration", "view_name", "view_split", "was_corrupted",
        "corruption_type", "gaussian_count", "visible_gaussian_count",
        "visibility_ratio", "radii_nonzero_count", "l1_loss", "ssim_loss",
        "total_loss", "densification_triggered", "gaussian_count_before_densification",
        "gaussian_count_after_densification", "gaussian_count_delta_densification",
        "birth_event_count", "clone_birth_count", "split_birth_count", "prune_death_count",
    ]
    corrupted_view_rows = [row for row in view_rows if row["was_corrupted"] == "true"]
    uncorrupted_view_rows = [row for row in view_rows if row["was_corrupted"] != "true"]

    def group_mean(rows: list[dict[str, Any]], field: str) -> float | None:
        return _mean([v for v in (_float_or_none(row.get(field)) for row in rows) if v is not None])

    top_prune = sorted(
        view_rows,
        key=lambda row: _int_or_none(row.get("prune_death_count_after_view")) or 0,
        reverse=True,
    )[:10]
    top_birth = sorted(
        view_rows,
        key=lambda row: _int_or_none(row.get("birth_event_count_after_view")) or 0,
        reverse=True,
    )[:10]
    low_survival = sorted(
        [
            row
            for row in view_rows
            if _float_or_none(row.get("birth_survival_ratio_after_view")) is not None
        ],
        key=lambda row: _float_or_none(row.get("birth_survival_ratio_after_view")) or 0.0,
    )[:10]
    warnings: list[str] = []
    if not corruption_summary and condition != "clean":
        warnings.append("corruption summary unavailable")
    if missing_view_identity:
        warnings.append(f"missing view identity rows: {missing_view_identity}")
    if missing_source_view:
        warnings.append(f"missing lifecycle source view rows: {missing_source_view}")
    if unexpected_non_train_sampled_views:
        warnings.append(
            "Training iteration metrics include non-train views. Use --eval during "
            "official 3DGS training to keep test cameras held out."
        )

    observation_only_sources = {
        "metadata": _bool_or_none(metadata.get("observation_only")),
        "summary": _bool_or_none(command_summary.get("observation_only")),
        "training_events": _bool_or_none(training_summary.get("observation_only")),
        "gaussian_lifecycle": _bool_or_none(lifecycle_summary.get("observation_only")),
    }
    known_observation_only_values = {
        value for value in observation_only_sources.values() if value is not None
    }
    if len(known_observation_only_values) > 1:
        warnings.append("observation_only source values disagree")
    observation_only = any(value is True for value in observation_only_sources.values())

    exact_gaussian_summary: dict[str, Any] = {}
    if enable_exact_gaussian_logging:
        if exact_gaussian_log_dir is None:
            raise ValueError("--exact-gaussian-log-dir is required when exact Gaussian logging is enabled")
        exact_config = _json_file(exact_gaussian_logging_config) if exact_gaussian_logging_config else {}
        if exact_config.get("labels", {}).get("use_corruption_labels_for_scoring") is True:
            raise ValueError("exact Gaussian logging config must not use corruption labels for scoring")
        exact_gaussian_summary = _write_exact_gaussian_logs_from_existing_lifecycle(
            run_dir=run_dir,
            output_dir=exact_gaussian_log_dir,
            scene=scene,
            condition=condition,
            subset_name=subset_name,
            run_id=exact_gaussian_run_id or f"{run_id}-exact-gaussian",
            corruption_labels=corruption_labels,
        )

    section_started = time.perf_counter()
    _write_csv(output_dir / "view_influence.csv", view_rows, view_fields)
    _write_csv(output_dir / "view_lifecycle_attribution.csv", attribution_rows, attribution_fields)
    _write_csv(output_dir / "view_iteration_events.csv", iteration_rows, iteration_fields)
    _write_csv(
        output_dir / "view_influence_artifact_manifest.csv",
        _artifact_rows(run_dir),
        ["relative_path", "exists", "file_type", "size_bytes", "required", "artifact_group"],
    )
    timing["write_outputs_s"] = time.perf_counter() - section_started

    runtime_s = time.perf_counter() - started
    camera_pool_view_count = _int_or_none(view_metrics_summary.get("view_count_total"))
    if camera_pool_view_count is None:
        camera_pool_view_count = len(all_view_names)
    summary = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "run_dir": str(run_dir),
        "run_id": run_id,
        "scene": scene,
        "condition": condition,
        "runtime_s": runtime_s,
        "timing": timing,
        "input_rows": {
            "training_event_rows": len(all_training_rows),
            "densification_event_rows": len(densification_rows),
            "lifecycle_event_rows": lifecycle_event_rows_count,
            "final_lifecycle_rows": len(final_rows),
            "view_metric_rows": len(view_metrics),
        },
        "throughput": {
            "lifecycle_events_per_second": _ratio(
                lifecycle_event_rows_count,
                timing.get("stream_lifecycle_events_s"),
            ),
            "final_lifecycle_rows_per_second": _ratio(
                len(final_rows),
                timing.get("load_final_lifecycle_s"),
            ),
        },
        "observation_only": observation_only,
        "exact_gaussian_logging": {
            "enabled": enable_exact_gaussian_logging,
            "output_dir": str(exact_gaussian_log_dir) if exact_gaussian_log_dir else "",
            "summary_schema": exact_gaussian_summary.get("schema_name", ""),
            "evidence_quality": exact_gaussian_summary.get("evidence_quality", ""),
            "integration_source": exact_gaussian_summary.get("integration_source", ""),
            "parent_mapping_source": exact_gaussian_summary.get("parent_mapping_source", ""),
        },
        "observation_only_sources": observation_only_sources,
        "training_event_invariants_ok": (_int_or_none(training_summary.get("invalid_training_event_rows")) or 0) == 0,
        "gaussian_lifecycle_invariants_ok": (_int_or_none(lifecycle_summary.get("invariant_violations")) or 0) == 0,
        "view_identity_available": bool(training_rows) and missing_view_identity == 0,
        "lifecycle_source_view_available": source_lifecycle_rows_count > 0 and missing_source_view == 0,
        "corruption_summary_available": bool(corruption_summary),
        "corrupted_view_count": len([row for row in corruption_labels.values() if row.get("was_corrupted") == "true"]),
        "camera_pool_view_count": camera_pool_view_count,
        "training_view_count": split_counts.get("train", 0),
        "views_sampled_count": len(samples_by_view),
        "sampled_train_view_count": split_counts.get("train", 0),
        "sampled_test_view_count": split_counts.get("test", 0),
        "sampled_target_view_count": split_counts.get("target", 0),
        "sampled_unknown_view_count": split_counts.get("unknown", 0),
        "unexpected_non_train_sampled_view_count": len(unexpected_non_train_sampled_views),
        "unexpected_non_train_sampled_views": unexpected_non_train_sampled_views,
        "total_iteration_metric_rows": len(training_rows),
        "total_lifecycle_event_rows": source_lifecycle_rows_count,
        "top_views_by_prune_death_count": [
            {
                "view_name": row["view_name"],
                "was_corrupted": row["was_corrupted"],
                "prune_death_count_after_view": row["prune_death_count_after_view"],
            }
            for row in top_prune
        ],
        "top_views_by_birth_event_count": [
            {
                "view_name": row["view_name"],
                "was_corrupted": row["was_corrupted"],
                "birth_event_count_after_view": row["birth_event_count_after_view"],
            }
            for row in top_birth
        ],
        "top_views_by_low_birth_survival_ratio": [
            {
                "view_name": row["view_name"],
                "was_corrupted": row["was_corrupted"],
                "birth_survival_ratio_after_view": row["birth_survival_ratio_after_view"],
            }
            for row in low_survival
        ],
        "corrupted_views_mean_prune_death_count": group_mean(corrupted_view_rows, "prune_death_count_after_view"),
        "uncorrupted_views_mean_prune_death_count": group_mean(uncorrupted_view_rows, "prune_death_count_after_view"),
        "corrupted_views_mean_birth_event_count": group_mean(corrupted_view_rows, "birth_event_count_after_view"),
        "uncorrupted_views_mean_birth_event_count": group_mean(uncorrupted_view_rows, "birth_event_count_after_view"),
        "corrupted_views_mean_birth_survival_ratio": group_mean(corrupted_view_rows, "birth_survival_ratio_after_view"),
        "uncorrupted_views_mean_birth_survival_ratio": group_mean(uncorrupted_view_rows, "birth_survival_ratio_after_view"),
        "warnings": warnings,
    }
    (output_dir / "view_influence_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _write_exact_gaussian_logs_from_existing_lifecycle(
    *,
    run_dir: Path,
    output_dir: Path,
    scene: str,
    condition: str,
    subset_name: str,
    run_id: str,
    corruption_labels: dict[str, dict[str, str]],
) -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from viewtrust.instrumentation.gaussian_identity_tracker import GaussianIdentityTracker

    event_rows = _csv_rows(run_dir / "tables" / "gaussian_lifecycle_events.csv")
    final_rows = _csv_rows(run_dir / "tables" / "gaussian_lifecycle_final.csv")
    view_group_map = {
        view_name: "direct_corrupted" if _truthy(label.get("was_corrupted")) else "other_clean"
        for view_name, label in corruption_labels.items()
    }
    tracker = GaussianIdentityTracker.from_existing_lifecycle_tables(
        event_rows=event_rows,
        final_rows=final_rows,
        scene=scene,
        condition=condition,
        subset_name=subset_name,
        run_id=run_id,
        view_group_map=view_group_map,
        output_dir=output_dir,
        integration_source="real_view_influence_runner",
    )
    for row in _csv_rows(run_dir / "tables" / "training_events.csv"):
        if row.get("event_type") != "iteration_metrics":
            continue
        view_name = str(row.get("view_name", "") or "")
        if not view_name:
            continue
        iteration = _int_or_none(row.get("iteration")) or 0
        active_count = len(tracker.active_ids)
        if active_count <= 0:
            continue
        visible_count = min(_int_or_none(row.get("visible_gaussian_count")) or 0, active_count)
        gaussian_count = _int_or_none(row.get("gaussian_count")) or active_count
        tracker.record_visibility_observation(
            list(range(visible_count)),
            iteration,
            view_name,
            view_index=_view_index_from_name(view_name),
        )
        tracker.record_update_observation(
            list(range(min(max(visible_count, 1), active_count))),
            iteration,
            view_name,
            view_index=_view_index_from_name(view_name),
            metadata={
                "notes": (
                    "real view influence runner proxy update observation; "
                    f"gaussian_count={gaussian_count}"
                )
            },
        )
    summary = tracker.write_outputs(output_dir)
    validation = _json_file(output_dir / "exact_gaussian_logging_validation.json")
    if not validation.get("identity_consistency_passed"):
        raise ValueError("exact Gaussian logging validation failed during view influence integration")
    return summary


def _markdown(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# View Influence Attribution Report",
            "",
            "PR12 reports observation-only temporal/source-view attribution. It is not exact per-pixel contribution attribution and not a trust score.",
            "",
            "## Run Identity",
            f"- Run ID: `{summary.get('run_id')}`",
            f"- Run dir: `{summary.get('run_dir')}`",
            f"- Scene: `{summary.get('scene')}`",
            f"- Condition: `{summary.get('condition')}`",
            "",
            "## Coverage",
            f"- View identity available: `{summary.get('view_identity_available')}`",
            f"- Lifecycle source view available: `{summary.get('lifecycle_source_view_available')}`",
            f"- Views sampled: `{summary.get('views_sampled_count')}`",
            f"- Sampled train/test/target/unknown: `{summary.get('sampled_train_view_count')}` / `{summary.get('sampled_test_view_count')}` / `{summary.get('sampled_target_view_count')}` / `{summary.get('sampled_unknown_view_count')}`",
            f"- Unexpected non-train sampled views: `{summary.get('unexpected_non_train_sampled_view_count')}`",
            f"- Lifecycle source events: `{summary.get('total_lifecycle_event_rows')}`",
            f"- Input rows: `{summary.get('input_rows')}`",
            f"- Runtime seconds: `{summary.get('runtime_s')}`",
            f"- Throughput: `{summary.get('throughput')}`",
            "",
            "## Observation Mode",
            f"- Observation only: `{summary.get('observation_only')}`",
            f"- Observation-only sources: `{summary.get('observation_only_sources')}`",
            "",
            "## Corruption Labels",
            f"- Corruption summary available: `{summary.get('corruption_summary_available')}`",
            f"- Corrupted view count: `{summary.get('corrupted_view_count')}`",
            "",
            "## Top Views By Lifecycle Influence",
            "- See `view_influence.csv` and `view_lifecycle_attribution.csv`.",
            "",
            "## Corrupted vs Uncorrupted Aggregate",
            f"- Corrupted mean prune deaths: `{summary.get('corrupted_views_mean_prune_death_count')}`",
            f"- Uncorrupted mean prune deaths: `{summary.get('uncorrupted_views_mean_prune_death_count')}`",
            f"- Corrupted mean births: `{summary.get('corrupted_views_mean_birth_event_count')}`",
            f"- Uncorrupted mean births: `{summary.get('uncorrupted_views_mean_birth_event_count')}`",
            f"- Corrupted mean birth survival ratio: `{summary.get('corrupted_views_mean_birth_survival_ratio')}`",
            f"- Uncorrupted mean birth survival ratio: `{summary.get('uncorrupted_views_mean_birth_survival_ratio')}`",
            "",
            "## Interpretation Guidance",
            "- Use phrases like view-associated lifecycle influence and observation-only attribution.",
            "- Do not interpret these tables as detection, classification, defense success, or a ViewTrust score.",
            "",
            "## Known Limitations",
            "- Lifecycle events are associated with the sampled view active during densification/pruning context.",
            "- This does not prove that the view alone caused the event.",
            "",
            "## Warnings",
            *[f"- {warning}" for warning in summary.get("warnings", [])],
            "" if summary.get("warnings") else "- None",
            "",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--scene", default="chair")
    parser.add_argument("--condition", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--corruption-condition-root", type=Path)
    parser.add_argument("--require-view-identity", action="store_true")
    parser.add_argument("--require-source-view", action="store_true")
    parser.add_argument("--progress-interval-rows", type=int, default=50000)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--write-markdown", action="store_true")
    parser.add_argument("--enable-exact-gaussian-logging", action="store_true")
    parser.add_argument("--exact-gaussian-log-dir", type=Path)
    parser.add_argument(
        "--exact-gaussian-logging-config",
        type=Path,
        default=Path("configs/offline_viewtrust_signal/default_pr191_exact_gaussian_logging.json"),
    )
    parser.add_argument("--exact-gaussian-run-id")
    parser.add_argument("--subset-name", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        summary = build_view_influence(
            run_dir=args.run_dir,
            data_root=args.data_root,
            scene=args.scene,
            condition=args.condition,
            output_dir=args.output_dir,
            corruption_condition_root=args.corruption_condition_root,
            require_view_identity=args.require_view_identity,
            require_source_view=args.require_source_view,
            progress_interval_rows=args.progress_interval_rows,
            quiet=args.quiet,
            enable_exact_gaussian_logging=args.enable_exact_gaussian_logging,
            exact_gaussian_log_dir=args.exact_gaussian_log_dir,
            exact_gaussian_logging_config=args.exact_gaussian_logging_config,
            exact_gaussian_run_id=args.exact_gaussian_run_id,
            subset_name=args.subset_name,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    if args.write_markdown:
        (args.output_dir / "view_influence_report.md").write_text(
            _markdown(summary),
            encoding="utf-8",
        )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
