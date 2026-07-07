#!/usr/bin/env python3
"""Compare clean and natural-corruption observed training runs."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

SCHEMA_NAME = "viewtrust.clean_vs_corrupt.summary"
SCHEMA_VERSION = 1


def _json_file(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _summary(run_dir: Path) -> dict[str, Any]:
    payload = _json_file(run_dir / "summary.json")
    return payload.get("summary", payload) if isinstance(payload, dict) else {}


def _metadata(run_dir: Path) -> dict[str, Any]:
    return _json_file(run_dir / "metadata.json")


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


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _max_from_csv(path: Path, field: str) -> float | None:
    values = [
        value
        for value in (_float_or_none(row.get(field)) for row in _read_csv_rows(path))
        if value is not None
    ]
    return max(values) if values else None


def _count_csv_rows(path: Path) -> int | None:
    if not path.exists():
        return None
    return len(_read_csv_rows(path))


def _find_point_cloud_path(run_dir: Path, requested_iterations: int | None) -> Path | None:
    point_root = run_dir / "trainer_output" / "point_cloud"
    if not point_root.exists():
        return None
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
    for path in candidates:
        if path.is_file():
            return path
    return None


def read_ply_vertex_count(path: Path) -> int | None:
    try:
        with path.open("rb") as handle:
            for _ in range(256):
                raw_line = handle.readline()
                if not raw_line:
                    break
                line = raw_line.decode("utf-8", errors="replace").strip()
                parts = line.split()
                if len(parts) == 3 and parts[0] == "element" and parts[1] == "vertex":
                    return _int_or_none(parts[2])
                if line == "end_header":
                    break
    except OSError:
        return None
    return None


def _final_gaussian_count(
    run_dir: Path,
    training_events: dict[str, Any],
    lifecycle: dict[str, Any],
    requested_iterations: int | None,
) -> int | None:
    for value in (
        training_events.get("final_gaussian_count"),
        lifecycle.get("final_gaussian_count"),
        lifecycle.get("alive_final_count"),
        _json_file(run_dir / "training_dynamics_summary.json").get("final_gaussian_count"),
    ):
        parsed = _int_or_none(value)
        if parsed is not None:
            return parsed
    point_cloud = _find_point_cloud_path(run_dir, requested_iterations)
    return read_ply_vertex_count(point_cloud) if point_cloud else None


def _training_event_invariants_ok(training_events: dict[str, Any]) -> bool:
    return bool(training_events) and _int_or_none(
        training_events.get("invalid_training_event_rows")
    ) == 0


def _lifecycle_invariants_ok(lifecycle: dict[str, Any]) -> bool:
    if not lifecycle:
        return False
    alive = _int_or_none(lifecycle.get("alive_final_count"))
    dead = _int_or_none(lifecycle.get("dead_final_count"))
    known = _int_or_none(lifecycle.get("known_gaussian_count"))
    final = _int_or_none(lifecycle.get("final_gaussian_count"))
    violations = _int_or_none(lifecycle.get("invariant_violations")) or 0
    return (
        violations == 0
        and (alive is None or final is None or alive == final)
        and (alive is None or dead is None or known is None or alive + dead == known)
    )


def _view_metric(summary: dict[str, Any], split: str, metric: str) -> float | None:
    metrics = summary.get("metrics", {})
    if not isinstance(metrics, dict):
        return None
    split_metrics = metrics.get(split, {})
    if not isinstance(split_metrics, dict):
        return None
    return _float_or_none(split_metrics.get(metric))


def _mean_view_metric(summary: dict[str, Any], metric: str) -> float | None:
    values = [
        value
        for split in ("train", "test", "target")
        if (value := _view_metric(summary, split, metric)) is not None
    ]
    return sum(values) / len(values) if values else None


def _run_info(run_dir: Path) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    metadata = _metadata(run_dir)
    command_summary = _summary(run_dir)
    training_events = _json_file(run_dir / "training_events_summary.json")
    lifecycle = _json_file(run_dir / "gaussian_lifecycle_summary.json")
    view_metrics = _json_file(run_dir / "view_metrics_summary.json")
    requested_iterations = _requested_iterations(metadata, training_events)
    final_count = _final_gaussian_count(
        run_dir,
        training_events,
        lifecycle,
        requested_iterations,
    )
    return {
        "run_dir": str(run_dir),
        "path": run_dir,
        "run_id": metadata.get("run_id") or run_dir.name,
        "metadata": metadata,
        "summary": command_summary,
        "training_events": training_events,
        "lifecycle": lifecycle,
        "view_metrics": view_metrics,
        "scene": metadata.get("scene") or training_events.get("scene") or lifecycle.get("scene"),
        "condition": metadata.get("condition")
        or training_events.get("condition")
        or lifecycle.get("condition")
        or view_metrics.get("condition"),
        "trainer": metadata.get("trainer") or training_events.get("trainer") or lifecycle.get("trainer"),
        "requested_iterations": requested_iterations,
        "success": command_summary.get("returncode") == 0,
        "observation_only": command_summary.get("observation_only") is True,
        "elapsed_s": _float_or_none(command_summary.get("elapsed_s")),
        "gpu_peak_mb": _gpu_peak_mb(run_dir),
        "final_gaussian_count": final_count,
        "training_event_invariants_ok": _training_event_invariants_ok(training_events),
        "gaussian_lifecycle_invariants_ok": _lifecycle_invariants_ok(lifecycle),
        "has_view_metrics": bool(view_metrics),
    }


def _delta(corrupt: int | float | None, clean: int | float | None) -> int | float | None:
    return corrupt - clean if clean is not None and corrupt is not None else None


def _ratio(delta: int | float | None, clean: int | float | None) -> float | None:
    return delta / clean if delta is not None and clean not in (None, 0) else None


def _metric_pair(
    name: str,
    clean: int | float | None,
    corrupt: int | float | None,
    group: str,
) -> dict[str, Any]:
    delta = _delta(corrupt, clean)
    return {
        "metric_group": group,
        "metric": name,
        "clean": clean,
        "corrupt": corrupt,
        "delta": delta,
        "delta_ratio": _ratio(delta, clean),
    }


def _artifact_type(path: Path) -> str:
    if path.is_dir():
        return "directory"
    suffix = path.suffix.lower()
    if suffix in {".json", ".csv", ".jsonl", ".log", ".ply", ".md"}:
        return suffix[1:]
    return "file"


def _artifact_row(run_dir: Path, run_label: str, relative_path: str, required: bool, group: str) -> dict[str, Any]:
    path = run_dir / relative_path
    exists = path.exists()
    return {
        "run_label": run_label,
        "relative_path": relative_path,
        "exists": str(exists).lower(),
        "file_type": _artifact_type(path) if exists else "",
        "size_bytes": path.stat().st_size if exists and path.is_file() else "",
        "required": str(required).lower(),
        "artifact_group": group,
    }


def _artifact_rows(clean_run_dir: Path, corrupt_run_dir: Path) -> list[dict[str, Any]]:
    required = [
        ("metadata.json", "run_metadata"),
        ("summary.json", "run_metadata"),
        ("training_events_summary.json", "training_events"),
        ("tables/training_events.csv", "training_events"),
        ("tables/densification_events.csv", "training_events"),
        ("tables/gaussian_count_timeseries.csv", "training_events"),
        ("gaussian_lifecycle_summary.json", "gaussian_lifecycle"),
        ("tables/gaussian_lifecycle_events.csv", "gaussian_lifecycle"),
        ("tables/gaussian_lifecycle_final.csv", "gaussian_lifecycle"),
    ]
    optional = [
        ("view_metrics_summary.json", "view_metrics"),
        ("tables/view_metrics.csv", "view_metrics"),
        ("tables/view_render_artifacts.csv", "view_metrics"),
    ]
    rows: list[dict[str, Any]] = []
    for label, root in (("clean", clean_run_dir), ("corrupt", corrupt_run_dir)):
        rows.extend(_artifact_row(root, label, path, True, group) for path, group in required)
        rows.extend(_artifact_row(root, label, path, False, group) for path, group in optional)
    return rows


def _resolve_corrupt_condition_root(
    corrupt_run: dict[str, Any],
    *,
    data_root: Path | None,
    scene: str,
    corruption_condition: str,
    corrupt_condition_root: Path | None,
) -> Path | None:
    candidates: list[Path] = []
    prepared = corrupt_run["metadata"].get("prepared_scene_root")
    if prepared:
        candidates.append(Path(str(prepared)))
    if corrupt_condition_root is not None:
        candidates.append(corrupt_condition_root)
    if data_root is not None:
        candidates.append(
            data_root
            / "viewtrust-mini"
            / "nerf_synthetic"
            / scene
            / corruption_condition
        )
    for candidate in candidates:
        if candidate and candidate.exists():
            return candidate.resolve()
    return None


def _corruption_artifacts(root: Path | None) -> dict[str, Any]:
    if root is None:
        return {
            "root": None,
            "summary": {},
            "manifest": {},
            "csv_rows": [],
            "available": False,
        }
    summary = _json_file(root / "corruption_summary.json")
    manifest = _json_file(root / "corruption_manifest.json")
    rows = _read_csv_rows(root / "corruption_manifest.csv")
    training_manifest = _json_file(root / "manifest.json")
    return {
        "root": root,
        "summary": summary,
        "manifest": manifest,
        "training_manifest": training_manifest,
        "csv_rows": rows,
        "available": bool(summary),
    }


def _view_key(row: dict[str, str]) -> tuple[str, str]:
    split = str(row.get("split", ""))
    image_name = str(row.get("image_name", ""))
    view_name = Path(image_name).stem
    return split, view_name


def _view_metric_rows(run_dir: Path) -> dict[tuple[str, str], dict[str, str]]:
    rows = _read_csv_rows(run_dir / "tables" / "view_metrics.csv")
    return {_view_key(row): row for row in rows}


def _corruption_by_view(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    result: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        result[(str(row.get("split", "")), str(row.get("view_name", "")))] = row
    return result


def _view_corruption_effect_rows(
    *,
    clean_run_dir: Path,
    corrupt_run_dir: Path,
    corruption_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    clean_rows = _view_metric_rows(clean_run_dir)
    corrupt_rows = _view_metric_rows(corrupt_run_dir)
    corruption_by_view = _corruption_by_view(corruption_rows)
    if not clean_rows or not corrupt_rows or not corruption_by_view:
        return []
    output: list[dict[str, Any]] = []
    for key in sorted(set(clean_rows) | set(corrupt_rows)):
        split, view_name = key
        clean = clean_rows.get(key, {})
        corrupt = corrupt_rows.get(key, {})
        corruption = corruption_by_view.get(key, {})
        clean_psnr = _float_or_none(clean.get("psnr"))
        corrupt_psnr = _float_or_none(corrupt.get("psnr"))
        clean_ssim = _float_or_none(clean.get("ssim"))
        corrupt_ssim = _float_or_none(corrupt.get("ssim"))
        clean_l1 = _float_or_none(clean.get("l1_mean"))
        corrupt_l1 = _float_or_none(corrupt.get("l1_mean"))
        output.append(
            {
                "view_name": view_name,
                "split": split,
                "was_corrupted": corruption.get("was_corrupted", "false"),
                "corruption_type": corruption.get("corruption_type", ""),
                "clean_psnr": clean_psnr,
                "corrupt_psnr": corrupt_psnr,
                "psnr_delta": _delta(corrupt_psnr, clean_psnr),
                "clean_ssim": clean_ssim,
                "corrupt_ssim": corrupt_ssim,
                "ssim_delta": _delta(corrupt_ssim, clean_ssim),
                "clean_l1": clean_l1,
                "corrupt_l1": corrupt_l1,
                "l1_delta": _delta(corrupt_l1, clean_l1),
            }
        )
    return output


def compare_runs(
    clean_run_dir: Path,
    corrupt_run_dir: Path,
    corruption_condition: str,
    *,
    data_root: Path | None = None,
    scene: str = "chair",
    corrupt_condition_root: Path | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    clean = _run_info(clean_run_dir)
    corrupt = _run_info(corrupt_run_dir)
    warnings: list[str] = []

    if clean["condition"] not in (None, "clean"):
        warnings.append(f"clean run condition is not clean: {clean['condition']}")
    if corrupt["condition"] not in (None, corruption_condition):
        warnings.append(
            f"corrupt run condition does not match requested condition: {corrupt['condition']}"
        )
    if not clean["has_view_metrics"] or not corrupt["has_view_metrics"]:
        warnings.append("view metrics unavailable for one or both runs")

    clean_count = clean["final_gaussian_count"]
    corrupt_count = corrupt["final_gaussian_count"]
    count_delta = _delta(corrupt_count, clean_count)
    lifecycle_clean = clean["lifecycle"]
    lifecycle_corrupt = corrupt["lifecycle"]
    train_clean = clean["training_events"]
    train_corrupt = corrupt["training_events"]

    metrics = [
        _metric_pair("elapsed_s", clean["elapsed_s"], corrupt["elapsed_s"], "run"),
        _metric_pair("gpu_peak_mb", clean["gpu_peak_mb"], corrupt["gpu_peak_mb"], "run"),
        _metric_pair("final_gaussian_count", clean_count, corrupt_count, "run"),
        _metric_pair(
            "training_event_rows",
            _int_or_none(train_clean.get("training_event_rows"))
            or _count_csv_rows(clean["path"] / "tables" / "training_events.csv"),
            _int_or_none(train_corrupt.get("training_event_rows"))
            or _count_csv_rows(corrupt["path"] / "tables" / "training_events.csv"),
            "training_events",
        ),
        _metric_pair(
            "invalid_training_event_rows",
            _int_or_none(train_clean.get("invalid_training_event_rows")),
            _int_or_none(train_corrupt.get("invalid_training_event_rows")),
            "training_events",
        ),
        _metric_pair(
            "densification_event_rows",
            _int_or_none(train_clean.get("densification_event_rows")),
            _int_or_none(train_corrupt.get("densification_event_rows")),
            "training_events",
        ),
        _metric_pair(
            "densification_trigger_count",
            _int_or_none(train_clean.get("densification_trigger_count")),
            _int_or_none(train_corrupt.get("densification_trigger_count")),
            "training_events",
        ),
        _metric_pair(
            "opacity_reset_count",
            _int_or_none(train_clean.get("opacity_reset_count")),
            _int_or_none(train_corrupt.get("opacity_reset_count")),
            "training_events",
        ),
        _metric_pair(
            "initial_gaussian_count",
            _int_or_none(train_clean.get("initial_gaussian_count")),
            _int_or_none(train_corrupt.get("initial_gaussian_count")),
            "training_events",
        ),
        _metric_pair(
            "max_visible_gaussian_count",
            _int_or_none(train_clean.get("max_visible_gaussian_count"))
            or _max_from_csv(clean["path"] / "tables" / "training_events.csv", "visible_gaussian_count"),
            _int_or_none(train_corrupt.get("max_visible_gaussian_count"))
            or _max_from_csv(corrupt["path"] / "tables" / "training_events.csv", "visible_gaussian_count"),
            "training_events",
        ),
        _metric_pair(
            "max_visibility_ratio",
            _float_or_none(train_clean.get("max_visibility_ratio"))
            or _max_from_csv(clean["path"] / "tables" / "training_events.csv", "visibility_ratio"),
            _float_or_none(train_corrupt.get("max_visibility_ratio"))
            or _max_from_csv(corrupt["path"] / "tables" / "training_events.csv", "visibility_ratio"),
            "training_events",
        ),
        _metric_pair(
            "known_gaussian_count",
            _int_or_none(lifecycle_clean.get("known_gaussian_count")),
            _int_or_none(lifecycle_corrupt.get("known_gaussian_count")),
            "gaussian_lifecycle",
        ),
        _metric_pair(
            "birth_event_count",
            _int_or_none(lifecycle_clean.get("birth_event_count")),
            _int_or_none(lifecycle_corrupt.get("birth_event_count")),
            "gaussian_lifecycle",
        ),
        _metric_pair(
            "clone_birth_count",
            _int_or_none(lifecycle_clean.get("clone_birth_count")),
            _int_or_none(lifecycle_corrupt.get("clone_birth_count")),
            "gaussian_lifecycle",
        ),
        _metric_pair(
            "split_birth_count",
            _int_or_none(lifecycle_clean.get("split_birth_count")),
            _int_or_none(lifecycle_corrupt.get("split_birth_count")),
            "gaussian_lifecycle",
        ),
        _metric_pair(
            "densification_birth_count",
            _int_or_none(lifecycle_clean.get("densification_birth_count")),
            _int_or_none(lifecycle_corrupt.get("densification_birth_count")),
            "gaussian_lifecycle",
        ),
        _metric_pair(
            "prune_death_count",
            _int_or_none(lifecycle_clean.get("prune_death_count")),
            _int_or_none(lifecycle_corrupt.get("prune_death_count")),
            "gaussian_lifecycle",
        ),
        _metric_pair(
            "alive_final_count",
            _int_or_none(lifecycle_clean.get("alive_final_count")),
            _int_or_none(lifecycle_corrupt.get("alive_final_count")),
            "gaussian_lifecycle",
        ),
        _metric_pair(
            "dead_final_count",
            _int_or_none(lifecycle_clean.get("dead_final_count")),
            _int_or_none(lifecycle_corrupt.get("dead_final_count")),
            "gaussian_lifecycle",
        ),
        _metric_pair(
            "final_lifecycle_rows",
            _int_or_none(lifecycle_clean.get("final_lifecycle_rows"))
            or _count_csv_rows(clean["path"] / "tables" / "gaussian_lifecycle_final.csv"),
            _int_or_none(lifecycle_corrupt.get("final_lifecycle_rows"))
            or _count_csv_rows(corrupt["path"] / "tables" / "gaussian_lifecycle_final.csv"),
            "gaussian_lifecycle",
        ),
        _metric_pair(
            "lifecycle_event_rows",
            _int_or_none(lifecycle_clean.get("lifecycle_event_rows"))
            or _count_csv_rows(clean["path"] / "tables" / "gaussian_lifecycle_events.csv"),
            _int_or_none(lifecycle_corrupt.get("lifecycle_event_rows"))
            or _count_csv_rows(corrupt["path"] / "tables" / "gaussian_lifecycle_events.csv"),
            "gaussian_lifecycle",
        ),
        _metric_pair(
            "invariant_violations",
            _int_or_none(lifecycle_clean.get("invariant_violations")),
            _int_or_none(lifecycle_corrupt.get("invariant_violations")),
            "gaussian_lifecycle",
        ),
    ]

    for split in ("train", "test", "target"):
        metrics.append(
            _metric_pair(
                f"{split}_mean_psnr",
                _view_metric(clean["view_metrics"], split, "psnr_mean"),
                _view_metric(corrupt["view_metrics"], split, "psnr_mean"),
                "view_metrics",
            )
        )
    metrics.extend(
        [
            _metric_pair(
                "mean_psnr",
                _mean_view_metric(clean["view_metrics"], "psnr_mean"),
                _mean_view_metric(corrupt["view_metrics"], "psnr_mean"),
                "view_metrics",
            ),
            _metric_pair(
                "mean_ssim",
                _mean_view_metric(clean["view_metrics"], "ssim_mean"),
                _mean_view_metric(corrupt["view_metrics"], "ssim_mean"),
                "view_metrics",
            ),
            _metric_pair(
                "view_count_total",
                _int_or_none(clean["view_metrics"].get("view_count_total")),
                _int_or_none(corrupt["view_metrics"].get("view_count_total")),
                "view_metrics",
            ),
        ]
    )

    resolved_scene = corrupt["scene"] or clean["scene"] or scene
    corruption_root = _resolve_corrupt_condition_root(
        corrupt,
        data_root=data_root.resolve() if data_root is not None else None,
        scene=resolved_scene,
        corruption_condition=corruption_condition,
        corrupt_condition_root=corrupt_condition_root.resolve()
        if corrupt_condition_root is not None
        else None,
    )
    corruption_artifacts = _corruption_artifacts(corruption_root)
    corruption = corruption_artifacts["summary"]
    if not corruption:
        warnings.append("corruption summary unavailable")
    view_corruption_effects = _view_corruption_effect_rows(
        clean_run_dir=clean["path"],
        corrupt_run_dir=corrupt["path"],
        corruption_rows=corruption_artifacts["csv_rows"],
    )
    if not view_corruption_effects and (clean["has_view_metrics"] and corrupt["has_view_metrics"]):
        warnings.append("view corruption effects unavailable")
    summary = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "scene": resolved_scene,
        "trainer": corrupt["trainer"] or clean["trainer"] or "gaussian-splatting",
        "clean_condition": clean["condition"] or "clean",
        "corrupt_condition": corruption_condition,
        "clean_run_dir": clean["run_dir"],
        "corrupt_run_dir": corrupt["run_dir"],
        "clean_run_id": clean["run_id"],
        "corrupt_run_id": corrupt["run_id"],
        "requested_iterations": corrupt["requested_iterations"] or clean["requested_iterations"],
        "clean_success": clean["success"],
        "corrupt_success": corrupt["success"],
        "clean_observation_only": clean["observation_only"],
        "corrupt_observation_only": corrupt["observation_only"],
        "clean_training_event_invariants_ok": clean["training_event_invariants_ok"],
        "corrupt_training_event_invariants_ok": corrupt["training_event_invariants_ok"],
        "clean_gaussian_lifecycle_invariants_ok": clean["gaussian_lifecycle_invariants_ok"],
        "corrupt_gaussian_lifecycle_invariants_ok": corrupt["gaussian_lifecycle_invariants_ok"],
        "clean_final_gaussian_count": clean_count,
        "corrupt_final_gaussian_count": corrupt_count,
        "final_gaussian_count_delta": count_delta,
        "final_gaussian_count_delta_ratio": _ratio(count_delta, clean_count),
        "clean_densification_trigger_count": _int_or_none(train_clean.get("densification_trigger_count")),
        "corrupt_densification_trigger_count": _int_or_none(train_corrupt.get("densification_trigger_count")),
        "densification_trigger_count_delta": _delta(
            _int_or_none(train_corrupt.get("densification_trigger_count")),
            _int_or_none(train_clean.get("densification_trigger_count")),
        ),
        "clean_birth_event_count": _int_or_none(lifecycle_clean.get("birth_event_count")),
        "corrupt_birth_event_count": _int_or_none(lifecycle_corrupt.get("birth_event_count")),
        "birth_event_count_delta": _delta(
            _int_or_none(lifecycle_corrupt.get("birth_event_count")),
            _int_or_none(lifecycle_clean.get("birth_event_count")),
        ),
        "clean_clone_birth_count": _int_or_none(lifecycle_clean.get("clone_birth_count")),
        "corrupt_clone_birth_count": _int_or_none(lifecycle_corrupt.get("clone_birth_count")),
        "clone_birth_count_delta": _delta(
            _int_or_none(lifecycle_corrupt.get("clone_birth_count")),
            _int_or_none(lifecycle_clean.get("clone_birth_count")),
        ),
        "clean_split_birth_count": _int_or_none(lifecycle_clean.get("split_birth_count")),
        "corrupt_split_birth_count": _int_or_none(lifecycle_corrupt.get("split_birth_count")),
        "split_birth_count_delta": _delta(
            _int_or_none(lifecycle_corrupt.get("split_birth_count")),
            _int_or_none(lifecycle_clean.get("split_birth_count")),
        ),
        "clean_prune_death_count": _int_or_none(lifecycle_clean.get("prune_death_count")),
        "corrupt_prune_death_count": _int_or_none(lifecycle_corrupt.get("prune_death_count")),
        "prune_death_count_delta": _delta(
            _int_or_none(lifecycle_corrupt.get("prune_death_count")),
            _int_or_none(lifecycle_clean.get("prune_death_count")),
        ),
        "clean_alive_final_count": _int_or_none(lifecycle_clean.get("alive_final_count")),
        "corrupt_alive_final_count": _int_or_none(lifecycle_corrupt.get("alive_final_count")),
        "clean_dead_final_count": _int_or_none(lifecycle_clean.get("dead_final_count")),
        "corrupt_dead_final_count": _int_or_none(lifecycle_corrupt.get("dead_final_count")),
        "view_metrics_available": clean["has_view_metrics"] and corrupt["has_view_metrics"],
        "corruption_summary_available": bool(corruption),
        "corruption_condition_root": str(corruption_root) if corruption_root else None,
        "corruption_summary": corruption,
        "corruption_manifest_available": bool(corruption_artifacts["manifest"]),
        "corruption_manifest_csv_available": bool(corruption_artifacts["csv_rows"]),
        "view_corruption_effects_available": bool(view_corruption_effects),
        "warnings": warnings,
    }
    artifacts = _artifact_rows(clean["path"], corrupt["path"])
    return summary, metrics, artifacts, view_corruption_effects


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _markdown(summary: dict[str, Any]) -> str:
    corruption = summary.get("corruption_summary") or {}
    return "\n".join(
        [
            "# Clean-vs-Corrupt Observation Report",
            "",
            "This comparison measures how a natural corruption condition changes observed training dynamics and Gaussian lifecycle relative to clean training.",
            "",
            "It does not claim corruption detection, poison detection, defense success, or view trust classification.",
            "",
            "## Run Identity",
            f"- Clean run: `{summary.get('clean_run_dir')}`",
            f"- Corrupt run: `{summary.get('corrupt_run_dir')}`",
            f"- Clean run ID: `{summary.get('clean_run_id')}`",
            f"- Corrupt run ID: `{summary.get('corrupt_run_id')}`",
            "",
            "## Conditions Compared",
            f"- Scene: `{summary.get('scene')}`",
            f"- Clean condition: `{summary.get('clean_condition')}`",
            f"- Corrupt condition: `{summary.get('corrupt_condition')}`",
            f"- Trainer: `{summary.get('trainer')}`",
            f"- Requested iterations: `{summary.get('requested_iterations')}`",
            "",
            "## Training Success",
            f"- Clean success: `{summary.get('clean_success')}`",
            f"- Corrupt success: `{summary.get('corrupt_success')}`",
            "",
            "## Observation Invariants",
            f"- Clean training events: `{summary.get('clean_training_event_invariants_ok')}`",
            f"- Corrupt training events: `{summary.get('corrupt_training_event_invariants_ok')}`",
            f"- Clean lifecycle: `{summary.get('clean_gaussian_lifecycle_invariants_ok')}`",
            f"- Corrupt lifecycle: `{summary.get('corrupt_gaussian_lifecycle_invariants_ok')}`",
            "",
            "## Final Gaussian Count Comparison",
            f"- Clean final count: `{summary.get('clean_final_gaussian_count')}`",
            f"- Corrupt final count: `{summary.get('corrupt_final_gaussian_count')}`",
            f"- Delta: `{summary.get('final_gaussian_count_delta')}`",
            f"- Delta ratio: `{summary.get('final_gaussian_count_delta_ratio')}`",
            "",
            "## Densification Event Comparison",
            f"- Clean triggers: `{summary.get('clean_densification_trigger_count')}`",
            f"- Corrupt triggers: `{summary.get('corrupt_densification_trigger_count')}`",
            f"- Delta: `{summary.get('densification_trigger_count_delta')}`",
            "",
            "## Gaussian Lifecycle Comparison",
            f"- Birth event delta: `{summary.get('birth_event_count_delta')}`",
            f"- Clone birth delta: `{summary.get('clone_birth_count_delta')}`",
            f"- Split birth delta: `{summary.get('split_birth_count_delta')}`",
            f"- Prune death delta: `{summary.get('prune_death_count_delta')}`",
            f"- Clean alive/dead final: `{summary.get('clean_alive_final_count')}` / `{summary.get('clean_dead_final_count')}`",
            f"- Corrupt alive/dead final: `{summary.get('corrupt_alive_final_count')}` / `{summary.get('corrupt_dead_final_count')}`",
            "",
            "## View Metrics Comparison",
            f"- View metrics available: `{summary.get('view_metrics_available')}`",
            "- See `clean_vs_corrupt_metrics.csv` for split-level metric deltas.",
            "",
            "## Corruption Manifest Summary",
            f"- Available: `{summary.get('corruption_summary_available')}`",
            f"- Corrupted images: `{corruption.get('corrupted_image_count')}`",
            f"- Selected train views: `{corruption.get('selected_train_views')}`",
            "",
            "## Interpretation Guidance",
            "- Natural corruptions are non-malicious perturbations.",
            "- Use these tables as evidence for later ViewTrust signal design.",
            "- A larger change in lifecycle or training dynamics is not itself a trust score.",
            "",
            "## Known Limitations",
            "- PR11 compares observations but does not classify views as trustworthy or untrustworthy.",
            "- PR11 does not implement ViewTrust scoring, a defense, poisoning, or training-time intervention.",
            "- Only one natural condition may be validated first on the server, though the full suite is supported.",
            "",
            "## Warnings",
            *[f"- {warning}" for warning in summary.get("warnings", [])],
            "" if summary.get("warnings") else "- None",
            "",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-run-dir", required=True, type=Path)
    parser.add_argument("--corrupt-run-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--scene", default="chair")
    parser.add_argument("--corruption-condition", required=True)
    parser.add_argument("--corrupt-condition-root", type=Path)
    parser.add_argument("--require-observation-invariants", action="store_true")
    parser.add_argument("--write-markdown", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary, metrics, artifacts, view_corruption_effects = compare_runs(
        args.clean_run_dir,
        args.corrupt_run_dir,
        args.corruption_condition,
        data_root=args.data_root,
        scene=args.scene,
        corrupt_condition_root=args.corrupt_condition_root,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "clean_vs_corrupt_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(
        args.output_dir / "clean_vs_corrupt_metrics.csv",
        metrics,
        ["metric_group", "metric", "clean", "corrupt", "delta", "delta_ratio"],
    )
    _write_csv(
        args.output_dir / "clean_vs_corrupt_artifact_manifest.csv",
        artifacts,
        [
            "run_label",
            "relative_path",
            "exists",
            "file_type",
            "size_bytes",
            "required",
            "artifact_group",
        ],
    )
    if view_corruption_effects:
        _write_csv(
            args.output_dir / "view_corruption_effects.csv",
            view_corruption_effects,
            [
                "view_name",
                "split",
                "was_corrupted",
                "corruption_type",
                "clean_psnr",
                "corrupt_psnr",
                "psnr_delta",
                "clean_ssim",
                "corrupt_ssim",
                "ssim_delta",
                "clean_l1",
                "corrupt_l1",
                "l1_delta",
            ],
        )
    if args.write_markdown:
        (args.output_dir / "clean_vs_corrupt_report.md").write_text(
            _markdown(summary),
            encoding="utf-8",
        )
    print(json.dumps(summary, indent=2, sort_keys=True))

    if args.require_observation_invariants:
        invariant_ok = (
            summary["clean_training_event_invariants_ok"]
            and summary["corrupt_training_event_invariants_ok"]
            and summary["clean_gaussian_lifecycle_invariants_ok"]
            and summary["corrupt_gaussian_lifecycle_invariants_ok"]
        )
        if not invariant_ok:
            print("ERROR: clean-vs-corrupt observation invariants failed.", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
