#!/usr/bin/env python3
"""Build PR12 view-to-Gaussian influence attribution tables."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
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
) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = _json_file(run_dir / "metadata.json")
    command_summary = _json_file(run_dir / "summary.json")
    training_summary = _json_file(run_dir / "training_events_summary.json")
    lifecycle_summary = _json_file(run_dir / "gaussian_lifecycle_summary.json")
    run_id = metadata.get("run_id") or run_dir.name

    training_rows = [
        row
        for row in _csv_rows(run_dir / "tables" / "training_events.csv")
        if row.get("event_type") == "iteration_metrics"
    ]
    missing_view_identity = sum(1 for row in training_rows if not row.get("view_name"))
    if require_view_identity and missing_view_identity:
        raise ValueError(f"missing view identity rows: {missing_view_identity}")

    lifecycle_rows = _csv_rows(run_dir / "tables" / "gaussian_lifecycle_events.csv")
    source_event_types = {"clone_birth", "split_birth", "densification_birth", "prune_death"}
    source_lifecycle_rows = [
        row for row in lifecycle_rows if row.get("event_type") in source_event_types
    ]
    missing_source_view = sum(1 for row in source_lifecycle_rows if not row.get("source_view_name"))
    if require_source_view and missing_source_view:
        raise ValueError(f"missing lifecycle source view rows: {missing_source_view}")

    final_rows = _csv_rows(run_dir / "tables" / "gaussian_lifecycle_final.csv")
    final_by_id = {str(row.get("gaussian_id", "")): row for row in final_rows}
    densification_by_iteration = {
        str(row.get("iteration", "")): row
        for row in _csv_rows(run_dir / "tables" / "densification_events.csv")
    }
    condition_root = _resolve_condition_root(
        run_dir=run_dir,
        data_root=data_root,
        scene=scene,
        condition=condition,
        corruption_condition_root=corruption_condition_root,
    )
    corruption_labels, corruption_summary = _corruption_labels(condition_root)
    view_metrics = _view_metric_by_name(run_dir)

    lifecycle_by_view: dict[str, list[dict[str, str]]] = defaultdict(list)
    lifecycle_by_iteration: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in source_lifecycle_rows:
        lifecycle_by_view[str(row.get("source_view_name", ""))].append(row)
        lifecycle_by_iteration[str(row.get("source_iteration") or row.get("iteration") or "")].append(row)

    iteration_rows: list[dict[str, Any]] = []
    samples_by_view: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in training_rows:
        view_name = str(row.get("view_name", ""))
        if not view_name:
            continue
        samples_by_view[view_name].append(row)
        iteration = str(row.get("iteration", ""))
        densification = densification_by_iteration.get(iteration, {})
        events = lifecycle_by_iteration.get(iteration, [])
        label = corruption_labels.get(view_name, {})
        birth_events = [event for event in events if "birth" in str(event.get("event_type", ""))]
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
                "birth_event_count": len(birth_events),
                "clone_birth_count": sum(1 for event in events if event.get("event_type") == "clone_birth"),
                "split_birth_count": sum(1 for event in events if event.get("event_type") == "split_birth"),
                "prune_death_count": sum(1 for event in events if event.get("event_type") == "prune_death"),
            }
        )

    all_view_names = sorted(set(samples_by_view) | set(corruption_labels))
    view_rows: list[dict[str, Any]] = []
    attribution_rows: list[dict[str, Any]] = []
    for view_name in all_view_names:
        samples = samples_by_view.get(view_name, [])
        label = corruption_labels.get(view_name, {})
        events = lifecycle_by_view.get(view_name, [])
        births = [event for event in events if "birth" in str(event.get("event_type", ""))]
        clone_births = [event for event in events if event.get("event_type") == "clone_birth"]
        split_births = [event for event in events if event.get("event_type") == "split_birth"]
        prune_deaths = [event for event in events if event.get("event_type") == "prune_death"]
        alive_births = [
            event
            for event in births
            if _truthy(final_by_id.get(str(event.get("gaussian_id", "")), {}).get("alive"))
        ]
        dead_births = [
            event
            for event in births
            if final_by_id.get(str(event.get("gaussian_id", "")), {}).get("alive") not in ("", None)
            and not _truthy(final_by_id.get(str(event.get("gaussian_id", "")), {}).get("alive"))
        ]
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
                "densification_context_count": len({event.get("source_iteration") for event in events if event.get("source_iteration")}),
                "birth_event_count_after_view": len(births),
                "clone_birth_count_after_view": len(clone_births),
                "split_birth_count_after_view": len(split_births),
                "prune_death_count_after_view": len(prune_deaths),
                "final_survivor_birth_count_after_view": len(alive_births),
                "dead_birth_count_after_view": len(dead_births),
                "birth_survival_ratio_after_view": _ratio(len(alive_births), len(births)),
                "mean_view_psnr": _float_or_none(metric.get("psnr")),
                "mean_view_ssim": _float_or_none(metric.get("ssim")),
                "mean_view_l1": _float_or_none(metric.get("l1_mean")),
                "warnings": "",
            }
        )

        for (iteration, event_type), grouped_events in sorted(
            {
                (event.get("source_iteration") or event.get("iteration") or "", event.get("event_type", "")):
                [
                    item
                    for item in events
                    if (item.get("source_iteration") or item.get("iteration") or "") == (event.get("source_iteration") or event.get("iteration") or "")
                    and item.get("event_type", "") == event.get("event_type", "")
                ]
                for event in events
            }.items()
        ):
            gaussian_ids = {event.get("gaussian_id", "") for event in grouped_events if event.get("gaussian_id")}
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
                    "source_view_split": grouped_events[0].get("source_view_split", ""),
                    "was_corrupted": label.get("was_corrupted", "false"),
                    "corruption_type": label.get("corruption_type", ""),
                    "source_iteration": iteration,
                    "event_type": event_type,
                    "lifecycle_action": event_type,
                    "event_count": len(grouped_events),
                    "clone_birth_count": sum(1 for event in grouped_events if event.get("event_type") == "clone_birth"),
                    "split_birth_count": sum(1 for event in grouped_events if event.get("event_type") == "split_birth"),
                    "prune_death_count": sum(1 for event in grouped_events if event.get("event_type") == "prune_death"),
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
    _write_csv(output_dir / "view_influence.csv", view_rows, view_fields)
    _write_csv(output_dir / "view_lifecycle_attribution.csv", attribution_rows, attribution_fields)
    _write_csv(output_dir / "view_iteration_events.csv", iteration_rows, iteration_fields)
    _write_csv(
        output_dir / "view_influence_artifact_manifest.csv",
        _artifact_rows(run_dir),
        ["relative_path", "exists", "file_type", "size_bytes", "required", "artifact_group"],
    )

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
    summary = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "run_dir": str(run_dir),
        "run_id": run_id,
        "scene": scene,
        "condition": condition,
        "observation_only": command_summary.get("observation_only") is True,
        "training_event_invariants_ok": (_int_or_none(training_summary.get("invalid_training_event_rows")) or 0) == 0,
        "gaussian_lifecycle_invariants_ok": (_int_or_none(lifecycle_summary.get("invariant_violations")) or 0) == 0,
        "view_identity_available": bool(training_rows) and missing_view_identity == 0,
        "lifecycle_source_view_available": bool(source_lifecycle_rows) and missing_source_view == 0,
        "corruption_summary_available": bool(corruption_summary),
        "corrupted_view_count": len([row for row in corruption_labels.values() if row.get("was_corrupted") == "true"]),
        "training_view_count": len(all_view_names),
        "views_sampled_count": len(samples_by_view),
        "total_iteration_metric_rows": len(training_rows),
        "total_lifecycle_event_rows": len(source_lifecycle_rows),
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
            f"- Lifecycle source events: `{summary.get('total_lifecycle_event_rows')}`",
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
    parser.add_argument("--write-markdown", action="store_true")
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
