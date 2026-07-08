"""Pure helpers for PR16 subset and scene bias probing."""

from __future__ import annotations

import csv
import hashlib
import json
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_SCENES = ["chair", "drum"]
DEFAULT_CONDITIONS = ["corrupt_occluder", "corrupt_noise", "corrupt_mixed"]
DEFAULT_SUBSET_NAMES = ["original", "seed_20260708", "seed_20260709"]
DEFAULT_SUBSET_SEEDS = [20260708, 20260709]

REQUIRED_OFFLINE_FILES = [
    "offline_viewtrust_summary.json",
    "offline_viewtrust_signals.csv",
    "offline_viewtrust_rankings.csv",
    "offline_viewtrust_group_metrics.csv",
    "offline_viewtrust_signal_ablation.csv",
    "offline_viewtrust_config.json",
    "offline_viewtrust_report.md",
    "offline_viewtrust_artifact_manifest.csv",
]

PLAN_OUTPUT_FILES = [
    "pr16_condition_matrix.csv",
    "pr16_subset_manifest.csv",
    "pr16_seed_reproducibility_summary.json",
    "pr16_run_commands.sh",
    "pr16_plan_report.md",
    "artifact_manifest.csv",
]

ANALYSIS_OUTPUT_FILES = [
    "pr16_bias_probe_summary.json",
    "pr16_scene_subset_condition_results.csv",
    "pr16_subset_bias_summary.csv",
    "pr16_scene_bias_summary.csv",
    "pr16_view_identity_bias_table.csv",
    "pr16_repeated_false_positive_table.csv",
    "pr16_component_comparison.csv",
    "pr16_missing_outputs.csv",
    "pr16_bias_probe_report.md",
    "artifact_manifest.csv",
]

COMPONENT_KEYWORDS = [
    "loss",
    "visibility",
    "birth",
    "prune",
    "survival",
    "delta",
    "lifecycle",
    "consistency",
]

REPORT_DISCLAIMER = """This PR16 report is offline observation only.
It is not a trust score used during training.
It is not a defense.
It is not a poison classifier.
It does not reject views, suppress updates, reweight loss, or gate densification.
Corruption labels are used only for evaluation summaries, not for scoring or ranking."""


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: "" if row.get(field) is None else row.get(field) for field in fieldnames})


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def normalize_bool(value: Any) -> bool | None:
    if value in ("", None):
        return None
    text = str(value).strip().lower()
    if value is True or text in {"true", "1", "yes"}:
        return True
    if value is False or text in {"false", "0", "no"}:
        return False
    return None


def normalize_float(value: Any) -> float | None:
    if value in ("", None):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number


def normalize_int(value: Any) -> int | None:
    number = normalize_float(value)
    return int(number) if number is not None else None


def _first(row: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        if row.get(key) not in ("", None):
            return row.get(key)
    return ""


def _view_name(row: dict[str, Any]) -> str:
    return str(_first(row, ["view_name", "source_view_name"]))


def _risk(row: dict[str, Any]) -> float | None:
    return normalize_float(_first(row, ["offline_viewtrust_risk", "risk", "score", "full_signal"]))


def _rank(row: dict[str, Any]) -> int | None:
    return normalize_int(_first(row, ["rank", "offline_viewtrust_rank"]))


def _was_corrupted(row: dict[str, Any]) -> bool | None:
    return normalize_bool(_first(row, ["was_corrupted", "is_corrupted", "corrupted"]))


def _main_reason(row: dict[str, Any]) -> str:
    return str(_first(row, ["main_reason", "top_reason", "reason"]))


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def stable_hash_view_list(view_names: list[str]) -> str:
    ordered = "\n".join(view_names)
    return hashlib.sha256(ordered.encode("utf-8")).hexdigest()[:16]


def _view_name_from_file_path(file_path: str) -> str:
    return Path(file_path).stem


def discover_train_views(data_root: Path, scene: str) -> list[str]:
    candidates = [
        data_root / "viewtrust-mini" / "nerf_synthetic" / scene / "clean" / "transforms_train.json",
        data_root / "raw" / "nerf_synthetic" / scene / "transforms_train.json",
        data_root / "nerf_synthetic" / scene / "clean" / "transforms_train.json",
        data_root / scene / "clean" / "transforms_train.json",
        data_root / scene / "transforms_train.json",
    ]
    for path in candidates:
        payload = load_json(path)
        frames = payload.get("frames", []) if payload else []
        views = [
            _view_name_from_file_path(str(frame.get("file_path", "")))
            for frame in frames
            if frame.get("file_path")
        ]
        if views:
            return sorted(dict.fromkeys(views))
    return []


def _condition_candidates(input_root: Path, scene: str, subset_name: str, condition: str) -> list[Path]:
    if not input_root.exists():
        return []
    names = [
        f"offline_viewtrust_{scene}_{condition}_{subset_name}_pr16_input",
        f"offline_viewtrust_{scene}_{condition}_{subset_name}_pr14_input",
    ]
    candidates = [
        input_root / name
        for name in names
        if (input_root / name).is_dir()
    ]
    prefixes = [
        f"offline_viewtrust_{scene}_{condition}_{subset_name}_pr13",
        f"offline_viewtrust_{scene}_{condition}_{subset_name}_pr16",
    ]
    for path in input_root.iterdir() if input_root.exists() else []:
        if path.is_dir() and any(path.name.startswith(prefix) for prefix in prefixes):
            candidates.append(path)
    if scene == "chair" and subset_name == "original":
        legacy = input_root / f"offline_viewtrust_{condition}_pr14_input"
        if legacy.is_dir():
            candidates.append(legacy)
        for path in input_root.iterdir() if input_root.exists() else []:
            if path.is_dir() and path.name.startswith(f"offline_viewtrust_{condition}_pr13"):
                candidates.append(path)
    unique = {path: None for path in candidates}
    return sorted(unique, key=lambda path: path.name, reverse=True)


def discover_pr16_condition_output(input_root: Path, scene: str, subset_name: str, condition: str) -> Path | None:
    exact = input_root / f"offline_viewtrust_{scene}_{condition}_{subset_name}_pr16_input"
    if exact.is_dir() and validate_condition_output(exact)[0] == "ok":
        return exact
    for path in _condition_candidates(input_root, scene, subset_name, condition):
        if validate_condition_output(path)[0] == "ok":
            return path
    candidates = _condition_candidates(input_root, scene, subset_name, condition)
    return candidates[0] if candidates else None


def infer_original_corrupted_subset(input_root: Path, scene: str, conditions: list[str]) -> tuple[list[str], str, list[str]]:
    warnings: list[str] = []
    manifest_candidates = [
        input_root / f"pr16_{scene}_original_subset_manifest.json",
        input_root / f"{scene}_original_subset_manifest.json",
    ]
    for path in manifest_candidates:
        payload = load_json(path)
        names = payload.get("corrupted_view_names")
        if isinstance(names, list) and names:
            return [str(name) for name in names], "original_manifest", warnings

    collected: list[str] = []
    for condition in conditions:
        signal_dir = discover_pr16_condition_output(input_root, scene, "original", condition)
        if signal_dir is None:
            continue
        for row in load_condition_rankings(signal_dir):
            if _was_corrupted(row) is True and _view_name(row):
                collected.append(_view_name(row))
        if collected:
            return sorted(dict.fromkeys(collected)), "original_inferred", warnings
    warnings.append("original subset could not be inferred from existing outputs")
    return [], "original_inferred", warnings


def generate_seeded_subset(train_views: list[str], seed: int, corrupted_view_count: int) -> list[str]:
    if corrupted_view_count <= 0:
        raise ValueError("corrupted_view_count must be positive")
    if len(train_views) < corrupted_view_count:
        raise ValueError("not enough training views for requested corrupted_view_count")
    rng = random.Random(seed)
    selected = rng.sample(sorted(train_views), corrupted_view_count)
    return sorted(selected)


def _seed_for_subset(subset_name: str, subset_seeds: list[int]) -> int | None:
    if subset_name == "original":
        return None
    if subset_name.startswith("seed_"):
        try:
            return int(subset_name.removeprefix("seed_"))
        except ValueError:
            return None
    return subset_seeds[0] if subset_seeds else None


def build_subset_manifest(
    *,
    data_root: Path,
    input_root: Path,
    scenes: list[str],
    subset_names: list[str],
    subset_seeds: list[int],
    conditions: list[str],
    corrupted_view_count: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    hashes_by_scene: dict[str, dict[str, str]] = defaultdict(dict)
    same_seed_reproducible = True
    different_seed_collision_count = 0

    for scene in scenes:
        train_views = discover_train_views(data_root, scene)
        if not train_views:
            warnings.append(f"{scene}: no training views discovered")
        for subset_name in subset_names:
            row_warnings: list[str] = []
            subset_seed = _seed_for_subset(subset_name, subset_seeds)
            source = "random_seed"
            status = "ok"
            if subset_name == "original":
                selected, source, inferred_warnings = infer_original_corrupted_subset(input_root, scene, conditions)
                row_warnings.extend(inferred_warnings)
                if not selected and train_views:
                    selected = train_views[: min(corrupted_view_count, len(train_views))]
                    row_warnings.append("fallback_first_train_views_used_for_original_plan_only")
                if train_views:
                    selected = [view for view in selected if view in set(train_views)]
            else:
                if subset_seed is None:
                    selected = []
                    status = "invalid"
                    row_warnings.append("subset seed is missing")
                else:
                    try:
                        selected = generate_seeded_subset(train_views, subset_seed, corrupted_view_count)
                        repeat = generate_seeded_subset(train_views, subset_seed, corrupted_view_count)
                        same_seed_reproducible = same_seed_reproducible and selected == repeat
                    except ValueError as exc:
                        selected = []
                        status = "invalid"
                        row_warnings.append(str(exc))
            if len(selected) != min(corrupted_view_count, len(train_views)) and train_views:
                status = "invalid"
                row_warnings.append("corrupted_view_count_not_satisfied")
            if any(view not in set(train_views) for view in selected):
                status = "invalid"
                row_warnings.append("subset_contains_non_training_view")
            view_hash = stable_hash_view_list(selected)
            if subset_name != "original":
                for other_subset, other_hash in hashes_by_scene[scene].items():
                    if other_subset != "original" and other_hash == view_hash:
                        different_seed_collision_count += 1
                        row_warnings.append(f"same hash as {other_subset}")
                hashes_by_scene[scene][subset_name] = view_hash
            rows.append(
                {
                    "scene": scene,
                    "subset_name": subset_name,
                    "subset_seed": "" if subset_seed is None else subset_seed,
                    "train_view_count": len(train_views),
                    "corrupted_view_count": len(selected),
                    "corrupted_view_names": ";".join(selected),
                    "corrupted_view_hash": view_hash,
                    "source": source,
                    "status": status,
                    "warnings": ";".join(row_warnings),
                }
            )
            warnings.extend(f"{scene}/{subset_name}: {warning}" for warning in row_warnings)

    summary = {
        "schema_name": "viewtrust.pr16.subset_seed_reproducibility.summary",
        "schema_version": 1,
        "scenes": scenes,
        "subset_names": subset_names,
        "subset_seeds": subset_seeds,
        "corrupted_view_count": corrupted_view_count,
        "same_seed_reproducible": same_seed_reproducible,
        "different_seed_collision_count": different_seed_collision_count,
        "warnings": warnings,
    }
    return rows, summary


def build_condition_matrix(
    *,
    scenes: list[str],
    subset_manifest_rows: list[dict[str, Any]],
    conditions: list[str],
    top_k: int,
) -> list[dict[str, Any]]:
    by_scene_subset = {
        (str(row.get("scene")), str(row.get("subset_name"))): row
        for row in subset_manifest_rows
    }
    rows: list[dict[str, Any]] = []
    for scene in scenes:
        subset_names = [str(row.get("subset_name")) for row in subset_manifest_rows if row.get("scene") == scene]
        for subset_name in subset_names:
            subset = by_scene_subset[(scene, subset_name)]
            for condition in conditions:
                key = f"{scene}_{subset_name}_{condition}"
                rows.append(
                    {
                        "scene": scene,
                        "subset_name": subset_name,
                        "subset_seed": subset.get("subset_seed", ""),
                        "condition": condition,
                        "corrupted_view_count": subset.get("corrupted_view_count", ""),
                        "top_k": top_k,
                        "expected_output_key": key,
                        "expected_offline_signal_dir": f"outputs/reports/offline_viewtrust_{scene}_{condition}_{subset_name}_pr16_input",
                        "expected_pr14_dir": f"outputs/reports/offline_viewtrust_multi_condition_{scene}_{subset_name}_pr16",
                        "expected_pr15_dir": f"outputs/reports/offline_viewtrust_rank_consistency_{scene}_{subset_name}_pr16",
                        "status": "planned" if subset.get("status") == "ok" else "subset_invalid",
                        "warnings": subset.get("warnings", ""),
                    }
                )
    return rows


def load_condition_rankings(signal_dir: Path) -> list[dict[str, Any]]:
    rankings = load_csv_rows(signal_dir / "offline_viewtrust_rankings.csv")
    signals_by_view = {
        _view_name(row): row
        for row in load_csv_rows(signal_dir / "offline_viewtrust_signals.csv")
        if _view_name(row)
    }
    merged: list[dict[str, Any]] = []
    for row in rankings:
        combined = dict(signals_by_view.get(_view_name(row), {}))
        combined.update(row)
        merged.append(combined)
    if any(_rank(row) is None for row in merged):
        merged = sorted(merged, key=lambda row: (_risk(row) or 0.0, _view_name(row)), reverse=True)
        for index, row in enumerate(merged, start=1):
            row["rank"] = index
    return sorted(merged, key=lambda row: _rank(row) or 10**9)


def load_condition_ablation(signal_dir: Path) -> list[dict[str, str]]:
    return load_csv_rows(signal_dir / "offline_viewtrust_signal_ablation.csv")


def validate_condition_output(signal_dir: Path) -> tuple[str, list[str]]:
    missing = [name for name in REQUIRED_OFFLINE_FILES if not (signal_dir / name).is_file()]
    if missing:
        return "invalid", [f"missing required file: {name}" for name in missing]
    summary = load_json(signal_dir / "offline_viewtrust_summary.json")
    warnings: list[str] = []
    checks = [
        ("observation_only", summary.get("observation_only") is True),
        ("training_intervention", summary.get("training_intervention") is False),
        ("defense_enabled", summary.get("defense_enabled") is False),
        ("uses_corruption_labels_for_scoring", summary.get("uses_corruption_labels_for_scoring") is False),
        ("uses_corruption_labels_for_evaluation", summary.get("uses_corruption_labels_for_evaluation") is True),
    ]
    for name, ok in checks:
        if not ok:
            warnings.append(f"invalid summary field: {name}")
    rankings = load_condition_rankings(signal_dir)
    if not rankings:
        warnings.append("offline_viewtrust_rankings.csv has no rows")
    elif not _view_name(rankings[0]) or _risk(rankings[0]) is None:
        warnings.append("ranking lacks compatible view or risk columns")
    return ("failed_validation", warnings) if warnings else ("ok", [])


def _ablation_lookup(rows: list[dict[str, Any]], signal_name: str) -> dict[str, Any]:
    return next((row for row in rows if row.get("signal_name") == signal_name), {})


def _mapped_ablation_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    mapped: list[dict[str, Any]] = []
    enriched = []
    for row in rows:
        enriched.append(
            {
                "signal_name": row.get("signal_name", ""),
                "precision_at_k": row.get("precision_at_k", ""),
                "recall_at_k": row.get("recall_at_k", ""),
                "risk_gap_corrupted_minus_uncorrupted": _first(row, ["risk_gap_corrupted_minus_uncorrupted", "score_gap"]),
                "corrupted_in_top_k": _first(row, ["corrupted_in_top_k", "corrupted_in_topk"]),
                "top1_view_name": row.get("top1_view_name", ""),
                "top1_was_corrupted": row.get("top1_was_corrupted", ""),
                "status": row.get("status", "ok"),
            }
        )
    ranked = sorted(
        enriched,
        key=lambda row: (
            -(normalize_float(row.get("recall_at_k")) or 0.0),
            -(normalize_float(row.get("precision_at_k")) or 0.0),
            -(normalize_float(row.get("risk_gap_corrupted_minus_uncorrupted")) or 0.0),
            str(row.get("signal_name", "")),
        ),
    )
    for index, row in enumerate(ranked, start=1):
        row["rank_within_condition"] = index
        mapped.append(row)
    return mapped


def _beats(left: dict[str, Any], right: dict[str, Any]) -> bool:
    if not left or not right:
        return False
    for key in ("recall_at_k", "precision_at_k", "risk_gap_corrupted_minus_uncorrupted"):
        left_value = normalize_float(left.get(key)) or 0.0
        right_value = normalize_float(right.get(key)) or 0.0
        if left_value > right_value:
            return True
        if left_value < right_value:
            return False
    return False


def build_scene_subset_condition_results(
    *,
    input_root: Path,
    subset_manifest_rows: list[dict[str, Any]],
    scenes: list[str],
    conditions: list[str],
    subset_names: list[str],
    top_k: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[tuple[str, str, str], Path | None]]:
    subset_by_key = {
        (str(row.get("scene")), str(row.get("subset_name"))): row
        for row in subset_manifest_rows
    }
    results: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    dirs: dict[tuple[str, str, str], Path | None] = {}
    for scene in scenes:
        for subset_name in subset_names:
            subset = subset_by_key.get((scene, subset_name), {})
            for condition in conditions:
                signal_dir = discover_pr16_condition_output(input_root, scene, subset_name, condition)
                dirs[(scene, subset_name, condition)] = signal_dir
                base = {
                    "scene": scene,
                    "subset_name": subset_name,
                    "subset_seed": subset.get("subset_seed", ""),
                    "condition": condition,
                    "top_k": top_k,
                }
                expected = f"offline_viewtrust_{scene}_{condition}_{subset_name}_pr16_input"
                if signal_dir is None:
                    row = {**base, "status": "missing", "warnings": "missing condition output"}
                    results.append(row)
                    missing_rows.append(
                        {
                            "scene": scene,
                            "subset_name": subset_name,
                            "condition": condition,
                            "expected_pattern": expected,
                            "status": "missing",
                            "details": "no matching PR16/PR13/PR14-input offline signal directory",
                        }
                    )
                    continue
                status, warnings = validate_condition_output(signal_dir)
                if status != "ok":
                    row = {**base, "status": status, "warnings": ";".join(warnings)}
                    results.append(row)
                    missing_rows.append(
                        {
                            "scene": scene,
                            "subset_name": subset_name,
                            "condition": condition,
                            "expected_pattern": expected,
                            "status": status,
                            "details": ";".join(warnings),
                        }
                    )
                    continue
                summary = load_json(signal_dir / "offline_viewtrust_summary.json")
                rankings = load_condition_rankings(signal_dir)
                top = rankings[:top_k]
                top1 = rankings[0] if rankings else {}
                ablations = _mapped_ablation_rows(load_condition_ablation(signal_dir))
                full = _ablation_lookup(ablations, "full_signal")
                loss = _ablation_lookup(ablations, "loss_only")
                lifecycle = _ablation_lookup(ablations, "lifecycle_only")
                results.append(
                    {
                        **base,
                        "view_count": summary.get("view_count", len(rankings)),
                        "corrupted_view_count": summary.get("corrupted_view_count", ""),
                        "corrupted_in_top_k": summary.get("corrupted_in_top_k", ""),
                        "precision_at_k": summary.get("precision_at_k", ""),
                        "recall_at_k": summary.get("recall_at_k", ""),
                        "mean_corrupted_risk": summary.get("mean_corrupted_risk", ""),
                        "mean_uncorrupted_risk": summary.get("mean_uncorrupted_risk", ""),
                        "risk_gap_corrupted_minus_uncorrupted": summary.get("risk_gap_corrupted_minus_uncorrupted", ""),
                        "top1_view_name": _view_name(top1),
                        "top1_was_corrupted": _was_corrupted(top1),
                        "top1_risk": _risk(top1),
                        "top1_main_reason": _main_reason(top1),
                        "top5_view_names": ";".join(_view_name(row) for row in top),
                        "top5_corrupted_count": sum(1 for row in top if _was_corrupted(row) is True),
                        "best_ablation_signal": summary.get("best_ablation_signal", ""),
                        "full_signal_precision": full.get("precision_at_k", ""),
                        "full_signal_recall": full.get("recall_at_k", ""),
                        "loss_only_precision": loss.get("precision_at_k", ""),
                        "loss_only_recall": loss.get("recall_at_k", ""),
                        "lifecycle_only_precision": lifecycle.get("precision_at_k", ""),
                        "lifecycle_only_recall": lifecycle.get("recall_at_k", ""),
                        "does_full_beat_loss": _beats(full, loss),
                        "does_full_beat_lifecycle": _beats(full, lifecycle),
                        "status": "ok",
                        "warnings": ";".join(warnings),
                    }
                )
    return results, missing_rows, dirs


def build_component_comparison(
    *,
    condition_dirs: dict[tuple[str, str, str], Path | None],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (scene, subset_name, condition), signal_dir in sorted(condition_dirs.items()):
        if signal_dir is None or validate_condition_output(signal_dir)[0] != "ok":
            continue
        for row in _mapped_ablation_rows(load_condition_ablation(signal_dir)):
            rows.append(
                {
                    "scene": scene,
                    "subset_name": subset_name,
                    "condition": condition,
                    "signal_name": row.get("signal_name", ""),
                    "precision_at_k": row.get("precision_at_k", ""),
                    "recall_at_k": row.get("recall_at_k", ""),
                    "risk_gap_corrupted_minus_uncorrupted": row.get("risk_gap_corrupted_minus_uncorrupted", ""),
                    "corrupted_in_top_k": row.get("corrupted_in_top_k", ""),
                    "top1_view_name": row.get("top1_view_name", ""),
                    "top1_was_corrupted": row.get("top1_was_corrupted", ""),
                    "rank_within_condition": row.get("rank_within_condition", ""),
                    "status": row.get("status", "ok"),
                }
            )
    return rows


def build_subset_bias_summary(
    results: list[dict[str, Any]],
    subset_manifest_rows: list[dict[str, Any]],
    top_k: int,
) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    manifest = {
        (str(row.get("scene")), str(row.get("subset_name"))): row
        for row in subset_manifest_rows
    }
    for row in results:
        by_key[(str(row.get("scene")), str(row.get("subset_name")))].append(row)
    output: list[dict[str, Any]] = []
    for key, items in sorted(by_key.items()):
        scene, subset_name = key
        valid = [row for row in items if row.get("status") == "ok"]
        precision = [value for row in valid if (value := normalize_float(row.get("precision_at_k"))) is not None]
        recall = [value for row in valid if (value := normalize_float(row.get("recall_at_k"))) is not None]
        gaps = [value for row in valid if (value := normalize_float(row.get("risk_gap_corrupted_minus_uncorrupted"))) is not None]
        top5_union = sorted(
            {
                view
                for row in valid
                for view in str(row.get("top5_view_names", "")).split(";")
                if view
            }
        )
        corrupted_names = set(str(manifest.get(key, {}).get("corrupted_view_names", "")).split(";"))
        output.append(
            {
                "scene": scene,
                "subset_name": subset_name,
                "subset_seed": manifest.get(key, {}).get("subset_seed", ""),
                "condition_count_valid": len(valid),
                "mean_precision_at_k": _mean(precision),
                "mean_recall_at_k": _mean(recall),
                "median_precision_at_k": _median(precision),
                "median_recall_at_k": _median(recall),
                "mean_risk_gap": _mean(gaps),
                "corrupted_view_names": manifest.get(key, {}).get("corrupted_view_names", ""),
                "top1_view_names_by_condition": ";".join(
                    f"{row.get('condition')}:{row.get('top1_view_name')}" for row in valid
                ),
                "top5_union_view_names": ";".join(top5_union),
                "top5_corrupted_union_count": sum(1 for view in top5_union if view in corrupted_names),
                "top5_uncorrupted_union_count": sum(1 for view in top5_union if view not in corrupted_names),
                "full_signal_win_count_over_loss": sum(1 for row in valid if row.get("does_full_beat_loss") is True),
                "full_signal_win_count_over_lifecycle": sum(
                    1 for row in valid if row.get("does_full_beat_lifecycle") is True
                ),
                "repeated_false_positive_views": "",
                "status": "ok" if valid else "missing",
                "warnings": ";".join(str(row.get("warnings", "")) for row in items if row.get("warnings")),
            }
        )
    return output


def _ranking_events(
    condition_dirs: dict[tuple[str, str, str], Path | None],
    top_k: int,
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for (scene, subset_name, condition), signal_dir in sorted(condition_dirs.items()):
        if signal_dir is None or validate_condition_output(signal_dir)[0] != "ok":
            continue
        for row in load_condition_rankings(signal_dir):
            rank = _rank(row)
            events.append(
                {
                    "scene": scene,
                    "subset_name": subset_name,
                    "condition": condition,
                    "view_name": _view_name(row),
                    "rank": rank,
                    "risk": _risk(row),
                    "was_corrupted": _was_corrupted(row),
                    "main_reason": _main_reason(row),
                    "in_top1": rank is not None and rank <= 1,
                    "in_top3": rank is not None and rank <= 3,
                    "in_top5": rank is not None and rank <= top_k,
                }
            )
    return events


def build_view_identity_bias_table(
    condition_dirs: dict[tuple[str, str, str], Path | None],
    top_k: int,
) -> list[dict[str, Any]]:
    events = _ranking_events(condition_dirs, top_k)
    by_key: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        if event.get("view_name"):
            by_key[(str(event.get("scene")), str(event.get("view_name")))].append(event)
    rows: list[dict[str, Any]] = []
    for (scene, view_name), items in sorted(by_key.items()):
        corrupted = [row for row in items if row.get("was_corrupted") is True]
        uncorrupted = [row for row in items if row.get("was_corrupted") is False]
        ranks_corrupt = [float(row["rank"]) for row in corrupted if row.get("rank") is not None]
        ranks_clean = [float(row["rank"]) for row in uncorrupted if row.get("rank") is not None]
        risks_corrupt = [float(row["risk"]) for row in corrupted if row.get("risk") is not None]
        risks_clean = [float(row["risk"]) for row in uncorrupted if row.get("risk") is not None]
        top5_uncorrupted = [row for row in uncorrupted if row.get("in_top5") is True]
        top5_corrupted = [row for row in corrupted if row.get("in_top5") is True]
        top1_count = sum(1 for row in items if row.get("in_top1") is True)
        top5_count = sum(1 for row in items if row.get("in_top5") is True)
        bias_flag = (
            len(top5_uncorrupted) >= 2
            or (top1_count >= 2 and not corrupted)
            or (top5_count >= 3 and len(top5_uncorrupted) > 0 and len(top5_corrupted) > 0)
        )
        row = {
            "scene": scene,
            "view_name": view_name,
            "observed_condition_count": len(items),
            "corrupted_count": len(corrupted),
            "uncorrupted_count": len(uncorrupted),
            "top1_count": top1_count,
            "top3_count": sum(1 for item in items if item.get("in_top3") is True),
            "top5_count": top5_count,
            "top5_when_corrupted_count": len(top5_corrupted),
            "top5_when_uncorrupted_count": len(top5_uncorrupted),
            "mean_rank_when_corrupted": _mean(ranks_corrupt),
            "mean_rank_when_uncorrupted": _mean(ranks_clean),
            "mean_risk_when_corrupted": _mean(risks_corrupt),
            "mean_risk_when_uncorrupted": _mean(risks_clean),
            "risk_lift_corrupted_minus_uncorrupted": (
                _mean(risks_corrupt) - _mean(risks_clean)
                if _mean(risks_corrupt) is not None and _mean(risks_clean) is not None
                else None
            ),
            "rank_lift_corrupted_minus_uncorrupted": (
                _mean(ranks_clean) - _mean(ranks_corrupt)
                if _mean(ranks_clean) is not None and _mean(ranks_corrupt) is not None
                else None
            ),
            "is_repeated_top_view": top5_count >= 2,
            "is_repeated_false_positive": len(top5_uncorrupted) >= 2,
            "view_identity_bias_flag": bias_flag,
            "conditions_top5_when_uncorrupted": ";".join(
                f"{row.get('subset_name')}:{row.get('condition')}" for row in top5_uncorrupted
            ),
            "conditions_top5_when_corrupted": ";".join(
                f"{row.get('subset_name')}:{row.get('condition')}" for row in top5_corrupted
            ),
        }
        rows.append(row)
    return sorted(rows, key=lambda row: (str(row["scene"]), not row["view_identity_bias_flag"], str(row["view_name"])))


def _interpret_false_positive(reason: str) -> str:
    text = reason.lower()
    if "visibility" in text:
        return "visibility_overlap"
    if "loss" in text:
        return "loss_high_but_uncorrupted"
    if "lifecycle" in text or "birth" in text or "prune" in text or "survival" in text:
        return "clean_high_impact_view"
    return "unknown"


def build_repeated_false_positive_table(
    condition_dirs: dict[tuple[str, str, str], Path | None],
    view_identity_rows: list[dict[str, Any]],
    top_k: int,
) -> list[dict[str, Any]]:
    identity = {
        (str(row.get("scene")), str(row.get("view_name"))): row
        for row in view_identity_rows
    }
    events = _ranking_events(condition_dirs, top_k)
    rows: list[dict[str, Any]] = []
    for event in events:
        if not (event.get("in_top5") is True and event.get("was_corrupted") is False):
            continue
        stats = identity.get((str(event.get("scene")), str(event.get("view_name"))), {})
        if (normalize_int(stats.get("top5_when_uncorrupted_count")) or 0) < 2:
            continue
        rows.append(
            {
                "scene": event.get("scene", ""),
                "view_name": event.get("view_name", ""),
                "subset_name": event.get("subset_name", ""),
                "condition": event.get("condition", ""),
                "rank": event.get("rank", ""),
                "risk": event.get("risk", ""),
                "main_reason": event.get("main_reason", ""),
                "top5_when_uncorrupted_count": stats.get("top5_when_uncorrupted_count", ""),
                "mean_rank_when_uncorrupted": stats.get("mean_rank_when_uncorrupted", ""),
                "mean_risk_when_uncorrupted": stats.get("mean_risk_when_uncorrupted", ""),
                "possible_interpretation": _interpret_false_positive(str(event.get("main_reason", ""))),
            }
        )
    return rows


def build_scene_bias_summary(
    results: list[dict[str, Any]],
    view_identity_rows: list[dict[str, Any]],
    repeated_false_positive_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in results:
        by_scene[str(row.get("scene", ""))].append(row)
    rows: list[dict[str, Any]] = []
    for scene, items in sorted(by_scene.items()):
        valid = [row for row in items if row.get("status") == "ok"]
        precision = [value for row in valid if (value := normalize_float(row.get("precision_at_k"))) is not None]
        recall = [value for row in valid if (value := normalize_float(row.get("recall_at_k"))) is not None]
        gaps = [value for row in valid if (value := normalize_float(row.get("risk_gap_corrupted_minus_uncorrupted"))) is not None]
        top1_counts = Counter(row.get("top1_view_name", "") for row in valid if row.get("top1_view_name"))
        static_top1_rate = (max(top1_counts.values()) / len(valid)) if valid and top1_counts else 0.0
        identity_flags = [
            row for row in view_identity_rows if row.get("scene") == scene and row.get("view_identity_bias_flag") is True
        ]
        rows.append(
            {
                "scene": scene,
                "subset_count_valid": len({row.get("subset_name") for row in valid}),
                "condition_count_valid": len(valid),
                "mean_precision_at_k": _mean(precision),
                "mean_recall_at_k": _mean(recall),
                "median_precision_at_k": _median(precision),
                "median_recall_at_k": _median(recall),
                "mean_risk_gap": _mean(gaps),
                "full_signal_win_rate_over_loss": (
                    sum(1 for row in valid if row.get("does_full_beat_loss") is True) / len(valid)
                    if valid else 0.0
                ),
                "full_signal_win_rate_over_lifecycle": (
                    sum(1 for row in valid if row.get("does_full_beat_lifecycle") is True) / len(valid)
                    if valid else 0.0
                ),
                "static_top1_rate": static_top1_rate,
                "repeated_false_positive_count": sum(
                    1 for row in repeated_false_positive_rows if row.get("scene") == scene
                ),
                "view_identity_bias_warning": bool(identity_flags),
                "status": "ok" if valid else "missing",
                "warnings": ";".join(str(row.get("warnings", "")) for row in items if row.get("warnings")),
            }
        )
    return rows


def build_pr16_summary(
    *,
    scenes: list[str],
    conditions: list[str],
    subset_names: list[str],
    top_k: int,
    results: list[dict[str, Any]],
    scene_rows: list[dict[str, Any]],
    view_identity_rows: list[dict[str, Any]],
    repeated_false_positive_rows: list[dict[str, Any]],
    warnings: list[str],
) -> dict[str, Any]:
    valid = [row for row in results if row.get("status") == "ok"]
    missing = [row for row in results if row.get("status") == "missing"]
    precision = [value for row in valid if (value := normalize_float(row.get("precision_at_k"))) is not None]
    recall = [value for row in valid if (value := normalize_float(row.get("recall_at_k"))) is not None]
    gaps = [value for row in valid if (value := normalize_float(row.get("risk_gap_corrupted_minus_uncorrupted"))) is not None]
    scene_lookup = {row.get("scene"): row for row in scene_rows}
    return {
        "schema_name": "viewtrust.pr16.subset_scene_bias.summary",
        "schema_version": 1,
        "scenes": scenes,
        "conditions": conditions,
        "subset_names": subset_names,
        "top_k": top_k,
        "scene_count": len(scenes),
        "subset_count": len(subset_names),
        "condition_count": len(conditions),
        "valid_result_count": len(valid),
        "missing_result_count": len(missing),
        "mean_precision_at_k": _mean(precision),
        "mean_recall_at_k": _mean(recall),
        "mean_risk_gap": _mean(gaps),
        "chair_mean_recall_at_k": scene_lookup.get("chair", {}).get("mean_recall_at_k"),
        "drum_mean_recall_at_k": scene_lookup.get("drum", {}).get("mean_recall_at_k"),
        "full_signal_win_count_over_loss": sum(1 for row in valid if row.get("does_full_beat_loss") is True),
        "full_signal_win_count_over_lifecycle": sum(
            1 for row in valid if row.get("does_full_beat_lifecycle") is True
        ),
        "view_identity_bias_flags": [
            {"scene": row.get("scene"), "view_name": row.get("view_name")}
            for row in view_identity_rows
            if row.get("view_identity_bias_flag") is True
        ],
        "repeated_false_positive_views": sorted(
            {f"{row.get('scene')}:{row.get('view_name')}" for row in repeated_false_positive_rows}
        ),
        "subset_following_summary": {
            scene: {
                "valid_result_count": sum(1 for row in valid if row.get("scene") == scene),
                "mean_recall_at_k": scene_lookup.get(scene, {}).get("mean_recall_at_k"),
            }
            for scene in scenes
        },
        "observation_only": True,
        "training_intervention": False,
        "defense_enabled": False,
        "uses_corruption_labels_for_scoring": False,
        "uses_corruption_labels_for_evaluation": True,
        "warnings": warnings,
    }


def write_pr16_report(
    path: Path,
    *,
    summary: dict[str, Any],
    subset_rows: list[dict[str, Any]],
    scene_rows: list[dict[str, Any]],
    view_identity_rows: list[dict[str, Any]],
    repeated_false_positive_rows: list[dict[str, Any]],
    component_rows: list[dict[str, Any]],
    missing_rows: list[dict[str, Any]],
) -> None:
    chair = next((row for row in scene_rows if row.get("scene") == "chair"), {})
    drum = next((row for row in scene_rows if row.get("scene") == "drum"), {})
    train_013 = [row for row in view_identity_rows if row.get("view_name") == "train_013"]
    train_014 = [row for row in view_identity_rows if row.get("view_name") == "train_014"]
    noise_rows = [row for row in component_rows if row.get("condition") == "corrupt_noise"]
    loss_noise_wins = [row for row in noise_rows if row.get("signal_name") == "loss_only" and str(row.get("rank_within_condition")) == "1"]
    repeated_lines = [
        f"- `{row.get('scene')}` `{row.get('view_name')}` subset=`{row.get('subset_name')}` condition=`{row.get('condition')}` rank=`{row.get('rank')}`"
        for row in repeated_false_positive_rows[:20]
    ]
    identity_lines = [
        f"- `{row.get('scene')}` `{row.get('view_name')}` top5_clean=`{row.get('top5_when_uncorrupted_count')}` top5_corrupt=`{row.get('top5_when_corrupted_count')}`"
        for row in view_identity_rows
        if row.get("view_identity_bias_flag") is True
    ][:20]
    component_lines = [
        f"- `{row.get('scene')}` `{row.get('subset_name')}` `{row.get('condition')}` `{row.get('signal_name')}` recall=`{row.get('recall_at_k')}` precision=`{row.get('precision_at_k')}`"
        for row in component_rows[:30]
    ]
    missing_lines = [
        f"- `{row.get('scene')}` `{row.get('subset_name')}` `{row.get('condition')}` status=`{row.get('status')}`"
        for row in missing_rows[:30]
    ]
    report = "\n".join(
        [
            "# PR16 Subset and Scene Bias Probe",
            "",
            "## Purpose",
            "PR16 probes whether offline ViewTrust signal behavior is tied to fixed corrupted subsets, fixed view identity, or a single scene.",
            "",
            "## Inputs",
            f"- Scenes: `{', '.join(summary.get('scenes', []))}`",
            f"- Conditions: `{', '.join(summary.get('conditions', []))}`",
            f"- Subsets: `{', '.join(summary.get('subset_names', []))}`",
            f"- Top-k: `{summary.get('top_k')}`",
            "",
            "## Offline-only guarantee",
            REPORT_DISCLAIMER,
            "",
            "## Subset seed design",
            "Seeded subsets are generated from discovered training views with deterministic sampling and stable hashes. The original subset is inferred from existing outputs when available.",
            "",
            "## Chair results",
            f"- Mean recall@k: `{chair.get('mean_recall_at_k', '')}`",
            f"- Mean precision@k: `{chair.get('mean_precision_at_k', '')}`",
            f"- View identity warning: `{chair.get('view_identity_bias_warning', '')}`",
            "",
            "## Drum results",
            f"- Mean recall@k: `{drum.get('mean_recall_at_k', '')}`",
            f"- Mean precision@k: `{drum.get('mean_precision_at_k', '')}`",
            f"- View identity warning: `{drum.get('view_identity_bias_warning', '')}`",
            "",
            "## Cross-scene comparison",
            "Drum is included as a preliminary second-scene sanity check. Similar scene summaries suggest the offline signal may be less chair-specific; divergent summaries require scene-specific diagnosis.",
            "",
            "## Subset-bias diagnosis",
            "Top-ranked views should change when the corrupted subset changes. Fixed top-ranked views across subsets are treated as a view identity bias warning.",
            *(f"- `{row.get('scene')}` `{row.get('subset_name')}` recall=`{row.get('mean_recall_at_k')}` top5_union=`{row.get('top5_union_view_names')}`" for row in subset_rows),
            "",
            "## View-identity bias diagnosis",
            f"- train_013 rows present: `{len(train_013)}`. When present, inspect whether it remains high-risk when not corrupted.",
            f"- train_014 rows present: `{len(train_014)}`. When present, inspect whether it remains a repeated false positive.",
            *(identity_lines or ["- No repeated view-identity bias flags were found in available outputs."]),
            "",
            "## False positive analysis",
            *(repeated_lines or ["- No repeated uncorrupted top-k views were found in available outputs."]),
            "",
            "## Component comparison",
            f"- Noise loss-only rank-1 component rows: `{len(loss_noise_wins)}`. Inspect `corrupt_noise` rows to decide whether noise remains loss-dominated.",
            *(component_lines or ["- No component comparison rows were available."]),
            "",
            "## Interpretation guidance",
            "- Treat this as an offline bias probe, not an operational decision system.",
            "- Corruption labels are used only for post-hoc evaluation summaries.",
            "- Repeated clean top-k views may be candidate high-impact views rather than simple errors.",
            "",
            "## Limitations",
            "- still only two scenes/classes",
            "- still natural corruption only",
            "- not yet malicious poisoning",
            "- not yet full-scale scene validation",
            "- not yet training-time intervention",
            "- not causal proof",
            "- possible residual view-prior bias",
            "",
            "## Recommended next experiments",
            "1. If PR16 passes, expand to more scenes/classes.",
            "2. If PR16 shows view identity bias, add clean-prior normalization or delta-risk scoring.",
            "3. If drum fails but chair passes, investigate scene-specific visibility and camera path effects.",
            "4. If full_signal fails to beat loss_only under subset changes, redesign lifecycle weighting.",
            "5. Only after subset/scene bias is controlled, proceed to synthetic target-poison benchmark or training-time intervention.",
            "",
            "## Missing outputs",
            *(missing_lines or ["- None"]),
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")


def _file_type(path: Path) -> str:
    return path.suffix.lstrip(".") if path.suffix else ("directory" if path.is_dir() else "")


def _manifest_rows(
    *,
    output_dir: Path,
    output_files: list[str],
    inputs: list[tuple[str, Path, bool, str, str]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for relative_path, path, required, group, description in inputs:
        rows.append(
            {
                "relative_path": relative_path,
                "path": str(path),
                "exists": str(path.exists()).lower(),
                "file_type": "directory" if path.is_dir() else _file_type(path),
                "size_bytes": path.stat().st_size if path.is_file() else "",
                "required": str(required).lower(),
                "artifact_group": group,
                "description": description,
            }
        )
    for name in output_files:
        path = output_dir / name
        rows.append(
            {
                "relative_path": name,
                "path": str(path),
                "exists": str(path.exists()).lower(),
                "file_type": _file_type(path),
                "size_bytes": path.stat().st_size if path.is_file() else "",
                "required": "true",
                "artifact_group": "output_pr16",
                "description": "PR16 output artifact",
            }
        )
    return rows


def write_artifact_manifest(
    path: Path,
    *,
    output_dir: Path,
    output_files: list[str],
    inputs: list[tuple[str, Path, bool, str, str]],
) -> None:
    fields = [
        "relative_path",
        "path",
        "exists",
        "file_type",
        "size_bytes",
        "required",
        "artifact_group",
        "description",
    ]
    write_csv_rows(path, _manifest_rows(output_dir=output_dir, output_files=output_files, inputs=inputs), fields)
    write_csv_rows(path, _manifest_rows(output_dir=output_dir, output_files=output_files, inputs=inputs), fields)
