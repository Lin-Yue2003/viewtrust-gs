"""Pure helpers for PR18 co-visibility spillover diagnosis."""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from viewtrust.analysis.clean_prior_normalization import load_csv_rows, load_json, normalize_bool
from viewtrust.analysis.offline_signals import safe_float


PR18_OUTPUT_FILES = [
    "pr18_covisibility_spillover_summary.json",
    "pr18_candidate_false_positive_diagnosis.csv",
    "pr18_camera_neighbor_table.csv",
    "pr18_view_pair_distance_table.csv",
    "pr18_gaussian_support_overlap.csv",
    "pr18_spillover_classification.csv",
    "pr18_condition_summary.csv",
    "pr18_view_identity_transition.csv",
    "pr18_missing_outputs.csv",
    "pr18_report.md",
    "artifact_manifest.csv",
]

REQUIRED_PR17_FILES = [
    "clean_prior_normalized_summary.json",
    "clean_prior_normalized_rows.csv",
    "clean_prior_normalized_rankings.csv",
]


def write_csv_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: "" if row.get(field) is None else row.get(field) for field in fieldnames})


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_pr17_rows(pr17_dir: Path) -> list[dict[str, str]]:
    return load_csv_rows(pr17_dir / "clean_prior_normalized_rows.csv")


def load_pr17_rankings(pr17_dir: Path) -> list[dict[str, str]]:
    return load_csv_rows(pr17_dir / "clean_prior_normalized_rankings.csv")


def load_pr16_plan(plan_dir: Path) -> dict[tuple[str, str], dict[str, Any]]:
    plan: dict[tuple[str, str], dict[str, Any]] = {}
    for row in load_csv_rows(plan_dir / "pr16_subset_manifest.csv"):
        scene = str(row.get("scene", ""))
        subset = str(row.get("subset_name", ""))
        views = _split_names(row.get("corrupted_view_names", ""))
        plan[(scene, subset)] = {
            "scene": scene,
            "subset_name": subset,
            "subset_seed": row.get("subset_seed", ""),
            "corrupted_view_names": views,
            "raw": row,
        }
    return plan


def _split_names(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    text = str(value or "")
    if not text:
        return []
    normalized = text.replace(",", ";")
    return [item.strip() for item in normalized.split(";") if item.strip()]


def _truth(value: Any) -> bool:
    return normalize_bool(value) is True


def _float(value: Any, default: float = 0.0) -> float:
    number = safe_float(value)
    return default if number is None else number


def _view_index(view_name: str) -> int | None:
    token = view_name.rsplit("_", 1)[-1]
    try:
        return int(token)
    except ValueError:
        return None


def _view_from_file_path(file_path: str) -> str:
    return Path(file_path).stem


def _transform_candidates(data_root: Path, scene: str) -> list[Path]:
    return [
        data_root / "viewtrust-mini" / "nerf_synthetic" / scene / "clean" / "transforms_train.json",
        data_root / "raw" / "nerf_synthetic" / scene / "transforms_train.json",
        data_root / "nerf_synthetic" / scene / "clean" / "transforms_train.json",
        data_root / scene / "clean" / "transforms_train.json",
        data_root / scene / "transforms_train.json",
    ]


def load_camera_poses(data_root: Path, scene: str) -> tuple[dict[str, dict[str, Any]], Path | None]:
    for path in _transform_candidates(data_root, scene):
        payload = load_json(path)
        frames = payload.get("frames", []) if payload else []
        poses: dict[str, dict[str, Any]] = {}
        for frame in frames:
            matrix = frame.get("transform_matrix")
            file_path = str(frame.get("file_path", ""))
            if not file_path or not _valid_matrix(matrix):
                continue
            view_name = _view_from_file_path(file_path)
            poses[view_name] = {
                "view_name": view_name,
                "matrix": matrix,
                "center": [float(matrix[0][3]), float(matrix[1][3]), float(matrix[2][3])],
                "rotation": [list(map(float, row[:3])) for row in matrix[:3]],
            }
        if poses:
            return poses, path
    return {}, None


def _valid_matrix(matrix: Any) -> bool:
    if not isinstance(matrix, list) or len(matrix) < 3:
        return False
    return all(isinstance(row, list) and len(row) >= 4 for row in matrix[:3])


def _center_distance(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((left - right) ** 2 for left, right in zip(a, b)))


def _rotation_angle_deg(a: list[list[float]], b: list[list[float]]) -> float:
    trace = 0.0
    for i in range(3):
        for j in range(3):
            trace += a[j][i] * b[j][i]
    cosine = max(-1.0, min(1.0, (trace - 1.0) / 2.0))
    return math.degrees(math.acos(cosine))


def _median_positive(values: list[float], default: float = 1.0) -> float:
    positive = [value for value in values if value > 0.0]
    return statistics.median(positive) if positive else default


def compute_camera_pair_distances(
    poses: dict[str, dict[str, Any]],
    *,
    center_weight: float = 1.0,
    rotation_weight: float = 1.0,
) -> tuple[list[dict[str, Any]], dict[tuple[str, str], dict[str, Any]], dict[str, float]]:
    names = sorted(poses)
    raw_pairs: list[dict[str, Any]] = []
    nearest_center: dict[str, float] = {}
    nearest_rotation: dict[str, float] = {}
    for index, view_a in enumerate(names):
        for view_b in names[index + 1 :]:
            center = _center_distance(poses[view_a]["center"], poses[view_b]["center"])
            rotation = _rotation_angle_deg(poses[view_a]["rotation"], poses[view_b]["rotation"])
            raw_pairs.append(
                {
                    "view_a": view_a,
                    "view_b": view_b,
                    "center_distance": center,
                    "rotation_angle_deg": rotation,
                    "index_gap": _index_gap(view_a, view_b),
                }
            )
            nearest_center[view_a] = min(nearest_center.get(view_a, center), center)
            nearest_center[view_b] = min(nearest_center.get(view_b, center), center)
            nearest_rotation[view_a] = min(nearest_rotation.get(view_a, rotation), rotation)
            nearest_rotation[view_b] = min(nearest_rotation.get(view_b, rotation), rotation)

    median_center = _median_positive(list(nearest_center.values()))
    median_rotation = _median_positive(list(nearest_rotation.values()))
    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for pair in raw_pairs:
        combined = (
            center_weight * (_float(pair["center_distance"]) / median_center)
            + rotation_weight * (_float(pair["rotation_angle_deg"]) / median_rotation)
        )
        row = {
            **pair,
            "combined_camera_distance": combined,
            "scene_median_center_distance": median_center,
            "scene_median_rotation_angle_deg": median_rotation,
        }
        rows.append(row)
        lookup[(row["view_a"], row["view_b"])] = row
        lookup[(row["view_b"], row["view_a"])] = {
            **row,
            "view_a": row["view_b"],
            "view_b": row["view_a"],
        }
    stats = {
        "median_center_distance": median_center,
        "median_rotation_angle_deg": median_rotation,
        "median_combined_nearest_distance": _median_positive(_nearest_combined_by_view(rows)),
    }
    return rows, lookup, stats


def _nearest_combined_by_view(pair_rows: list[dict[str, Any]]) -> list[float]:
    nearest: dict[str, float] = {}
    for row in pair_rows:
        value = _float(row.get("combined_camera_distance"))
        for key in ("view_a", "view_b"):
            view = str(row[key])
            nearest[view] = min(nearest.get(view, value), value)
    return list(nearest.values())


def _index_gap(view_a: str, view_b: str) -> int | None:
    a = _view_index(view_a)
    b = _view_index(view_b)
    if a is None or b is None:
        return None
    return abs(a - b)


def compute_corrupted_neighbor_features(
    view_names: list[str],
    corrupted_views: list[str],
    pair_lookup: dict[tuple[str, str], dict[str, Any]],
    pair_rows: list[dict[str, Any]],
    stats: dict[str, float],
    *,
    neighbor_k: int,
    median_nn_distance_factor: float,
) -> dict[str, dict[str, Any]]:
    all_neighbors: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in pair_rows:
        all_neighbors[str(row["view_a"])].append({"view": row["view_b"], **row})
        reverse = pair_lookup.get((str(row["view_b"]), str(row["view_a"])))
        if reverse:
            all_neighbors[str(row["view_b"])].append({"view": row["view_a"], **reverse})
    corrupted = [view for view in corrupted_views if view in view_names]
    output: dict[str, dict[str, Any]] = {}
    threshold = stats.get("median_combined_nearest_distance", 1.0) * median_nn_distance_factor
    for view_name in view_names:
        distances = []
        for corrupted_view in corrupted:
            if corrupted_view == view_name:
                continue
            pair = pair_lookup.get((view_name, corrupted_view))
            if pair:
                distances.append({"corrupted_view": corrupted_view, **pair})
        nearest = min(distances, key=lambda row: _float(row.get("combined_camera_distance")), default={})
        neighbor_order = sorted(
            all_neighbors.get(view_name, []),
            key=lambda row: (_float(row.get("combined_camera_distance")), str(row.get("view", ""))),
        )
        rank = None
        for index, row in enumerate(neighbor_order, start=1):
            if row.get("view") == nearest.get("corrupted_view"):
                rank = index
                break
        sorted_corrupted = [
            f"{row['corrupted_view']}:{row.get('combined_camera_distance')}"
            for row in sorted(distances, key=lambda row: _float(row.get("combined_camera_distance")))
        ]
        nearest_combined = safe_float(nearest.get("combined_camera_distance"))
        camera_evidence = bool(
            nearest
            and (
                (rank is not None and rank <= neighbor_k)
                or (nearest_combined is not None and nearest_combined <= threshold)
            )
        )
        output[view_name] = {
            "nearest_corrupted_view": nearest.get("corrupted_view", ""),
            "nearest_corrupted_center_distance": nearest.get("center_distance", ""),
            "nearest_corrupted_rotation_angle_deg": nearest.get("rotation_angle_deg", ""),
            "nearest_corrupted_combined_distance": nearest.get("combined_camera_distance", ""),
            "corrupted_neighbor_rank": rank,
            "mean_distance_to_corrupted_views": statistics.fmean(
                [_float(row.get("combined_camera_distance")) for row in distances]
            )
            if distances
            else "",
            "is_camera_neighbor_of_corrupted": camera_evidence,
            "camera_neighbor_evidence": camera_evidence,
            "all_corrupted_neighbors_sorted": ";".join(sorted_corrupted),
        }
    return output


def compute_index_neighbor_features(
    view_names: list[str],
    corrupted_views: list[str],
    *,
    max_index_gap: int,
) -> dict[str, dict[str, Any]]:
    corrupted_indices = {
        view: _view_index(view)
        for view in corrupted_views
        if _view_index(view) is not None
    }
    output: dict[str, dict[str, Any]] = {}
    index_values = sorted(corrupted_indices.values())
    for view_name in view_names:
        index = _view_index(view_name)
        gaps = [
            abs(index - corrupted_index)
            for corrupted_index in corrupted_indices.values()
            if index is not None and corrupted_index is not None
        ]
        nearest_gap = min(gaps) if gaps else None
        between = bool(index_values and index is not None and min(index_values) < index < max(index_values))
        neighbors = [
            view
            for view, corrupted_index in corrupted_indices.items()
            if index is not None and corrupted_index is not None and abs(index - corrupted_index) <= max_index_gap
        ]
        adjacent = nearest_gap is not None and nearest_gap <= max_index_gap
        output[view_name] = {
            "nearest_corrupted_index_gap": nearest_gap,
            "between_corrupted_indices": between,
            "adjacent_to_corrupted_index": adjacent,
            "corrupted_index_neighbors": ";".join(sorted(neighbors)),
            "index_neighbor_evidence": adjacent or (between and nearest_gap is not None and nearest_gap <= max_index_gap),
        }
    return output


def resolve_offline_artifact_inputs(signal_dir: Path) -> dict[str, Path]:
    output: dict[str, Path] = {}
    for row in load_csv_rows(signal_dir / "offline_viewtrust_artifact_manifest.csv"):
        raw_path = str(row.get("path", ""))
        if not raw_path:
            continue
        path = Path(raw_path)
        relative = str(row.get("relative_path", ""))
        group = str(row.get("artifact_group", ""))
        if group == "input_clean" or relative.startswith("input_clean/"):
            output.setdefault("input_clean_dir", path.parent)
        elif group == "input_corrupt" or relative.startswith("input_corrupt/"):
            output.setdefault("input_corrupt_dir", path.parent)
        elif group == "input_comparison" or relative.startswith("input_comparison/"):
            output.setdefault("input_comparison_dir", path.parent)
    return output


def _support_sets_from_rows(rows: list[dict[str, str]], view_key: str) -> dict[str, set[str]]:
    supports: dict[str, set[str]] = defaultdict(set)
    id_keys = ["gaussian_id", "parent_id", "child_gaussian_id", "source_gaussian_id"]
    list_keys = ["gaussian_ids", "affected_gaussian_ids"]
    for row in rows:
        view_name = str(row.get(view_key, "") or row.get("view_name", ""))
        if not view_name:
            continue
        for key in id_keys:
            value = str(row.get(key, "") or "")
            if value:
                supports[view_name].add(value)
        for key in list_keys:
            for value in _split_names(row.get(key, "")):
                supports[view_name].add(value)
    return supports


def _proxy_vectors(paths: dict[str, Path]) -> dict[str, dict[str, float]]:
    candidates = []
    comparison_dir = paths.get("input_comparison_dir")
    corrupt_dir = paths.get("input_corrupt_dir")
    if comparison_dir:
        candidates.append(comparison_dir / "view_influence_comparison.csv")
    if corrupt_dir:
        candidates.append(corrupt_dir / "view_influence.csv")
    keys = [
        "birth_event_count_delta",
        "prune_death_count_delta",
        "visibility_ratio_delta",
        "birth_survival_ratio_delta",
        "birth_event_count_after_view",
        "prune_death_count_after_view",
        "mean_visibility_ratio",
    ]
    for path in candidates:
        rows = load_csv_rows(path)
        vectors: dict[str, dict[str, float]] = {}
        for row in rows:
            view_name = str(row.get("view_name", ""))
            vector = {
                key: abs(value)
                for key in keys
                if (value := _float(row.get(key), default=float("nan"))) == value
            }
            if view_name and vector:
                vectors[view_name] = vector
        if vectors:
            return vectors
    return {}


def _weighted_jaccard(left: dict[str, float], right: dict[str, float]) -> float | None:
    keys = set(left) | set(right)
    if not keys:
        return None
    numerator = sum(min(left.get(key, 0.0), right.get(key, 0.0)) for key in keys)
    denominator = sum(max(left.get(key, 0.0), right.get(key, 0.0)) for key in keys)
    return numerator / denominator if denominator > 0.0 else None


def compute_gaussian_support_overlap_best_effort(
    *,
    signal_dir: Path | None,
    candidate_views: list[str],
    corrupted_views: list[str],
    mode: str,
    evidence_threshold: float = 0.25,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    by_candidate: dict[str, dict[str, Any]] = {}
    if mode == "off" or signal_dir is None:
        return _unavailable_overlap(candidate_views, corrupted_views, "unavailable", "signal_dir unavailable")
    paths = resolve_offline_artifact_inputs(signal_dir)
    supports: dict[str, set[str]] = {}
    overlap_source = "unavailable"
    if mode in {"auto", "exact"}:
        for key in ("input_corrupt_dir", "input_clean_dir"):
            root = paths.get(key)
            if not root:
                continue
            supports.update(_support_sets_from_rows(load_csv_rows(root / "view_lifecycle_attribution.csv"), "source_view_name"))
            supports.update(_support_sets_from_rows(load_csv_rows(root / "view_iteration_events.csv"), "view_name"))
        if any(supports.values()):
            overlap_source = "exact"
    vectors = {}
    if overlap_source == "unavailable" and mode in {"auto", "proxy"}:
        vectors = _proxy_vectors(paths)
        if vectors:
            overlap_source = "proxy"
    if overlap_source == "unavailable":
        return _unavailable_overlap(candidate_views, corrupted_views, "unavailable", "no exact or proxy support fields available")

    for candidate in candidate_views:
        best = {"jaccard": None, "corrupted_view": ""}
        jaccards: list[float] = []
        for corrupted in corrupted_views:
            if overlap_source == "exact":
                left = supports.get(candidate, set())
                right = supports.get(corrupted, set())
                union = left | right
                intersection = left & right
                jaccard = len(intersection) / len(union) if union else None
                row = {
                    "candidate_support_count": len(left),
                    "corrupted_support_count": len(right),
                    "intersection_count": len(intersection),
                    "union_count": len(union),
                }
            else:
                jaccard = _weighted_jaccard(vectors.get(candidate, {}), vectors.get(corrupted, {}))
                row = {
                    "candidate_support_count": len(vectors.get(candidate, {})),
                    "corrupted_support_count": len(vectors.get(corrupted, {})),
                    "intersection_count": "",
                    "union_count": "",
                }
            available = jaccard is not None
            if available:
                jaccards.append(jaccard)
                if best["jaccard"] is None or jaccard > best["jaccard"]:
                    best = {"jaccard": jaccard, "corrupted_view": corrupted}
            rows.append(
                {
                    "candidate_view": candidate,
                    "corrupted_view": corrupted,
                    "overlap_source": overlap_source,
                    **row,
                    "jaccard": jaccard,
                    "overlap_available": available,
                    "warnings": "",
                }
            )
        by_candidate[candidate] = {
            "max_gaussian_support_jaccard_with_corrupted": best["jaccard"],
            "mean_gaussian_support_jaccard_with_corrupted": statistics.fmean(jaccards) if jaccards else None,
            "nearest_corrupted_by_gaussian_overlap": best["corrupted_view"],
            "gaussian_overlap_source": overlap_source,
            "gaussian_overlap_evidence": best["jaccard"] is not None and best["jaccard"] >= evidence_threshold,
        }
    return rows, by_candidate


def _unavailable_overlap(
    candidate_views: list[str],
    corrupted_views: list[str],
    source: str,
    warning: str,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    rows = [
        {
            "candidate_view": candidate,
            "corrupted_view": corrupted,
            "overlap_source": source,
            "candidate_support_count": "",
            "corrupted_support_count": "",
            "intersection_count": "",
            "union_count": "",
            "jaccard": "",
            "overlap_available": False,
            "warnings": warning,
        }
        for candidate in candidate_views
        for corrupted in corrupted_views
    ]
    by_candidate = {
        candidate: {
            "max_gaussian_support_jaccard_with_corrupted": "",
            "mean_gaussian_support_jaccard_with_corrupted": "",
            "nearest_corrupted_by_gaussian_overlap": "",
            "gaussian_overlap_source": source,
            "gaussian_overlap_evidence": False,
        }
        for candidate in candidate_views
    }
    return rows, by_candidate


def _quantile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    index = max(0, min(len(ordered) - 1, math.ceil(fraction * (len(ordered) - 1))))
    return ordered[index]


def classify_spillover_candidates(
    *,
    candidate_rows: list[dict[str, Any]],
    group_rows: list[dict[str, Any]],
    camera_features: dict[str, dict[str, Any]],
    index_features: dict[str, dict[str, Any]],
    gaussian_features: dict[str, dict[str, Any]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    lift_config = config.get("lift_pattern", {})
    low_prior_threshold = _quantile(
        [_float(row.get("clean_prior_risk")) for row in group_rows],
        _float(lift_config.get("low_prior_quantile"), 0.5),
    )
    high_delta_threshold = _quantile(
        [_float(row.get("positive_delta_risk")) for row in group_rows],
        _float(lift_config.get("high_delta_quantile"), 0.75),
    )
    high_lift_threshold = _quantile(
        [_float(row.get("rank_lift_score")) for row in group_rows],
        _float(lift_config.get("high_rank_lift_quantile"), 0.75),
    )
    output: list[dict[str, Any]] = []
    for row in candidate_rows:
        view_name = str(row.get("view_name", ""))
        camera = camera_features.get(view_name, {})
        index = index_features.get(view_name, {})
        gaussian = gaussian_features.get(view_name, {})
        was_corrupted = _truth(row.get("was_corrupted"))
        stable_prior_pattern = (
            _float(row.get("clean_prior_risk")) > low_prior_threshold
            and _float(row.get("delta_risk")) <= 0.25 * max(low_prior_threshold, 1.0)
        )
        collateral_lift_pattern = (
            _float(row.get("clean_prior_risk")) <= low_prior_threshold
            and _float(row.get("positive_delta_risk")) >= high_delta_threshold
            and _float(row.get("rank_lift_score")) >= high_lift_threshold
        )
        camera_evidence = _truth(camera.get("camera_neighbor_evidence"))
        index_evidence = _truth(index.get("index_neighbor_evidence"))
        gaussian_evidence = _truth(gaussian.get("gaussian_overlap_evidence"))
        if stable_prior_pattern:
            spillover_class = "clean_prior_false_positive"
            confidence = "medium"
            explanation = "high clean-prior risk with small delta/rank lift"
        elif (
            not was_corrupted
            and _truth(row.get("normalized_top_k"))
            and collateral_lift_pattern
            and (camera_evidence or index_evidence or gaussian_evidence)
        ):
            spillover_class = "co_visible_collateral"
            confidence = "high" if camera_evidence or gaussian_evidence else "medium"
            explanation = "low clean prior plus positive normalized lift with co-visibility evidence"
        elif not was_corrupted and _truth(row.get("normalized_top_k")):
            spillover_class = "unexplained_false_positive"
            confidence = "low"
            explanation = "normalized top-k clean view without clean-prior or co-visibility evidence"
        else:
            spillover_class = "prior_demoted"
            confidence = "medium" if stable_prior_pattern else "low"
            explanation = "raw false positive not retained in normalized top-k"
        output.append(
            {
                **row,
                **camera,
                **index,
                **gaussian,
                "stable_prior_pattern": stable_prior_pattern,
                "collateral_lift_pattern": collateral_lift_pattern,
                "camera_neighbor_evidence": camera_evidence,
                "index_neighbor_evidence": index_evidence,
                "gaussian_overlap_evidence": gaussian_evidence,
                "spillover_class": spillover_class,
                "spillover_confidence": confidence,
                "explanation": explanation,
            }
        )
    return output


def compute_spillover_summary(
    *,
    scenes: list[str],
    conditions: list[str],
    subset_names: list[str],
    top_k: int,
    condition_rows: list[dict[str, Any]],
    classification_rows: list[dict[str, Any]],
    missing_rows: list[dict[str, Any]],
    gaussian_overlap_rows: list[dict[str, Any]],
    warnings: list[str],
) -> dict[str, Any]:
    normalized_fp = [row for row in classification_rows if _truth(row.get("normalized_false_positive"))]
    class_counts = defaultdict(int)
    for row in normalized_fp:
        class_counts[str(row.get("spillover_class", ""))] += 1
    by_view = {(row.get("scene", ""), row.get("view_name", "")): row for row in classification_rows}

    def classes_for(view_name: str) -> list[str]:
        return sorted(
            {
                str(row.get("spillover_class"))
                for row in classification_rows
                if row.get("view_name") == view_name and row.get("spillover_class")
            }
        )

    return {
        "schema_name": "viewtrust.pr18.covisibility_spillover.summary",
        "schema_version": 1,
        "scenes": scenes,
        "conditions": conditions,
        "subset_names": subset_names,
        "top_k": top_k,
        "valid_condition_count": sum(1 for row in condition_rows if row.get("status") == "ok"),
        "missing_condition_count": len(missing_rows),
        "normalized_false_positive_count": len(normalized_fp),
        "co_visible_collateral_count": class_counts["co_visible_collateral"],
        "clean_prior_false_positive_count": class_counts["clean_prior_false_positive"],
        "unexplained_false_positive_count": class_counts["unexplained_false_positive"],
        "collateral_view_names": sorted({row["view_name"] for row in normalized_fp if row.get("spillover_class") == "co_visible_collateral"}),
        "unexplained_view_names": sorted({row["view_name"] for row in normalized_fp if row.get("spillover_class") == "unexplained_false_positive"}),
        "train_014_spillover_class": classes_for("train_014"),
        "train_007_spillover_class": classes_for("train_007"),
        "train_013_spillover_class": classes_for("train_013"),
        "gaussian_overlap_available_count": sum(1 for row in gaussian_overlap_rows if _truth(row.get("overlap_available"))),
        "gaussian_overlap_unavailable_count": sum(1 for row in gaussian_overlap_rows if not _truth(row.get("overlap_available"))),
        "uses_corruption_labels_for_scoring": False,
        "uses_corruption_labels_for_evaluation": True,
        "observation_only": True,
        "training_intervention": False,
        "defense_enabled": False,
        "warnings": warnings,
        "_view_lookup_count": len(by_view),
    }


def build_view_identity_transition(classification_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in classification_rows:
        grouped[(str(row.get("scene", "")), str(row.get("view_name", "")))].append(row)
    output: list[dict[str, Any]] = []
    for (scene, view_name), rows in sorted(grouped.items()):
        raw_count = sum(1 for row in rows if _truth(row.get("raw_false_positive")))
        norm_count = sum(1 for row in rows if _truth(row.get("normalized_false_positive")))
        collateral = [row for row in rows if row.get("spillover_class") == "co_visible_collateral"]
        clean_prior = [row for row in rows if row.get("spillover_class") == "clean_prior_false_positive"]
        unexplained = [row for row in rows if row.get("spillover_class") == "unexplained_false_positive"]
        if collateral:
            transition = "collateral_after_normalization"
        elif clean_prior and norm_count == 0:
            transition = "clean_prior_demoted"
        elif unexplained:
            transition = "unexplained_after_normalization"
        else:
            transition = "mixed_or_not_repeated"
        output.append(
            {
                "scene": scene,
                "view_name": view_name,
                "raw_false_positive_count": raw_count,
                "normalized_false_positive_count": norm_count,
                "co_visible_collateral_count": len(collateral),
                "clean_prior_false_positive_count": len(clean_prior),
                "unexplained_false_positive_count": len(unexplained),
                "transition_type": transition,
                "conditions_as_collateral": ";".join(sorted(f"{row.get('subset_name')}:{row.get('condition')}" for row in collateral)),
                "conditions_unexplained": ";".join(sorted(f"{row.get('subset_name')}:{row.get('condition')}" for row in unexplained)),
                "conditions_clean_prior": ";".join(sorted(f"{row.get('subset_name')}:{row.get('condition')}" for row in clean_prior)),
            }
        )
    return output


def write_artifact_manifest(output_dir: Path, pr17_dir: Path, plan_dir: Path, data_root: Path, input_root: Path) -> None:
    fields = ["relative_path", "path", "exists", "file_type", "size_bytes", "required", "artifact_group"]

    def rows() -> list[dict[str, Any]]:
        items: list[tuple[str, Path, bool, str]] = [
            ("input_pr17_dir", pr17_dir, True, "input"),
            ("input_plan_dir", plan_dir, True, "input"),
            ("input_data_root", data_root, False, "input"),
            ("input_root", input_root, False, "input"),
        ]
        items.extend((name, output_dir / name, True, "output_pr18") for name in PR18_OUTPUT_FILES)
        output = []
        for relative, path, required, group in items:
            output.append(
                {
                    "relative_path": relative,
                    "path": str(path),
                    "exists": str(path.exists()).lower(),
                    "file_type": "directory" if path.is_dir() else path.suffix.lstrip("."),
                    "size_bytes": path.stat().st_size if path.is_file() else "",
                    "required": str(required).lower(),
                    "artifact_group": group,
                }
            )
        return output

    manifest = output_dir / "artifact_manifest.csv"
    write_csv_rows(manifest, rows(), fields)
    write_csv_rows(manifest, rows(), fields)
