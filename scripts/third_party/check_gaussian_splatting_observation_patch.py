#!/usr/bin/env python3
"""Check whether ViewTrust observation patches are applied to Gaussian Splatting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PATCH_NAME_PR7 = "pr7_training_events"
PATCH_NAME_PR8 = "pr8_gaussian_lifecycle"
PATCH_NAME_PR12 = "pr12_view_influence_attribution"
SUPPORTED_PATCHES = {PATCH_NAME_PR7, PATCH_NAME_PR8, PATCH_NAME_PR12}
START = "# VIEWTRUST PR7 OBSERVATION START"
END = "# VIEWTRUST PR7 OBSERVATION END"
START_PR8 = "# VIEWTRUST PR8 GAUSSIAN LIFECYCLE START"
END_PR8 = "# VIEWTRUST PR8 GAUSSIAN LIFECYCLE END"
START_PR12 = "# VIEWTRUST PR12 VIEW INFLUENCE START"
END_PR12 = "# VIEWTRUST PR12 VIEW INFLUENCE END"


def inspect_pr7_patch(third_party_root: Path) -> dict[str, object]:
    train_path = third_party_root / "gaussian-splatting" / "train.py"
    exists = train_path.is_file()
    text = train_path.read_text(encoding="utf-8") if exists else ""
    marker_count = text.count(START)
    applied = marker_count > 0 and END in text
    checks = {
        "has_markers": applied,
        "has_env_gate": "VIEWTRUST_ENABLE_TRAINING_EVENTS" in text,
        "imports_observer": "TrainingEventObserver" in text,
        "uses_observer_environment": "TrainingEventObserver.from_environment()" in text,
        "has_finalize": "\"finalize\"" in text or ".finalize(" in text,
        "has_densification_hook": "\"log_densification_event\"" in text,
        "uses_bool_visibility_count": ".bool().sum()" in text,
        "uses_gaussian_count_visibility_ratio": "visible_count / gaussian_count" in text,
        "captures_render_time_gaussian_count": "viewtrust_gaussian_count_render = _viewtrust_pr7_count_gaussians(gaussians)" in text,
        "uses_render_count_for_iteration_metrics": "gaussian_count=viewtrust_gaussian_count_render" in text,
        "logs_after_render_count_stage": "stage=\"after_render\"" in text,
        "passes_requested_iterations": "requested_iterations=opt.iterations" in text,
        "keeps_densify_call": "gaussians.densify_and_prune(" in text,
        "keeps_optimizer_step": "gaussians.optimizer.step(" in text,
    }
    ok = exists and applied and all(checks.values())
    return {
        "patch": PATCH_NAME_PR7,
        "train_path": str(train_path),
        "exists": exists,
        "applied": applied,
        "marker_count": marker_count,
        "ok": ok,
        "checks": checks,
    }


def inspect_pr8_patch(third_party_root: Path) -> dict[str, object]:
    train_path = third_party_root / "gaussian-splatting" / "train.py"
    gaussian_model_path = third_party_root / "gaussian-splatting" / "scene" / "gaussian_model.py"
    exists = train_path.is_file()
    model_exists = gaussian_model_path.is_file()
    text = train_path.read_text(encoding="utf-8") if exists else ""
    model_text = gaussian_model_path.read_text(encoding="utf-8") if model_exists else ""
    lifecycle_marker_count = text.count(START_PR8) + model_text.count(START_PR8)
    checks = {
        "has_pr7_dependency": START in text and "VIEWTRUST_ENABLE_TRAINING_EVENTS" in text,
        "has_lifecycle_train_markers": START_PR8 in text and END_PR8 in text,
        "has_lifecycle_model_markers": START_PR8 in model_text and END_PR8 in model_text,
        "has_lifecycle_env_gate": "VIEWTRUST_ENABLE_GAUSSIAN_LIFECYCLE" in text,
        "imports_lifecycle_observer": "GaussianLifecycleObserver" in text,
        "attaches_lifecycle_observer": "viewtrust_lifecycle_observer" in text,
        "has_lifecycle_finalize": "\"finalize\"" in text
        and "viewtrust_lifecycle_observer" in text,
        "hooks_clone": "\"on_after_clone\"" in model_text,
        "hooks_split": "\"on_after_split\"" in model_text,
        "hooks_prune_before": "\"on_before_prune\"" in model_text,
        "hooks_prune_after": "\"on_after_prune\"" in model_text,
    }
    applied = checks["has_lifecycle_train_markers"] and checks["has_lifecycle_model_markers"]
    ok = exists and model_exists and applied and all(checks.values())
    return {
        "patch": PATCH_NAME_PR8,
        "train_path": str(train_path),
        "gaussian_model_path": str(gaussian_model_path),
        "exists": exists,
        "gaussian_model_exists": model_exists,
        "applied": applied,
        "lifecycle_marker_count": lifecycle_marker_count,
        "ok": ok,
        "checks": checks,
    }


def inspect_pr12_patch(third_party_root: Path) -> dict[str, object]:
    train_path = third_party_root / "gaussian-splatting" / "train.py"
    exists = train_path.is_file()
    text = train_path.read_text(encoding="utf-8") if exists else ""
    marker_count = text.count(START_PR12)
    checks = {
        "has_pr7_dependency": START in text and "VIEWTRUST_ENABLE_TRAINING_EVENTS" in text,
        "has_pr8_dependency": START_PR8 in text and "VIEWTRUST_ENABLE_GAUSSIAN_LIFECYCLE" in text,
        "has_pr12_markers": START_PR12 in text and END_PR12 in text,
        "has_view_name_helper": "_viewtrust_pr12_view_name" in text,
        "has_camera_uid_helper": "_viewtrust_pr12_camera_uid" in text,
        "has_view_split_helper": "_viewtrust_pr12_view_split" in text,
        "logs_training_event_view_name": "view_name=viewtrust_pr12_view_name" in text,
        "logs_training_event_camera_uid": "camera_uid=viewtrust_pr12_camera_uid" in text,
        "logs_training_event_view_split": "view_split=viewtrust_pr12_view_split" in text,
        "sets_lifecycle_source_context": "\"set_source_view_context\"" in text,
        "clears_lifecycle_source_context": "\"clear_source_view_context\"" in text,
        "keeps_densify_call": "gaussians.densify_and_prune(" in text,
        "keeps_optimizer_step": "gaussians.optimizer.step(" in text,
    }
    ok = exists and all(checks.values())
    return {
        "patch": PATCH_NAME_PR12,
        "train_path": str(train_path),
        "exists": exists,
        "applied": checks["has_pr12_markers"],
        "marker_count": marker_count,
        "ok": ok,
        "checks": checks,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--third-party-root", required=True, type=Path)
    parser.add_argument("--patch", default=PATCH_NAME_PR7)
    parser.add_argument("--require-applied", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.patch not in SUPPORTED_PATCHES:
        raise SystemExit(f"unsupported patch: {args.patch}")
    if args.patch == PATCH_NAME_PR7:
        report = inspect_pr7_patch(args.third_party_root)
    elif args.patch == PATCH_NAME_PR8:
        report = inspect_pr8_patch(args.third_party_root)
    else:
        report = inspect_pr12_patch(args.third_party_root)
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.require_applied and not report["ok"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
