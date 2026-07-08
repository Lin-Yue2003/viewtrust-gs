#!/usr/bin/env python3
"""Run a prepared chair condition through Priority 0 observed-command logging."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


def _bootstrap_project_imports() -> Path:
    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root))
    return project_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trainer", default="gaussian-splatting")
    parser.add_argument("--data-root", default=os.environ.get("VIEWTRUST_DATA_ROOT", "./data"))
    parser.add_argument(
        "--third-party-root",
        default=os.environ.get("VIEWTRUST_THIRD_PARTY_ROOT", "./third_party"),
    )
    parser.add_argument("--output-root", default=os.environ.get("VIEWTRUST_OUTPUT_ROOT", "./outputs"))
    parser.add_argument("--scene", default="chair")
    parser.add_argument("--condition", default="clean")
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--sample-interval-s", type=float, default=1.0)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--enable-training-events", action="store_true")
    parser.add_argument("--training-event-log-interval", type=int, default=10)
    parser.add_argument("--training-event-strict", action="store_true")
    parser.add_argument("--enable-gaussian-lifecycle", action="store_true")
    parser.add_argument("--gaussian-lifecycle-strict", action="store_true")
    parser.add_argument(
        "--gaussian-lifecycle-log-snapshot-stats",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--eval-split",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pass official Gaussian Splatting --eval so test cameras stay held out.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _resolve_path(project_root: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else (project_root / path).resolve()


def _metadata(
    *,
    args: argparse.Namespace,
    prepared_scene_root: Path,
    trainer_path: Path,
    command: list[str],
) -> dict[str, object]:
    return {
        "stage": "PR3_clean_baseline",
        "trainer": args.trainer,
        "scene": args.scene,
        "condition": args.condition,
        "iterations": args.iterations,
        "gpu": args.gpu,
        "prepared_scene_root": str(prepared_scene_root),
        "trainer_path": str(trainer_path),
        "command": command,
        "training_behavior_modified": False,
        "third_party_modified": False,
        "viewtrust_scoring_enabled": False,
        "defense_enabled": False,
        "training_events_enabled": args.enable_training_events,
        "training_event_log_interval": args.training_event_log_interval,
        "training_event_strict": args.training_event_strict,
        "gaussian_lifecycle_enabled": args.enable_gaussian_lifecycle,
        "gaussian_lifecycle_strict": args.gaussian_lifecycle_strict,
        "gaussian_lifecycle_log_snapshot_stats": args.gaussian_lifecycle_log_snapshot_stats,
        "eval_split": args.eval_split,
        "training_split_protocol": (
            "official_3dgs_eval_mode_train_only"
            if args.eval_split
            else "official_3dgs_non_eval_mode_may_merge_test"
        ),
    }


def main() -> int:
    project_root = _bootstrap_project_imports()

    from viewtrust.logging.writer import Priority0Logger
    from viewtrust.measurement.command_runner import run_observed_command
    from viewtrust.training.baseline import (
        BaselineTrainingConfig,
        build_baseline_label,
        build_gaussian_splatting_command,
        build_gaussian_lifecycle_env,
        build_training_event_env,
        preflight_gaussian_lifecycle_observer_import,
        preflight_training_event_observer_import,
        resolve_prepared_scene_root,
        resolve_trainer_path,
        validate_prepared_scene,
    )

    args = parse_args()
    config = BaselineTrainingConfig(
        trainer=args.trainer,
        data_root=_resolve_path(project_root, args.data_root),
        third_party_root=_resolve_path(project_root, args.third_party_root),
        output_root=_resolve_path(project_root, args.output_root),
        scene=args.scene,
        condition=args.condition,
        iterations=args.iterations,
        gpu=args.gpu,
        sample_interval_s=args.sample_interval_s,
        enable_training_events=args.enable_training_events,
        training_event_log_interval=args.training_event_log_interval,
        training_event_strict=args.training_event_strict,
        enable_gaussian_lifecycle=args.enable_gaussian_lifecycle,
        gaussian_lifecycle_strict=args.gaussian_lifecycle_strict,
        gaussian_lifecycle_log_snapshot_stats=args.gaussian_lifecycle_log_snapshot_stats,
        eval_split=args.eval_split,
    )

    label = build_baseline_label(config.scene, config.condition, config.trainer)
    prepared_scene_root = resolve_prepared_scene_root(
        config.data_root,
        config.scene,
        config.condition,
    )
    try:
        validate_prepared_scene(prepared_scene_root)
        trainer_path = resolve_trainer_path(config.trainer, config.third_party_root)
    except (FileNotFoundError, ValueError) as exc:
        print("ERROR: Prepared scene baseline preflight failed.", file=sys.stderr)
        print("", file=sys.stderr)
        print(str(exc), file=sys.stderr)
        return 2

    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = config.output_root / "baseline" / label / run_id
    trainer_output_dir = run_dir / "trainer_output"
    command = build_gaussian_splatting_command(
        trainer_path=trainer_path,
        prepared_scene_root=prepared_scene_root,
        trainer_output_dir=trainer_output_dir,
        iterations=config.iterations,
        eval_split=config.eval_split,
    )
    metadata = _metadata(
        args=args,
        prepared_scene_root=prepared_scene_root,
        trainer_path=trainer_path,
        command=command,
    )
    training_event_env = build_training_event_env(
        enabled=config.enable_training_events,
        project_root=project_root,
        run_dir=run_dir,
        run_id=run_id,
        scene=config.scene,
        condition=config.condition,
        trainer=config.trainer,
        log_interval=config.training_event_log_interval,
        strict=config.training_event_strict,
    )
    if config.enable_training_events:
        preflight = preflight_training_event_observer_import(
            python_executable=Path(command[0]),
            env_overrides={"CUDA_VISIBLE_DEVICES": str(config.gpu), **training_event_env},
            cwd=trainer_path.parent,
        )
        if preflight.returncode != 0:
            print(
                "ERROR: training events are enabled, but the official trainer child "
                "environment cannot import viewtrust.observation.training_events.",
                file=sys.stderr,
            )
            if preflight.stdout:
                print(preflight.stdout, file=sys.stderr)
            if preflight.stderr:
                print(preflight.stderr, file=sys.stderr)
            return 2

    gaussian_lifecycle_env = build_gaussian_lifecycle_env(
        enabled=config.enable_gaussian_lifecycle,
        project_root=project_root,
        run_dir=run_dir,
        run_id=run_id,
        scene=config.scene,
        condition=config.condition,
        trainer=config.trainer,
        strict=config.gaussian_lifecycle_strict,
        log_snapshot_stats=config.gaussian_lifecycle_log_snapshot_stats,
    )
    if config.enable_gaussian_lifecycle:
        preflight = preflight_gaussian_lifecycle_observer_import(
            python_executable=Path(command[0]),
            env_overrides={"CUDA_VISIBLE_DEVICES": str(config.gpu), **gaussian_lifecycle_env},
            cwd=trainer_path.parent,
        )
        if preflight.returncode != 0:
            print(
                "ERROR: Gaussian lifecycle logging is enabled, but the official "
                "trainer child environment cannot import "
                "viewtrust.observation.gaussian_lifecycle.",
                file=sys.stderr,
            )
            if preflight.stdout:
                print(preflight.stdout, file=sys.stderr)
            if preflight.stderr:
                print(preflight.stderr, file=sys.stderr)
            return 2

    config_snapshot = {
        "stage": "PR3_clean_baseline",
        "baseline": metadata,
        "output": {
            "run_dir": str(run_dir),
            "trainer_output_dir": str(trainer_output_dir),
            "training_events_dir": str(run_dir / "training_events"),
            "gaussian_lifecycle_dir": str(run_dir / "gaussian_lifecycle"),
        },
        "training_events": {
            "enabled": config.enable_training_events,
            "log_interval": config.training_event_log_interval,
            "strict": config.training_event_strict,
            "pythonpath_injected": "PYTHONPATH" in training_event_env,
        },
        "gaussian_lifecycle": {
            "enabled": config.enable_gaussian_lifecycle,
            "strict": config.gaussian_lifecycle_strict,
            "log_snapshot_stats": config.gaussian_lifecycle_log_snapshot_stats,
            "pythonpath_injected": "PYTHONPATH" in gaussian_lifecycle_env,
        },
        "observation_only": True,
    }

    dry_run_report = {
        "mode": "dry-run" if args.dry_run else "run",
        "validation": "ok",
        "run_dir": str(run_dir),
        "prepared_scene_root": str(prepared_scene_root),
        "trainer_path": str(trainer_path),
        "trainer_output_dir": str(trainer_output_dir),
        "cuda_visible_devices": str(config.gpu),
        "command": command,
        "training_events_enabled": config.enable_training_events,
        "training_event_env_keys": sorted(training_event_env),
        "training_event_strict": config.training_event_strict,
        "gaussian_lifecycle_enabled": config.enable_gaussian_lifecycle,
        "gaussian_lifecycle_env_keys": sorted(gaussian_lifecycle_env),
        "gaussian_lifecycle_strict": config.gaussian_lifecycle_strict,
        "eval_split": config.eval_split,
        "training_split_protocol": (
            "official_3dgs_eval_mode_train_only"
            if config.eval_split
            else "official_3dgs_non_eval_mode_may_merge_test"
        ),
    }
    print(json.dumps(dry_run_report, indent=2, sort_keys=True))

    if args.dry_run:
        print("prepared chair baseline dry-run ok")
        return 0

    logger = Priority0Logger(run_dir=run_dir, run_id=run_id)
    result = run_observed_command(
        command=command,
        logger=logger,
        config=config_snapshot,
        sample_interval_s=config.sample_interval_s,
        label=label,
        env_overrides={
            "CUDA_VISIBLE_DEVICES": str(config.gpu),
            **training_event_env,
            **gaussian_lifecycle_env,
        },
        metadata_extra=metadata,
    )
    print(f"prepared chair baseline run_dir: {result.run_dir}")
    print(f"returncode: {result.returncode}")
    print(f"elapsed_s: {result.elapsed_s:.6f}")
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
