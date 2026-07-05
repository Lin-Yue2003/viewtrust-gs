"""Clean baseline training wrapper helpers.

This module is local-safe and pure Python. It composes external training
commands but does not import CUDA libraries or modify trainer internals.
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BaselineTrainingConfig:
    trainer: str
    data_root: Path
    third_party_root: Path
    output_root: Path
    scene: str
    condition: str
    iterations: int
    gpu: int
    sample_interval_s: float
    enable_training_events: bool = False
    training_event_log_interval: int = 10
    training_event_strict: bool = False
    enable_gaussian_lifecycle: bool = False
    gaussian_lifecycle_strict: bool = False
    gaussian_lifecycle_log_snapshot_stats: bool = True


def build_baseline_label(scene: str, condition: str, trainer: str) -> str:
    normalized_trainer = trainer.replace("-", "_")
    return f"{scene}_{condition}_{normalized_trainer}"


def resolve_prepared_scene_root(data_root: Path, scene: str, condition: str) -> Path:
    return (
        data_root
        / "viewtrust-mini"
        / "nerf_synthetic"
        / scene
        / condition
    ).resolve()


def validate_prepared_scene(prepared_scene_root: Path) -> None:
    required_files = [
        "transforms_train.json",
        "transforms_test.json",
        "manifest.json",
    ]
    missing = [
        str(prepared_scene_root / name)
        for name in required_files
        if not (prepared_scene_root / name).is_file()
    ]
    images_dir = prepared_scene_root / "images"
    if not images_dir.is_dir():
        missing.append(str(images_dir))

    if missing:
        details = "\n".join(f"  {path}" for path in missing)
        raise FileNotFoundError(
            "Prepared NeRF Synthetic chair subset is missing required files:\n"
            f"{details}\n\n"
            "Prepare it with scripts/data/prepare_nerf_synthetic_subset.py first."
        )


def resolve_trainer_path(trainer: str, third_party_root: Path) -> Path:
    if trainer != "gaussian-splatting":
        raise ValueError("PR3 supports only --trainer gaussian-splatting")

    trainer_path = (third_party_root / "gaussian-splatting" / "train.py").resolve()
    if not trainer_path.is_file():
        raise FileNotFoundError(
            "Gaussian Splatting trainer was not found.\n\n"
            f"Expected:\n  {trainer_path}\n\n"
            "Clone or symlink the official repo under:\n"
            f"  {third_party_root / 'gaussian-splatting'}\n\n"
            "Do not vendor third_party code into the ViewTrust-GS repo."
        )
    return trainer_path


def build_gaussian_splatting_command(
    *,
    trainer_path: Path,
    prepared_scene_root: Path,
    trainer_output_dir: Path,
    iterations: int,
) -> list[str]:
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    return [
        sys.executable,
        str(trainer_path),
        "-s",
        str(prepared_scene_root),
        "-m",
        str(trainer_output_dir),
        "--iterations",
        str(iterations),
    ]


def build_training_event_env(
    *,
    enabled: bool,
    project_root: Path,
    run_dir: Path,
    run_id: str,
    scene: str,
    condition: str,
    trainer: str,
    log_interval: int,
    strict: bool = False,
    base_env: dict[str, str] | None = None,
) -> dict[str, str]:
    """Build opt-in environment variables for PR7 training event observation."""

    if not enabled:
        return {}
    if log_interval <= 0:
        raise ValueError("training_event_log_interval must be positive")
    resolved_project_root = project_root.resolve()
    existing_pythonpath = (base_env or os.environ).get("PYTHONPATH", "")
    pythonpath_parts = [
        str(resolved_project_root),
        *[
            part
            for part in existing_pythonpath.split(os.pathsep)
            if part and part != str(resolved_project_root)
        ],
    ]
    env = {
        "PYTHONPATH": os.pathsep.join(pythonpath_parts),
        "VIEWTRUST_PROJECT_ROOT": str(resolved_project_root),
        "VIEWTRUST_ENABLE_TRAINING_EVENTS": "1",
        "VIEWTRUST_TRAINING_EVENTS_DIR": str(run_dir / "training_events"),
        "VIEWTRUST_RUN_ID": run_id,
        "VIEWTRUST_SCENE": scene,
        "VIEWTRUST_CONDITION": condition,
        "VIEWTRUST_TRAINER": trainer,
        "VIEWTRUST_OBSERVATION_ONLY": "1",
        "VIEWTRUST_TRAINING_EVENT_LOG_INTERVAL": str(log_interval),
    }
    if strict:
        env["VIEWTRUST_OBSERVER_STRICT"] = "1"
    return env


def build_gaussian_lifecycle_env(
    *,
    enabled: bool,
    project_root: Path,
    run_dir: Path,
    run_id: str,
    scene: str,
    condition: str,
    trainer: str,
    strict: bool = False,
    log_snapshot_stats: bool = True,
    base_env: dict[str, str] | None = None,
) -> dict[str, str]:
    """Build opt-in environment variables for PR8 lifecycle observation."""

    if not enabled:
        return {}
    resolved_project_root = project_root.resolve()
    existing_pythonpath = (base_env or os.environ).get("PYTHONPATH", "")
    pythonpath_parts = [
        str(resolved_project_root),
        *[
            part
            for part in existing_pythonpath.split(os.pathsep)
            if part and part != str(resolved_project_root)
        ],
    ]
    env = {
        "PYTHONPATH": os.pathsep.join(pythonpath_parts),
        "VIEWTRUST_PROJECT_ROOT": str(resolved_project_root),
        "VIEWTRUST_ENABLE_GAUSSIAN_LIFECYCLE": "1",
        "VIEWTRUST_GAUSSIAN_LIFECYCLE_DIR": str(run_dir / "gaussian_lifecycle"),
        "VIEWTRUST_GAUSSIAN_LIFECYCLE_LOG_SNAPSHOT_STATS": "1"
        if log_snapshot_stats
        else "0",
        "VIEWTRUST_RUN_ID": run_id,
        "VIEWTRUST_SCENE": scene,
        "VIEWTRUST_CONDITION": condition,
        "VIEWTRUST_TRAINER": trainer,
        "VIEWTRUST_OBSERVATION_ONLY": "1",
    }
    if strict:
        env["VIEWTRUST_GAUSSIAN_LIFECYCLE_STRICT"] = "1"
    return env


def preflight_training_event_observer_import(
    *,
    python_executable: Path,
    env_overrides: dict[str, str],
    cwd: Path | None = None,
    base_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Check that the child trainer environment can import the PR7 observer."""

    child_env = dict(base_env or os.environ)
    child_env.update(env_overrides)
    return subprocess.run(
        [
            str(python_executable),
            "-c",
            (
                "from viewtrust.observation.training_events import "
                "TrainingEventObserver, TrainingEventObserverConfig; "
                "print('observer import ok')"
            ),
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=child_env,
        cwd=str(cwd) if cwd else None,
    )


def preflight_gaussian_lifecycle_observer_import(
    *,
    python_executable: Path,
    env_overrides: dict[str, str],
    cwd: Path | None = None,
    base_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Check that the child trainer environment can import the PR8 observer."""

    child_env = dict(base_env or os.environ)
    child_env.update(env_overrides)
    return subprocess.run(
        [
            str(python_executable),
            "-c",
            (
                "from viewtrust.observation.gaussian_lifecycle import "
                "GaussianLifecycleObserver, GaussianLifecycleConfig; "
                "print('gaussian lifecycle import ok')"
            ),
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=child_env,
        cwd=str(cwd) if cwd else None,
    )
