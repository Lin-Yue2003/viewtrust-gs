"""Clean baseline training wrapper helpers.

This module is local-safe and pure Python. It composes external training
commands but does not import CUDA libraries or modify trainer internals.
"""

from __future__ import annotations

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
