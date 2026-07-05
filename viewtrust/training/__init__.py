"""Training command composition helpers.

These helpers do not modify training internals. They only validate inputs and
compose commands for external trainers.
"""

from viewtrust.training.baseline import (
    BaselineTrainingConfig,
    build_baseline_label,
    build_gaussian_splatting_command,
    resolve_prepared_scene_root,
    resolve_trainer_path,
    validate_prepared_scene,
)

__all__ = [
    "BaselineTrainingConfig",
    "build_baseline_label",
    "build_gaussian_splatting_command",
    "resolve_prepared_scene_root",
    "resolve_trainer_path",
    "validate_prepared_scene",
]
