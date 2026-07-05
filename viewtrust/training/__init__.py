"""Training command composition helpers.

These helpers do not modify training internals. They only validate inputs and
compose commands for external trainers.
"""

from viewtrust.training.baseline import (
    BaselineTrainingConfig,
    build_baseline_label,
    build_gaussian_splatting_command,
    build_training_event_env,
    resolve_prepared_scene_root,
    resolve_trainer_path,
    validate_prepared_scene,
)
from viewtrust.training.dynamics import (
    TrainingDynamicsExtractionConfig,
    extract_training_dynamics,
)

__all__ = [
    "BaselineTrainingConfig",
    "TrainingDynamicsExtractionConfig",
    "build_baseline_label",
    "build_gaussian_splatting_command",
    "build_training_event_env",
    "extract_training_dynamics",
    "resolve_prepared_scene_root",
    "resolve_trainer_path",
    "validate_prepared_scene",
]
