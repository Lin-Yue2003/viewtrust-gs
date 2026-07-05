#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"

python scripts/smoke/mock_cpu_smoke_test.py
python scripts/smoke/priority0_logging_smoke_test.py
python scripts/smoke/measurement_format_smoke_test.py
python scripts/smoke/observed_command_smoke_test.py
python scripts/smoke/nerf_synthetic_subset_smoke_test.py
python scripts/smoke/training_wrapper_dry_run_smoke_test.py
python scripts/smoke/baseline_run_inspection_smoke_test.py
python scripts/smoke/training_dynamics_extraction_smoke_test.py
python scripts/smoke/view_render_wrapper_dry_run_smoke_test.py
python scripts/smoke/view_metrics_extraction_smoke_test.py
python scripts/smoke/training_events_child_env_smoke_test.py
python scripts/smoke/training_event_observer_smoke_test.py
python scripts/smoke/training_event_sanity_smoke_test.py
python scripts/smoke/gaussian_splatting_observation_patch_smoke_test.py
python scripts/data/install_datasets.py --manifest configs/datasets.example.json --data-root ./data

echo "mock checks ok"
