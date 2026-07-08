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
python scripts/smoke/training_split_protocol_smoke_test.py
python scripts/smoke/baseline_run_inspection_smoke_test.py
python scripts/smoke/training_dynamics_extraction_smoke_test.py
python scripts/smoke/view_render_wrapper_dry_run_smoke_test.py
python scripts/smoke/view_metrics_extraction_smoke_test.py
python scripts/smoke/training_events_child_env_smoke_test.py
python scripts/smoke/training_event_observer_smoke_test.py
python scripts/smoke/training_event_sanity_smoke_test.py
python scripts/smoke/gaussian_lifecycle_observer_smoke_test.py
python scripts/smoke/gaussian_lifecycle_invariant_smoke_test.py
python scripts/smoke/gaussian_lifecycle_child_env_smoke_test.py
python scripts/smoke/gaussian_splatting_observation_patch_smoke_test.py
python scripts/smoke/noop_equivalence_smoke_test.py
python scripts/smoke/priority0_report_smoke_test.py
python scripts/smoke/natural_corruption_generation_smoke_test.py
python scripts/smoke/natural_corruption_inspector_smoke_test.py
python scripts/smoke/clean_vs_corrupt_comparison_smoke_test.py
python scripts/smoke/corruption_manifest_linking_smoke_test.py
python scripts/smoke/view_influence_table_smoke_test.py
python scripts/smoke/view_influence_table_performance_smoke_test.py
python scripts/smoke/view_influence_summary_schema_smoke_test.py
python scripts/smoke/view_influence_comparison_smoke_test.py
python scripts/smoke/offline_viewtrust_signals_smoke_test.py
python scripts/smoke/offline_viewtrust_multi_condition_smoke_test.py
python scripts/smoke/offline_viewtrust_rank_consistency_smoke_test.py
python scripts/smoke/pr16_subset_scene_bias_smoke_test.py
python scripts/smoke/clean_prior_normalized_viewtrust_smoke_test.py
python scripts/data/install_datasets.py --manifest configs/datasets.example.json --data-root ./data

echo "mock checks ok"
