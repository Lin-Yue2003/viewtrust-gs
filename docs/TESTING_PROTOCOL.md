# Testing Protocol

Every future ViewTrust-GS task should state which checks are LOCAL-SAFE and which are SERVER-REQUIRED.

## LOCAL-SAFE

LOCAL-SAFE checks can run on the local Mac without CUDA:

```text
documentation checks
Python syntax compilation
config loading
relative path resolution
run ID generation
metadata JSON writing
schema placeholder validation
CPU-only mock smoke tests
tiny fake NeRF Synthetic subset preparation
training wrapper dry-run smoke test
training dynamics extraction smoke test
view render wrapper dry-run smoke test
view metrics extraction smoke test
training events child environment smoke test
training event observer smoke test
training event sanity inspector smoke test
Gaussian lifecycle observer smoke test
Gaussian lifecycle invariant inspector smoke test
Gaussian lifecycle child environment smoke test
Gaussian Splatting observation patch dry-run/check smoke test
no-op equivalence smoke test
Priority 0 report smoke test
natural corruption generation smoke test with a fake clean mini scene
natural corruption inspector smoke test with a fake generated condition
clean-vs-corrupt comparison smoke test with fake observed runs
corruption manifest linking smoke test
view influence table smoke test
view influence comparison smoke test
offline ViewTrust rank consistency smoke test
PR16 subset and scene bias smoke test
```

Commands:

```bash
bash scripts/checks/run_static_checks.sh
bash scripts/checks/run_mock_checks.sh
```

`run_mock_checks.sh` includes the CPU-only scaffold smoke test and the Priority
0 logging, measurement format, observed command wrapper, and dataset installer
dry-run smoke tests. It also runs `nerf_synthetic_subset_smoke_test.py` with a
tiny fake scene and `training_wrapper_dry_run_smoke_test.py` with a fake trainer.
It also runs `training_dynamics_extraction_smoke_test.py` on a fake observed run
with a tiny PLY file.
It also runs `view_render_wrapper_dry_run_smoke_test.py` and
`view_metrics_extraction_smoke_test.py` without CUDA.
It also runs `training_events_child_env_smoke_test.py`,
`training_event_observer_smoke_test.py`,
`training_event_sanity_smoke_test.py`,
`gaussian_lifecycle_observer_smoke_test.py`,
`gaussian_lifecycle_invariant_smoke_test.py`,
`gaussian_lifecycle_child_env_smoke_test.py`, and
`gaussian_splatting_observation_patch_smoke_test.py`,
`noop_equivalence_smoke_test.py`, and `priority0_report_smoke_test.py`
without touching real `third_party` source. It also runs
`natural_corruption_generation_smoke_test.py` and
`natural_corruption_inspector_smoke_test.py` on a tiny fake clean mini scene
without CUDA. It also runs `clean_vs_corrupt_comparison_smoke_test.py`, which
checks PR10.1 manifest compatibility and PR11 comparison outputs with fake
observed clean/corrupt runs. PR12 adds
`corruption_manifest_linking_smoke_test.py`,
`view_influence_table_smoke_test.py`, and
`view_influence_comparison_smoke_test.py`. PR12.1 adds
`training_split_protocol_smoke_test.py`,
`view_influence_table_performance_smoke_test.py`, and
`view_influence_summary_schema_smoke_test.py`; it also extends
`training_event_sanity_smoke_test.py` to verify train-only sampling checks.
PR13 adds `offline_viewtrust_signals_smoke_test.py`, which builds offline
candidate signal outputs from fake split-correct view influence tables and
checks robust normalization, ranking, group metrics, ablation metrics, and
label-use boundaries. PR14 adds
`offline_viewtrust_multi_condition_smoke_test.py`, which validates discovery of
the newest PR13 output per condition, helper-generated
`offline_viewtrust_<condition>_pr14_input` directories, partial
missing-condition behavior, strict missing-condition failure, cross-condition
outputs, failure cases, report wording, and artifact manifest self-validation.
PR15 adds `offline_viewtrust_rank_consistency_smoke_test.py`, which validates
cross-condition repeated top views, false positive top-k summaries, corrupted
view rank distributions, component diagnosis tables, offline-only summary
fields, report wording, and artifact manifest self-validation using tiny fake
PR14 and per-condition PR13 outputs.
PR16 adds `pr16_subset_scene_bias_smoke_test.py`, which validates
deterministic subset planning, chair/drum scene coverage, subset-seed manifests,
view identity bias rows, repeated false positives, component comparisons,
offline-only report wording, and artifact manifest self-validation with fake
offline outputs.
PR17 adds `clean_prior_normalized_viewtrust_smoke_test.py`, which validates
clean-prior normalization with fake offline signal directories, artifact
manifest input resolution, missing-output reporting, false-positive reduction,
and offline-only label-use guarantees.
PR18 adds `pr18_covisibility_spillover_smoke_test.py`, which validates
camera-neighbor spillover diagnosis with fake PR17 rows, fake PR16 subset
metadata, synthetic camera transforms, and offline-only label-use guarantees.

## Observed Command Checks

Observed command checks validate external-process observation before any
training-loop instrumentation.

LOCAL-SAFE:

```text
observed command mock test
observed sleep test
```

Command:

```bash
bash scripts/checks/run_observed_checks.sh
```

GPU sampling may be empty locally when `nvidia-smi` is unavailable.

SERVER-REQUIRED:

```text
observed gsplat CUDA smoke test
future observed 3DGS training run
```

Server-required observed gsplat command:

```bash
python scripts/measure/run_observed_command.py \
  --label gsplat-smoke-observed \
  --sample-interval-s 0.2 \
  -- python scripts/smoke/gsplat_cuda_smoke_test.py
```

## SERVER-REQUIRED

SERVER-REQUIRED checks must run on the remote Ubuntu GPU server:

```text
server environment activation
CUDA_HOME validation
nvcc validation
PyTorch CUDA validation
GPU inventory
gsplat import
gsplat CUDA rasterization smoke test
real 3DGS training
Priority 0 GPU memory and timing measurement
preparing real NeRF Synthetic chair subset from raw data
future training on prepared chair subset
real clean chair baseline training
observed GPU memory sampling during training
official Gaussian Splatting CUDA submodule import validation
extract_training_dynamics.py on a successful clean chair baseline run
render_clean_views.py on a successful clean chair baseline run
extract_view_metrics.py on rendered train/test/target views
manual PR7 observation patch application/check
instrumented clean chair baseline with --enable-training-events
strict child observer import validation with --training-event-strict
inspect_training_events.py on the instrumented run
training event scalar sanity validation
instrumented clean chair baseline with --enable-gaussian-lifecycle
inspect_gaussian_lifecycle.py on the lifecycle run
compare_noop_runs.py for uninstrumented vs PR7+PR8 observed runs
build_priority0_report.py for the observed Priority 0 run
generating real PR10 natural corruption chair conditions from clean mini data
inspecting real PR10 natural corruption chair conditions
training a natural corruption chair condition
compare_clean_corrupt_observations.py for clean vs natural-corrupt observed runs
future multi-condition clean-vs-corrupt suites
offline rank consistency analysis on real PR14.1 outputs
PR16 planner and analyzer on real chair/drum subset outputs
```

Command:

```bash
bash scripts/checks/run_server_checks.sh
```

This command is not expected to pass on the local Mac.

When the official trainer is present under `third_party/gaussian-splatting`,
run the extended server environment check:

```bash
bash scripts/env/check_server_environment.sh --require-gaussian-splatting
```

After a clean chair baseline run, inspect the observed artifacts with:

```bash
python scripts/measure/inspect_baseline_run.py \
  --run-dir outputs/baseline/chair_clean_gaussian_splatting/<run_id> \
  --require-success
```

Then extract PR5 training dynamics:

```bash
python scripts/measure/extract_training_dynamics.py \
  --run-dir outputs/baseline/chair_clean_gaussian_splatting/<run_id> \
  --require-success
```

Then render and extract PR6 clean view metrics:

```bash
python scripts/evaluate/render_clean_views.py \
  --run-dir outputs/baseline/chair_clean_gaussian_splatting/<run_id> \
  --data-root "$VIEWTRUST_DATA_ROOT" \
  --third-party-root ./third_party \
  --trainer gaussian-splatting \
  --scene chair \
  --condition clean \
  --iteration 500 \
  --splits train test target \
  --gpu 0 \
  --sample-interval-s 1.0 \
  --overwrite

python scripts/measure/extract_view_metrics.py \
  --run-dir outputs/baseline/chair_clean_gaussian_splatting/<run_id> \
  --scene chair \
  --condition clean \
  --iteration 500 \
  --require-renders
```

The PR6 render wrapper must pass `--eval` to official Gaussian Splatting
`render.py`. For Blender datasets, omitting `--eval` merges test cameras into
train and causes the mini chair split counts to become `train=25, test=0`
instead of `train=20, test=5`. Target-as-test rendering also depends on
`--eval`.

PR7 server validation adds:

```bash
python scripts/third_party/apply_gaussian_splatting_observation_patch.py \
  --third-party-root ./third_party \
  --patch pr7_training_events

python scripts/third_party/check_gaussian_splatting_observation_patch.py \
  --third-party-root ./third_party \
  --patch pr7_training_events \
  --require-applied

python scripts/third_party/apply_gaussian_splatting_observation_patch.py \
  --third-party-root ./third_party \
  --patch pr8_gaussian_lifecycle

python scripts/third_party/check_gaussian_splatting_observation_patch.py \
  --third-party-root ./third_party \
  --patch pr8_gaussian_lifecycle \
  --require-applied

python scripts/train/run_clean_chair_baseline.py \
  --trainer gaussian-splatting \
  --data-root "$VIEWTRUST_DATA_ROOT" \
  --third-party-root ./third_party \
  --output-root ./outputs \
  --scene chair \
  --condition clean \
  --iterations 500 \
  --gpu 0 \
  --sample-interval-s 1.0 \
  --enable-training-events \
  --training-event-log-interval 10 \
  --training-event-strict

python scripts/measure/inspect_training_events.py \
  --run-dir outputs/baseline/chair_clean_gaussian_splatting/<run_id> \
  --require-events

python scripts/measure/inspect_gaussian_lifecycle.py \
  --run-dir outputs/baseline/chair_clean_gaussian_splatting/<run_id> \
  --require-lifecycle \
  --require-no-invariant-violations

python scripts/measure/compare_noop_runs.py \
  --baseline-run-dir "$BASELINE_RUN_DIR" \
  --observed-run-dir "$OBSERVED_RUN_DIR" \
  --output-dir outputs/reports/priority0_noop_$(date +%Y%m%dT%H%M%S) \
  --require-success \
  --require-observation-invariants \
  --write-markdown

python scripts/measure/build_priority0_report.py \
  --run-dir "$OBSERVED_RUN_DIR" \
  --output-dir outputs/reports/priority0_report_$(date +%Y%m%dT%H%M%S) \
  --include-view-metrics \
  --include-training-events \
  --include-gaussian-lifecycle \
  --require-priority0-complete \
  --write-markdown
```

PR10 natural corruption condition generation is server-required for the real
chair mini dataset because it reads `$VIEWTRUST_DATA_ROOT`, but the generation
and inspector code are CPU-only:

```bash
python scripts/data/generate_default_natural_corruption_suite.py \
  --data-root "$VIEWTRUST_DATA_ROOT" \
  --scene chair \
  --source-condition clean \
  --seed 20260706 \
  --num-corrupt-train-views 4 \
  --copy-mode symlink \
  --overwrite

python scripts/measure/inspect_natural_corruption_dataset.py \
  --data-root "$VIEWTRUST_DATA_ROOT" \
  --scene chair \
  --condition corrupt_occluder \
  --require-valid \
  --require-corrupted-count 4
```

PR11 clean-vs-corrupt validation starts with `corrupt_occluder`:

```bash
python scripts/train/run_clean_chair_baseline.py \
  --trainer gaussian-splatting \
  --data-root "$VIEWTRUST_DATA_ROOT" \
  --third-party-root ./third_party \
  --output-root ./outputs \
  --scene chair \
  --condition corrupt_occluder \
  --iterations 700 \
  --gpu 0 \
  --sample-interval-s 1.0 \
  --enable-training-events \
  --training-event-log-interval 10 \
  --training-event-strict \
  --enable-gaussian-lifecycle \
  --gaussian-lifecycle-strict

python scripts/measure/compare_clean_corrupt_observations.py \
  --clean-run-dir "$CLEAN_RUN_DIR" \
  --corrupt-run-dir "$CORRUPT_RUN_DIR" \
  --corruption-condition corrupt_occluder \
  --output-dir outputs/reports/clean_vs_corrupt_occluder_$(date +%Y%m%dT%H%M%S) \
  --require-observation-invariants \
  --write-markdown
```

The PR7.2/PR11.1 inspector sanity check requires:

```text
0 <= visible_gaussian_count <= gaussian_count
0 <= visibility_ratio <= 1
0 <= radii_nonzero_count <= gaussian_count
densification gaussian_count_after >= 0
densification gaussian_count_delta == gaussian_count_after - gaussian_count_before
```

For `iteration_metrics`, `gaussian_count` is the render-time/pre-prune count,
not the post-densification count. Post-prune counts belong in
`densification_events.csv`.

PR8 lifecycle validation requires:

```text
alive_final_count == final_gaussian_count
alive_final_count + dead_final_count == known_gaussian_count
gaussian_lifecycle_final.csv row count == known_gaussian_count
no duplicate alive final_index
```

PR12 source-view validation adds:

```bash
python scripts/measure/inspect_training_events.py \
  --run-dir "$RUN_DIR" \
  --require-events \
  --require-view-identity \
  --require-train-only-sampling

python scripts/measure/inspect_gaussian_lifecycle.py \
  --run-dir "$RUN_DIR" \
  --require-lifecycle \
  --require-no-invariant-violations \
  --require-source-view
```

PR12 view influence table generation is read-only:

```bash
python scripts/measure/build_view_influence_table.py \
  --run-dir "$RUN_DIR" \
  --data-root "$VIEWTRUST_DATA_ROOT" \
  --scene chair \
  --condition corrupt_occluder \
  --output-dir outputs/reports/view_influence_corrupt_occluder_$(date +%Y%m%dT%H%M%S) \
  --require-view-identity \
  --require-source-view \
  --progress-interval-rows 50000 \
  --write-markdown
```

PR12.1 requires split-correct training runs. The baseline wrapper passes
official Gaussian Splatting `--eval` by default, and
`inspect_training_events.py --require-train-only-sampling` should report zero
unexpected non-train sampled views. The view influence builder reports
`runtime_s`, per-stage `timing`, `input_rows`, throughput estimates, and
split-aware sampled-view counts in `view_influence_summary.json`; use `--quiet`
when progress logs are not wanted.

PR13 offline signal generation is read-only:

```bash
python scripts/measure/build_offline_viewtrust_signals.py \
  --clean-view-influence-dir "$CLEAN_VIEW_INFLUENCE_DIR" \
  --corrupt-view-influence-dir "$CORRUPT_VIEW_INFLUENCE_DIR" \
  --view-influence-comparison-dir "$VIEW_INFLUENCE_COMPARE_DIR" \
  --output-dir outputs/reports/offline_viewtrust_corrupt_occluder_pr13_$(date +%Y%m%dT%H%M%S) \
  --write-markdown \
  --top-k 5
```

Expected PR13 summary invariants:

```text
observation_only = true
uses_corruption_labels_for_scoring = false
uses_corruption_labels_for_evaluation = true
training_intervention = false
defense_enabled = false
```

PR14 multi-condition offline aggregation is read-only:

```bash
python scripts/measure/aggregate_offline_viewtrust_results.py \
  --input-root outputs/reports \
  --output-dir outputs/reports/offline_viewtrust_multi_condition_pr14_partial_$(date +%Y%m%dT%H%M%S) \
  --scene chair \
  --clean-condition clean \
  --conditions corrupt_occluder corrupt_blur corrupt_exposure corrupt_color_shift corrupt_noise corrupt_mixed \
  --top-k 5 \
  --write-markdown
```

Use strict mode after all condition outputs exist:

```bash
python scripts/measure/aggregate_offline_viewtrust_results.py \
  --input-root outputs/reports \
  --output-dir outputs/reports/offline_viewtrust_multi_condition_pr14_full_$(date +%Y%m%dT%H%M%S) \
  --scene chair \
  --clean-condition clean \
  --conditions corrupt_occluder corrupt_blur corrupt_exposure corrupt_color_shift corrupt_noise corrupt_mixed \
  --top-k 5 \
  --require-all-conditions \
  --write-markdown
```

Expected PR14 summary invariants:

```text
uses_corruption_labels_for_scoring = false
uses_corruption_labels_for_evaluation = true
training_intervention = false
defense_enabled = false
```

## PR15 Offline Rank Consistency Diagnosis

PR15 rank consistency analysis is read-only:

```bash
python scripts/measure/analyze_offline_viewtrust_rank_consistency.py \
  --multi-condition-dir "$PR14_FULL_DIR" \
  --input-root outputs/reports \
  --scene chair \
  --conditions corrupt_occluder corrupt_blur corrupt_exposure corrupt_color_shift corrupt_noise corrupt_mixed \
  --top-k 5 \
  --output-dir "$PR15_DIR" \
  --write-markdown
```

LOCAL-SAFE:

```bash
python scripts/smoke/offline_viewtrust_rank_consistency_smoke_test.py
```

SERVER-REQUIRED:

```text
Run the PR15 analyzer on real PR14.1 outputs after all per-condition PR13 /
PR14-input directories have been produced on the server.
```

Expected PR15 summary invariants:

```text
schema_name = viewtrust.offline_signal.rank_consistency.summary
observation_only = true
training_intervention = false
defense_enabled = false
uses_corruption_labels_for_scoring = false
uses_corruption_labels_for_evaluation = true
```

## PR16 Subset and Scene Bias Probe

PR16 planning and analysis are offline and observation-only:

```bash
python scripts/experiments/plan_pr16_subset_scene_bias_probe.py \
  --data-root "$VIEWTRUST_DATA_ROOT" \
  --output-dir "$PR16_PLAN_DIR" \
  --scenes chair drum \
  --conditions corrupt_occluder corrupt_noise corrupt_mixed \
  --subset-names original seed_20260708 seed_20260709 \
  --subset-seeds 20260708 20260709 \
  --corrupted-view-count 4 \
  --top-k 5 \
  --write-commands

python scripts/measure/analyze_pr16_subset_scene_bias.py \
  --input-root outputs/reports \
  --plan-dir "$PR16_PLAN_DIR" \
  --output-dir "$PR16_ANALYSIS_DIR" \
  --scenes chair drum \
  --conditions corrupt_occluder corrupt_noise corrupt_mixed \
  --subset-names original seed_20260708 seed_20260709 \
  --top-k 5 \
  --write-markdown
```

LOCAL-SAFE:

```bash
python scripts/smoke/pr16_subset_scene_bias_smoke_test.py
```

SERVER-REQUIRED:

```text
Run the PR16 analyzer on real chair/drum, subset, and condition outputs after
the corresponding PR13 / PR14-input offline signal directories have been
created.
```

Expected PR16 summary invariants:

```text
schema_name = viewtrust.pr16.subset_scene_bias.summary
observation_only = true
training_intervention = false
defense_enabled = false
uses_corruption_labels_for_scoring = false
uses_corruption_labels_for_evaluation = true
```

## PR17 Clean-Prior Normalized Offline ViewTrust

PR17 clean-prior normalization is offline analysis only:

```bash
python scripts/measure/analyze_clean_prior_normalized_viewtrust.py \
  --input-root outputs/reports \
  --plan-dir "$PR16_PLAN_ANALYSIS_DIR" \
  --output-dir "$PR17_DIR" \
  --scenes chair drums \
  --conditions corrupt_occluder corrupt_noise corrupt_mixed \
  --subset-names original seed_20260710 \
  --top-k 5 \
  --allow-missing \
  --write-markdown
```

LOCAL-SAFE:

```bash
python scripts/smoke/clean_prior_normalized_viewtrust_smoke_test.py
```

Expected PR17 summary invariants:

```text
schema_name = viewtrust.pr17.clean_prior_normalized.summary
observation_only = true
training_intervention = false
defense_enabled = false
uses_corruption_labels_for_scoring = false
uses_corruption_labels_for_evaluation = true
```

## PR18 Co-visibility Spillover Diagnosis

PR18 is offline analysis only. It diagnoses remaining normalized false
positives without changing PR17 scores or any training behavior:

```bash
python scripts/measure/analyze_pr18_covisibility_spillover.py \
  --data-root "$VIEWTRUST_DATA_ROOT" \
  --input-root outputs/reports \
  --plan-dir "$PR16_PLAN_ANALYSIS_DIR" \
  --pr17-dir "$PR17_DIR" \
  --output-dir "$PR18_DIR" \
  --scenes chair drums \
  --conditions corrupt_occluder corrupt_noise corrupt_mixed \
  --subset-names original seed_20260710 \
  --top-k 5 \
  --allow-missing \
  --write-markdown
```

LOCAL-SAFE:

```bash
python scripts/smoke/pr18_covisibility_spillover_smoke_test.py
```

Expected PR18 summary invariants:

```text
schema_name = viewtrust.pr18.covisibility_spillover.summary
observation_only = true
training_intervention = false
defense_enabled = false
uses_corruption_labels_for_scoring = false
uses_corruption_labels_for_evaluation = true
```

Recommended server validation flow:

```bash
deactivate 2>/dev/null || true

export MAMBA_ROOT_PREFIX=/trainingData/sage/yue/.mamba-root
eval "$(/trainingData/sage/yue/tools/micromamba/bin/micromamba shell hook -s bash)"
micromamba activate /trainingData/sage/yue/envs/viewtrust-p0

source scripts/env/activate_server_viewtrust_p0.sh
bash scripts/checks/run_server_checks.sh
```

## OPTIONAL-GPU

OPTIONAL-GPU checks may run on any machine with a valid CUDA setup, but they are not required for local Mac development. They can be used for extra confidence, but the official GPU validation target is the remote server.

## Current Stage

Current stage:

```text
Priority 0 = observation-only infrastructure
```

This stage does not implement ViewTrust scoring, defense logic, densification gating, pruning changes, loss changes, optimizer changes, rendering changes, or dataset sampling changes.
