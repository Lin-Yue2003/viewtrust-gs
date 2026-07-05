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
Gaussian Splatting observation patch dry-run/check smoke test
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
`training_event_observer_smoke_test.py`, and
`gaussian_splatting_observation_patch_smoke_test.py` without touching real
`third_party` source.

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
