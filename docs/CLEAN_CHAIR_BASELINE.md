# Clean Chair Baseline

PR3 creates a clean chair baseline wrapper for the prepared NeRF Synthetic mini
subset.

This is still observation-only. It does not modify training internals, edit
third-party code, enable ViewTrust scoring, add defenses, or gate
densification. It composes an external trainer command and runs it through
`scripts/measure/run_observed_command.py`.

## Requirements

Prepared dataset:

```text
$VIEWTRUST_DATA_ROOT/viewtrust-mini/nerf_synthetic/chair/clean/
  transforms_train.json
  transforms_test.json
  manifest.json
  images/
```

Preferred trainer, if available:

```text
third_party/gaussian-splatting/train.py
```

The trainer repo is optional local state and is not vendored by ViewTrust-GS. If
it is missing, clone or symlink the official Gaussian Splatting repo under
`third_party/gaussian-splatting`.

Validated server baseline:

```text
run_dir: outputs/baseline/chair_clean_gaussian_splatting/20260705T064007Z
trainer: official graphdeco Gaussian Splatting
scene: NeRF Synthetic chair clean mini subset
iterations: 500
gpu: 0
returncode: 0
elapsed_s: 16.850461
```

This validation did not require modifying ViewTrust training behavior. It did
require server-local dependency setup for the official trainer.

## Official Trainer Server Dependencies

The official Gaussian Splatting CUDA submodules may need to be installed in the
active server environment:

```bash
python -m pip install "setuptools<82" wheel ninja
python -m pip install plyfile tqdm opencv-python joblib
python -m pip install --no-build-isolation -e third_party/gaussian-splatting/submodules/diff-gaussian-rasterization
python -m pip install --no-build-isolation -e third_party/gaussian-splatting/submodules/simple-knn
python -m pip install --no-build-isolation -e third_party/gaussian-splatting/submodules/fused-ssim
```

Keep `setuptools<82` for the validated `torch 2.12.1+cu126` environment. Do not
upgrade to `setuptools` 83+ unless the CUDA extension build is revalidated.

After activation, `scripts/env/activate_server_viewtrust_p0.sh` adds PyTorch's
shared library directory to `LD_LIBRARY_PATH`. This is required by the official
Gaussian Splatting CUDA extension imports on the validated server.

Validate the trainer-side dependencies with:

```bash
bash scripts/env/check_server_environment.sh --require-gaussian-splatting
```

## Known Third-Party Compatibility Patch

On the validated server, the official Gaussian Splatting checkout required a
local NumPy compatibility edit in:

```text
third_party/gaussian-splatting/scene/dataset_readers.py
```

The server-local change was:

```text
np.byte -> np.uint8
```

inside the `Image.fromarray(np.array(arr * 255.0, dtype=np.uint8), "RGB")`
conversion path.

Do not commit or vendor this patch into ViewTrust-GS. Keep it as documented
server-local trainer state unless a future PR intentionally manages
third-party patches.

## Dry Run

Run this on the server after activating the ViewTrust environment:

```bash
source scripts/env/activate_server_viewtrust_p0.sh
export VIEWTRUST_DATA_ROOT=/trainingData/sage/yue/viewtrust-data
export CUDA_VISIBLE_DEVICES=0

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
  --dry-run
```

The dry run validates:

```text
prepared scene exists
required transforms and manifest exist
images directory exists
trainer path exists
baseline command can be composed
```

It does not execute training.

## Real Server Run

GPU 1 may already have memory in use. Start with GPU 0:

```bash
python scripts/train/run_clean_chair_baseline.py \
  --trainer gaussian-splatting \
  --data-root "$VIEWTRUST_DATA_ROOT" \
  --third-party-root ./third_party \
  --output-root ./outputs \
  --scene chair \
  --condition clean \
  --iterations 500 \
  --gpu 0 \
  --sample-interval-s 1.0
```

The wrapper sets `CUDA_VISIBLE_DEVICES` for the observed child process.

## Expected Output

Observed run directory:

```text
outputs/baseline/chair_clean_gaussian_splatting/<run_id>/
```

Observed artifacts:

```text
metadata.json
config_snapshot.json
events.jsonl
stdout.log
stderr.log
summary.json
stats.json
tables/command_summary.csv
tables/gpu_memory_samples.csv
trainer_output/
```

The actual trainer output is placed under:

```text
outputs/baseline/chair_clean_gaussian_splatting/<run_id>/trainer_output/
```

## Inspect Results

Compact JSON inspector:

```bash
python scripts/measure/inspect_baseline_run.py \
  --run-dir outputs/baseline/chair_clean_gaussian_splatting/20260705T064007Z \
  --require-success
```

The inspector is local-safe. It reads existing files only and reports fields
such as `returncode`, `elapsed_s`, `has_stdout`, `has_stderr`,
`has_gpu_samples`, `gpu_sample_count`, `trainer_output_exists`,
`trainer_output_file_count`, `detected_iterations`, and `observation_only`.

Manual inspection:

```bash
RUN_DIR=$(find outputs/baseline/chair_clean_gaussian_splatting -mindepth 1 -maxdepth 1 -type d | sort | tail -1)
echo "$RUN_DIR"
find "$RUN_DIR" -maxdepth 2 -type f | sort
cat "$RUN_DIR/summary.json"
tail -80 "$RUN_DIR/stdout.log"
tail -80 "$RUN_DIR/stderr.log"
cat "$RUN_DIR/tables/gpu_memory_samples.csv" 2>/dev/null || true
```

## Missing Dependency Errors

If the wrapper reports that `third_party/gaussian-splatting/train.py` is
missing, the official trainer is unavailable in the selected `--third-party-root`.
Clone or symlink it there manually. The wrapper will not download or vendor it.

If the trainer starts but fails, inspect `stderr.log` first. Common causes are
missing trainer dependencies, CUDA environment mismatch, or trainer assumptions
about dataset format. ViewTrust-GS does not patch these trainer internals in PR3.
