# Server Environment

This document records the known validated GPU server for ViewTrust-GS Priority 0 work.

## Known Server

```text
OS: Ubuntu 20.04.6 LTS
GPU: 2 x NVIDIA GeForce RTX 3090, 24GB each
NVIDIA driver: 560.35.03
Driver CUDA runtime: 12.6
Working project root area: /trainingData/sage/yue
Working micromamba environment: /trainingData/sage/yue/envs/viewtrust-p0
CUDA_HOME on server: /trainingData/sage/yue/envs/viewtrust-p0
gsplat CUDA smoke test: passed
```

CUDA headers are expected in the micromamba environment. The server setup may
need both include roots:

```text
$CUDA_HOME/targets/x86_64-linux/include
$CUDA_HOME/include
```

## Important CUDA Note

Do not use system CUDA at:

```text
/usr/local/cuda-11.0
```

That `nvcc` is too old for the current `gsplat` build because it does not support the required C++20 compilation mode. Use the micromamba environment CUDA toolchain instead.

Do not stack the old uv `.venv-viewtrust-p0` environment with the micromamba
environment. If `VIRTUAL_ENV` is set, deactivate it before activating the
server environment.

## Activation

After syncing this repository to the server, run:

```bash
cd /trainingData/sage/yue/viewtrust-gs
source scripts/env/activate_server_viewtrust_p0.sh
```

Expected tool locations:

```text
python: /trainingData/sage/yue/envs/viewtrust-p0/bin/python
pip: /trainingData/sage/yue/envs/viewtrust-p0/bin/pip
nvcc: /trainingData/sage/yue/envs/viewtrust-p0/bin/nvcc
```

Expected runtime checks:

```text
torch.cuda.is_available(): True
gsplat import: ok
```

The activation script also attempts to resolve PyTorch's shared library
directory and prepend it to `LD_LIBRARY_PATH`:

```text
$CONDA_PREFIX/lib/python*/site-packages/torch/lib
```

This is required for the official Gaussian Splatting CUDA submodules on the
validated server. If the activation script warns that it cannot resolve
`TORCH_LIB_DIR`, fix that before running the official trainer.

## Full Server Validation Flow

Run this exact flow from the repository root on the server:

```bash
deactivate 2>/dev/null || true

export MAMBA_ROOT_PREFIX=/trainingData/sage/yue/.mamba-root
eval "$(/trainingData/sage/yue/tools/micromamba/bin/micromamba shell hook -s bash)"
micromamba activate /trainingData/sage/yue/envs/viewtrust-p0

source scripts/env/activate_server_viewtrust_p0.sh
bash scripts/checks/run_server_checks.sh
```

## Server Smoke Test

Run:

```bash
bash scripts/checks/run_server_checks.sh
```

This is SERVER-REQUIRED and is not expected to pass on the local Mac.

## Official Gaussian Splatting Trainer Checks

The ViewTrust-GS repository does not vendor the official trainer. If the trainer
is present under `third_party/gaussian-splatting`, validate its CUDA submodule
imports on the server with:

```bash
bash scripts/env/check_server_environment.sh --require-gaussian-splatting
```

This additional check requires:

```text
third_party/gaussian-splatting/train.py
diff-gaussian-rasterization
simple-knn
fused-ssim
PyTorch shared library directory in LD_LIBRARY_PATH
```

Validated server install commands:

```bash
python -m pip install "setuptools<82" wheel ninja
python -m pip install plyfile tqdm opencv-python joblib
python -m pip install --no-build-isolation -e third_party/gaussian-splatting/submodules/diff-gaussian-rasterization
python -m pip install --no-build-isolation -e third_party/gaussian-splatting/submodules/simple-knn
python -m pip install --no-build-isolation -e third_party/gaussian-splatting/submodules/fused-ssim
```

Do not upgrade to `setuptools` 83+ in the validated `torch 2.12.1+cu126`
environment unless the CUDA extension builds are revalidated.

Optional dependency for PR5 loss-curve extraction from official Gaussian
Splatting TensorBoard event files:

```bash
python -m pip install tensorboard
```

This is not required for local smoke tests or for PR5 artifact/PLY summaries.

Known server-local compatibility patch for the official trainer:

```text
third_party/gaussian-splatting/scene/dataset_readers.py
np.byte -> np.uint8
```

This patch is documented server-local state. Do not commit third-party source
changes to ViewTrust-GS.

## PR7 Observation Patch

PR7 uses an explicit local patch to add observation-only training event hooks to
the server's official Gaussian Splatting clone. Apply it manually:

```bash
python scripts/third_party/apply_gaussian_splatting_observation_patch.py \
  --third-party-root ./third_party \
  --patch pr7_training_events
```

Check it manually:

```bash
python scripts/third_party/check_gaussian_splatting_observation_patch.py \
  --third-party-root ./third_party \
  --patch pr7_training_events \
  --require-applied
```

The server environment check can also require it explicitly:

```bash
bash scripts/env/check_server_environment.sh --require-gaussian-splatting --require-observation-patch
```

This patch is opt-in and activates only when the clean baseline wrapper is run
with `--enable-training-events`.

For PR7.1 server validation, use strict training event mode so child-environment
observer import issues fail before or during training instead of producing a run
without event outputs:

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
  --sample-interval-s 1.0 \
  --enable-training-events \
  --training-event-log-interval 10 \
  --training-event-strict
```

The wrapper injects the repository root into the trainer child `PYTHONPATH` and
preflights `viewtrust.observation.training_events` before launching training.

## PR8 Gaussian Lifecycle

PR8 is a separate lifecycle patch applied after PR7 training events. If PR7 is
already applied on the server, run only the PR8 apply/check commands:

```bash
python scripts/third_party/apply_gaussian_splatting_observation_patch.py \
  --third-party-root ./third_party \
  --patch pr8_gaussian_lifecycle

python scripts/third_party/check_gaussian_splatting_observation_patch.py \
  --third-party-root ./third_party \
  --patch pr8_gaussian_lifecycle \
  --require-applied
```

Server validation command:

```bash
python scripts/train/run_clean_chair_baseline.py \
  --trainer gaussian-splatting \
  --data-root "$VIEWTRUST_DATA_ROOT" \
  --third-party-root ./third_party \
  --output-root ./outputs \
  --scene chair \
  --condition clean \
  --iterations 700 \
  --gpu 0 \
  --sample-interval-s 1.0 \
  --enable-training-events \
  --training-event-log-interval 10 \
  --training-event-strict \
  --enable-gaussian-lifecycle \
  --gaussian-lifecycle-strict
```
