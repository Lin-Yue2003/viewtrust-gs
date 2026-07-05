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
