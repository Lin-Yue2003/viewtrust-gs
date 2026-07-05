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

## Important CUDA Note

Do not use system CUDA at:

```text
/usr/local/cuda-11.0
```

That `nvcc` is too old for the current `gsplat` build because it does not support the required C++20 compilation mode. Use the micromamba environment CUDA toolchain instead.

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

## Server Smoke Test

Run:

```bash
bash scripts/checks/run_server_checks.sh
```

This is SERVER-REQUIRED and is not expected to pass on the local Mac.
