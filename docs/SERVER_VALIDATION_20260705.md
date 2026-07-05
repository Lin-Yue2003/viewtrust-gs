# Server Validation 2026-07-05

Date: 2026-07-05

## Environment

```text
Server OS: Ubuntu 20.04.6
GPU: 2 x NVIDIA GeForce RTX 3090 24GB
Driver: 560.35.03
CUDA toolkit used by env: 12.6
Python: /trainingData/sage/yue/envs/viewtrust-p0/bin/python
nvcc: /trainingData/sage/yue/envs/viewtrust-p0/bin/nvcc
torch: 2.12.1+cu126
torch CUDA: 12.6
```

## Results

```text
torch.cuda.is_available(): True
GPU count: 2
GPU 0: NVIDIA GeForce RTX 3090
GPU 1: NVIDIA GeForce RTX 3090
gsplat import: ok
gsplat CUDA smoke test: passed
observed command sleep test: passed
GPU memory sampling: passed
```

The observed sleep test created:

```text
config_snapshot.json
events.jsonl
metadata.json
stats.json
stderr.log
stdout.log
summary.json
tables/command_summary.csv
tables/gpu_memory_samples.csv
```

`gpu_memory_samples.csv` included samples from both RTX 3090 GPUs.

## Notes

Do not use:

```text
/usr/local/cuda-11.0
```

Do not stack the old uv `.venv-viewtrust-p0` with the micromamba environment.
Deactivate old virtual environments before activating `/trainingData/sage/yue/envs/viewtrust-p0`.
