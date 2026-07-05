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
```

Commands:

```bash
bash scripts/checks/run_static_checks.sh
bash scripts/checks/run_mock_checks.sh
```

`run_mock_checks.sh` includes the CPU-only scaffold smoke test and the Priority
0 logging smoke test.

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
```

Command:

```bash
bash scripts/checks/run_server_checks.sh
```

This command is not expected to pass on the local Mac.

## OPTIONAL-GPU

OPTIONAL-GPU checks may run on any machine with a valid CUDA setup, but they are not required for local Mac development. They can be used for extra confidence, but the official GPU validation target is the remote server.

## Current Stage

Current stage:

```text
Priority 0 = observation-only infrastructure
```

This stage does not implement ViewTrust scoring, defense logic, densification gating, pruning changes, loss changes, optimizer changes, rendering changes, or dataset sampling changes.
