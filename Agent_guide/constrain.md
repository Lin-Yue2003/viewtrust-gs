# ViewTrust-GS Codex Project Guide: Local Mac Development, Remote Server Execution

## 0. Critical Development Constraint

This project is developed on a local Mac with Codex, but the actual GPU environment only exists on the remote Linux server.

Therefore:

```text id="ejq874"
Codex can edit code locally.
Codex may run lightweight syntax checks or mock tests locally.
Codex must not assume CUDA, gsplat CUDA kernels, 3DGS training, or Priority 0 GPU tests can run on the Mac.
All real execution must happen after the user syncs the git repo to the server.
```

The project must be designed as a normal reproducible research repo, but with a clear separation between:

```text id="17n0ou"
local development
remote GPU execution
environment documentation
portable configs
```

Do not hard-code server paths into core Python code.

Server paths may appear only in:

```text id="45i1ey"
docs/SERVER_ENVIRONMENT.md
scripts/env/activate_server_viewtrust_p0.sh
configs/server.example.yaml
.env.server.example
```

Core code must use relative paths, command-line arguments, config files, or environment variables.

---

## 1. Known Development Machines

### 1.1 Local Mac

Purpose:

```text id="wfe4gs"
code editing
repository organization
documentation
unit tests with mock data
schema validation
format/lint checks
small CPU-only tests
```

Limitations:

```text id="w3bsmf"
No reliable CUDA.
No gsplat CUDA execution.
No full 3DGS training.
No Priority 0 real logging run.
No assumption that imports involving CUDA extensions will work.
```

Codex must not mark GPU-dependent tasks as complete based only on Mac tests.

### 1.2 Remote Server

Purpose:

```text id="rl1jle"
real environment validation
gsplat CUDA smoke test
3DGS training
Priority 0 logging
dataset experiments
GPU memory / time measurement
```

Known server information:

```text id="q2x10b"
OS: Ubuntu 20.04.6 LTS
GPU: 2 × NVIDIA GeForce RTX 3090, 24GB each
Driver: 560.35.03
Driver CUDA runtime: 12.6
Working project root used by user: /trainingData/sage/yue
Priority 0 environment: /trainingData/sage/yue/envs/viewtrust-p0
CUDA_HOME on server: /trainingData/sage/yue/envs/viewtrust-p0
gsplat CUDA smoke test: passed
```

Important server note:

```text id="piyrr5"
System nvcc at /usr/local/cuda-11.0/bin/nvcc is too old.
Do not rely on system CUDA 11.0.
The working setup uses micromamba environment CUDA 12.x / nvcc.
```

---

## 2. Path Policy

Core code must never hard-code:

```text id="rldil3"
/trainingData/sage/yue
/trainingData/sage/yue/envs/viewtrust-p0
/Users/...
/home/sage/...
absolute dataset paths
absolute output paths
```

Instead use:

```text id="8ndwd6"
VIEWTRUST_PROJECT_ROOT
VIEWTRUST_DATA_ROOT
VIEWTRUST_OUTPUT_ROOT
VIEWTRUST_THIRD_PARTY_ROOT
CUDA_HOME
CUDA_PATH
```

All scripts should infer project root from the script location when possible.

Example:

```bash id="w4petm"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
```

Python code should accept path arguments:

```text id="pafkyo"
--config
--data-root
--output-root
--run-id
--scene
--condition
```

If a default is needed, use relative paths such as:

```text id="963xmf"
./outputs
./data
./third_party
./configs
```

---

## 3. Required Repository Structure

Create or maintain this structure:

```text id="ksp5i1"
viewtrust-gs/
  docs/
    SERVER_ENVIRONMENT.md
    LOCAL_DEVELOPMENT.md
    IMPLEMENTATION_PLAN.md
    TESTING_PROTOCOL.md
  scripts/
    env/
      activate_server_viewtrust_p0.sh
      check_server_environment.sh
    smoke/
      gsplat_cuda_smoke_test.py
      mock_cpu_smoke_test.py
    checks/
      run_static_checks.sh
      run_mock_checks.sh
      run_server_checks.sh
  configs/
    default.yaml
    local.example.yaml
    server.example.yaml
  viewtrust/
    __init__.py
    configs/
    logging/
    checks/
    analysis/
    utils/
  tests/
    unit/
    mock/
  outputs/
    .gitkeep
  third_party/
    .gitkeep
  .env.example
  .env.server.example
```

Do not commit real datasets, model outputs, compiled CUDA caches, or large logs.

---

## 4. Environment Files

### 4.1 `.env.example`

This file should be portable and not server-specific:

```bash id="d14dni"
# Copy this file to .env and edit locally.

VIEWTRUST_PROJECT_ROOT=.
VIEWTRUST_DATA_ROOT=./data
VIEWTRUST_OUTPUT_ROOT=./outputs
VIEWTRUST_THIRD_PARTY_ROOT=./third_party

# Optional. Only needed on GPU server.
CUDA_HOME=
CUDA_PATH=
```

### 4.2 `.env.server.example`

This may document the known server setup, but code must not require these exact paths:

```bash id="hjvixx"
# Example for the current remote server.
# Do not hard-code these paths in Python source.

VIEWTRUST_PROJECT_ROOT=/trainingData/sage/yue/viewtrust-gs
VIEWTRUST_DATA_ROOT=/trainingData/sage/yue/datasets
VIEWTRUST_OUTPUT_ROOT=/trainingData/sage/yue/viewtrust_outputs
VIEWTRUST_THIRD_PARTY_ROOT=/trainingData/sage/yue/viewtrust-gs/third_party

CUDA_HOME=/trainingData/sage/yue/envs/viewtrust-p0
CUDA_PATH=/trainingData/sage/yue/envs/viewtrust-p0
```

---

## 5. Server Activation Script

Create:

```text id="dgandn"
scripts/env/activate_server_viewtrust_p0.sh
```

It should be a sourceable script:

```bash id="qu1kfb"
#!/usr/bin/env bash
# Usage:
#   source scripts/env/activate_server_viewtrust_p0.sh

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export MAMBA_ROOT_PREFIX="/trainingData/sage/yue/.mamba-root"

if [ -f "/trainingData/sage/yue/tools/micromamba/bin/micromamba" ]; then
  eval "$(/trainingData/sage/yue/tools/micromamba/bin/micromamba shell hook -s bash)"
  micromamba activate /trainingData/sage/yue/envs/viewtrust-p0
else
  echo "micromamba not found at expected server path."
  return 1 2>/dev/null || exit 1
fi

export CUDA_HOME="/trainingData/sage/yue/envs/viewtrust-p0"
export CUDA_PATH="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:$PATH"
export CPATH="$CUDA_HOME/targets/x86_64-linux/include:$CUDA_HOME/include:${CPATH:-}"
export LIBRARY_PATH="$CUDA_HOME/targets/x86_64-linux/lib:$CUDA_HOME/lib:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$CUDA_HOME/targets/x86_64-linux/lib:$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"

export VIEWTRUST_PROJECT_ROOT="$PROJECT_ROOT"
export VIEWTRUST_OUTPUT_ROOT="${VIEWTRUST_OUTPUT_ROOT:-$PROJECT_ROOT/outputs}"
export VIEWTRUST_THIRD_PARTY_ROOT="${VIEWTRUST_THIRD_PARTY_ROOT:-$PROJECT_ROOT/third_party}"

echo "===== ViewTrust-GS server environment ====="
echo "PROJECT_ROOT=$PROJECT_ROOT"
echo "CONDA_PREFIX=${CONDA_PREFIX:-}"
echo "CUDA_HOME=$CUDA_HOME"
echo "VIEWTRUST_OUTPUT_ROOT=$VIEWTRUST_OUTPUT_ROOT"
echo

echo "===== binaries ====="
which python || true
python --version || true
which pip || true
which nvcc || true
nvcc --version || true

echo "===== torch / gsplat ====="
python - <<'PY'
import os
import torch

print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(i, torch.cuda.get_device_name(i))

try:
    import gsplat
    print("gsplat import: ok")
    print("gsplat file:", gsplat.__file__)
except Exception as e:
    print("gsplat import: failed")
    print(repr(e))

print("CUDA_HOME:", os.environ.get("CUDA_HOME"))
print("CUDA_PATH:", os.environ.get("CUDA_PATH"))
PY
```

This script may contain the current server path because it is server-specific. Core Python code must not.

---

## 6. Testing Philosophy

Every task must clearly label tests as one of:

```text id="my8q8m"
LOCAL-SAFE
SERVER-REQUIRED
OPTIONAL-GPU
```

### 6.1 LOCAL-SAFE tests

These can run on the Mac:

```text id="lrx4sy"
Python syntax checks
config parsing
metadata schema validation
path resolution
mock logger writes
small fake data tests
unit tests that do not import CUDA-only modules
```

Examples:

```bash id="v6jleq"
python -m py_compile viewtrust/**/*.py
python scripts/smoke/mock_cpu_smoke_test.py
bash scripts/checks/run_mock_checks.sh
```

### 6.2 SERVER-REQUIRED tests

These must run on the remote server:

```text id="l7rtpy"
torch.cuda.is_available()
gsplat CUDA rasterization
training loop execution
GPU memory logging
CUDA extension build
actual Priority 0 logging
3DGS dataset training
```

Examples:

```bash id="60jb9n"
source scripts/env/activate_server_viewtrust_p0.sh
python scripts/smoke/gsplat_cuda_smoke_test.py
bash scripts/checks/run_server_checks.sh
```

Codex must not claim these passed unless the user provides server output.

---

## 7. Git Sync Workflow

The user will develop locally and execute remotely.

Expected workflow:

```text id="un3iv9"
1. Codex edits code on local Mac.
2. Codex runs only local-safe checks.
3. User commits and pushes to GitHub.
4. User pulls the repo on the server.
5. User activates server environment.
6. User runs server-required checks.
7. User pastes logs back for Codex to debug.
```

Codex must write instructions assuming this workflow.

Every PR or task should include:

```text id="w7l2lj"
Local checks I can run
Server checks the user must run
Expected output
What logs to paste back if it fails
```

---

## 8. First Required Files

Before touching training code, create these files:

```text id="0a2hkk"
docs/LOCAL_DEVELOPMENT.md
docs/SERVER_ENVIRONMENT.md
docs/IMPLEMENTATION_PLAN.md
docs/TESTING_PROTOCOL.md
scripts/env/activate_server_viewtrust_p0.sh
scripts/env/check_server_environment.sh
scripts/smoke/gsplat_cuda_smoke_test.py
scripts/smoke/mock_cpu_smoke_test.py
scripts/checks/run_static_checks.sh
scripts/checks/run_mock_checks.sh
scripts/checks/run_server_checks.sh
configs/default.yaml
configs/local.example.yaml
configs/server.example.yaml
.env.example
.env.server.example
```

---

## 9. First Codex Task

Do not start training instrumentation yet.

First implement only:

```text id="08svru"
environment documentation
path policy
activation script
smoke tests
mock local tests
project scaffold
```

Definition of done:

```text id="tjc4fm"
1. Mac can run mock_cpu_smoke_test.py without CUDA.
2. Server can run activate_server_viewtrust_p0.sh.
3. Server can run gsplat_cuda_smoke_test.py.
4. No core Python code contains hard-coded /trainingData paths.
5. docs/IMPLEMENTATION_PLAN.md explains how future Priority 0 logging will be added.
```

---

## 10. Server Smoke Test Script

Create:

```text id="mxq8qr"
scripts/smoke/gsplat_cuda_smoke_test.py
```

It should contain the known successful gsplat rasterization test and print:

```text id="oo70p3"
torch version
torch cuda version
cuda availability
GPU names
gsplat import path
render output shapes
meta keys
smoke test ok
```

It should fail loudly if CUDA is unavailable.

---

## 11. Local Mock Smoke Test

Create:

```text id="jjpo54"
scripts/smoke/mock_cpu_smoke_test.py
```

This should not import gsplat CUDA or require GPU.

It should test:

```text id="63cofh"
config loading
run_id generation
output directory creation
metadata JSON writing
basic logger schema
path resolution with relative paths
```

It should write to:

```text id="df6coq"
./outputs/mock_smoke_test/
```

and print:

```text id="7dqx35"
mock smoke test ok
```

---

## 12. Coding Rules for Codex

Codex must follow these rules:

```text id="4blgz9"
1. Do not hard-code server absolute paths in core code.
2. Do not assume CUDA is available on local Mac.
3. Do not import gsplat at module import time in general utilities.
4. Put GPU-specific imports inside GPU-specific functions or scripts.
5. Keep local tests CPU-only.
6. Keep server tests explicit.
7. Do not modify training behavior until scaffold and environment checks are complete.
8. Do not implement ViewTrust defense yet.
9. Do not add attack repos as required dependencies.
10. Do not add huge outputs, datasets, caches, or compiled artifacts to git.
```

---

## 13. Priority 0 After Environment Scaffold

Only after the above is complete, proceed to Priority 0 logging in this order:

```text id="jm1fjj"
PR1: run metadata logging
PR2: view-level loss and residual summary logging
PR3: training dynamics logging
PR4: Gaussian lifecycle ID tracking
PR5: Gaussian summary checkpoints
PR6: sparse visibility / top contributor logging
PR7: correctness checks
PR8: Priority 0 analysis report
```

Do not jump to trust scoring, defense, gating, or poisoning experiments yet.

---

## 14. Required Response Format from Codex

For every task, Codex should report:

```text id="uplkvi"
Changed files:
- ...

Local checks:
- command
- expected result
- actual result, if runnable locally

Server checks for user:
- command
- expected result
- what log to paste back if it fails

Assumptions:
- ...

Next step:
- ...
```

If a check cannot be run locally due to Mac limitations, say so explicitly and provide the exact server command.

---

## 15. Current First Prompt to Codex

Use this prompt to start:

```text id="nk7tzz"
We are building ViewTrust-GS Priority 0.

Important architecture constraint:
I develop with Codex on a local Mac, but the real GPU environment exists only on a remote Ubuntu server. The Mac may not have CUDA or gsplat working. Therefore, code must be portable, config-driven, and tested in two tiers: local mock tests and server GPU tests.

Known server:
- Ubuntu 20.04.6
- 2 × RTX 3090 24GB
- NVIDIA driver 560.35.03
- Driver CUDA runtime 12.6
- Working micromamba env: /trainingData/sage/yue/envs/viewtrust-p0
- CUDA_HOME on server: /trainingData/sage/yue/envs/viewtrust-p0
- gsplat CUDA smoke test has passed on server
- Do not use system /usr/local/cuda-11.0

Do not hard-code server paths in core Python code.
Server-specific paths may appear only in docs, .env.server.example, configs/server.example.yaml, or scripts/env/activate_server_viewtrust_p0.sh.

Before touching training code, create the environment/documentation/testing scaffold:
- docs/LOCAL_DEVELOPMENT.md
- docs/SERVER_ENVIRONMENT.md
- docs/IMPLEMENTATION_PLAN.md
- docs/TESTING_PROTOCOL.md
- scripts/env/activate_server_viewtrust_p0.sh
- scripts/env/check_server_environment.sh
- scripts/smoke/gsplat_cuda_smoke_test.py
- scripts/smoke/mock_cpu_smoke_test.py
- scripts/checks/run_static_checks.sh
- scripts/checks/run_mock_checks.sh
- scripts/checks/run_server_checks.sh
- configs/default.yaml
- configs/local.example.yaml
- configs/server.example.yaml
- .env.example
- .env.server.example

The first PR must not modify training behavior.
The first PR must not implement ViewTrust defense.
The first PR must not require CUDA on local Mac.
The first PR must not import gsplat in general CPU-only modules.

Please first inspect the current repo structure, then write docs/IMPLEMENTATION_PLAN.md before implementing. Include changed files, local checks, server checks, assumptions, and next steps.
```
