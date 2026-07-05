# ViewTrust-GS Implementation Plan

## 1. Current Repository Structure

At project start, the repository contains only the project guide and third-party source trees:

```text
viewtrust-gs/
  Agent_guide/
    constrain.md
  third_party/
    gaussian-splatting/
    gsplat/
```

The ViewTrust project scaffold, configs, scripts, docs, package modules, and tests do not yet exist.

## 2. Files to Create

This first task creates the observation-only scaffold:

```text
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

viewtrust/__init__.py
viewtrust/configs/
viewtrust/logging/
viewtrust/checks/
viewtrust/analysis/
viewtrust/utils/

tests/unit/
tests/mock/

outputs/.gitkeep
third_party/.gitkeep
```

Package directories will include `__init__.py` files so static syntax checks can import the scaffold cleanly.

## 3. Files to Modify

No training-related code will be modified in this first task.

The existing third-party training sources under `third_party/gaussian-splatting/` and `third_party/gsplat/` are treated as read-only for this scaffold pass.

## 4. Local Mac Testing Strategy

Local Mac checks are limited to LOCAL-SAFE work:

```text
python -m compileall viewtrust scripts
bash scripts/checks/run_static_checks.sh
bash scripts/checks/run_mock_checks.sh
```

The local mock smoke test must not import `gsplat`, require CUDA, or execute real 3DGS training. It validates config loading, relative path resolution, run ID generation, metadata JSON writing, output directory creation, and a placeholder logger schema.

## 5. Remote Server Testing Strategy

Remote server checks run only after the repository is synced to the Ubuntu GPU server and the server environment is activated:

```bash
source scripts/env/activate_server_viewtrust_p0.sh
bash scripts/checks/run_server_checks.sh
```

Server checks validate CUDA, `nvcc`, PyTorch CUDA availability, GPU inventory, `gsplat` import, and a minimal CUDA rasterization smoke test.

## 6. Path Policy

Core Python code must not hard-code machine-specific absolute paths, including:

```text
the known server project root
the known server environment prefix
Linux home-directory paths
local Mac user-directory paths
absolute dataset paths
absolute output paths
```

Core code must use relative defaults, command-line arguments, config files, or environment variables:

```text
VIEWTRUST_PROJECT_ROOT
VIEWTRUST_DATA_ROOT
VIEWTRUST_OUTPUT_ROOT
VIEWTRUST_THIRD_PARTY_ROOT
CUDA_HOME
CUDA_PATH
```

Server-specific absolute paths may appear only in:

```text
docs/SERVER_ENVIRONMENT.md
.env.server.example
configs/server.example.yaml
scripts/env/activate_server_viewtrust_p0.sh
scripts/env/check_server_environment.sh
```

## 7. Environment Activation Policy

Local development does not require CUDA activation. Local scripts infer the project root from their location and default to relative paths.

Server execution must use the micromamba environment at the known server location and must use the environment CUDA 12.x toolchain, not the old system CUDA 11.0 toolchain. CUDA headers may be provided under the environment target include directory as well as the top-level include directory.

## 8. GPU-Specific Import Isolation

GPU-specific imports such as `torch` CUDA calls and `gsplat` imports are isolated to server-only scripts or functions:

```text
scripts/env/check_server_environment.sh
scripts/smoke/gsplat_cuda_smoke_test.py
scripts/checks/run_server_checks.sh
```

CPU-only project modules and mock tests must not import `gsplat`.

## 9. LOCAL-SAFE

LOCAL-SAFE tasks may run on the local Mac:

```text
documentation checks
config parsing
relative path resolution
CPU-only mock smoke test
Python syntax compilation
schema placeholder validation
small output writes under ./outputs/mock_smoke_test/
```

## 10. SERVER-REQUIRED

SERVER-REQUIRED tasks must run on the remote Ubuntu GPU server:

```text
CUDA validation
nvcc validation
torch.cuda.is_available() validation
GPU enumeration
gsplat import validation
gsplat CUDA rasterization smoke test
real 3DGS training
Priority 0 GPU memory and timing measurement
```

## 11. Definition of Done for This First Task

This first task is done when:

```text
the scaffold files and directories exist
local documentation describes the two-tier workflow
server documentation records the known validated environment
configs use relative local defaults
server examples document server paths only in approved files
local static checks pass
local mock checks pass without CUDA or gsplat
server checks are documented but not required to pass on Mac
training-related code remains unchanged
```

## 12. Explicit Non-Modification Statement

Training behavior will not be modified in this first task.

This task does not implement ViewTrust scoring, a defense, densification gating, pruning changes, rendering changes, loss changes, optimizer changes, dataset sampling changes, or attack generation.

## Future Priority 0 Roadmap

After the scaffold is validated locally and on the server, future Priority 0 work can add observation-only logging around training execution. That future work must preserve training behavior and record metadata, timing, GPU memory, configuration snapshots, and view/iteration observations without changing optimization or rendering decisions.

## Priority 0 Logging Foundation

The next implementation step adds a local-safe Priority 0 logger that writes run
metadata, config snapshots, and JSONL observation events. This foundation is
intended to be called later by server-side training wrappers, but it does not
modify training code or import GPU-specific dependencies.

## PR Progression

```text
PR0: environment and scaffold
PR1: observed command validation
PR2: minimal dataset policy and NeRF Synthetic chair subset recipe
PR3: baseline clean training wrapper
PR4: training-internal Priority 0 logging
```

PR1 validates external observation before any training-loop instrumentation. It
uses `scripts/measure/run_observed_command.py` to observe subprocesses from the
outside and record Priority 0 artifacts without modifying training behavior.

PR2 creates a minimal NeRF Synthetic chair subset preparation recipe. It is
still observation preparation only and does not modify training behavior.

PR3 creates a clean chair baseline training wrapper. It uses the observed
command infrastructure to run an external trainer and does not modify training
internals.
