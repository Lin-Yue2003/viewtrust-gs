# Local Development

ViewTrust-GS is edited locally on a Mac with Codex, while the real GPU environment lives on the remote Ubuntu server.

The local Mac is for:

```text
code editing
documentation
config validation
schema placeholder validation
CPU-only mock tests
static syntax checks
```

The local Mac may not have CUDA, `gsplat` CUDA kernels, or the real 3DGS training environment. Local checks must not assume CUDA is available.

## Local-Safe Commands

From the repository root:

```bash
bash scripts/checks/run_static_checks.sh
bash scripts/checks/run_mock_checks.sh
```

These commands are expected to run without CUDA and without importing `gsplat`.

Local Mac development is limited to static checks, CPU-only mock tests, config
parsing, path resolution, and metadata/log artifact writing. CUDA, `gsplat`
rasterization, real training, and real Priority 0 GPU measurements are server
work only.

## Not Local-Safe

Do not treat the following as complete based on local Mac execution:

```text
CUDA validation
gsplat CUDA rasterization
real 3DGS training
Priority 0 GPU memory measurement
Priority 0 timing measurement
```

Those checks must run after syncing the repository to the remote server.

On the server, use the micromamba environment directly. Do not stack an old uv
virtual environment such as `.venv-viewtrust-p0` with the micromamba
environment.

## Development Rule

This scaffold is observation-only infrastructure. It does not modify training behavior, rendering behavior, losses, optimizers, densification, pruning, or dataset sampling.
