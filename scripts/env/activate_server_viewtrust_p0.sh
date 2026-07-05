#!/usr/bin/env bash

# Source this file from the repository root on the GPU server:
#   source scripts/env/activate_server_viewtrust_p0.sh

_viewtrust_return() {
  return "$1" 2>/dev/null || exit "$1"
}

export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-/trainingData/sage/yue}"
export VIEWTRUST_ENV_PREFIX="/trainingData/sage/yue/envs/viewtrust-p0"

_VIEWTRUST_ACTIVATE_SCRIPT="${BASH_SOURCE[0]:-$0}"
_VIEWTRUST_ACTIVATE_DIR="$(cd "$(dirname "${_VIEWTRUST_ACTIVATE_SCRIPT}")" && pwd)"
export VIEWTRUST_PROJECT_ROOT="$(cd "${_VIEWTRUST_ACTIVATE_DIR}/../.." && pwd)"

if command -v micromamba >/dev/null 2>&1; then
  eval "$(micromamba shell hook --shell bash)"
  micromamba activate "${VIEWTRUST_ENV_PREFIX}" || _viewtrust_return 1
elif [ -f "${VIEWTRUST_ENV_PREFIX}/bin/activate" ]; then
  # Fallback for environments that expose a standard activation script.
  # shellcheck disable=SC1091
  source "${VIEWTRUST_ENV_PREFIX}/bin/activate" || _viewtrust_return 1
else
  echo "ERROR: micromamba was not found and no activation script exists at ${VIEWTRUST_ENV_PREFIX}/bin/activate" >&2
  _viewtrust_return 1
fi

export CUDA_HOME="${VIEWTRUST_ENV_PREFIX}"
export CUDA_PATH="${CUDA_HOME}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export CPATH="${CUDA_HOME}/include:${CPATH:-}"
export LIBRARY_PATH="${CUDA_HOME}/lib:${CUDA_HOME}/lib64:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

export VIEWTRUST_DATA_ROOT="${VIEWTRUST_DATA_ROOT:-./data}"
export VIEWTRUST_OUTPUT_ROOT="${VIEWTRUST_OUTPUT_ROOT:-./outputs}"
export VIEWTRUST_THIRD_PARTY_ROOT="${VIEWTRUST_THIRD_PARTY_ROOT:-./third_party}"

echo "ViewTrust-GS server environment diagnostics"
echo "PROJECT_ROOT=${VIEWTRUST_PROJECT_ROOT}"
echo "which python: $(command -v python || true)"
python --version || true
echo "which pip: $(command -v pip || true)"
echo "which nvcc: $(command -v nvcc || true)"
nvcc --version || true
echo "CUDA_HOME=${CUDA_HOME}"
echo "CUDA_PATH=${CUDA_PATH}"

python - <<'PY'
import importlib.util

try:
    import torch
except Exception as exc:
    print(f"torch import failed: {exc}")
else:
    print(f"torch version: {torch.__version__}")
    print(f"torch CUDA version: {torch.version.cuda}")
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU count: {torch.cuda.device_count()}")
        for index in range(torch.cuda.device_count()):
            print(f"GPU {index}: {torch.cuda.get_device_name(index)}")

spec = importlib.util.find_spec("gsplat")
if spec is None:
    print("gsplat import status: not found")
else:
    print(f"gsplat import status: found at {spec.origin}")
PY
