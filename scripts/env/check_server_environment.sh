#!/usr/bin/env bash
set -euo pipefail

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

echo "ViewTrust-GS server environment check"
echo "CONDA_PREFIX=${CONDA_PREFIX:-}"
echo "VIRTUAL_ENV=${VIRTUAL_ENV:-}"
echo "CUDA_HOME=${CUDA_HOME:-}"
echo "CUDA_PATH=${CUDA_PATH:-}"
echo "PATH=${PATH}"

[ -n "${CUDA_HOME:-}" ] || fail "CUDA_HOME is missing. Source scripts/env/activate_server_viewtrust_p0.sh first."

EXPECTED_PREFIX="${VIEWTRUST_ENV_PREFIX:-${CUDA_HOME}}"
echo "EXPECTED_PREFIX=${EXPECTED_PREFIX}"

NVCC_PATH="$(command -v nvcc || true)"
echo "which nvcc: ${NVCC_PATH:-<not found>}"
[ -n "${NVCC_PATH}" ] || fail "nvcc is not found on PATH."

case "${NVCC_PATH}" in
  *cuda-11.0*)
    fail "nvcc resolves to an unsupported CUDA 11.0 toolchain: ${NVCC_PATH}"
    ;;
esac

EXPECTED_NVCC="${CUDA_HOME}/bin/nvcc"
if [ "${NVCC_PATH}" != "${EXPECTED_NVCC}" ]; then
  case "${NVCC_PATH}" in
    "${CUDA_HOME}"/bin/nvcc)
      ;;
    *)
      fail "nvcc is not under CUDA_HOME/bin. nvcc=${NVCC_PATH}, expected=${EXPECTED_NVCC}"
      ;;
  esac
fi

if [ ! -f "${CUDA_HOME}/include/cuda_runtime.h" ] && [ ! -f "${CUDA_HOME}/targets/x86_64-linux/include/cuda_runtime.h" ]; then
  echo "Checked CUDA header locations:" >&2
  echo "  ${CUDA_HOME}/include/cuda_runtime.h" >&2
  echo "  ${CUDA_HOME}/targets/x86_64-linux/include/cuda_runtime.h" >&2
  fail "cuda_runtime.h was not found under CUDA_HOME include paths."
fi

PYTHON_PATH="$(command -v python || true)"
echo "which python: ${PYTHON_PATH:-<not found>}"
[ -n "${PYTHON_PATH}" ] || fail "python is not found on PATH."

case "${PYTHON_PATH}" in
  "${EXPECTED_PREFIX}"/*)
    ;;
  *)
    fail "python is not from the active ViewTrust environment. python=${PYTHON_PATH}, expected prefix=${EXPECTED_PREFIX}"
    ;;
esac

echo "which pip: $(command -v pip || true)"
echo "nvcc --version:"
nvcc --version

python - <<'PY'
import importlib.util
import sys

try:
    import torch
except Exception as exc:
    raise SystemExit(f"torch import failed: {exc}")

print(f"torch version: {torch.__version__}")
print(f"torch CUDA version: {torch.version.cuda}")
print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")

if not torch.cuda.is_available():
    raise SystemExit("torch.cuda.is_available() is False")

print(f"GPU count: {torch.cuda.device_count()}")
for index in range(torch.cuda.device_count()):
    print(f"GPU {index}: {torch.cuda.get_device_name(index)}")

spec = importlib.util.find_spec("gsplat")
if spec is None:
    raise SystemExit("gsplat cannot be imported")

import gsplat

print(f"gsplat import ok: {getattr(gsplat, '__file__', '<unknown>')}")
PY

echo "server environment check ok"
