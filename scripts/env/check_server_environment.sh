#!/usr/bin/env bash
set -euo pipefail

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

[ -n "${CUDA_HOME:-}" ] || fail "CUDA_HOME is missing. Source scripts/env/activate_server_viewtrust_p0.sh first."

NVCC_PATH="$(command -v nvcc || true)"
[ -n "${NVCC_PATH}" ] || fail "nvcc is not found on PATH."

case "${NVCC_PATH}" in
  *cuda-11.0*)
    fail "nvcc resolves to an unsupported CUDA 11.0 toolchain: ${NVCC_PATH}"
    ;;
esac

PYTHON_PATH="$(command -v python || true)"
[ -n "${PYTHON_PATH}" ] || fail "python is not found on PATH."

EXPECTED_PREFIX="${VIEWTRUST_ENV_PREFIX:-${CUDA_HOME}}"
case "${PYTHON_PATH}" in
  "${EXPECTED_PREFIX}"/*)
    ;;
  *)
    fail "python is not from the active ViewTrust environment. python=${PYTHON_PATH}, expected prefix=${EXPECTED_PREFIX}"
    ;;
esac

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
