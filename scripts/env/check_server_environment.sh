#!/usr/bin/env bash
set -euo pipefail

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

REQUIRE_GAUSSIAN_SPLATTING=0
REQUIRE_OBSERVATION_PATCH=0
for arg in "$@"; do
  case "${arg}" in
    --require-gaussian-splatting)
      REQUIRE_GAUSSIAN_SPLATTING=1
      ;;
    --require-observation-patch)
      REQUIRE_OBSERVATION_PATCH=1
      ;;
    *)
      fail "unknown argument: ${arg}"
      ;;
  esac
done

path_contains() {
  local needle="$1"
  local haystack="$2"
  case ":${haystack}:" in
    *":${needle}:"*) return 0 ;;
    *) return 1 ;;
  esac
}

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

echo "ViewTrust-GS server environment check"
echo "CONDA_PREFIX=${CONDA_PREFIX:-}"
echo "VIRTUAL_ENV=${VIRTUAL_ENV:-}"
echo "CUDA_HOME=${CUDA_HOME:-}"
echo "CUDA_PATH=${CUDA_PATH:-}"
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}"
echo "PATH=${PATH}"
echo "REQUIRE_GAUSSIAN_SPLATTING=${REQUIRE_GAUSSIAN_SPLATTING}"
echo "REQUIRE_OBSERVATION_PATCH=${REQUIRE_OBSERVATION_PATCH}"

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

if [ "${REQUIRE_GAUSSIAN_SPLATTING}" -eq 1 ]; then
  THIRD_PARTY_ROOT="${VIEWTRUST_THIRD_PARTY_ROOT:-${PROJECT_ROOT}/third_party}"
  TRAINER_PATH="${THIRD_PARTY_ROOT}/gaussian-splatting/train.py"
  echo "Gaussian Splatting trainer path: ${TRAINER_PATH}"

  [ -f "${TRAINER_PATH}" ] || fail "third_party/gaussian-splatting/train.py was not found."

  TORCH_LIB_DIR="$(
    python - <<'PY'
from pathlib import Path

try:
    import torch
except Exception as exc:
    raise SystemExit(f"torch import failed while resolving torch lib dir: {exc}")

print(Path(torch.__file__).resolve().parent / "lib")
PY
  )"
  echo "torch shared library dir: ${TORCH_LIB_DIR}"
  [ -d "${TORCH_LIB_DIR}" ] || fail "torch shared library directory does not exist: ${TORCH_LIB_DIR}"

  if ! path_contains "${TORCH_LIB_DIR}" "${LD_LIBRARY_PATH:-}"; then
    fail "torch shared library directory is not in LD_LIBRARY_PATH. Source scripts/env/activate_server_viewtrust_p0.sh again."
  fi

  python - <<'PY'
try:
    from diff_gaussian_rasterization import (
        GaussianRasterizationSettings,
        GaussianRasterizer,
    )
    from simple_knn._C import distCUDA2
    import fused_ssim
except Exception as exc:
    raise SystemExit(f"official Gaussian Splatting CUDA submodule import failed: {exc}")

print("official Gaussian Splatting CUDA submodule imports ok")
print(f"GaussianRasterizationSettings={GaussianRasterizationSettings}")
print(f"GaussianRasterizer={GaussianRasterizer}")
print(f"distCUDA2={distCUDA2}")
print(f"fused_ssim={getattr(fused_ssim, '__file__', '<unknown>')}")
PY
fi

if [ "${REQUIRE_OBSERVATION_PATCH}" -eq 1 ]; then
  THIRD_PARTY_ROOT="${VIEWTRUST_THIRD_PARTY_ROOT:-${PROJECT_ROOT}/third_party}"
  python scripts/third_party/check_gaussian_splatting_observation_patch.py \
    --third-party-root "${THIRD_PARTY_ROOT}" \
    --patch pr7_training_events \
    --require-applied
fi

echo "server environment check ok"
