#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"

echo "SERVER-REQUIRED: this command is not expected to pass on a local Mac without CUDA."

bash scripts/env/check_server_environment.sh
python scripts/smoke/gsplat_cuda_smoke_test.py

echo "server checks ok"
