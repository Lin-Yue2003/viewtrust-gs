#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"

python scripts/smoke/mock_cpu_smoke_test.py
python scripts/smoke/priority0_logging_smoke_test.py

echo "mock checks ok"
