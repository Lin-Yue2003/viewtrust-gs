#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"

python scripts/smoke/observed_command_smoke_test.py

python scripts/measure/run_observed_command.py \
  --label observed-sleep-test \
  --sample-interval-s 0.5 \
  -- python -c "import time; print('sleep test start'); time.sleep(3); print('sleep test end')"

echo "observed checks ok"
