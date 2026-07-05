#!/usr/bin/env python3
"""LOCAL-SAFE smoke test for observed command measurement."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path


def _bootstrap_project_imports() -> Path:
    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root))
    return project_root


def main() -> int:
    project_root = _bootstrap_project_imports()

    from viewtrust.configs.loader import load_simple_yaml
    from viewtrust.logging.writer import Priority0Logger
    from viewtrust.measurement.command_runner import run_observed_command
    from viewtrust.utils.paths import ensure_child_path

    config = load_simple_yaml(project_root / "configs" / "default.yaml")
    output_root = (project_root / "outputs").resolve()
    run_id = "observed-command-mock-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = output_root / "observed_command_smoke_test" / run_id
    ensure_child_path(output_root, run_dir)

    logger = Priority0Logger(run_dir=run_dir, run_id=run_id)
    result = run_observed_command(
        command=[
            sys.executable,
            "-c",
            "print('viewtrust observed command mock')",
        ],
        logger=logger,
        config=config,
        sample_interval_s=0.1,
        label="observed_command_smoke_test",
    )
    if result.returncode != 0:
        raise SystemExit(result.returncode)

    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    if summary["summary"]["returncode"] != 0:
        raise ValueError("observed command summary did not record returncode 0")
    if "viewtrust observed command mock" not in (run_dir / "stdout.log").read_text(encoding="utf-8"):
        raise ValueError("stdout log did not capture mock command output")

    print(f"observed command smoke run_dir: {run_dir}")
    print("observed command smoke test ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
