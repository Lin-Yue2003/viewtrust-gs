#!/usr/bin/env python3
"""LOCAL-SAFE smoke test for PR8 child trainer lifecycle imports."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path


def _bootstrap_project_imports() -> Path:
    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root))
    return project_root


def main() -> int:
    project_root = _bootstrap_project_imports()

    from viewtrust.training.baseline import (
        build_gaussian_lifecycle_env,
        preflight_gaussian_lifecycle_observer_import,
    )

    with tempfile.TemporaryDirectory(prefix="viewtrust-lifecycle-child-env-") as tmp_name:
        tmp = Path(tmp_name)
        env = build_gaussian_lifecycle_env(
            enabled=True,
            project_root=project_root,
            run_dir=tmp / "run",
            run_id="lifecycle-child-env-smoke",
            scene="chair",
            condition="clean",
            trainer="gaussian-splatting",
            strict=True,
            log_snapshot_stats=True,
            base_env={},
        )
        if env["PYTHONPATH"].split(os.pathsep)[0] != str(project_root.resolve()):
            raise ValueError("project root was not prepended to PYTHONPATH")
        if env.get("VIEWTRUST_GAUSSIAN_LIFECYCLE_STRICT") != "1":
            raise ValueError("strict lifecycle mode was not encoded")

        base_env = {"PATH": os.environ.get("PATH", "")}
        preflight = preflight_gaussian_lifecycle_observer_import(
            python_executable=Path(sys.executable),
            env_overrides=env,
            cwd=tmp,
            base_env=base_env,
        )
        if preflight.returncode != 0:
            raise RuntimeError(preflight.stderr or preflight.stdout)
        if "gaussian lifecycle import ok" not in preflight.stdout:
            raise ValueError("lifecycle import preflight did not print success")

        missing_pythonpath = preflight_gaussian_lifecycle_observer_import(
            python_executable=Path(sys.executable),
            env_overrides={},
            cwd=tmp,
            base_env=base_env,
        )
        if missing_pythonpath.returncode == 0:
            raise ValueError("lifecycle import unexpectedly succeeded without PYTHONPATH")

    print("gaussian lifecycle child env smoke test ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
