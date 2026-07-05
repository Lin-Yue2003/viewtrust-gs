#!/usr/bin/env python3
"""LOCAL-SAFE smoke test for PR7.1 child trainer observer imports."""

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
        build_training_event_env,
        preflight_training_event_observer_import,
    )

    with tempfile.TemporaryDirectory(prefix="viewtrust-child-env-") as tmp_name:
        tmp = Path(tmp_name)
        run_dir = tmp / "run"
        env = build_training_event_env(
            enabled=True,
            project_root=project_root,
            run_dir=run_dir,
            run_id="child-env-smoke",
            scene="chair",
            condition="clean",
            trainer="gaussian-splatting",
            log_interval=10,
            strict=True,
            base_env={},
        )
        pythonpath_parts = env["PYTHONPATH"].split(os.pathsep)
        if pythonpath_parts[0] != str(project_root.resolve()):
            raise ValueError("project root was not prepended to PYTHONPATH")
        if env.get("VIEWTRUST_OBSERVER_STRICT") != "1":
            raise ValueError("strict observer mode was not encoded in child env")

        base_env = {"PATH": os.environ.get("PATH", "")}
        preflight = preflight_training_event_observer_import(
            python_executable=Path(sys.executable),
            env_overrides=env,
            cwd=tmp,
            base_env=base_env,
        )
        if preflight.returncode != 0:
            raise RuntimeError(preflight.stderr or preflight.stdout)
        if "observer import ok" not in preflight.stdout:
            raise ValueError("observer import preflight did not print success")

        missing_pythonpath = preflight_training_event_observer_import(
            python_executable=Path(sys.executable),
            env_overrides={},
            cwd=tmp,
            base_env=base_env,
        )
        if missing_pythonpath.returncode == 0:
            raise ValueError("observer import unexpectedly succeeded without PYTHONPATH")
        if "viewtrust" not in missing_pythonpath.stderr:
            raise ValueError(
                "observer import failure did not mention the missing viewtrust module"
            )

    print("training events child env smoke test ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
