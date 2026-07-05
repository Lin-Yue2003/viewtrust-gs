#!/usr/bin/env python3
"""LOCAL-SAFE smoke test for PR8 Gaussian lifecycle observer."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
from pathlib import Path


def _bootstrap_project_imports() -> None:
    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root))


class FakeGaussians:
    def __init__(self, count: int):
        self.resize(count)

    def resize(self, count: int) -> None:
        self._xyz = [[float(i), float(i + 1), float(i + 2)] for i in range(count)]
        self._opacity = [[0.1] for _ in range(count)]
        self._scaling = [[1.0, 2.0, 3.0] for _ in range(count)]
        self._rotation = [[1.0, 0.0, 0.0, 0.0] for _ in range(count)]

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_opacity(self):
        return self._opacity

    @property
    def get_scaling(self):
        return self._scaling

    @property
    def get_rotation(self):
        return self._rotation


def _row_count(path: Path) -> int:
    with path.open(newline="", encoding="utf-8") as handle:
        return max(0, len(list(csv.reader(handle))) - 1)


def main() -> int:
    _bootstrap_project_imports()

    from viewtrust.observation.gaussian_lifecycle import (
        GaussianLifecycleConfig,
        GaussianLifecycleObserver,
    )

    with tempfile.TemporaryDirectory(prefix="viewtrust-lifecycle-") as tmp:
        output_dir = Path(tmp) / "run" / "gaussian_lifecycle"
        observer = GaussianLifecycleObserver(
            GaussianLifecycleConfig(
                output_dir=output_dir,
                run_id="mock-lifecycle",
                scene="chair",
                condition="clean",
                trainer="gaussian-splatting",
            )
        )
        gaussians = FakeGaussians(5)
        observer.on_after_scene_init(iteration=0, gaussians=gaussians)

        observer.on_before_prune(
            iteration=3,
            prune_mask=[False, True, False, True, False],
            gaussians=gaussians,
        )
        gaussians.resize(3)
        observer.on_after_prune(iteration=3, gaussians=gaussians)
        if len(observer.current_ids) != 3:
            raise ValueError("prune did not leave 3 alive lifecycle IDs")

        gaussians.resize(5)
        observer.on_after_clone(
            iteration=4,
            source_mask=[True, False, True],
            gaussians=gaussians,
        )
        if len(set(observer.current_ids)) != 5:
            raise ValueError("append did not preserve unique lifecycle IDs")
        summary = observer.finalize(
            iteration=10,
            requested_iterations=10,
            gaussians=gaussians,
            final_gaussian_count=5,
        )
        if summary["known_gaussian_count"] != 7:
            raise ValueError("known_gaussian_count mismatch")
        if summary["alive_final_count"] != 5:
            raise ValueError("alive_final_count mismatch")
        if summary["dead_final_count"] != 2:
            raise ValueError("dead_final_count mismatch")
        if summary["prune_death_count"] != 2:
            raise ValueError("prune_death_count mismatch")
        if summary["clone_birth_count"] != 2:
            raise ValueError("clone_birth_count mismatch")
        if summary["invariant_violations"] != 0:
            raise ValueError("unexpected invariant violation")
        if _row_count(output_dir / "gaussian_lifecycle_final.csv") != 7:
            raise ValueError("final lifecycle row count mismatch")
        if _row_count(output_dir / "gaussian_lifecycle_events.csv") < 4:
            raise ValueError("expected clone/prune lifecycle events")
        loaded = json.loads((output_dir / "gaussian_lifecycle_summary.json").read_text())
        if loaded["observation_only"] is not True:
            raise ValueError("summary observation_only mismatch")

    print("gaussian lifecycle observer smoke test ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
