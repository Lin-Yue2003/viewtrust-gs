#!/usr/bin/env python3
"""LOCAL-SAFE smoke test for PR7 training event observer."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
from pathlib import Path


def _bootstrap_project_imports() -> None:
    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root))


def _row_count(path: Path) -> int:
    with path.open(newline="", encoding="utf-8") as handle:
        return max(0, len(list(csv.reader(handle))) - 1)


def main() -> int:
    _bootstrap_project_imports()

    from viewtrust.observation.training_events import (
        TrainingEventObserver,
        TrainingEventObserverConfig,
    )

    with tempfile.TemporaryDirectory(prefix="viewtrust-training-events-") as tmp:
        output_dir = Path(tmp) / "run" / "training_events"
        observer = TrainingEventObserver(
            TrainingEventObserverConfig(
                output_dir=output_dir,
                run_id="mock-run",
                scene="chair",
                condition="clean",
                trainer="gaussian-splatting",
                flush_every=1,
            )
        )
        observer.log_gaussian_count(
            iteration=0,
            stage="after_scene_init",
            gaussian_count=100000,
        )
        observer.log_iteration_metrics(
            iteration=1,
            event_type="iteration_metrics",
            camera_index=0,
            camera_image_name="train_000",
            loss=0.5,
            l1_loss=0.2,
            ssim=0.8,
            gaussian_count=100000,
            visible_gaussian_count=1234,
            visibility_ratio=0.01234,
            densification_eligible=True,
            densification_triggered=False,
            opacity_reset_triggered=False,
            optimizer_step=True,
        )
        observer.log_iteration_metrics(
            iteration=2,
            event_type="iteration_metrics",
            camera_index=1,
            camera_image_name="train_001",
            loss=0.4,
            l1_loss=0.18,
            ssim=0.82,
            gaussian_count=100000,
            visible_gaussian_count=1300,
            visibility_ratio=0.013,
            densification_eligible=True,
            densification_triggered=True,
            opacity_reset_triggered=False,
            optimizer_step=True,
        )
        observer.log_densification_event(
            iteration=2,
            densification_eligible=True,
            densification_triggered=True,
            densify_from_iter=1,
            densify_until_iter=10,
            densification_interval=1,
            densify_grad_threshold=0.0002,
            size_threshold="",
            gaussian_count_before=100000,
            gaussian_count_after=100010,
            opacity_reset_triggered=False,
        )
        summary = observer.finalize(iteration=2, final_gaussian_count=100010)

        required = [
            output_dir / "training_events.csv",
            output_dir / "densification_events.csv",
            output_dir / "gaussian_count_timeseries.csv",
            output_dir / "training_events_summary.json",
            output_dir.parent / "tables" / "training_events.csv",
            output_dir.parent / "tables" / "densification_events.csv",
            output_dir.parent / "tables" / "gaussian_count_timeseries.csv",
            output_dir.parent / "training_events_summary.json",
        ]
        for path in required:
            if not path.exists():
                raise FileNotFoundError(path)
        if _row_count(output_dir / "training_events.csv") != 2:
            raise ValueError("training_events.csv row count mismatch")
        if _row_count(output_dir / "densification_events.csv") != 1:
            raise ValueError("densification_events.csv row count mismatch")
        if _row_count(output_dir / "gaussian_count_timeseries.csv") < 2:
            raise ValueError("gaussian_count_timeseries.csv row count mismatch")
        loaded = json.loads((output_dir / "training_events_summary.json").read_text())
        if loaded["observation_only"] is not True:
            raise ValueError("summary observation_only mismatch")
        if summary["final_gaussian_count"] != 100010:
            raise ValueError("summary final_gaussian_count mismatch")

    print("training event observer smoke test ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
