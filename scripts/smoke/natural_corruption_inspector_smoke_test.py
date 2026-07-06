#!/usr/bin/env python3
"""LOCAL-SAFE smoke test for PR10 natural corruption inspector."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.smoke.natural_corruption_generation_smoke_test import _make_fake_clean


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="viewtrust-corrupt-inspect-") as tmp:
        data_root = Path(tmp) / "data"
        _make_fake_clean(data_root)
        generate_script = PROJECT_ROOT / "scripts" / "data" / "generate_natural_corruptions.py"
        inspect_script = PROJECT_ROOT / "scripts" / "measure" / "inspect_natural_corruption_dataset.py"
        generated = _run(
            [
                sys.executable,
                str(generate_script),
                "--data-root",
                str(data_root),
                "--output-condition",
                "corrupt_occluder",
                "--corruption-type",
                "occluder",
                "--num-corrupt-train-views",
                "2",
                "--copy-mode",
                "copy",
                "--overwrite",
            ]
        )
        if generated.returncode != 0:
            raise RuntimeError(generated.stderr or generated.stdout)

        ok = _run(
            [
                sys.executable,
                str(inspect_script),
                "--data-root",
                str(data_root),
                "--scene",
                "chair",
                "--condition",
                "corrupt_occluder",
                "--require-valid",
                "--require-corrupted-count",
                "2",
            ]
        )
        if ok.returncode != 0:
            raise RuntimeError(ok.stderr or ok.stdout)
        report = json.loads(ok.stdout)
        if report["valid"] is not True:
            raise ValueError("inspector should report valid")
        if report["has_manifest"] is not True:
            raise ValueError("inspector should validate manifest.json")
        if report["test_target_uncorrupted"] is not True:
            raise ValueError("test/target should remain uncorrupted")
        if report["all_transform_images_exist"] is not True:
            raise ValueError("all transform images should exist")

        wrong_count = _run(
            [
                sys.executable,
                str(inspect_script),
                "--data-root",
                str(data_root),
                "--scene",
                "chair",
                "--condition",
                "corrupt_occluder",
                "--require-valid",
                "--require-corrupted-count",
                "3",
            ]
        )
        if wrong_count.returncode == 0:
            raise ValueError("wrong corrupted count should fail")

        manifest_path = (
            data_root
            / "viewtrust-mini"
            / "nerf_synthetic"
            / "chair"
            / "corrupt_occluder"
            / "manifest.json"
        )
        manifest_path.unlink()
        missing_manifest = _run(
            [
                sys.executable,
                str(inspect_script),
                "--data-root",
                str(data_root),
                "--scene",
                "chair",
                "--condition",
                "corrupt_occluder",
                "--require-valid",
            ]
        )
        if missing_manifest.returncode == 0:
            raise ValueError("missing manifest.json should fail under --require-valid")

    print("natural corruption inspector smoke test ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
