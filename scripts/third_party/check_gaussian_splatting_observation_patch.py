#!/usr/bin/env python3
"""Check whether the PR7 observation patch is applied to Gaussian Splatting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PATCH_NAME = "pr7_training_events"
START = "# VIEWTRUST PR7 OBSERVATION START"
END = "# VIEWTRUST PR7 OBSERVATION END"


def inspect_patch(third_party_root: Path) -> dict[str, object]:
    train_path = third_party_root / "gaussian-splatting" / "train.py"
    exists = train_path.is_file()
    text = train_path.read_text(encoding="utf-8") if exists else ""
    marker_count = text.count(START)
    applied = marker_count > 0 and END in text
    checks = {
        "has_markers": applied,
        "has_env_gate": "VIEWTRUST_ENABLE_TRAINING_EVENTS" in text,
        "imports_observer": "TrainingEventObserver" in text,
        "uses_observer_environment": "TrainingEventObserver.from_environment()" in text,
        "has_finalize": "\"finalize\"" in text or ".finalize(" in text,
        "has_densification_hook": "\"log_densification_event\"" in text,
        "uses_bool_visibility_count": ".bool().sum()" in text,
        "uses_gaussian_count_visibility_ratio": "visible_count / gaussian_count" in text,
        "passes_requested_iterations": "requested_iterations=opt.iterations" in text,
        "keeps_densify_call": "gaussians.densify_and_prune(" in text,
        "keeps_optimizer_step": "gaussians.optimizer.step(" in text,
    }
    ok = exists and applied and all(checks.values())
    return {
        "patch": PATCH_NAME,
        "train_path": str(train_path),
        "exists": exists,
        "applied": applied,
        "marker_count": marker_count,
        "ok": ok,
        "checks": checks,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--third-party-root", required=True, type=Path)
    parser.add_argument("--patch", default=PATCH_NAME)
    parser.add_argument("--require-applied", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.patch != PATCH_NAME:
        raise SystemExit(f"unsupported patch: {args.patch}")
    report = inspect_patch(args.third_party_root)
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.require_applied and not report["ok"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
