#!/usr/bin/env python3
"""Install or dry-run dataset manifests for ViewTrust-GS."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _bootstrap_project_imports() -> Path:
    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root))
    return project_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default="configs/datasets.example.json",
        help="Dataset manifest JSON path.",
    )
    parser.add_argument(
        "--data-root",
        default=os.environ.get("VIEWTRUST_DATA_ROOT", "./data"),
        help="Dataset root. Defaults to VIEWTRUST_DATA_ROOT or ./data.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Actually download datasets. Omit for local-safe dry-run.",
    )
    parser.add_argument(
        "--extract",
        action="store_true",
        help="Extract downloaded archives after download.",
    )
    parser.add_argument(
        "--plan-json",
        default=None,
        help="Optional path to write the install plan JSON.",
    )
    return parser.parse_args()


def main() -> int:
    project_root = _bootstrap_project_imports()

    from viewtrust.datasets.installer import install_from_manifest
    from viewtrust.utils.paths import resolve_relative_path

    args = parse_args()
    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = project_root / manifest_path

    data_root = Path(args.data_root)
    if not data_root.is_absolute():
        data_root = resolve_relative_path(project_root, args.data_root)

    plans = install_from_manifest(
        manifest_path=manifest_path,
        data_root=data_root,
        download=args.download,
        extract=args.extract,
    )
    document = {
        "schema_name": "viewtrust.datasets.install_plan",
        "schema_version": 1,
        "mode": "download" if args.download else "dry_run",
        "manifest_path": str(manifest_path),
        "data_root": str(data_root),
        "datasets": [plan.as_dict() for plan in plans],
    }

    print(json.dumps(document, indent=2, sort_keys=True))

    if args.plan_json:
        plan_path = Path(args.plan_json)
        if not plan_path.is_absolute():
            plan_path = project_root / plan_path
        plan_path.parent.mkdir(parents=True, exist_ok=True)
        plan_path.write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    if args.download:
        print("dataset install ok")
    else:
        print("dataset install dry-run ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
