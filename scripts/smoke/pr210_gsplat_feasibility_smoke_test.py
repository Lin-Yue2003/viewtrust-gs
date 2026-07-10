#!/usr/bin/env python3
"""LOCAL-SAFE smoke test for PR21.0 gsplat feasibility probing."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import zlib
from pathlib import Path


REQUIRED_OUTPUTS = [
    "pr210_gsplat_feasibility_summary.json",
    "pr210_dependency_probe.json",
    "pr210_run_artifact_audit.csv",
    "pr210_ply_schema_audit.csv",
    "pr210_camera_schema_audit.csv",
    "pr210_selected_view_audit.csv",
    "pr210_gsplat_api_audit.csv",
    "pr210_checkpoint_conversion_audit.csv",
    "pr210_render_replay_audit.csv",
    "pr210_render_parity_metrics.csv",
    "pr210_blockers.csv",
    "pr210_recommendations.json",
    "pr210_report.md",
    "pr210_missing_inputs.csv",
    "artifact_manifest.csv",
]


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    body = kind + payload
    return len(payload).to_bytes(4, "big") + body + zlib.crc32(body).to_bytes(4, "big")


def _write_png(path: Path, width: int = 8, height: int = 8) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = b"".join(b"\x00" + bytes([x % 256, y % 256, 64]) * width for y in range(height) for x in [0])
    payload = (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk("IHDR".encode("ascii"), width.to_bytes(4, "big") + height.to_bytes(4, "big") + b"\x08\x02\x00\x00\x00")
        + _png_chunk("IDAT".encode("ascii"), zlib.compress(raw))
        + _png_chunk("IEND".encode("ascii"), b"")
    )
    path.write_bytes(payload)


def _write_fake_ply(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = """ply
format ascii 1.0
element vertex 2
property float x
property float y
property float z
property float nx
property float ny
property float nz
property float f_dc_0
property float f_dc_1
property float f_dc_2
property float opacity
property float scale_0
property float scale_1
property float scale_2
property float rot_0
property float rot_1
property float rot_2
property float rot_3
end_header
"""
    rows = [
        "0 0 3 0 0 0 1 0 0 0.5 -2 -2 -2 1 0 0 0",
        "1 0 3 0 0 0 0 1 0 0.4 -2 -2 -2 1 0 0 0",
    ]
    path.write_text(header + "\n".join(rows) + "\n", encoding="utf-8")


def _write_fake_run(run_dir: Path) -> None:
    _write_fake_ply(run_dir / "trainer_output" / "point_cloud" / "iteration_700" / "point_cloud.ply")
    (run_dir / "trainer_output" / "input.ply").write_text("ply\nformat ascii 1.0\nend_header\n", encoding="utf-8")
    (run_dir / "trainer_output" / "cfg_args").write_text("Namespace(iterations=700)\n", encoding="utf-8")
    (run_dir / "trainer_output" / "exposure.json").write_text("{}\n", encoding="utf-8")
    cameras = [
        {
            "id": 4,
            "img_name": "train_004",
            "width": 8,
            "height": 8,
            "fx": 10.0,
            "fy": 10.0,
            "rotation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "position": [0, 0, 0],
        },
        {
            "id": 9,
            "img_name": "train_009",
            "width": 8,
            "height": 8,
            "fx": 10.0,
            "fy": 10.0,
            "rotation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "position": [0, 0, 0],
        },
    ]
    (run_dir / "trainer_output" / "cameras.json").write_text(json.dumps(cameras, indent=2) + "\n", encoding="utf-8")
    render_root = run_dir / "view_evaluation" / "render_models" / "train_test_model" / "train" / "ours_700" / "renders"
    gt_root = run_dir / "view_evaluation" / "render_models" / "train_test_model" / "train" / "ours_700" / "gt"
    for name in ["train_004.png", "train_009.png"]:
        _write_png(render_root / name)
        _write_png(gt_root / name)
    for name in ["metadata.json", "config_snapshot.json", "stats.json", "summary.json"]:
        (run_dir / name).write_text("{}\n", encoding="utf-8")


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def main() -> int:
    project_root = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory(prefix="viewtrust-pr210-") as tmp:
        root = Path(tmp)
        run_dir = root / "fake_run"
        output_dir = root / "pr210"
        _write_fake_run(run_dir)
        result = _run(
            [
                sys.executable,
                str(project_root / "scripts" / "measure" / "probe_pr210_gsplat_feasibility.py"),
                "--run-dir",
                str(run_dir),
                "--scene",
                "chair",
                "--condition",
                "corrupt_occluder",
                "--subset-name",
                "seed_20260710",
                "--iteration",
                "700",
                "--split",
                "train",
                "--views",
                "train_004",
                "train_009",
                "--output-dir",
                str(output_dir),
                "--metadata-only",
                "--write-markdown",
            ]
        )
        if result.returncode != 0:
            print(result.stdout)
            print(result.stderr, file=sys.stderr)
            return result.returncode
        for name in REQUIRED_OUTPUTS:
            path = output_dir / name
            assert path.exists(), name
            assert path.stat().st_size > 0, name
        summary = json.loads((output_dir / "pr210_gsplat_feasibility_summary.json").read_text(encoding="utf-8"))
        assert summary["schema_name"] == "viewtrust.pr210.gsplat_feasibility.summary"
        assert summary["observation_only"] is True
        assert summary["training_intervention"] is False
        assert summary["defense_enabled"] is False
        assert summary["densification_gating_enabled"] is False
        assert summary["third_party_modified"] is False
        assert summary["exact_sparse_attribution_ready"] is False
        assert summary["official_point_cloud_found"] is True
        assert summary["official_cameras_json_found"] is True
        assert summary["official_render_root_found"] is True
        assert summary["official_gt_root_found"] is True
        assert summary["ply_vertex_count"] == 2
        assert summary["camera_count"] == 2
        assert summary["selected_view_count_requested"] == 2
        assert summary["selected_view_count_available"] == 2
        assert summary["gsplat_render_replay_supported"] is False
    print("pr210 gsplat feasibility smoke test ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
