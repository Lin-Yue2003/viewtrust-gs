"""PR21.0 gsplat feasibility and official-checkpoint replay audit helpers."""

from __future__ import annotations

import csv
import importlib
import inspect
import json
import math
import struct
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PR210_OUTPUT_FILES = [
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

REQUIRED_PLY_PROPERTIES = [
    "x",
    "y",
    "z",
    "opacity",
    "scale_0",
    "scale_1",
    "scale_2",
    "rot_0",
    "rot_1",
    "rot_2",
    "rot_3",
    "f_dc_0",
    "f_dc_1",
    "f_dc_2",
]

RUN_ARTIFACT_FIELDS = [
    "scene",
    "condition",
    "subset_name",
    "run_dir",
    "iteration",
    "point_cloud_path",
    "point_cloud_exists",
    "cameras_json_path",
    "cameras_json_exists",
    "cfg_args_path",
    "cfg_args_exists",
    "exposure_json_exists",
    "render_root",
    "render_root_exists",
    "gt_root",
    "gt_root_exists",
    "render_file_count",
    "gt_file_count",
    "sample_render_shape",
    "sample_gt_shape",
    "notes",
]
PLY_SCHEMA_FIELDS = ["scene", "ply_path", "vertex_count", "property_name", "property_type", "required_for_gsplat", "present", "notes"]
CAMERA_SCHEMA_FIELDS = [
    "scene",
    "cameras_json_path",
    "camera_count",
    "sample_camera_keys",
    "has_img_name",
    "has_width_height",
    "has_fx_fy_or_fov",
    "has_rotation_translation_or_transform",
    "image_name_examples",
    "schema_supported",
    "blockers",
]
SELECTED_VIEW_FIELDS = [
    "scene",
    "split",
    "requested_view_name",
    "found_in_cameras_json",
    "matched_camera_id",
    "matched_camera_img_name",
    "official_render_path",
    "official_render_exists",
    "official_gt_path",
    "official_gt_exists",
    "image_index",
    "notes",
]
GSPLAT_API_FIELDS = ["api_name", "available", "module_path", "callable", "signature_if_available", "notes"]
CONVERSION_FIELDS = ["scene", "ply_path", "conversion_step", "supported", "tensor_shape", "dtype", "device", "notes"]
RENDER_REPLAY_FIELDS = ["scene", "view_name", "replay_attempted", "replay_supported", "official_render_path", "gsplat_render_path", "shape_match", "blocker", "notes"]
RENDER_PARITY_FIELDS = ["scene", "view_name", "mean_l1", "max_l1", "psnr", "official_render_path", "gsplat_render_path"]
BLOCKER_FIELDS = ["severity", "component", "blocker", "evidence", "recommended_action"]
MISSING_FIELDS = ["input_name", "path", "exists", "required", "details"]
MANIFEST_FIELDS = ["relative_path", "path", "exists", "file_type", "size_bytes", "required", "artifact_group"]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _bool_text(value: Any) -> str:
    return "true" if bool(value) else "false"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _safe_import(name: str) -> tuple[Any | None, str | None]:
    try:
        return importlib.import_module(name), None
    except Exception as exc:
        return None, str(exc)


def probe_dependencies() -> dict[str, Any]:
    probe: dict[str, Any] = {
        "sys_executable": sys.executable,
        "python_version": sys.version,
        "torch_import_ok": False,
        "torch_version": None,
        "torch_cuda_version": None,
        "cuda_available": False,
        "gpu_count": 0,
        "gpu_names": [],
        "gsplat_import_ok": False,
        "gsplat_version": None,
        "gsplat_module_path": None,
        "gsplat_import_error": None,
    }
    torch, torch_error = _safe_import("torch")
    if torch is None:
        probe["torch_import_error"] = torch_error
    else:
        probe["torch_import_ok"] = True
        probe["torch_version"] = getattr(torch, "__version__", None)
        probe["torch_cuda_version"] = getattr(getattr(torch, "version", None), "cuda", None)
        try:
            cuda_available = bool(torch.cuda.is_available())
            probe["cuda_available"] = cuda_available
            probe["gpu_count"] = int(torch.cuda.device_count()) if cuda_available else 0
            probe["gpu_names"] = [torch.cuda.get_device_name(index) for index in range(int(probe["gpu_count"]))]
        except Exception as exc:
            probe["cuda_probe_error"] = str(exc)
    gsplat, gsplat_error = _safe_import("gsplat")
    if gsplat is None:
        probe["gsplat_import_error"] = gsplat_error
    else:
        probe["gsplat_import_ok"] = True
        probe["gsplat_version"] = getattr(gsplat, "__version__", None)
        probe["gsplat_module_path"] = getattr(gsplat, "__file__", None)
    return probe


def _api_row(api_name: str, obj: Any | None, module_path: str = "", notes: str = "") -> dict[str, Any]:
    signature = ""
    if obj is not None and callable(obj):
        try:
            signature = str(inspect.signature(obj))
        except Exception as exc:
            signature = f"<signature unavailable: {exc}>"
    return {
        "api_name": api_name,
        "available": _bool_text(obj is not None),
        "module_path": module_path,
        "callable": _bool_text(callable(obj)),
        "signature_if_available": signature,
        "notes": notes,
    }


def audit_gsplat_api() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    gsplat, error = _safe_import("gsplat")
    if gsplat is None:
        for name in [
            "gsplat",
            "gsplat.rendering.rasterization",
            "gsplat.cuda",
            "gsplat.cuda._wrapper",
            "rasterize_to_indices_in_range",
            "accumulate",
        ]:
            rows.append(_api_row(name, None, notes=f"gsplat import failed: {error}"))
        return rows

    rows.append(_api_row("gsplat", gsplat, getattr(gsplat, "__file__", ""), "root module imported"))
    rendering, rendering_error = _safe_import("gsplat.rendering")
    rasterization = getattr(rendering, "rasterization", None) if rendering is not None else getattr(gsplat, "rasterization", None)
    rows.append(_api_row("gsplat.rendering.rasterization", rasterization, getattr(rendering, "__file__", ""), rendering_error or ""))

    for module_name in ["gsplat.cuda", "gsplat.cuda._wrapper"]:
        module, module_error = _safe_import(module_name)
        rows.append(_api_row(module_name, module, getattr(module, "__file__", "") if module else "", module_error or ""))

    searchable_modules = [module for module in [gsplat, rendering, _safe_import("gsplat.cuda")[0], _safe_import("gsplat.cuda._wrapper")[0]] if module is not None]
    for api_name in ["rasterize_to_indices_in_range", "accumulate"]:
        found_obj = None
        found_path = ""
        for module in searchable_modules:
            if hasattr(module, api_name):
                found_obj = getattr(module, api_name)
                found_path = getattr(module, "__file__", "")
                break
        rows.append(_api_row(api_name, found_obj, found_path, "public API probe"))
    return rows


def _image_shape(path: Path) -> str:
    if not path.is_file():
        return ""
    try:
        with path.open("rb") as handle:
            header = handle.read(32)
        if header.startswith(b"\x89PNG\r\n\x1a\n") and len(header) >= 24:
            width, height = struct.unpack(">II", header[16:24])
            return f"{width}x{height}"
        if header[:2] == b"\xff\xd8":
            with path.open("rb") as handle:
                handle.read(2)
                while True:
                    marker_start = handle.read(1)
                    if not marker_start:
                        break
                    if marker_start != b"\xff":
                        continue
                    marker = handle.read(1)
                    if marker in {b"\xc0", b"\xc2"}:
                        block = handle.read(7)
                        height, width = struct.unpack(">HH", block[3:7])
                        return f"{width}x{height}"
                    length_bytes = handle.read(2)
                    if len(length_bytes) != 2:
                        break
                    length = struct.unpack(">H", length_bytes)[0]
                    handle.seek(length - 2, 1)
    except Exception:
        return ""
    return ""


def _image_files(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    suffixes = {".png", ".jpg", ".jpeg"}
    return sorted(path for path in root.iterdir() if path.is_file() and path.suffix.lower() in suffixes)


def audit_run_artifacts(run_dir: Path, iteration: int, split: str, scene: str, condition: str, subset_name: str) -> dict[str, Any]:
    trainer_output = run_dir / "trainer_output"
    point_cloud = trainer_output / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"
    cameras = trainer_output / "cameras.json"
    cfg_args = trainer_output / "cfg_args"
    exposure = trainer_output / "exposure.json"
    render_root = run_dir / "view_evaluation" / "render_models" / "train_test_model" / split / f"ours_{iteration}" / "renders"
    gt_root = run_dir / "view_evaluation" / "render_models" / "train_test_model" / split / f"ours_{iteration}" / "gt"
    render_files = _image_files(render_root)
    gt_files = _image_files(gt_root)
    notes = []
    if not point_cloud.exists():
        notes.append("missing point cloud")
    if not cameras.exists():
        notes.append("missing cameras.json")
    return {
        "scene": scene,
        "condition": condition,
        "subset_name": subset_name,
        "run_dir": str(run_dir),
        "iteration": iteration,
        "point_cloud_path": str(point_cloud),
        "point_cloud_exists": _bool_text(point_cloud.exists()),
        "cameras_json_path": str(cameras),
        "cameras_json_exists": _bool_text(cameras.exists()),
        "cfg_args_path": str(cfg_args),
        "cfg_args_exists": _bool_text(cfg_args.exists()),
        "exposure_json_exists": _bool_text(exposure.exists()),
        "render_root": str(render_root),
        "render_root_exists": _bool_text(render_root.exists()),
        "gt_root": str(gt_root),
        "gt_root_exists": _bool_text(gt_root.exists()),
        "render_file_count": len(render_files),
        "gt_file_count": len(gt_files),
        "sample_render_shape": _image_shape(render_files[0]) if render_files else "",
        "sample_gt_shape": _image_shape(gt_files[0]) if gt_files else "",
        "notes": "; ".join(notes),
        "_point_cloud_path": point_cloud,
        "_cameras_json_path": cameras,
        "_render_root": render_root,
        "_gt_root": gt_root,
    }


def parse_ply_header(ply_path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"path": str(ply_path), "exists": ply_path.exists(), "vertex_count": None, "format": "", "properties": [], "header_lines": []}
    if not ply_path.is_file():
        return result
    in_vertex = False
    try:
        with ply_path.open("rb") as handle:
            for _ in range(10000):
                raw = handle.readline()
                if not raw:
                    break
                line = raw.decode("utf-8", errors="replace").strip()
                result["header_lines"].append(line)
                if line.startswith("format "):
                    result["format"] = line.split(maxsplit=2)[1] if len(line.split()) >= 2 else ""
                if line.startswith("element "):
                    parts = line.split()
                    in_vertex = len(parts) >= 3 and parts[1] == "vertex"
                    if in_vertex:
                        try:
                            result["vertex_count"] = int(parts[2])
                        except ValueError:
                            result["vertex_count"] = None
                    continue
                if in_vertex and line.startswith("property "):
                    parts = line.split()
                    if len(parts) >= 3:
                        result["properties"].append({"property_type": parts[1], "property_name": parts[-1]})
                if line == "end_header":
                    break
    except Exception as exc:
        result["error"] = str(exc)
    return result


def audit_ply_schema(ply_path: Path, scene: str) -> tuple[list[dict[str, Any]], bool, int | None]:
    header = parse_ply_header(ply_path)
    properties = {str(item["property_name"]): str(item["property_type"]) for item in header.get("properties", [])}
    vertex_count = header.get("vertex_count")
    rows: list[dict[str, Any]] = []
    for name in REQUIRED_PLY_PROPERTIES:
        present = name in properties
        rows.append(
            {
                "scene": scene,
                "ply_path": str(ply_path),
                "vertex_count": vertex_count if vertex_count is not None else "",
                "property_name": name,
                "property_type": properties.get(name, ""),
                "required_for_gsplat": "true",
                "present": _bool_text(present),
                "notes": "official 3DGS parameter field; activation/conversion required" if present else "missing required field",
            }
        )
    f_rest = sorted(name for name in properties if name.startswith("f_rest_"))
    rows.append(
        {
            "scene": scene,
            "ply_path": str(ply_path),
            "vertex_count": vertex_count if vertex_count is not None else "",
            "property_name": "f_rest_*",
            "property_type": "mixed" if f_rest else "",
            "required_for_gsplat": "false",
            "present": _bool_text(bool(f_rest)),
            "notes": f"{len(f_rest)} SH rest fields present" if f_rest else "SH rest fields absent; DC color-only conversion may still be possible",
        }
    )
    supported = bool(vertex_count) and all(name in properties for name in REQUIRED_PLY_PROPERTIES)
    return rows, supported, int(vertex_count) if isinstance(vertex_count, int) else None


def load_cameras_json(cameras_json_path: Path) -> Any:
    if not cameras_json_path.is_file():
        return []
    with cameras_json_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _camera_list(cameras: Any) -> list[dict[str, Any]]:
    if isinstance(cameras, list):
        return [item for item in cameras if isinstance(item, dict)]
    if isinstance(cameras, dict):
        for key in ["cameras", "frames"]:
            value = cameras.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    return []


def _camera_name(camera: dict[str, Any]) -> str:
    for key in ["img_name", "image_name", "file_path", "path", "name"]:
        value = camera.get(key)
        if value not in (None, ""):
            return str(value)
    return ""


def _has_any(camera: dict[str, Any], keys: list[str]) -> bool:
    return any(key in camera and camera.get(key) not in (None, "") for key in keys)


def audit_camera_schema(cameras_json_path: Path, scene: str) -> tuple[list[dict[str, Any]], bool, int]:
    cameras = _camera_list(load_cameras_json(cameras_json_path))
    sample = cameras[0] if cameras else {}
    has_img_name = any(_camera_name(camera) for camera in cameras)
    has_width_height = any(_has_any(camera, ["width", "w"]) and _has_any(camera, ["height", "h"]) for camera in cameras)
    has_intrinsics = any(_has_any(camera, ["fx", "fl_x", "FovX", "fov_x", "camera_angle_x"]) for camera in cameras)
    has_pose = any(
        _has_any(camera, ["rotation", "R", "transform_matrix"]) and _has_any(camera, ["position", "translation", "T", "transform_matrix"])
        for camera in cameras
    )
    blockers = []
    if not cameras:
        blockers.append("no cameras")
    if not has_img_name:
        blockers.append("no image names")
    if not has_width_height:
        blockers.append("no width/height")
    if not has_intrinsics:
        blockers.append("no fx/fy/fov intrinsics")
    if not has_pose:
        blockers.append("no rotation/translation/transform")
    supported = not blockers
    rows = [
        {
            "scene": scene,
            "cameras_json_path": str(cameras_json_path),
            "camera_count": len(cameras),
            "sample_camera_keys": ";".join(sorted(str(key) for key in sample.keys())),
            "has_img_name": _bool_text(has_img_name),
            "has_width_height": _bool_text(has_width_height),
            "has_fx_fy_or_fov": _bool_text(has_intrinsics),
            "has_rotation_translation_or_transform": _bool_text(has_pose),
            "image_name_examples": ";".join(_camera_name(camera) for camera in cameras[:5]),
            "schema_supported": _bool_text(supported),
            "blockers": "; ".join(blockers),
        }
    ]
    return rows, supported, len(cameras)


def _view_index(view_name: str) -> int | None:
    digits = "".join(ch for ch in view_name if ch.isdigit())
    if digits == "":
        return None
    try:
        return int(digits)
    except ValueError:
        return None


def _norm_name(value: str) -> str:
    stem = Path(value).stem.lower()
    return stem.replace("r_", "").replace("-", "_")


def _name_matches(requested: str, candidate: str) -> bool:
    if not candidate:
        return False
    requested_norm = _norm_name(requested)
    candidate_norm = _norm_name(candidate)
    if requested_norm == candidate_norm or requested_norm in candidate_norm or candidate_norm in requested_norm:
        return True
    requested_index = _view_index(requested)
    candidate_index = _view_index(candidate_norm)
    return requested_index is not None and candidate_index == requested_index


def _find_view_file(root: Path, requested: str) -> Path | None:
    files = _image_files(root)
    requested_index = _view_index(requested)
    expected = {_norm_name(requested)}
    if requested_index is not None:
        expected.update({f"{requested_index:05d}", f"{requested_index:04d}", f"{requested_index:03d}", str(requested_index)})
    for path in files:
        if _norm_name(path.name) in expected or _name_matches(requested, path.name):
            return path
    return None


def match_selected_views(
    cameras_json: Any,
    render_root: Path,
    gt_root: Path,
    requested_views: list[str],
    scene: str,
    split: str,
) -> list[dict[str, Any]]:
    cameras = _camera_list(cameras_json)
    rows: list[dict[str, Any]] = []
    for requested in requested_views:
        matched_camera: dict[str, Any] | None = None
        for camera in cameras:
            if _name_matches(requested, _camera_name(camera)):
                matched_camera = camera
                break
        render_path = _find_view_file(render_root, requested)
        gt_path = _find_view_file(gt_root, requested)
        camera_name = _camera_name(matched_camera or {})
        rows.append(
            {
                "scene": scene,
                "split": split,
                "requested_view_name": requested,
                "found_in_cameras_json": _bool_text(matched_camera is not None),
                "matched_camera_id": (matched_camera or {}).get("id", (matched_camera or {}).get("uid", "")),
                "matched_camera_img_name": camera_name,
                "official_render_path": str(render_path or ""),
                "official_render_exists": _bool_text(render_path is not None and render_path.exists()),
                "official_gt_path": str(gt_path or ""),
                "official_gt_exists": _bool_text(gt_path is not None and gt_path.exists()),
                "image_index": _view_index(requested) if _view_index(requested) is not None else "",
                "notes": "" if matched_camera is not None else "requested view not matched in cameras.json",
            }
        )
    return rows


def audit_checkpoint_conversion(ply_path: Path, cameras_json_path: Path, scene: str, device: str) -> tuple[list[dict[str, Any]], bool]:
    header = parse_ply_header(ply_path)
    vertex_count = header.get("vertex_count") or ""
    properties = {str(item["property_name"]) for item in header.get("properties", [])}
    camera_rows, camera_supported, _ = audit_camera_schema(cameras_json_path, scene)
    steps = [
        ("positions", {"x", "y", "z"}, f"({vertex_count}, 3)"),
        ("opacities", {"opacity"}, f"({vertex_count},)"),
        ("scales", {"scale_0", "scale_1", "scale_2"}, f"({vertex_count}, 3)"),
        ("rotations", {"rot_0", "rot_1", "rot_2", "rot_3"}, f"({vertex_count}, 4)"),
        ("colors / f_dc", {"f_dc_0", "f_dc_1", "f_dc_2"}, f"({vertex_count}, 3)"),
        ("SH rest", set(), f"({vertex_count}, N)"),
    ]
    rows: list[dict[str, Any]] = []
    all_required_supported = True
    for step, required, shape in steps:
        if step == "SH rest":
            supported = any(name.startswith("f_rest_") for name in properties)
            notes = "optional SH rest fields present" if supported else "optional SH rest fields absent; DC-only probe still possible"
        else:
            supported = bool(vertex_count) and required.issubset(properties)
            notes = "PLY fields available for tensor conversion" if supported else f"missing fields: {sorted(required - properties)}"
            all_required_supported = all_required_supported and supported
        rows.append(
            {
                "scene": scene,
                "ply_path": str(ply_path),
                "conversion_step": step,
                "supported": _bool_text(supported),
                "tensor_shape": shape if supported else "",
                "dtype": "float32" if supported else "",
                "device": device if supported else "",
                "notes": notes,
            }
        )
    rows.append(
        {
            "scene": scene,
            "ply_path": str(ply_path),
            "conversion_step": "camera intrinsics",
            "supported": camera_rows[0]["has_fx_fy_or_fov"] if camera_rows else "false",
            "tensor_shape": "(1, 3, 3)" if camera_supported else "",
            "dtype": "float32" if camera_supported else "",
            "device": device if camera_supported else "",
            "notes": "camera intrinsics appear convertible" if camera_supported else "camera schema blocker",
        }
    )
    rows.append(
        {
            "scene": scene,
            "ply_path": str(ply_path),
            "conversion_step": "camera extrinsics",
            "supported": camera_rows[0]["has_rotation_translation_or_transform"] if camera_rows else "false",
            "tensor_shape": "(1, 4, 4)" if camera_supported else "",
            "dtype": "float32" if camera_supported else "",
            "device": device if camera_supported else "",
            "notes": "camera extrinsics appear convertible" if camera_supported else "camera schema blocker",
        }
    )
    return rows, all_required_supported and camera_supported


def maybe_attempt_gsplat_replay(
    *,
    scene: str,
    selected_rows: list[dict[str, Any]],
    skip_render: bool,
    metadata_only: bool,
    compare_official_renders: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], bool, bool | None, float | None, float | None]:
    del compare_official_renders
    replay_rows: list[dict[str, Any]] = []
    parity_rows: list[dict[str, Any]] = []
    blocker = "render replay is deferred to PR21.1; PR21.0 audits compatibility only"
    if skip_render:
        blocker = "render replay skipped by --skip-render"
    if metadata_only:
        blocker = "metadata-only mode requested"
    for row in selected_rows:
        replay_rows.append(
            {
                "scene": scene,
                "view_name": row.get("requested_view_name", ""),
                "replay_attempted": "false",
                "replay_supported": "false",
                "official_render_path": row.get("official_render_path", ""),
                "gsplat_render_path": "",
                "shape_match": "",
                "blocker": blocker,
                "notes": "no fake render parity emitted",
            }
        )
    parity_rows.append(
        {
            "scene": scene,
            "view_name": "",
            "mean_l1": "",
            "max_l1": "",
            "psnr": "",
            "official_render_path": "",
            "gsplat_render_path": "",
        }
    )
    return replay_rows, parity_rows, False, None, None, None


def _metadata_api_supported(api_rows: list[dict[str, Any]]) -> bool:
    target_names = {"rasterize_to_indices_in_range", "accumulate"}
    return any(row.get("api_name") in target_names and row.get("available") == "true" for row in api_rows)


def _api_availability_map(api_rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for row in api_rows:
        key = "api_available_" + str(row.get("api_name", "")).replace(".", "_").replace("/", "_").replace(" ", "_")
        if key != "api_available_":
            result[key] = row.get("available") == "true"
    return result


def _blocker_rows(
    *,
    dependency: dict[str, Any],
    artifact_row: dict[str, Any],
    ply_schema_supported: bool,
    camera_schema_supported: bool,
    conversion_supported: bool,
    selected_rows: list[dict[str, Any]],
    metadata_api_supported: bool,
    render_replay_supported: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not dependency.get("gsplat_import_ok"):
        rows.append(
            {
                "severity": "warning",
                "component": "dependency",
                "blocker": "gsplat unavailable",
                "evidence": str(dependency.get("gsplat_import_error") or ""),
                "recommended_action": "run on server environment with installed gsplat",
            }
        )
    if not dependency.get("cuda_available"):
        rows.append(
            {
                "severity": "warning",
                "component": "dependency",
                "blocker": "CUDA unavailable",
                "evidence": "torch.cuda.is_available() is false or torch missing",
                "recommended_action": "run server validation after activation",
            }
        )
    for key, name in [
        ("point_cloud_exists", "point cloud"),
        ("cameras_json_exists", "cameras.json"),
        ("render_root_exists", "official render root"),
        ("gt_root_exists", "official GT root"),
    ]:
        if artifact_row.get(key) != "true":
            rows.append(
                {
                    "severity": "error",
                    "component": "run_artifact",
                    "blocker": f"missing {name}",
                    "evidence": str(artifact_row.get(key.replace("_exists", "_path"), artifact_row.get(key, ""))),
                    "recommended_action": "check run directory and iteration",
                }
            )
    if not ply_schema_supported:
        rows.append(
            {
                "severity": "error",
                "component": "ply_schema",
                "blocker": "PLY schema lacks required official 3DGS fields",
                "evidence": "see pr210_ply_schema_audit.csv",
                "recommended_action": "inspect checkpoint export format before gsplat conversion",
            }
        )
    if not camera_schema_supported:
        rows.append(
            {
                "severity": "error",
                "component": "camera_schema",
                "blocker": "camera schema is not fully supported",
                "evidence": "see pr210_camera_schema_audit.csv",
                "recommended_action": "add a safe camera conversion adapter",
            }
        )
    if not conversion_supported:
        rows.append(
            {
                "severity": "error",
                "component": "checkpoint_conversion",
                "blocker": "checkpoint conversion is not fully supported",
                "evidence": "see pr210_checkpoint_conversion_audit.csv",
                "recommended_action": "fix PLY/camera conversion before exact attribution",
            }
        )
    missing_views = [row.get("requested_view_name", "") for row in selected_rows if row.get("found_in_cameras_json") != "true"]
    if missing_views:
        rows.append(
            {
                "severity": "error",
                "component": "selected_views",
                "blocker": "selected views not matched in cameras.json",
                "evidence": ";".join(str(view) for view in missing_views),
                "recommended_action": "verify view naming and camera matching",
            }
        )
    if not metadata_api_supported:
        rows.append(
            {
                "severity": "warning",
                "component": "gsplat_api",
                "blocker": "public contributor metadata APIs not confirmed",
                "evidence": "rasterize_to_indices_in_range/accumulate not found in public probe",
                "recommended_action": "inspect installed gsplat API on server before PR21.1",
            }
        )
    if not render_replay_supported:
        rows.append(
            {
                "severity": "info",
                "component": "render_replay",
                "blocker": "render replay not implemented in PR21.0",
                "evidence": "compatibility audit only",
                "recommended_action": "implement sparse replay in PR21.1 only after metadata API path is confirmed",
            }
        )
    return rows


def _recommendations(summary: dict[str, Any]) -> dict[str, Any]:
    proceed = bool(summary.get("pr21_ready_for_exact_attribution"))
    return {
        "recommended_next_step": (
            "Proceed to PR21.1 exact sparse pixel-to-Gaussian attribution replay using installed gsplat."
            if proceed
            else "Fix checkpoint/camera conversion or gsplat metadata API blockers before exact attribution."
        ),
        "should_proceed_to_pr21_1_exact_sparse_attribution": proceed,
        "should_clone_gsplat_to_third_party": False,
        "should_modify_official_rasterizer_now": False,
        "safe_next_actions": [
            "run PR21.0 on the server with installed gsplat",
            "inspect gsplat API audit and checkpoint conversion audit",
            "design PR21.1 sparse replay only after metadata path is confirmed",
        ],
        "unsafe_next_actions": [
            "claim exact attribution from PR20 proxy rows",
            "modify official training or rasterizer behavior",
            "use PR21.0 outputs for view rejection or densification gating",
            "vendor gsplat source into this repository",
        ],
    }


def _write_report(path: Path, summary: dict[str, Any], blockers: list[dict[str, Any]]) -> None:
    lines = [
        "# PR21.0 gsplat Feasibility",
        "",
        "PR21.0 is observation-only. It does not implement exact attribution, defense, view rejection, update suppression, or densification gating.",
        "",
        "## Summary",
        f"- Scene: `{summary.get('scene')}`",
        f"- Condition: `{summary.get('condition')}`",
        f"- gsplat available: `{summary.get('gsplat_available')}`",
        f"- PLY schema supported: `{summary.get('ply_schema_supported')}`",
        f"- Camera schema supported: `{summary.get('camera_schema_supported')}`",
        f"- Checkpoint conversion supported: `{summary.get('checkpoint_conversion_supported')}`",
        f"- gsplat render replay supported: `{summary.get('gsplat_render_replay_supported')}`",
        f"- Exact sparse attribution ready: `{summary.get('exact_sparse_attribution_ready')}`",
        f"- Recommended next step: `{summary.get('recommended_next_step')}`",
        "",
        "## Dependency probe",
        f"- Python: `{summary.get('torch_version')}` torch, CUDA `{summary.get('torch_cuda_version')}`",
        f"- CUDA available: `{summary.get('cuda_available')}`",
        f"- GPU count: `{summary.get('gpu_count')}`",
        f"- gsplat module path: `{summary.get('gsplat_module_path')}`",
        "",
        "## Run artifact audit",
        f"- Point cloud found: `{summary.get('official_point_cloud_found')}`",
        f"- Cameras JSON found: `{summary.get('official_cameras_json_found')}`",
        f"- Official render root found: `{summary.get('official_render_root_found')}`",
        f"- Official GT root found: `{summary.get('official_gt_root_found')}`",
        "",
        "## PLY schema audit",
        f"- Vertex count: `{summary.get('ply_vertex_count')}`",
        "",
        "## Camera schema audit",
        f"- Camera count: `{summary.get('camera_count')}`",
        f"- Selected views available: `{summary.get('selected_view_count_available')}` / `{summary.get('selected_view_count_requested')}`",
        "",
        "## gsplat API audit",
        f"- Metadata probe supported: `{summary.get('gsplat_metadata_probe_supported')}`",
        "",
        "## Conversion feasibility",
        f"- Checkpoint conversion supported: `{summary.get('checkpoint_conversion_supported')}`",
        "",
        "## Render replay feasibility",
        "Render replay is not implemented in PR21.0. This PR writes explicit blockers instead of fake parity metrics.",
        "",
        "## Blockers",
    ]
    if blockers:
        for row in blockers:
            lines.append(f"- `{row.get('severity')}` `{row.get('component')}`: {row.get('blocker')} ({row.get('recommended_action')})")
    else:
        lines.append("- No blockers recorded by the audit.")
    lines.extend(["", "## Recommendation", str(summary.get("recommended_next_step", "")), ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _artifact_rows(items: list[tuple[str, Path, bool, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for relative, path, required, group in items:
        rows.append(
            {
                "relative_path": relative,
                "path": str(path),
                "exists": _bool_text(path.exists()),
                "file_type": "directory" if path.is_dir() else path.suffix.lstrip("."),
                "size_bytes": path.stat().st_size if path.is_file() else "",
                "required": _bool_text(required),
                "artifact_group": group,
            }
        )
    return rows


def write_artifact_manifest(output_dir: Path, run_dir: Path) -> None:
    items = [("run_dir", run_dir, True, "input")]
    items.extend((name, output_dir / name, True, "output_pr210") for name in PR210_OUTPUT_FILES)
    manifest = output_dir / "artifact_manifest.csv"
    _write_csv(manifest, _artifact_rows(items), MANIFEST_FIELDS)
    _write_csv(manifest, _artifact_rows(items), MANIFEST_FIELDS)


def run_gsplat_feasibility_probe(
    *,
    run_dir: Path,
    scene: str,
    condition: str,
    subset_name: str,
    iteration: int,
    split: str,
    views: list[str],
    output_dir: Path,
    device: str = "cuda:0",
    max_views: int = 6,
    allow_missing: bool = False,
    strict: bool = False,
    compare_official_renders: bool = False,
    metadata_only: bool = False,
    skip_render: bool = False,
    image_width: int | None = None,
    image_height: int | None = None,
    write_markdown: bool = False,
) -> tuple[dict[str, Any], int]:
    del image_width, image_height, write_markdown
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_views = views[:max_views] if max_views > 0 else views
    dependency = probe_dependencies()
    api_rows = audit_gsplat_api()
    artifact = audit_run_artifacts(run_dir, iteration, split, scene, condition, subset_name)
    point_cloud = artifact["_point_cloud_path"]
    cameras_json_path = artifact["_cameras_json_path"]
    render_root = artifact["_render_root"]
    gt_root = artifact["_gt_root"]
    missing_rows: list[dict[str, Any]] = []
    for input_name, path, required in [
        ("point_cloud", point_cloud, True),
        ("cameras_json", cameras_json_path, True),
        ("render_root", render_root, True),
        ("gt_root", gt_root, True),
    ]:
        if not path.exists():
            missing_rows.append({"input_name": input_name, "path": str(path), "exists": "false", "required": _bool_text(required), "details": "missing PR21.0 input"})

    ply_rows, ply_supported, vertex_count = audit_ply_schema(point_cloud, scene)
    camera_rows, camera_supported, camera_count = audit_camera_schema(cameras_json_path, scene)
    cameras_json = load_cameras_json(cameras_json_path) if cameras_json_path.is_file() else []
    selected_rows = match_selected_views(cameras_json, render_root, gt_root, selected_views, scene, split)
    conversion_rows, conversion_supported = audit_checkpoint_conversion(point_cloud, cameras_json_path, scene, device)
    replay_rows, parity_rows, replay_supported, replay_passed, mean_l1, max_l1 = maybe_attempt_gsplat_replay(
        scene=scene,
        selected_rows=selected_rows,
        skip_render=skip_render,
        metadata_only=metadata_only,
        compare_official_renders=compare_official_renders,
    )
    metadata_supported = _metadata_api_supported(api_rows)
    blockers = _blocker_rows(
        dependency=dependency,
        artifact_row=artifact,
        ply_schema_supported=ply_supported,
        camera_schema_supported=camera_supported,
        conversion_supported=conversion_supported,
        selected_rows=selected_rows,
        metadata_api_supported=metadata_supported,
        render_replay_supported=replay_supported,
    )
    selected_available = sum(1 for row in selected_rows if row.get("found_in_cameras_json") == "true")
    pr21_ready = bool(dependency.get("gsplat_import_ok") and ply_supported and camera_supported and conversion_supported and metadata_supported)
    summary = {
        "schema_name": "viewtrust.pr210.gsplat_feasibility.summary",
        "schema_version": 1,
        "created_at_utc": _utc_now(),
        "observation_only": True,
        "training_intervention": False,
        "defense_enabled": False,
        "view_rejection_enabled": False,
        "densification_gating_enabled": False,
        "training_behavior_modified": False,
        "rendering_behavior_modified_for_training": False,
        "third_party_modified": False,
        "scene": scene,
        "condition": condition,
        "subset_name": subset_name,
        "run_dir": str(run_dir),
        "iteration": iteration,
        "split": split,
        "gsplat_available": bool(dependency.get("gsplat_import_ok")),
        "gsplat_version": dependency.get("gsplat_version"),
        "gsplat_module_path": dependency.get("gsplat_module_path"),
        "torch_version": dependency.get("torch_version"),
        "torch_cuda_version": dependency.get("torch_cuda_version"),
        "cuda_available": bool(dependency.get("cuda_available")),
        "gpu_count": dependency.get("gpu_count", 0),
        "official_point_cloud_found": artifact.get("point_cloud_exists") == "true",
        "official_cameras_json_found": artifact.get("cameras_json_exists") == "true",
        "official_render_root_found": artifact.get("render_root_exists") == "true",
        "official_gt_root_found": artifact.get("gt_root_exists") == "true",
        "ply_vertex_count": vertex_count,
        "camera_count": camera_count,
        "selected_view_count_requested": len(selected_views),
        "selected_view_count_available": selected_available,
        "ply_schema_supported": ply_supported,
        "camera_schema_supported": camera_supported,
        "checkpoint_conversion_supported": conversion_supported,
        "gsplat_render_replay_supported": replay_supported,
        "gsplat_metadata_probe_supported": metadata_supported,
        "render_parity_attempted": False,
        "render_parity_passed": replay_passed,
        "mean_l1_to_official_render": mean_l1,
        "max_l1_to_official_render": max_l1,
        "exact_sparse_attribution_ready": False,
        "pr21_ready_for_exact_attribution": pr21_ready,
        "recommended_next_step": (
            "Proceed to PR21.1 exact sparse pixel-to-Gaussian attribution replay using installed gsplat."
            if pr21_ready
            else "Fix checkpoint/camera conversion or gsplat metadata API blockers before exact attribution."
        ),
        "blocker_count": len(blockers),
        "warnings": [row["blocker"] for row in blockers if row.get("severity") in {"warning", "error"}],
    }
    recommendations = _recommendations(summary)

    artifact_row = {key: value for key, value in artifact.items() if not key.startswith("_")}
    _write_json(output_dir / "pr210_gsplat_feasibility_summary.json", summary)
    _write_json(output_dir / "pr210_dependency_probe.json", dependency | {"api_probe_count": len(api_rows)} | _api_availability_map(api_rows))
    _write_csv(output_dir / "pr210_run_artifact_audit.csv", [artifact_row], RUN_ARTIFACT_FIELDS)
    _write_csv(output_dir / "pr210_ply_schema_audit.csv", ply_rows, PLY_SCHEMA_FIELDS)
    _write_csv(output_dir / "pr210_camera_schema_audit.csv", camera_rows, CAMERA_SCHEMA_FIELDS)
    _write_csv(output_dir / "pr210_selected_view_audit.csv", selected_rows, SELECTED_VIEW_FIELDS)
    _write_csv(output_dir / "pr210_gsplat_api_audit.csv", api_rows, GSPLAT_API_FIELDS)
    _write_csv(output_dir / "pr210_checkpoint_conversion_audit.csv", conversion_rows, CONVERSION_FIELDS)
    _write_csv(output_dir / "pr210_render_replay_audit.csv", replay_rows, RENDER_REPLAY_FIELDS)
    _write_csv(output_dir / "pr210_render_parity_metrics.csv", parity_rows, RENDER_PARITY_FIELDS)
    _write_csv(output_dir / "pr210_blockers.csv", blockers, BLOCKER_FIELDS)
    _write_json(output_dir / "pr210_recommendations.json", recommendations)
    _write_csv(output_dir / "pr210_missing_inputs.csv", missing_rows, MISSING_FIELDS)
    _write_report(output_dir / "pr210_report.md", summary, blockers)
    write_artifact_manifest(output_dir, run_dir)

    missing_required = [name for name in PR210_OUTPUT_FILES if not (output_dir / name).is_file()]
    if missing_required:
        raise RuntimeError(f"missing PR21.0 outputs: {missing_required}")
    if strict and (missing_rows or not dependency.get("gsplat_import_ok") or not conversion_supported):
        return summary, 1
    if missing_rows and not allow_missing:
        return summary, 1
    return summary, 0
