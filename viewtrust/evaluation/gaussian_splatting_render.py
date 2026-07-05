"""Helpers for rendering clean views with the official Gaussian Splatting renderer."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from viewtrust.training.dynamics import load_observed_summary
from viewtrust.utils.paths import ensure_child_path

RENDER_SUMMARY_SCHEMA = "viewtrust.view_render.summary"
RENDER_SUMMARY_VERSION = 1
SUPPORTED_SPLITS = {"train", "test", "target"}


@dataclass(frozen=True)
class ViewRenderConfig:
    run_dir: Path
    data_root: Path
    third_party_root: Path
    scene: str = "chair"
    condition: str = "clean"
    iteration: int = 500
    gpu: int = 0
    sample_interval_s: float = 1.0
    splits: tuple[str, ...] = ("train", "test", "target")
    trainer: str = "gaussian-splatting"
    dry_run: bool = False
    overwrite: bool = False


def resolve_prepared_scene_root(config: ViewRenderConfig) -> Path:
    return (
        config.data_root
        / "viewtrust-mini"
        / "nerf_synthetic"
        / config.scene
        / config.condition
    ).resolve()


def resolve_trainer_output_dir(run_dir: Path) -> Path:
    return (run_dir / "trainer_output").resolve()


def resolve_official_render_script(third_party_root: Path) -> Path:
    return (third_party_root / "gaussian-splatting" / "render.py").resolve()


def _validate_splits(splits: tuple[str, ...]) -> tuple[str, ...]:
    unknown = sorted(set(splits).difference(SUPPORTED_SPLITS))
    if unknown:
        raise ValueError(f"unsupported render splits: {unknown}")
    return tuple(dict.fromkeys(splits))


def validate_render_preflight(config: ViewRenderConfig) -> dict[str, Any]:
    run_dir = config.run_dir.resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"run_dir does not exist: {run_dir}")

    summary = load_observed_summary(run_dir)
    if summary.get("returncode") != 0:
        raise RuntimeError(
            "baseline run returncode is not 0. Refusing to render view metrics from a failed run."
        )

    if config.trainer != "gaussian-splatting":
        raise ValueError("PR6 supports only trainer=gaussian-splatting")

    splits = _validate_splits(config.splits)
    trainer_output_dir = resolve_trainer_output_dir(run_dir)
    point_cloud_path = (
        trainer_output_dir
        / "point_cloud"
        / f"iteration_{config.iteration}"
        / "point_cloud.ply"
    )
    if not point_cloud_path.is_file():
        raise FileNotFoundError(
            "point cloud not found for iteration "
            f"{config.iteration}:\n{point_cloud_path}"
        )

    prepared_scene_root = resolve_prepared_scene_root(config)
    if not prepared_scene_root.is_dir():
        raise FileNotFoundError(f"prepared scene root does not exist: {prepared_scene_root}")

    required_transforms = ["transforms_train.json"]
    if "test" in splits:
        required_transforms.append("transforms_test.json")
    if "target" in splits:
        required_transforms.append("transforms_target.json")
    missing_transforms = [
        name for name in required_transforms if not (prepared_scene_root / name).is_file()
    ]
    if "transforms_target.json" in missing_transforms:
        raise FileNotFoundError(
            "target split requested but transforms_target.json is missing from prepared scene."
        )
    if missing_transforms:
        raise FileNotFoundError(
            "prepared scene is missing required transforms: "
            + ", ".join(missing_transforms)
        )

    render_script = resolve_official_render_script(config.third_party_root)
    if not render_script.is_file():
        raise FileNotFoundError(
            "official Gaussian Splatting render.py not found:\n" f"{render_script}"
        )

    return {
        "run_dir": str(run_dir),
        "returncode": summary.get("returncode"),
        "prepared_scene_root": str(prepared_scene_root),
        "trainer_output_dir": str(trainer_output_dir),
        "point_cloud_path": str(point_cloud_path),
        "render_script": str(render_script),
        "splits": list(splits),
    }


def _reset_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"output already exists: {path}. Use --overwrite to replace it.")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _replace_symlink(target: Path, link_path: Path) -> None:
    if link_path.exists() or link_path.is_symlink():
        if link_path.is_dir() and not link_path.is_symlink():
            shutil.rmtree(link_path)
        else:
            link_path.unlink()
    os.symlink(target.resolve(), link_path)


def create_eval_model_dir(
    run_dir: Path,
    name: str,
    trainer_output_dir: Path,
    *,
    overwrite: bool = False,
) -> Path:
    run_dir = run_dir.resolve()
    model_dir = run_dir / "view_evaluation" / "render_models" / name
    ensure_child_path(run_dir, model_dir)
    _reset_dir(model_dir, overwrite)

    point_cloud_source = trainer_output_dir / "point_cloud"
    _replace_symlink(point_cloud_source, model_dir / "point_cloud")

    cfg_args_source = trainer_output_dir / "cfg_args"
    if cfg_args_source.is_file():
        shutil.copy2(cfg_args_source, model_dir / "cfg_args")

    return model_dir


def create_target_eval_scene(
    run_dir: Path,
    prepared_scene_root: Path,
    *,
    overwrite: bool = False,
) -> Path:
    run_dir = run_dir.resolve()
    prepared_scene_root = prepared_scene_root.resolve()
    scene_dir = run_dir / "view_evaluation" / "eval_scenes" / "target_as_test"
    ensure_child_path(run_dir, scene_dir)
    _reset_dir(scene_dir, overwrite)

    transforms_target = prepared_scene_root / "transforms_target.json"
    if not transforms_target.is_file():
        raise FileNotFoundError(
            "target split requested but transforms_target.json is missing from prepared scene."
        )

    shutil.copy2(prepared_scene_root / "transforms_train.json", scene_dir / "transforms_train.json")
    shutil.copy2(transforms_target, scene_dir / "transforms_test.json")
    images_source = prepared_scene_root / "images"
    if not images_source.is_dir():
        raise FileNotFoundError(f"prepared scene images directory is missing: {images_source}")
    _replace_symlink(images_source, scene_dir / "images")
    return scene_dir


def build_render_command(
    python_executable: Path,
    render_script: Path,
    source_path: Path,
    model_path: Path,
    iteration: int,
    split_mode: str,
) -> list[str]:
    command = [
        str(python_executable),
        str(render_script),
        "-s",
        str(source_path),
        "-m",
        str(model_path),
        "--iteration",
        str(iteration),
    ]
    if split_mode == "train_only":
        command.append("--skip_test")
    elif split_mode == "test_only":
        command.append("--skip_train")
    elif split_mode == "train_test":
        pass
    else:
        raise ValueError(f"unsupported split_mode: {split_mode}")
    return command


def build_render_plan(config: ViewRenderConfig) -> dict[str, Any]:
    preflight = validate_render_preflight(config)
    run_dir = config.run_dir.resolve()
    prepared_scene_root = resolve_prepared_scene_root(config)
    render_script = resolve_official_render_script(config.third_party_root)
    trainer_output_dir = resolve_trainer_output_dir(run_dir)
    splits = _validate_splits(config.splits)

    commands: list[dict[str, Any]] = []
    if "train" in splits or "test" in splits:
        if "train" in splits and "test" in splits:
            split_mode = "train_test"
        elif "train" in splits:
            split_mode = "train_only"
        else:
            split_mode = "test_only"
        commands.append(
            {
                "name": "train_test",
                "splits": [split for split in ("train", "test") if split in splits],
                "source_path": str(prepared_scene_root),
                "model_path": str(run_dir / "view_evaluation" / "render_models" / "train_test_model"),
                "split_mode": split_mode,
                "command": build_render_command(
                    Path(sys.executable),
                    render_script,
                    prepared_scene_root,
                    run_dir / "view_evaluation" / "render_models" / "train_test_model",
                    config.iteration,
                    split_mode,
                ),
            }
        )

    if "target" in splits:
        commands.append(
            {
                "name": "target",
                "splits": ["target"],
                "source_path": str(run_dir / "view_evaluation" / "eval_scenes" / "target_as_test"),
                "model_path": str(run_dir / "view_evaluation" / "render_models" / "target_model"),
                "split_mode": "test_only",
                "command": build_render_command(
                    Path(sys.executable),
                    render_script,
                    run_dir / "view_evaluation" / "eval_scenes" / "target_as_test",
                    run_dir / "view_evaluation" / "render_models" / "target_model",
                    config.iteration,
                    "test_only",
                ),
            }
        )

    return {
        "schema_name": RENDER_SUMMARY_SCHEMA,
        "schema_version": RENDER_SUMMARY_VERSION,
        "run_dir": str(run_dir),
        "run_id": run_dir.name,
        "scene": config.scene,
        "condition": config.condition,
        "iteration": config.iteration,
        "observation_only": True,
        "dry_run": config.dry_run,
        "preflight": preflight,
        "commands": commands,
        "render_logs_dir": str(run_dir / "view_evaluation" / "render_logs"),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "warnings": [],
    }


def render_clean_views(config: ViewRenderConfig) -> dict[str, Any]:
    plan = build_render_plan(config)
    if config.dry_run:
        plan["executed"] = False
        return plan

    run_dir = config.run_dir.resolve()
    prepared_scene_root = resolve_prepared_scene_root(config)
    trainer_output_dir = resolve_trainer_output_dir(run_dir)
    splits = _validate_splits(config.splits)

    if "train" in splits or "test" in splits:
        create_eval_model_dir(
            run_dir,
            "train_test_model",
            trainer_output_dir,
            overwrite=config.overwrite,
        )
    if "target" in splits:
        create_eval_model_dir(
            run_dir,
            "target_model",
            trainer_output_dir,
            overwrite=config.overwrite,
        )
        create_target_eval_scene(
            run_dir,
            prepared_scene_root,
            overwrite=config.overwrite,
        )

    logs_dir = run_dir / "view_evaluation" / "render_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(config.gpu)
    command_results: list[dict[str, Any]] = []
    for command_info in plan["commands"]:
        stdout_path = logs_dir / f"{command_info['name']}_stdout.log"
        stderr_path = logs_dir / f"{command_info['name']}_stderr.log"
        with stdout_path.open("w", encoding="utf-8") as stdout_handle, stderr_path.open(
            "w",
            encoding="utf-8",
        ) as stderr_handle:
            completed = subprocess.run(
                command_info["command"],
                check=False,
                stdout=stdout_handle,
                stderr=stderr_handle,
                text=True,
                env=env,
            )
        result = {
            "name": command_info["name"],
            "returncode": completed.returncode,
            "stdout_log": stdout_path.relative_to(run_dir).as_posix(),
            "stderr_log": stderr_path.relative_to(run_dir).as_posix(),
        }
        command_results.append(result)
        if completed.returncode != 0:
            raise RuntimeError(
                f"render command failed for {command_info['name']} with returncode "
                f"{completed.returncode}. See {stderr_path}"
            )

    plan["executed"] = True
    plan["command_results"] = command_results
    summary_path = run_dir / "view_evaluation" / "view_render_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return plan
