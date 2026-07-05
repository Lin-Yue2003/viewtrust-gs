#!/usr/bin/env python3
"""Apply the PR7 observation-only patch to a local Gaussian Splatting clone."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

PATCH_NAME = "pr7_training_events"
START = "# VIEWTRUST PR7 OBSERVATION START"
END = "# VIEWTRUST PR7 OBSERVATION END"


HELPER_SNIPPET = f'''
{START}
def _viewtrust_pr7_scalar(value):
    if value is None:
        return ""
    try:
        if hasattr(value, "detach"):
            return float(value.detach().item())
        return float(value)
    except Exception:
        return ""


def _viewtrust_pr7_count_gaussians(gaussians):
    try:
        return int(gaussians.get_xyz.shape[0])
    except Exception:
        return ""


def _viewtrust_pr7_sum(value):
    try:
        return int(value.detach().sum().item())
    except Exception:
        return ""


def _viewtrust_pr7_numel(value):
    try:
        return int(value.numel())
    except Exception:
        return ""


def _viewtrust_pr7_camera_name(camera):
    for attr in ("image_name", "image_path", "uid"):
        try:
            value = getattr(camera, attr)
            if value is not None:
                return str(value)
        except Exception:
            pass
    return ""


def _viewtrust_pr7_radii_stats(radii):
    stats = {{
        "radii_min": "",
        "radii_mean": "",
        "radii_max": "",
        "radii_nonzero_count": "",
    }}
    try:
        detached = radii.detach()
        if detached.numel() == 0:
            return stats
        stats["radii_min"] = float(detached.min().item())
        stats["radii_mean"] = float(detached.float().mean().item())
        stats["radii_max"] = float(detached.max().item())
        stats["radii_nonzero_count"] = int((detached > 0).sum().item())
    except Exception:
        pass
    return stats


def _viewtrust_pr7_grad_stats(viewspace_point_tensor):
    stats = {{"position_grad_mean": "", "position_grad_max": ""}}
    try:
        grad = viewspace_point_tensor.grad
        if grad is None or grad.numel() == 0:
            return stats
        detached = grad.detach().abs()
        stats["position_grad_mean"] = float(detached.mean().item())
        stats["position_grad_max"] = float(detached.max().item())
    except Exception:
        pass
    return stats


def _viewtrust_pr7_init_observer(dataset, opt, first_iter, scene, gaussians):
    if os.environ.get("VIEWTRUST_ENABLE_TRAINING_EVENTS") != "1":
        return None
    try:
        from viewtrust.observation.training_events import TrainingEventObserver

        observer = TrainingEventObserver.from_environment()
        if observer is not None:
            observer.log_gaussian_count(
                iteration=first_iter,
                stage="after_scene_init",
                gaussian_count=_viewtrust_pr7_count_gaussians(gaussians),
            )
        return observer
    except Exception as exc:
        if os.environ.get("VIEWTRUST_OBSERVER_STRICT") == "1":
            raise
        print(f"[ViewTrust] Training event observer initialization failed: {{exc!r}}")
        print("[ViewTrust] Training event logging disabled.")
        return None


def _viewtrust_pr7_call(observer, method_name, **kwargs):
    if observer is None:
        return None
    try:
        return getattr(observer, method_name)(**kwargs)
    except Exception as exc:
        if os.environ.get("VIEWTRUST_OBSERVER_STRICT") == "1":
            raise
        print(f"VIEWTRUST PR7 observer call failed: {{exc}}")
        return None
{END}
'''


def _insert_after(text: str, needle: str, insertion: str) -> str:
    if needle not in text:
        raise ValueError(f"patch anchor not found: {needle[:80]!r}")
    return text.replace(needle, needle + insertion, 1)


def apply_patch_text(text: str) -> str:
    if START in text:
        raise ValueError("PR7 observation patch is already applied")
    if "def training(dataset, opt, pipe" not in text:
        raise ValueError("train.py does not look like the expected official training file")

    text = _insert_after(text, "except:\n    SPARSE_ADAM_AVAILABLE = False\n", "\n" + HELPER_SNIPPET + "\n")

    checkpoint_block = (
        "    if checkpoint:\n"
        "        (model_params, first_iter) = torch.load(checkpoint)\n"
        "        gaussians.restore(model_params, opt)\n"
    )
    observer_init = (
        "\n"
        f"    {START}\n"
        "    viewtrust_observer = _viewtrust_pr7_init_observer(dataset, opt, first_iter, scene, gaussians)\n"
        f"    {END}\n"
    )
    text = _insert_after(text, checkpoint_block, observer_init)

    original_report = (
        "            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), "
        "testing_iterations, scene, render, (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp), dataset.train_test_exp)\n"
    )
    replacement_report = (
        f"            {START}\n"
        "            viewtrust_iter_time_ms = iter_start.elapsed_time(iter_end)\n"
        f"            {END}\n"
        "            training_report(tb_writer, iteration, Ll1, loss, l1_loss, viewtrust_iter_time_ms, "
        "testing_iterations, scene, render, (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp), dataset.train_test_exp)\n"
    )
    if original_report not in text:
        raise ValueError("training_report anchor not found")
    text = text.replace(original_report, replacement_report, 1)

    save_block = (
        "            if (iteration in saving_iterations):\n"
        "                print(\"\\n[ITER {}] Saving Gaussians\".format(iteration))\n"
        "                scene.save(iteration)\n"
    )
    save_patch = (
        save_block
        +
        f"                {START}\n"
        "                _viewtrust_pr7_call(viewtrust_observer, \"log_iteration_metrics\", event_type=\"save\", iteration=iteration, gaussian_count=_viewtrust_pr7_count_gaussians(gaussians), status=\"ok\")\n"
        "                _viewtrust_pr7_call(viewtrust_observer, \"log_gaussian_count\", iteration=iteration, stage=\"after_save\", gaussian_count=_viewtrust_pr7_count_gaussians(gaussians))\n"
        f"                {END}\n"
    )
    if save_block not in text:
        raise ValueError("save block anchor not found")
    text = text.replace(save_block, save_patch, 1)

    densification_header = "            # Densification\n"
    densification_vars = (
        f"            {START}\n"
        "            viewtrust_densification_eligible = iteration < opt.densify_until_iter\n"
        "            viewtrust_densification_triggered = False\n"
        "            viewtrust_opacity_reset_triggered = False\n"
        f"            {END}\n"
    )
    text = _insert_after(text, densification_header, densification_vars)

    densify_call = (
        "                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None\n"
        "                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, radii)\n"
    )
    densify_patch = (
        "                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None\n"
        f"                    {START}\n"
        "                    viewtrust_gaussian_count_before = _viewtrust_pr7_count_gaussians(gaussians)\n"
        f"                    {END}\n"
        "                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, radii)\n"
        f"                    {START}\n"
        "                    viewtrust_densification_triggered = True\n"
        "                    viewtrust_gaussian_count_after = _viewtrust_pr7_count_gaussians(gaussians)\n"
        "                    _viewtrust_pr7_call(viewtrust_observer, \"log_densification_event\", iteration=iteration, densification_eligible=True, densification_triggered=True, densify_from_iter=opt.densify_from_iter, densify_until_iter=opt.densify_until_iter, densification_interval=opt.densification_interval, densify_grad_threshold=opt.densify_grad_threshold, size_threshold=size_threshold, gaussian_count_before=viewtrust_gaussian_count_before, gaussian_count_after=viewtrust_gaussian_count_after, opacity_reset_triggered=False, status=\"ok\")\n"
        "                    _viewtrust_pr7_call(viewtrust_observer, \"log_gaussian_count\", iteration=iteration, stage=\"after_densification\", gaussian_count=viewtrust_gaussian_count_after)\n"
        f"                    {END}\n"
    )
    if densify_call not in text:
        raise ValueError("densify_and_prune anchor not found")
    text = text.replace(densify_call, densify_patch, 1)

    reset_call = "                    gaussians.reset_opacity()\n"
    reset_patch = (
        "                    gaussians.reset_opacity()\n"
        f"                    {START}\n"
        "                    viewtrust_opacity_reset_triggered = True\n"
        "                    _viewtrust_pr7_call(viewtrust_observer, \"log_gaussian_count\", iteration=iteration, stage=\"after_opacity_reset\", gaussian_count=_viewtrust_pr7_count_gaussians(gaussians))\n"
        f"                    {END}\n"
    )
    if reset_call not in text:
        raise ValueError("reset_opacity anchor not found")
    text = text.replace(reset_call, reset_patch, 1)

    optimizer_header = "            # Optimizer step\n"
    metrics_patch = (
        f"            {START}\n"
        "            viewtrust_radii_stats = _viewtrust_pr7_radii_stats(radii)\n"
        "            viewtrust_grad_stats = _viewtrust_pr7_grad_stats(viewspace_point_tensor)\n"
        "            viewtrust_visible_count = _viewtrust_pr7_sum(visibility_filter)\n"
        "            viewtrust_visibility_total = _viewtrust_pr7_numel(visibility_filter)\n"
        "            viewtrust_visibility_ratio = (viewtrust_visible_count / viewtrust_visibility_total) if viewtrust_visible_count != \"\" and viewtrust_visibility_total else \"\"\n"
        "            _viewtrust_pr7_call(viewtrust_observer, \"log_iteration_metrics\", iteration=iteration, event_type=\"iteration_metrics\", camera_index=vind, camera_image_name=_viewtrust_pr7_camera_name(viewpoint_cam), loss=_viewtrust_pr7_scalar(loss), l1_loss=_viewtrust_pr7_scalar(Ll1), ssim=_viewtrust_pr7_scalar(ssim_value), depth_l1=Ll1depth, iter_time_ms=viewtrust_iter_time_ms, gaussian_count=_viewtrust_pr7_count_gaussians(gaussians), visible_gaussian_count=viewtrust_visible_count, visibility_ratio=viewtrust_visibility_ratio, densification_eligible=viewtrust_densification_eligible, densification_triggered=viewtrust_densification_triggered, opacity_reset_triggered=viewtrust_opacity_reset_triggered, optimizer_step=iteration < opt.iterations, status=\"ok\", **viewtrust_radii_stats, **viewtrust_grad_stats)\n"
        "            _viewtrust_pr7_call(viewtrust_observer, \"log_gaussian_count\", iteration=iteration, stage=\"iteration_end\", gaussian_count=_viewtrust_pr7_count_gaussians(gaussians))\n"
        f"            {END}\n"
    )
    text = _insert_after(text, optimizer_header, metrics_patch)

    optimizer_block_end = (
        "                else:\n"
        "                    gaussians.optimizer.step()\n"
        "                    gaussians.optimizer.zero_grad(set_to_none = True)\n"
    )
    optimizer_patch = (
        optimizer_block_end
        +
        f"                {START}\n"
        "                _viewtrust_pr7_call(viewtrust_observer, \"log_optimizer_step\", iteration=iteration, gaussian_count=_viewtrust_pr7_count_gaussians(gaussians))\n"
        f"                {END}\n"
    )
    if optimizer_block_end not in text:
        raise ValueError("optimizer step anchor not found")
    text = text.replace(optimizer_block_end, optimizer_patch, 1)

    checkpoint_block = (
        "            if (iteration in checkpoint_iterations):\n"
        "                print(\"\\n[ITER {}] Saving Checkpoint\".format(iteration))\n"
        "                torch.save((gaussians.capture(), iteration), scene.model_path + \"/chkpnt\" + str(iteration) + \".pth\")\n"
    )
    checkpoint_patch = (
        checkpoint_block
        +
        f"                {START}\n"
        "                _viewtrust_pr7_call(viewtrust_observer, \"log_iteration_metrics\", event_type=\"checkpoint\", iteration=iteration, gaussian_count=_viewtrust_pr7_count_gaussians(gaussians), status=\"ok\")\n"
        f"                {END}\n"
    )
    if checkpoint_block not in text:
        raise ValueError("checkpoint block anchor not found")
    text = text.replace(checkpoint_block, checkpoint_patch, 1)

    finalize_patch = (
        f"\n    {START}\n"
        "    _viewtrust_pr7_call(viewtrust_observer, \"finalize\", iteration=opt.iterations, final_gaussian_count=_viewtrust_pr7_count_gaussians(gaussians))\n"
        f"    {END}\n"
    )
    text = _insert_after(text, checkpoint_patch, finalize_patch)
    return text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--third-party-root", required=True, type=Path)
    parser.add_argument("--patch", default=PATCH_NAME)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.patch != PATCH_NAME:
        raise SystemExit(f"unsupported patch: {args.patch}")

    train_path = args.third_party_root / "gaussian-splatting" / "train.py"
    if not train_path.is_file():
        raise SystemExit(f"ERROR: train.py not found: {train_path}")
    original = train_path.read_text(encoding="utf-8")
    already_applied = START in original and "VIEWTRUST_ENABLE_TRAINING_EVENTS" in original
    if already_applied and not args.force:
        raise SystemExit("ERROR: PR7 observation patch is already applied. Use --force to reapply.")

    source = original
    if already_applied and args.force:
        raise SystemExit("ERROR: --force reapply is intentionally not supported for patched files. Restore backup first.")

    patched = apply_patch_text(source)
    backup_path = train_path.with_name("train.py.viewtrust-pr7-backup")
    report = {
        "patch": args.patch,
        "train_path": str(train_path),
        "backup_path": str(backup_path),
        "dry_run": args.dry_run,
        "changed": patched != original,
        "markers_inserted": patched.count(START),
    }
    if not args.dry_run:
        if backup_path.exists() and not args.force:
            raise SystemExit(f"ERROR: backup already exists: {backup_path}")
        shutil.copy2(train_path, backup_path)
        train_path.write_text(patched, encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
