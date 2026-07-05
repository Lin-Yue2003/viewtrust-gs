#!/usr/bin/env python3
"""LOCAL-SAFE smoke test for PR7 Gaussian Splatting observation patch scripts."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


def _bootstrap_project_imports() -> Path:
    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root))
    return project_root


FAKE_TRAIN = '''import os
import torch

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False
def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        vind = viewpoint_indices.pop(rand_idx)
        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        Ll1 = l1_loss(image, gt_image)
        ssim_value = ssim(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)
        Ll1depth = 0
        loss.backward()
        with torch.no_grad():
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp), dataset.train_test_exp)
            if (iteration in saving_iterations):
                print("\\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, radii)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                if use_sparse_adam:
                    gaussians.optimizer.step(visible, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none = True)
                else:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):
    return None
'''


FAKE_GAUSSIAN_MODEL = '''import os
import torch
from torch import nn

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
except:
    pass

class GaussianModel:
    def __init__(self):
        self.optimizer = None
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._opacity = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.tmp_radii = torch.empty(0)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_scaling(self):
        return self._scaling

    def _prune_optimizer(self, mask):
        return {
            "xyz": self._xyz,
            "f_dc": self._features_dc,
            "f_rest": self._features_rest,
            "opacity": self._opacity,
            "scaling": self._scaling,
            "rotation": self._rotation,
        }

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        self.tmp_radii = self.tmp_radii[valid_points_mask]

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tmp_radii):
        pass

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        selected_pts_mask = grads
        new_xyz = grads
        new_scaling = grads
        new_rotation = grads
        new_features_dc = grads
        new_features_rest = grads
        new_opacity = grads
        new_tmp_radii = grads

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_tmp_radii)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        selected_pts_mask = grads
        new_xyz = grads
        new_features_dc = grads
        new_features_rest = grads
        new_opacities = grads
        new_scaling = grads
        new_rotation = grads

        new_tmp_radii = grads

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tmp_radii)
'''


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def main() -> int:
    project_root = _bootstrap_project_imports()

    with tempfile.TemporaryDirectory(prefix="viewtrust-pr7-patch-") as tmp:
        third_party_root = Path(tmp) / "third_party"
        train_path = third_party_root / "gaussian-splatting" / "train.py"
        model_path = third_party_root / "gaussian-splatting" / "scene" / "gaussian_model.py"
        train_path.parent.mkdir(parents=True)
        model_path.parent.mkdir(parents=True)
        train_path.write_text(FAKE_TRAIN, encoding="utf-8")
        model_path.write_text(FAKE_GAUSSIAN_MODEL, encoding="utf-8")

        apply_script = project_root / "scripts" / "third_party" / "apply_gaussian_splatting_observation_patch.py"
        check_script = project_root / "scripts" / "third_party" / "check_gaussian_splatting_observation_patch.py"

        dry_run = _run(
            [
                sys.executable,
                str(apply_script),
                "--third-party-root",
                str(third_party_root),
                "--patch",
                "pr7_training_events",
                "--dry-run",
            ]
        )
        if dry_run.returncode != 0:
            raise RuntimeError(dry_run.stderr or dry_run.stdout)
        dry_report = json.loads(dry_run.stdout)
        if dry_report["markers_inserted"] <= 0:
            raise ValueError("dry-run did not report inserted markers")
        if (train_path.with_name("train.py.viewtrust-pr7-backup")).exists():
            raise ValueError("dry-run should not create backup")
        if (model_path.with_name("gaussian_model.py.viewtrust-pr8-backup")).exists():
            raise ValueError("dry-run should not create model backup")

        applied = _run(
            [
                sys.executable,
                str(apply_script),
                "--third-party-root",
                str(third_party_root),
                "--patch",
                "pr7_training_events",
            ]
        )
        if applied.returncode != 0:
            raise RuntimeError(applied.stderr or applied.stdout)
        if not train_path.with_name("train.py.viewtrust-pr7-backup").exists():
            raise FileNotFoundError("backup was not created")
        if not model_path.with_name("gaussian_model.py.viewtrust-pr8-backup").exists():
            raise FileNotFoundError("model backup was not created")
        patched_text = train_path.read_text(encoding="utf-8")
        patched_model_text = model_path.read_text(encoding="utf-8")
        if "[ViewTrust] Training event observer initialization failed" not in patched_text:
            raise ValueError("patched trainer does not include visible init failure message")
        if "[ViewTrust] Training event logging disabled." not in patched_text:
            raise ValueError("patched trainer does not include disabled logging message")
        if "VIEWTRUST_OBSERVER_STRICT" not in patched_text:
            raise ValueError("patched trainer does not include strict observer mode")
        if "VIEWTRUST_ENABLE_GAUSSIAN_LIFECYCLE" not in patched_text:
            raise ValueError("patched trainer does not include lifecycle env gate")
        if "GaussianLifecycleObserver" not in patched_text:
            raise ValueError("patched trainer does not import lifecycle observer")
        for expected in ("on_after_clone", "on_after_split", "on_before_prune", "on_after_prune"):
            if expected not in patched_model_text:
                raise ValueError(f"patched model does not include {expected}")

        check = _run(
            [
                sys.executable,
                str(check_script),
                "--third-party-root",
                str(third_party_root),
                "--patch",
                "pr7_training_events",
                "--require-applied",
            ]
        )
        if check.returncode != 0:
            raise RuntimeError(check.stderr or check.stdout)
        check_report = json.loads(check.stdout)
        if not check_report["ok"]:
            raise ValueError(check_report)

        compile_result = _run(
            [sys.executable, "-m", "py_compile", str(train_path), str(model_path)]
        )
        if compile_result.returncode != 0:
            raise RuntimeError(compile_result.stderr or compile_result.stdout)

    print("gaussian splatting observation patch smoke test ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
