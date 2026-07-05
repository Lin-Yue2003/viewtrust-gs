#!/usr/bin/env python3
"""SERVER-REQUIRED gsplat CUDA smoke test."""

from __future__ import annotations

import sys
from pathlib import Path


def _ensure_repo_gsplat_on_path() -> None:
    project_root = Path(__file__).resolve().parents[2]
    repo_gsplat = project_root / "third_party" / "gsplat"
    if repo_gsplat.exists():
        sys.path.insert(0, str(repo_gsplat))


def main() -> int:
    _ensure_repo_gsplat_on_path()

    try:
        import torch
    except Exception as exc:
        raise SystemExit(f"torch import failed: {exc}")

    print(f"torch version: {torch.__version__}")
    print(f"torch CUDA version: {torch.version.cuda}")
    print(f"cuda availability: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; this SERVER-REQUIRED smoke test cannot run.")

    print(f"GPU count: {torch.cuda.device_count()}")
    for index in range(torch.cuda.device_count()):
        print(f"GPU {index}: {torch.cuda.get_device_name(index)}")

    try:
        import gsplat
    except Exception as exc:
        raise SystemExit(f"gsplat import failed: {exc}")

    print(f"gsplat import path: {getattr(gsplat, '__file__', '<unknown>')}")

    device = torch.device("cuda:0")
    means = torch.tensor([[0.0, 0.0, 3.0]], dtype=torch.float32, device=device)
    quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32, device=device)
    scales = torch.tensor([[0.2, 0.2, 0.2]], dtype=torch.float32, device=device)
    opacities = torch.tensor([0.9], dtype=torch.float32, device=device)
    colors = torch.tensor([[1.0, 0.2, 0.1]], dtype=torch.float32, device=device)
    viewmats = torch.eye(4, dtype=torch.float32, device=device).unsqueeze(0)
    Ks = torch.tensor(
        [[50.0, 0.0, 32.0], [0.0, 50.0, 32.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)

    with torch.no_grad():
        render_colors, render_alphas, meta = gsplat.rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=viewmats,
            Ks=Ks,
            width=64,
            height=64,
        )

    torch.cuda.synchronize()

    print(f"render output shapes: colors={tuple(render_colors.shape)}, alphas={tuple(render_alphas.shape)}")
    print(f"meta keys: {sorted(meta.keys())}")
    print("smoke test ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
