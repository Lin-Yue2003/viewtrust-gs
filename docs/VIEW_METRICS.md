# View Metrics

PR6 adds post-hoc view-level clean metrics for successful baseline runs.

This is observation-only. It does not rerun training, modify official Gaussian
Splatting code, change rendering behavior inside `third_party`, write ViewTrust
scores, add defenses, or classify views as good/bad/poisoned.

## Purpose

Given a completed clean chair baseline run, PR6 renders train, test, and target
views from the final trained Gaussian model, then computes per-view metrics
against ground-truth images.

## Output Layout

PR6 writes derived evaluation data under the run directory:

```text
view_evaluation/
  render_models/
    train_test_model/
    target_model/
  eval_scenes/
    target_as_test/
  render_logs/
  tables/
    view_metrics.csv
    view_render_artifacts.csv
  view_metrics_summary.json
tables/
  view_metrics.csv
  view_render_artifacts.csv
view_metrics_summary.json
```

The root-level `tables/` files are the canonical ViewTrust tables. The
`view_evaluation/tables/` copies make the render package self-contained.

## Rendering Strategy

PR6 calls the official Gaussian Splatting `render.py` from outside the trainer:

```bash
python third_party/gaussian-splatting/render.py \
  -s <prepared_scene_root> \
  -m <evaluation_model_dir> \
  --iteration 500 \
  --eval
```

`--eval` is required for Blender datasets. Without it, official Gaussian
Splatting merges test cameras into train and clears the test camera list. On
the mini chair subset this produces the incorrect state `train=25, test=0`
instead of `train=20, test=5`.

The evaluation model directories live under:

```text
<run_dir>/view_evaluation/render_models/
```

They symlink `point_cloud` back to the completed baseline `trainer_output` and
copy small config files such as `cfg_args` when present. PR6 does not mutate
`trainer_output/`.

## Target Views

Official Gaussian Splatting renders train and test splits. PR6 evaluates target
views by creating:

```text
<run_dir>/view_evaluation/eval_scenes/target_as_test/
```

In that derived scene, `transforms_test.json` is copied from the prepared
scene's `transforms_target.json`. The original prepared dataset is not
modified.

Target rendering also requires `--eval`; otherwise the target-as-test cameras
are cleared when official `render.py` runs with `--skip_train`.

## Metrics

`view_metrics.csv` columns:

```text
run_id,scene,condition,split,iteration,view_index,image_name,render_relative_path,gt_relative_path,width,height,l1_mean,mse,psnr,ssim,ssim_method,residual_mean,residual_median,residual_p95,residual_p99,residual_max,status,warning
```

Definitions:

```text
l1_mean: mean absolute RGB error normalized to [0, 1]
mse: mean squared RGB error normalized to [0, 1]
psnr: 20 * log10(1.0 / sqrt(mse)); exact matches use explicit value 100.0
residual_*: statistics over absolute RGB residuals
```

SSIM uses `skimage.metrics.structural_similarity` when available. If `skimage`
is unavailable, PR6 uses a simple global NumPy fallback and marks:

```text
ssim_method=global_numpy_fallback
```

If SSIM cannot be computed, it records:

```text
ssim_method=unavailable
```

## Server Commands

```bash
cd /trainingData/sage/yue/viewtrust-gs
source scripts/env/activate_server_viewtrust_p0.sh
export VIEWTRUST_DATA_ROOT=/trainingData/sage/yue/viewtrust-data

RUN_DIR=$(find outputs/baseline/chair_clean_gaussian_splatting -mindepth 1 -maxdepth 1 -type d | sort | tail -1)
echo "$RUN_DIR"

python scripts/evaluate/render_clean_views.py \
  --run-dir "$RUN_DIR" \
  --data-root "$VIEWTRUST_DATA_ROOT" \
  --third-party-root ./third_party \
  --trainer gaussian-splatting \
  --scene chair \
  --condition clean \
  --iteration 500 \
  --splits train test target \
  --gpu 0 \
  --sample-interval-s 1.0 \
  --overwrite

python scripts/measure/extract_view_metrics.py \
  --run-dir "$RUN_DIR" \
  --scene chair \
  --condition clean \
  --iteration 500 \
  --require-renders

cat "$RUN_DIR/view_metrics_summary.json"
head -20 "$RUN_DIR/tables/view_metrics.csv"
head -20 "$RUN_DIR/tables/view_render_artifacts.csv"

python scripts/measure/inspect_baseline_run.py \
  --run-dir "$RUN_DIR" \
  --require-success
```

## Known Limitations

PR6 covers clean train/test/target views only. It does not add trust scores,
natural corruption or poison conditions, Gaussian lifecycle tracking,
densification event attribution, or training-loop hooks.

If target rendering fails because of official renderer constraints, report the
failure and keep train/test metrics visible. Do not silently drop target.

PR7 adds opt-in global training event and densification schedule logging after
PR6. PR7 still does not add trust scores or Gaussian lifecycle tracking.
