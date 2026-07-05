# NeRF Synthetic Chair Mini Subset

PR2 defines the first small real dataset recipe for ViewTrust-GS Priority 0:

```text
dataset: NeRF Synthetic / Blender
scene: chair
condition: clean
```

The goal is not SOTA quality. The goal is a tiny, reproducible, storage-conscious
scene subset for validating Priority 0 metadata, config snapshots, view identity
consistency, and future observation logs.

## Why Chair First

NeRF Synthetic chair is compact, widely used, and simple enough for fast
iteration. It is a better first Priority 0 target than larger datasets because
we need to validate infrastructure before spending server disk and GPU time.

We do not start with Mip-NeRF 360, Tanks and Temples, or Deep Blending because
those datasets are larger, slower to prepare, and more likely to fill local or
server storage during early experiments.

## Raw Dataset Layout

The first server-tested source was:

```text
Hugging Face: rishitdagli/nerf-gs-datasets
folder: chair
```

Place the already downloaded raw scene at either:

```text
data/raw/nerf_synthetic/chair/
```

or on the server:

```text
$VIEWTRUST_DATA_ROOT/raw/nerf_synthetic/chair/
```

Expected raw scene layout:

```text
chair/
  README.txt
  transforms_train.json
  transforms_val.json
  transforms_test.json
  train/
  val/
  test/
```

`transforms_val.json` may exist, but PR2 ignores it.

Observed server raw chair size:

```text
126M
```

## Prepared Output Layout

The clean mini subset is written to:

```text
data/viewtrust-mini/nerf_synthetic/chair/clean/
```

or, when `VIEWTRUST_DATA_ROOT` is set:

```text
$VIEWTRUST_DATA_ROOT/viewtrust-mini/nerf_synthetic/chair/clean/
```

Output files:

```text
clean/
  transforms_train.json
  transforms_test.json
  transforms_target.json
  images/
  manifest.json
  README.md
```

## Default Subset

```text
max_train_views: 20
max_test_views: 5
max_target_views: 3
max_image_width: 400
condition: clean only
copy_mode: symlink
```

The tested server subset selected 20 train views, 5 test views, and 3 target
views. On the tested source, `will_resize=true` because source images were wider
than 400 pixels. In that case resized output images are written even when
`copy_mode=symlink`.

Frames are selected by deterministic uniform sampling across each split. This
avoids selecting only adjacent views.

## Dry Run

```bash
python scripts/data/prepare_nerf_synthetic_subset.py \
  --raw-scene-root data/raw/nerf_synthetic/chair \
  --output-root data/viewtrust-mini/nerf_synthetic/chair \
  --scene chair \
  --condition clean \
  --max-train-views 20 \
  --max-test-views 5 \
  --max-target-views 3 \
  --max-image-width 400 \
  --copy-mode symlink \
  --dry-run
```

## Real Preparation

```bash
python scripts/data/prepare_nerf_synthetic_subset.py \
  --raw-scene-root data/raw/nerf_synthetic/chair \
  --output-root data/viewtrust-mini/nerf_synthetic/chair \
  --scene chair \
  --condition clean \
  --max-train-views 20 \
  --max-test-views 5 \
  --max-target-views 3 \
  --max-image-width 400 \
  --copy-mode symlink \
  --overwrite
```

On the server, prefer:

```bash
python scripts/data/prepare_nerf_synthetic_subset.py \
  --data-root "$VIEWTRUST_DATA_ROOT" \
  --raw-scene-root "$VIEWTRUST_DATA_ROOT/raw/nerf_synthetic/chair" \
  --output-root "$VIEWTRUST_DATA_ROOT/viewtrust-mini/nerf_synthetic/chair" \
  --scene chair \
  --condition clean \
  --max-train-views 20 \
  --max-test-views 5 \
  --max-target-views 3 \
  --max-image-width 400 \
  --copy-mode symlink \
  --overwrite
```

## Inspect Output

Check:

```bash
find "$VIEWTRUST_DATA_ROOT/viewtrust-mini/nerf_synthetic/chair/clean" -maxdepth 2 -type f | sort
du -sh "$VIEWTRUST_DATA_ROOT/viewtrust-mini/nerf_synthetic/chair/clean"
python - <<'PY'
import json, os
from pathlib import Path

root = Path(os.environ["VIEWTRUST_DATA_ROOT"]) / "viewtrust-mini/nerf_synthetic/chair/clean"
manifest = json.loads((root / "manifest.json").read_text())
print("image_count:", manifest["image_count"])
print("train:", len(manifest["selected_train_frames"]))
print("test:", len(manifest["selected_test_frames"]))
print("target:", len(manifest["selected_target_frames"]))
for name in ["transforms_train.json", "transforms_test.json", "transforms_target.json"]:
    data = json.loads((root / name).read_text())
    print(name, len(data["frames"]))
PY
```

The prepared transform `file_path` entries should be relative paths under
`images/`.

`manifest.json` records dataset roots relative to `VIEWTRUST_DATA_ROOT`; it
should not contain machine-specific absolute paths.

## Regenerate Portable Manifest

After PR2 manifest cleanup is pulled on the server, regenerate the prepared
subset:

```bash
python scripts/data/prepare_nerf_synthetic_subset.py \
  --data-root "$VIEWTRUST_DATA_ROOT" \
  --raw-scene-root "$VIEWTRUST_DATA_ROOT/raw/nerf_synthetic/chair" \
  --output-root "$VIEWTRUST_DATA_ROOT/viewtrust-mini/nerf_synthetic/chair" \
  --scene chair \
  --condition clean \
  --max-train-views 20 \
  --max-test-views 5 \
  --max-target-views 3 \
  --max-image-width 400 \
  --copy-mode symlink \
  --overwrite
```

## Storage Policy

The default `copy_mode` is `symlink` to avoid duplicating raw images and filling
server disks.

Supported modes:

```text
symlink
hardlink
copy
```

If resizing is needed because source images are wider than `max_image_width`,
new resized image files are written under `images/` even when `copy_mode` is
`symlink` or `hardlink`.

## Clean First

PR2 prepares only the clean condition. It does not generate motion blur,
lighting shifts, pose noise, local patch poison, poisoning attacks, natural
corruptions, scoring, defenses, or gating.

Future conditions can be added later under:

```text
data/viewtrust-mini/nerf_synthetic/chair/motion_blur/
data/viewtrust-mini/nerf_synthetic/chair/lighting_shift/
data/viewtrust-mini/nerf_synthetic/chair/pose_noise/
data/viewtrust-mini/nerf_synthetic/chair/local_patch_poison/
```
