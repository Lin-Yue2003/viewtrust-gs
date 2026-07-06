# Natural Corruption Conditions

PR10 adds storage-conscious natural corruption condition generation for the
prepared NeRF Synthetic chair mini dataset.

This is observation preparation only. It does not change training behavior,
trainer internals, ViewTrust scoring, defense logic, densification, pruning,
rendering, or sampling.

## Input Dataset

The clean mini dataset must already exist:

```text
$VIEWTRUST_DATA_ROOT/viewtrust-mini/nerf_synthetic/chair/clean/
  images/
  transforms_train.json
  transforms_test.json
  transforms_target.json
  manifest.json
```

The transform `file_path` values remain official Gaussian Splatting compatible
and extensionless:

```json
{
  "file_path": "images/train_000",
  "transform_matrix": []
}
```

The corresponding image file is expected at:

```text
images/train_000.png
```

## Default Conditions

The default PR10 suite creates:

```text
corrupt_occluder
corrupt_blur
corrupt_exposure
corrupt_color_shift
corrupt_noise
corrupt_mixed
```

By default, each condition corrupts only 4 train views from the 20-view clean
mini train split, using seed `20260706`. Test and target views are copied or
symlinked unchanged so clean-vs-corrupt comparisons keep evaluation views
stable.

## Output Layout

Each condition is written under:

```text
$VIEWTRUST_DATA_ROOT/viewtrust-mini/nerf_synthetic/chair/<condition>/
  images/
  transforms_train.json
  transforms_test.json
  transforms_target.json
  manifest.json
  corruption_manifest.json
  corruption_manifest.csv
  corruption_summary.json
  preview/preview_grid.png
```

The transform files are copied from the clean condition so camera metadata and
extensionless `file_path` entries are preserved.

`manifest.json` uses the existing
`viewtrust.nerf_synthetic_subset.manifest` schema so the baseline training
wrapper can run natural corruption conditions directly.

## Storage Policy

Default `--copy-mode symlink` avoids duplicating uncorrupted images. Corrupted
train views are always written as new PNG files because their pixels change.

Use `--copy-mode copy` when symlinks are not convenient for an archive or
portable test fixture.

## Dry Run

```bash
python scripts/data/generate_natural_corruptions.py \
  --data-root "$VIEWTRUST_DATA_ROOT" \
  --scene chair \
  --source-condition clean \
  --output-condition corrupt_occluder \
  --corruption-type occluder \
  --seed 20260706 \
  --num-corrupt-train-views 4 \
  --copy-mode symlink \
  --dry-run
```

## Generate One Condition

```bash
python scripts/data/generate_natural_corruptions.py \
  --data-root "$VIEWTRUST_DATA_ROOT" \
  --scene chair \
  --source-condition clean \
  --output-condition corrupt_occluder \
  --corruption-type occluder \
  --seed 20260706 \
  --num-corrupt-train-views 4 \
  --copy-mode symlink \
  --overwrite
```

## Generate Default Suite

```bash
python scripts/data/generate_default_natural_corruption_suite.py \
  --data-root "$VIEWTRUST_DATA_ROOT" \
  --scene chair \
  --source-condition clean \
  --seed 20260706 \
  --num-corrupt-train-views 4 \
  --copy-mode symlink \
  --overwrite
```

## Inspect One Condition

```bash
python scripts/measure/inspect_natural_corruption_dataset.py \
  --data-root "$VIEWTRUST_DATA_ROOT" \
  --scene chair \
  --condition corrupt_occluder \
  --require-valid \
  --require-corrupted-count 4
```

The inspector checks that transform images exist, transform paths are
extensionless, test and target views are uncorrupted, selected train views are
valid, `manifest.json` exists, corruption manifests exist, summaries exist, and the preview grid exists when
requested.

## PR10.1 Training Compatibility

Natural corruption conditions are generated from the clean mini scene but are
still prepared scene roots. Each generated condition writes:

```text
manifest.json
```

This file records the condition name, source condition, natural corruption
type, train/test/target counts, corrupted image count, and links to
`corruption_manifest.json` and `corruption_summary.json`.

Because of this compatibility manifest, the existing baseline wrapper can run:

```bash
python scripts/train/run_clean_chair_baseline.py \
  --data-root "$VIEWTRUST_DATA_ROOT" \
  --third-party-root ./third_party \
  --scene chair \
  --condition corrupt_occluder
```

## Why Natural Corruption First

Natural corruptions provide a small, controlled bridge between the clean
Priority 0 baseline and later adversarial or poison settings. PR10 intentionally
does not generate poisoning, patch attacks, trust scores, defenses, or
training-time interventions.

Future conditions can add other non-clean layouts under the same scene root,
for example:

```text
motion_blur/
lighting_shift/
pose_noise/
local_patch_poison/
```

Those are intentionally out of scope for PR10.
