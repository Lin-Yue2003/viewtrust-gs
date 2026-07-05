# Training Dynamics

PR5 adds post-hoc training dynamics extraction for completed clean baseline
runs. It reads an existing observed run directory and writes standardized
derived tables and a compact summary.

This is observation-only. It does not rerun training, change official trainer
behavior, edit trainer outputs, import CUDA, add ViewTrust scoring, add defense
logic, or track view-level or per-Gaussian lifecycle state.

## Inputs

Expected observed baseline run:

```text
outputs/baseline/chair_clean_gaussian_splatting/<run_id>/
  summary.json
  metadata.json
  config_snapshot.json
  stdout.log
  stderr.log
  tables/command_summary.csv
  tables/gpu_memory_samples.csv
  trainer_output/
```

## Outputs

PR5 writes derived files into the same run directory:

```text
tables/training_dynamics.csv
tables/training_artifacts.csv
tables/final_gaussian_summary.csv
training_dynamics_summary.json
```

`training_dynamics.csv` records TensorBoard scalar rows when available:

```text
run_id,source,iteration,tag,value,wall_time
```

`training_artifacts.csv` lists files under `trainer_output/` without reading
large binary contents:

```text
run_id,relative_path,file_type,size_bytes,modified_time
```

`final_gaussian_summary.csv` records the final discovered point cloud:

```text
run_id,iteration,point_cloud_path,exists,gaussian_count,size_bytes,parse_status
```

## TensorBoard Handling

Official Gaussian Splatting writes TensorBoard scalars only when TensorBoard is
available in the trainer environment. PR5 searches for event files under
`trainer_output/`.

If event files exist and the `tensorboard` Python package is installed, PR5
extracts scalar rows such as:

```text
train_loss_patches/l1_loss
train_loss_patches/total_loss
iter_time
test/loss_viewpoint - l1_loss
test/loss_viewpoint - psnr
train/loss_viewpoint - l1_loss
train/loss_viewpoint - psnr
total_points
```

If TensorBoard is unavailable or no event files exist, extraction still
succeeds by default. `training_dynamics.csv` is written with only its header,
and `training_dynamics_summary.json` records explicit warnings. PR5 never
pretends missing loss curves exist.

Optional server dependency for loss curve extraction:

```bash
python -m pip install tensorboard
```

This is not required for local smoke tests.

## Gaussian Count

PR5 searches for:

```text
trainer_output/point_cloud/iteration_*/point_cloud.ply
```

It selects the largest numeric iteration and reads only the PLY header. The
Gaussian count comes from:

```text
element vertex <N>
```

The binary body is not parsed.

## Server Extraction

After a successful baseline run:

```bash
RUN_DIR=$(find outputs/baseline/chair_clean_gaussian_splatting -mindepth 1 -maxdepth 1 -type d | sort | tail -1)

python scripts/measure/extract_training_dynamics.py \
  --run-dir "$RUN_DIR" \
  --require-success

cat "$RUN_DIR/training_dynamics_summary.json"
cat "$RUN_DIR/tables/final_gaussian_summary.csv"
head -20 "$RUN_DIR/tables/training_artifacts.csv"
head -20 "$RUN_DIR/tables/training_dynamics.csv"
```

Inspect the run again with PR5 fields:

```bash
python scripts/measure/inspect_baseline_run.py \
  --run-dir "$RUN_DIR" \
  --require-success
```

## Known Limitations

PR5 records global dynamics only. It does not include view-level metrics,
per-view residuals, per-Gaussian lifecycle IDs, birth/prune tracking,
densification source attribution, trust scores, defenses, or any training
behavior changes.

Those belong to later PRs.

## Next Step

PR6 consumes a successful baseline run after PR5 and adds per-view clean render
metrics for train, test, and target views. PR6 remains post-hoc and
observation-only; it does not add trust scores or modify training behavior.

PR7 then adds opt-in global training event logging through a local
observation-only trainer patch. PR7 still does not add Gaussian lifecycle IDs
or trust scores.
