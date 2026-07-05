# Measurement Format

Priority 0 measurements use an append-friendly run directory format.

## Run Directory

Each run writes:

```text
metadata.json
config_snapshot.json
events.jsonl
summary.json
stats.json
tables/
  iteration_observations.csv
  view_observations.csv
  command_summary.csv
  gpu_memory_samples.csv
```

Additional tables may be added without changing existing files. New fields
should be appended as new columns or added under event payloads.

## Stable JSON Fields

All JSON documents include:

```text
schema_name
schema_version
run_id
created_at_utc
```

Events are written to `events.jsonl` with:

```text
event_type
payload
```

The JSONL file is the complete event stream. CSV tables are derived or
table-friendly snapshots for analysis.

## Current Tables

`iteration_observations.csv` is intended for per-iteration metrics such as:

```text
iteration
elapsed_ms
gpu_memory_allocated_mb
visible_gaussians
```

`view_observations.csv` is intended for per-view metadata such as:

```text
iteration
view_id
camera_uid
width
height
```

`command_summary.csv` records observed subprocess status:

```text
label
returncode
elapsed_s
stdout_path
stderr_path
```

`gpu_memory_samples.csv` records sampled GPU state when `nvidia-smi` is
available:

```text
elapsed_s
gpu_index
gpu_name
memory_used_mb
memory_total_mb
utilization_gpu_percent
```

Future fields should preserve observation-only semantics. They may record what
happened, but they must not drive optimizer, rendering, densification, pruning,
sampling, loss, or defense decisions.

## Statistics

`stats.json` stores numeric summaries with:

```text
count
minimum
maximum
mean
population_std
```

This keeps quick comparisons available without losing the full raw event stream.

## Observed Command Wrapper

Use this wrapper to measure an existing training command without modifying that
training code:

```bash
python scripts/measure/run_observed_command.py \
  --label baseline-3dgs \
  --output-root "$VIEWTRUST_OUTPUT_ROOT" \
  --sample-interval-s 1.0 \
  -- \
  python third_party/gaussian-splatting/train.py <training args>
```

The wrapper records wall time, stdout/stderr, optional GPU samples from
`nvidia-smi`, summary JSON, stats JSON, and CSV tables.
