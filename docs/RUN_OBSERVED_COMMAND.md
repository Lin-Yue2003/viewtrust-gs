# Run Observed Command

`scripts/measure/run_observed_command.py` wraps an external command and records
Priority 0 observation artifacts. It does not modify the command, training
code, optimizer behavior, rendering, densification, pruning, sampling, or loss.

Basic form:

```bash
python scripts/measure/run_observed_command.py -- <command>
```

Useful options:

```text
--label                 Output subdirectory label.
--sample-interval-s     GPU sampling interval in seconds.
--config                Config snapshot path.
--output-root           Output root, default VIEWTRUST_OUTPUT_ROOT or ./outputs.
--run-id                Optional explicit run id.
```

## Examples

### A. Sleep Test

This is LOCAL-SAFE and does not require CUDA. GPU sampling may be empty when
`nvidia-smi` is unavailable.

```bash
python scripts/measure/run_observed_command.py \
  --label observed-sleep-test \
  --sample-interval-s 0.5 \
  -- python -c "import time; print('sleep test start'); time.sleep(3); print('sleep test end')"
```

### B. gsplat CUDA Smoke Test Under Observation

This is SERVER-REQUIRED because it runs the CUDA/gsplat smoke test.

```bash
python scripts/measure/run_observed_command.py \
  --label gsplat-smoke-observed \
  --sample-interval-s 0.2 \
  -- python scripts/smoke/gsplat_cuda_smoke_test.py
```

### C. Future Training Wrapper Placeholder

Placeholder only. Do not use this as training integration yet.

```bash
python scripts/measure/run_observed_command.py \
  --label future-3dgs-training-placeholder \
  --sample-interval-s 1.0 \
  -- python path/to/train.py --example-args
```

The wrapper observes the external process from the outside. It is intended to
validate Priority 0 artifact capture before any training-loop instrumentation.

## Output Files

`metadata.json`

Records the run label, command, run id, measurement mode, and the explicit fact
that training behavior was not modified. This anchors each Priority 0 run.

`config_snapshot.json`

Stores the config used by the wrapper. This makes later comparisons
reproducible even if defaults change.

`events.jsonl`

Append-only event stream. It records command start/end, timing observations,
config snapshots, table snapshots, and GPU samples when available. This is the
most complete Priority 0 record.

`stdout.log`

Captured standard output from the observed command. This preserves the command's
native logs without changing the command itself.

`stderr.log`

Captured standard error from the observed command. This is important for
failures, warnings, CUDA errors, and dependency diagnostics.

`summary.json`

Small run-level summary with return code, elapsed time, GPU sample count, and
paths to logs. This is the first file to inspect after a run.

`stats.json`

Numeric summaries such as elapsed time and GPU memory statistics. It supports
quick tables and comparisons while keeping raw records in `events.jsonl`.

`tables/command_summary.csv`

CSV row for the observed command: label, return code, elapsed time, stdout path,
and stderr path. This is convenient for spreadsheets and batch comparisons.

`tables/gpu_memory_samples.csv`

CSV table of `nvidia-smi` samples, including GPU index, name, used memory, total
memory, utilization, and elapsed time. It may be absent or empty on machines
without `nvidia-smi`; that is acceptable for LOCAL-SAFE checks.
