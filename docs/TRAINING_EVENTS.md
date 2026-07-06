# Training Events

PR7 adds observation-only global training event logging for official Gaussian
Splatting clean baseline runs.

This is the first ViewTrust-GS stage that uses an explicit local patch around
the official trainer loop. The patch is opt-in, marker-delimited, and applied
only by a user command on the server. Patched `third_party` source is not
committed to ViewTrust-GS.

## Scope

PR7 logs global training events:

```text
iteration metrics
selected train camera identity when available
loss, L1, SSIM, optional depth loss
iteration timing
Gaussian count
visibility and radii statistics
densification schedule and trigger state
opacity reset trigger state
optimizer step status
save and checkpoint events
```

PR7 does not implement Gaussian lifecycle tracking, parent-child clone/split
IDs, view attribution, trust scores, defenses, poisoning, corruptions, or any
training behavior changes.

## Why Densification May Be Absent

The 500-iteration clean chair baseline may have zero triggered densification
events. Official defaults commonly require:

```text
iteration > densify_from_iter
```

and `densify_from_iter` is often 500. A 500-iteration run can therefore finish
before the first densification trigger. This is valid and should be recorded as
a warning, not treated as failure.

## Patch Workflow

Apply the observation patch manually on the server:

```bash
python scripts/third_party/apply_gaussian_splatting_observation_patch.py \
  --third-party-root ./third_party \
  --patch pr7_training_events
```

Check the patch:

```bash
python scripts/third_party/check_gaussian_splatting_observation_patch.py \
  --third-party-root ./third_party \
  --patch pr7_training_events \
  --require-applied
```

The patch activates only when:

```text
VIEWTRUST_ENABLE_TRAINING_EVENTS=1
```

The clean baseline wrapper sets this only when
`--enable-training-events` is passed.

PR7.1 also makes the wrapper inject the ViewTrust-GS project root into the
child trainer `PYTHONPATH` and run an observer import preflight before training
starts. If the official trainer child environment cannot import
`viewtrust.observation.training_events`, the wrapper stops before launching the
training command.

For stricter validation, pass:

```text
--training-event-strict
```

This sets `VIEWTRUST_OBSERVER_STRICT=1` for the child trainer. In strict mode,
observer import or logging failures raise instead of silently disabling event
logging.

## Instrumented Baseline

```bash
python scripts/train/run_clean_chair_baseline.py \
  --trainer gaussian-splatting \
  --data-root "$VIEWTRUST_DATA_ROOT" \
  --third-party-root ./third_party \
  --output-root ./outputs \
  --scene chair \
  --condition clean \
  --iterations 500 \
  --gpu 0 \
  --sample-interval-s 1.0 \
  --enable-training-events \
  --training-event-log-interval 10 \
  --training-event-strict
```

## Outputs

For an instrumented run:

```text
training_events/
  training_events.csv
  densification_events.csv
  gaussian_count_timeseries.csv
  observer_warnings.jsonl
  training_events_summary.json
tables/
  training_events.csv
  densification_events.csv
  gaussian_count_timeseries.csv
training_events_summary.json
```

`training_events.csv` records global iteration/save/checkpoint events.
`densification_events.csv` records actual densification calls when they occur.
`gaussian_count_timeseries.csv` records count checkpoints such as
`after_scene_init`, `iteration_end`, and `final`.

PR7.2 enforces scalar sanity for visibility and radii fields:

```text
0 <= visible_gaussian_count <= gaussian_count
0 <= visibility_ratio <= 1
0 <= radii_nonzero_count <= gaussian_count
```

`visible_gaussian_count` is computed from `visibility_filter.detach().bool().sum()`.
`visibility_ratio` uses the current Gaussian count as the denominator. The
observer stores Python scalars only and does not retain tensors.

## Inspect

```bash
RUN_DIR=$(find outputs/baseline/chair_clean_gaussian_splatting -mindepth 1 -maxdepth 1 -type d | sort | tail -1)

python scripts/measure/inspect_training_events.py \
  --run-dir "$RUN_DIR" \
  --require-events

cat "$RUN_DIR/training_events_summary.json"
head -20 "$RUN_DIR/tables/training_events.csv"
head -20 "$RUN_DIR/tables/densification_events.csv"
head -20 "$RUN_DIR/tables/gaussian_count_timeseries.csv"
```

With `--require-events`, the inspector fails if any training event row violates
the visibility invariants. The compact report includes:

```text
invalid_training_event_rows
max_visible_gaussian_count
max_visibility_ratio
max_gaussian_count
requested_iterations
logged_iteration_count
```

`logged_iteration_count` is the count of logged iteration rows, not the total
trainer iteration request. `requested_iterations` records the trainer target
when the patched trainer passes `opt.iterations` to the observer.

## Safety Rules

The observer must not affect training:

```text
no backward calls
no tensor mutation
no optimizer calls
no loss modification
no sampling modification
no densification condition changes
no pruning or opacity reset decision changes
no retained GPU tensor references
```

Observer initialization failures print:

```text
[ViewTrust] Training event observer initialization failed: <repr(error)>
[ViewTrust] Training event logging disabled.
```

Logging failures disable the observer by default. Strict failure mode is opt-in:

```text
VIEWTRUST_OBSERVER_STRICT=1
```

## Known Limitations

PR7 is global event logging only. It does not track per-Gaussian lifecycle IDs
or clone/split parent-child relationships. That belongs to PR8.

PR8 preserves these PR7 outputs and adds separate Gaussian lifecycle outputs
under `gaussian_lifecycle/` and `tables/gaussian_lifecycle_*.csv`.

PR9 consumes PR7 summaries and tables for no-op equivalence checks and the
Priority 0 report. It does not change PR7 logging behavior.
