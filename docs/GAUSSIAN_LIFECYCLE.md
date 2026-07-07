# Gaussian Lifecycle

PR8 adds observation-only Gaussian lifecycle logging for official Gaussian
Splatting clean baseline runs.

Lifecycle IDs are non-trainable metadata. They are not optimizer parameters,
are not rendered, do not receive gradients, and do not affect loss, sampling,
densification, pruning, opacity reset, or rendering decisions.

## Patch Strategy

PR8 is a separate marker-delimited patch that is applied after PR7 training
events. This avoids the confusing state where an already-patched PR7 checkout
cannot be upgraded. The patch generator updates:

```text
third_party/gaussian-splatting/train.py
third_party/gaussian-splatting/scene/gaussian_model.py
```

Patched `third_party` source is server-local state and is not committed to
ViewTrust-GS.

Apply/check sequence:

```bash
python scripts/third_party/apply_gaussian_splatting_observation_patch.py \
  --third-party-root ./third_party \
  --patch pr7_training_events

python scripts/third_party/apply_gaussian_splatting_observation_patch.py \
  --third-party-root ./third_party \
  --patch pr8_gaussian_lifecycle

python scripts/third_party/apply_gaussian_splatting_observation_patch.py \
  --third-party-root ./third_party \
  --patch pr12_view_influence_attribution

python scripts/third_party/check_gaussian_splatting_observation_patch.py \
  --third-party-root ./third_party \
  --patch pr8_gaussian_lifecycle \
  --require-applied
```

If PR7 is already applied on the server, skip the PR7 apply command and run the
PR8 apply command directly.

## Enable

```bash
python scripts/train/run_clean_chair_baseline.py \
  --trainer gaussian-splatting \
  --data-root "$VIEWTRUST_DATA_ROOT" \
  --third-party-root ./third_party \
  --output-root ./outputs \
  --scene chair \
  --condition clean \
  --iterations 700 \
  --gpu 0 \
  --sample-interval-s 1.0 \
  --enable-training-events \
  --training-event-log-interval 10 \
  --training-event-strict \
  --enable-gaussian-lifecycle \
  --gaussian-lifecycle-strict
```

The wrapper injects the project root into the child trainer `PYTHONPATH` and
preflights `viewtrust.observation.gaussian_lifecycle` before training starts.

## Outputs

```text
gaussian_lifecycle/
  gaussian_lifecycle_events.csv
  gaussian_lifecycle_final.csv
  gaussian_lifecycle_summary.json
  gaussian_lifecycle_warnings.jsonl
tables/
  gaussian_lifecycle_events.csv
  gaussian_lifecycle_final.csv
gaussian_lifecycle_summary.json
```

`gaussian_lifecycle_events.csv` records birth and prune/death events.
`gaussian_lifecycle_final.csv` records one row per known Gaussian ID.

PR12 adds source-view context fields to lifecycle event rows:

```text
source_iteration
source_view_name
source_camera_uid
source_view_split
event_context
```

Birth events use `event_context=densification_after_view` and prune/death
events use `event_context=prune_after_view` when a sampled-view context is
available.

## Invariants

The lifecycle observer checks:

```text
len(lifecycle_ids) == current_gaussian_count
all alive lifecycle IDs are unique
every alive ID exists in lifecycle state
no dead ID remains alive
alive_final_count == final_gaussian_count
dead_final_count + alive_final_count == known_gaussian_count
```

The inspector also validates:

```text
gaussian_lifecycle_final.csv row count == known_gaussian_count
alive rows have final_index populated
dead rows have death_iteration populated
lifetime_iterations >= 0
no duplicate gaussian_id
no duplicate final_index among alive rows
```

## Inspect

```bash
python scripts/measure/inspect_gaussian_lifecycle.py \
  --run-dir "$RUN_DIR" \
  --require-lifecycle \
  --require-no-invariant-violations \
  --require-source-view
```

## Lineage

PR8 records clone/split parent IDs when the official trainer exposes a safe
source mask at the hook point. If exact mapping is unavailable, new Gaussians
receive unique IDs with `birth_type=densification_unknown` and an explicit
warning. PR8 does not implement trust scores, view attribution, poison
detection, or defenses.

PR9 consumes lifecycle summaries and tables for no-op equivalence checks and
the Priority 0 report. PR12 consumes source-view lifecycle context for
observation-only view influence attribution.
