# PR19.2 Exact Gaussian Logging Runner Integration

PR19.2 connects the PR19.1 stable Gaussian identity schema to the real
ViewTrust view influence runner:

```text
scripts/measure/build_view_influence_table.py
```

This runner already consumes real training artifacts:

- `tables/gaussian_lifecycle_events.csv`
- `tables/gaussian_lifecycle_final.csv`
- `tables/training_events.csv`
- corruption manifests

PR19.2 replays those real lifecycle rows into the PR19.1 exact Gaussian logging
schema when exact logging is explicitly enabled.

## Safety

Exact logging is disabled by default. PR19.2 does not modify `third_party`,
training behavior, rendering behavior, optimization, PR13 scoring, PR17
normalization, PR18 diagnosis, or PR19 scoring.

## New Runner Flags

```bash
--enable-exact-gaussian-logging
--exact-gaussian-log-dir <path>
--exact-gaussian-logging-config configs/offline_viewtrust_signal/default_pr191_exact_gaussian_logging.json
--exact-gaussian-run-id <optional string>
--subset-name <subset name>
```

Recommended placement:

```text
<view_influence_output_dir>/exact_gaussian_logging/
```

PR19 exact-mode discovery can inspect exact logs when they are placed under a
view influence directory.

## Example

```bash
python scripts/measure/build_view_influence_table.py \
  --run-dir "$RUN_DIR" \
  --data-root "$VIEWTRUST_DATA_ROOT" \
  --scene chair \
  --condition corrupt_occluder \
  --subset-name seed_20260710 \
  --output-dir "$VIEW_INFLUENCE_DIR" \
  --enable-exact-gaussian-logging \
  --exact-gaussian-log-dir "$VIEW_INFLUENCE_DIR/exact_gaussian_logging" \
  --exact-gaussian-logging-config configs/offline_viewtrust_signal/default_pr191_exact_gaussian_logging.json \
  --require-view-identity \
  --require-source-view \
  --write-markdown
```

Then validate:

```bash
python scripts/measure/validate_pr191_exact_gaussian_logging.py \
  --exact-log-dir "$VIEW_INFLUENCE_DIR/exact_gaussian_logging" \
  --output-dir "$PR192_VALIDATE_DIR" \
  --write-markdown
```

## Evidence Quality

The integration source is:

```text
real_view_influence_runner
```

Parent mapping source is reported as:

- `exact_clone_split_masks` when replayed lifecycle rows contain parent IDs.
- `partial` when parent mapping is missing or densification is unknown.

Proxy update/visibility observations are derived from existing training event
visibility counts. They are observation-only support hints, not optimization
changes.

## Outputs

The exact log directory contains:

- `gaussian_identity_table.csv`
- `gaussian_lifecycle_events.csv`
- `view_gaussian_event_attribution.csv`
- `gaussian_support_summary.csv`
- `exact_gaussian_logging_summary.json`
- `exact_gaussian_logging_validation.json`
- `artifact_manifest.csv`

The summary records `uses_row_index_as_stable_id = false`.

## Limitations

PR19.2 replays existing ViewTrust lifecycle artifacts. It does not insert a new
hook into official 3DGS internals. If a real run lacks parent IDs, the output is
marked `partial` rather than overclaimed as exact.
