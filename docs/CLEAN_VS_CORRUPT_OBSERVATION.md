# Clean-vs-Corrupt Observation

PR11 compares a clean observed 3DGS run with a natural-corruption observed run.

This is evidence gathering only. It does not implement ViewTrust scoring,
defense logic, poison detection, camera sampling changes, rendering changes,
loss changes, optimizer changes, densification changes, pruning changes, or
training-time intervention.

## Purpose

PR11 asks:

```text
How do natural non-malicious corruptions change 3DGS training dynamics and
Gaussian lifecycle compared with clean training?
```

The comparison uses PR7 training event logs and PR8 Gaussian lifecycle logs.
It should not be read as corruption detection.

## Required Inputs

Clean observed run:

```text
outputs/baseline/chair_clean_gaussian_splatting/<run_id>/
```

Corrupt observed run, for example:

```text
outputs/baseline/chair_corrupt_occluder_gaussian_splatting/<run_id>/
```

Both runs should ideally contain:

```text
training_events_summary.json
gaussian_lifecycle_summary.json
tables/training_events.csv
tables/densification_events.csv
tables/gaussian_count_timeseries.csv
tables/gaussian_lifecycle_events.csv
tables/gaussian_lifecycle_final.csv
```

View metrics are optional for the comparison script. If available, add:

```text
view_metrics_summary.json
tables/view_metrics.csv
tables/view_render_artifacts.csv
```

## Train a Corrupt Condition

The existing baseline wrapper accepts any prepared condition via
`--condition`. The script name remains historical.

```bash
python scripts/train/run_clean_chair_baseline.py \
  --trainer gaussian-splatting \
  --data-root "$VIEWTRUST_DATA_ROOT" \
  --third-party-root ./third_party \
  --output-root ./outputs \
  --scene chair \
  --condition corrupt_occluder \
  --iterations 700 \
  --gpu 0 \
  --sample-interval-s 1.0 \
  --enable-training-events \
  --training-event-log-interval 10 \
  --training-event-strict \
  --enable-gaussian-lifecycle \
  --gaussian-lifecycle-strict
```

## Compare Runs

```bash
python scripts/measure/compare_clean_corrupt_observations.py \
  --clean-run-dir "$CLEAN_RUN_DIR" \
  --corrupt-run-dir "$CORRUPT_RUN_DIR" \
  --data-root "$VIEWTRUST_DATA_ROOT" \
  --scene chair \
  --corruption-condition corrupt_occluder \
  --output-dir outputs/reports/clean_vs_corrupt_occluder_$(date +%Y%m%dT%H%M%S) \
  --require-observation-invariants \
  --write-markdown
```

Outputs:

```text
clean_vs_corrupt_summary.json
clean_vs_corrupt_report.md
clean_vs_corrupt_metrics.csv
clean_vs_corrupt_artifact_manifest.csv
view_corruption_effects.csv
```

PR11.2 resolves corruption manifests from the corrupt run metadata
`prepared_scene_root`, explicit `--corrupt-condition-root`, or
`--data-root --scene --corruption-condition`. When per-view metrics are
available, `view_corruption_effects.csv` joins clean/corrupt view metrics with
`corruption_manifest.csv`.

## Report Interpretation

The report measures changes in observed training dynamics and Gaussian
lifecycle under a natural corruption condition. It does not classify views as
trustworthy or untrustworthy.

Avoid interpreting larger deltas as detection. PR11 is a measurement layer for
future ViewTrust signal design.

## Known Limitations

Only `corrupt_occluder` is required for first server validation, though the full
six-condition PR10 suite is supported.

PR11 does not implement trust scores, defenses, poison attacks, or
training-time interventions.
