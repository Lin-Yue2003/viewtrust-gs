# View Influence Attribution

PR12 adds observation-only view-to-Gaussian influence attribution.

It links sampled training views to training-event rows and Gaussian lifecycle
birth/prune contexts. It does not compute trust scores, classify views, defend
against corruptions, suppress updates, gate densification, change loss, change
optimizer behavior, change rendering, or change camera sampling.

## What PR12 Measures

PR12 asks which sampled views are temporally associated with:

```text
iteration metrics
densification contexts
Gaussian clone and split births
Gaussian prune deaths
final birth survival
```

This is source-view attribution, not exact dense per-pixel contribution
attribution. A lifecycle event is associated with the sampled view active
during the densification/pruning context; this does not prove that the view
alone caused the event.

## Patch Order

Apply observation patches on the server in this order:

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
```

PR12 adds sampled view identity to `training_events.csv` and stamps PR8
lifecycle birth/prune events with source-view context.

## Split-Correct Training Protocol

PR12.1 makes the clean/corrupt baseline wrapper pass official 3DGS `--eval` by
default. For Blender/NeRF Synthetic scenes this keeps test cameras held out
instead of merging them into the training camera pool.

New PR12.1 runs should validate train-only sampling:

```bash
python scripts/measure/inspect_training_events.py \
  --run-dir "$RUN_DIR" \
  --require-events \
  --require-view-identity \
  --require-train-only-sampling
```

Older PR12 runs created without `--eval` may contain sampled `test_*` views in
`training_events.csv`. Those runs can still be inspected as historical
artifacts, but they should not be used as split-correct evidence.

## Build View Influence Tables

```bash
python scripts/measure/build_view_influence_table.py \
  --run-dir "$CORRUPT_RUN_DIR" \
  --data-root "$VIEWTRUST_DATA_ROOT" \
  --scene chair \
  --condition corrupt_occluder \
  --output-dir outputs/reports/view_influence_corrupt_occluder_$(date +%Y%m%dT%H%M%S) \
  --require-view-identity \
  --require-source-view \
  --progress-interval-rows 50000 \
  --write-markdown
```

Outputs:

```text
view_influence_summary.json
view_influence_report.md
view_influence.csv
view_lifecycle_attribution.csv
view_iteration_events.csv
view_influence_artifact_manifest.csv
```

`view_influence_summary.json` includes `runtime_s`, per-stage `timing`,
`input_rows`, throughput estimates, observation-only source fields, and
split-aware sampled-view counts. Use `--quiet` to suppress lifecycle progress
logs during scripted runs.

## Compare Clean and Corrupt Influence

```bash
python scripts/measure/compare_view_influence_tables.py \
  --clean-view-influence-dir "$CLEAN_VIEW_INFLUENCE_DIR" \
  --corrupt-view-influence-dir "$CORRUPT_VIEW_INFLUENCE_DIR" \
  --output-dir outputs/reports/view_influence_clean_vs_corrupt_occluder_$(date +%Y%m%dT%H%M%S) \
  --write-markdown
```

Outputs:

```text
view_influence_comparison_summary.json
view_influence_comparison.csv
view_influence_comparison_report.md
```

## Offline Signal Design

PR13 consumes split-correct PR12.1 view influence tables to build offline
candidate ViewTrust signals:

```bash
python scripts/measure/build_offline_viewtrust_signals.py \
  --clean-view-influence-dir "$CLEAN_VIEW_INFLUENCE_DIR" \
  --corrupt-view-influence-dir "$CORRUPT_VIEW_INFLUENCE_DIR" \
  --view-influence-comparison-dir "$VIEW_INFLUENCE_COMPARE_DIR" \
  --output-dir outputs/reports/offline_viewtrust_corrupt_occluder_pr13_$(date +%Y%m%dT%H%M%S) \
  --write-markdown \
  --top-k 5
```

PR13 remains offline and observation-only. It does not implement a training-time
trust score, defense, poison classifier, loss reweighting, update suppression,
or densification gating. Corruption labels are used only after scoring for
post-hoc evaluation.

## Interpretation

Use these reports as exploratory evidence for later offline trust signal
design. Preferred language:

```text
view-associated lifecycle influence
corruption-associated lifecycle shift
observation-only attribution
evidence for later trust signal design
```

Avoid language such as poison detection, bad-view detection, defense success,
or trust score.

## Known Limitations

PR12 attribution is temporal/source-view attribution. It does not prove
causality and does not classify any view as trustworthy or untrustworthy.

The PR12.1 table builder streams lifecycle events and stores grouped counters
instead of repeatedly rescanning every lifecycle row. This is intended to keep
the first 700-iteration chair runs tractable while preserving the existing CSV
schemas.
