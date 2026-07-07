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
