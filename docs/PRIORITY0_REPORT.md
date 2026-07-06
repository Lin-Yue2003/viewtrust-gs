# Priority 0 Report

PR9 builds a consolidated report for one Priority 0 clean baseline run.

The report is read-only. It collects existing run artifacts and writes a
machine-readable summary, a human-readable Markdown report, and an artifact
manifest.

## Run

```bash
python scripts/measure/build_priority0_report.py \
  --run-dir "$OBSERVED_RUN_DIR" \
  --output-dir outputs/reports/priority0_report_$(date +%Y%m%dT%H%M%S) \
  --include-view-metrics \
  --include-training-events \
  --include-gaussian-lifecycle \
  --require-priority0-complete \
  --write-markdown
```

Outputs:

```text
priority0_report_summary.json
priority0_report.md
priority0_artifact_manifest.csv
```

The artifact manifest records:

```text
relative_path
exists
file_type
size_bytes
required
artifact_group
description
```

## Scope

PR9 does not add trust scores, defenses, poison detection, corruption
conditions, or training-time intervention. It consolidates Priority 0 artifacts
so later clean/corrupt/poison comparisons have a stable baseline report.

## PR10 Context

PR10 adds natural corruption condition generation as input preparation for
future clean-vs-corrupt observation runs. The generated condition manifests and
inspection summaries are dataset artifacts, not training interventions.

Future report extensions can reference PR10 condition summaries when comparing
clean and corrupt runs. PR10 itself does not modify the PR9 no-op equivalence
report, training observers, or trainer behavior.

## PR11 Clean-vs-Corrupt Reports

PR11 adds a separate comparison report:

```bash
python scripts/measure/compare_clean_corrupt_observations.py \
  --clean-run-dir "$CLEAN_RUN_DIR" \
  --corrupt-run-dir "$CORRUPT_RUN_DIR" \
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
```

This report compares natural corruption observations against clean training.
It does not claim detection, does not classify views, and does not implement a
trust score or defense.
