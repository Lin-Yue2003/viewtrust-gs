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
