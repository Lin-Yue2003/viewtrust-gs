# PR19.3 Exact View-Group Binding

PR19.3 is an offline, post-hoc binding step. It takes PR19.2 exact Gaussian
logs and attaches PR17/PR18 view-group semantics:

- `direct_corrupted`
- `co_visible_collateral`
- `clean_prior_demoted`
- `other_clean`

It does not rerun training, modify rendering, change PR13/PR17/PR18 scoring, or
change PR19 risk scoring. Corruption labels are used only for grouping and
evaluation, not for scoring.

## Inputs

```bash
python scripts/measure/bind_pr193_exact_view_groups.py \
  --exact-log-dir "$PR192_EXACT_DIR" \
  --pr17-dir "$PR17_DIR" \
  --pr18-dir "$PR18_DIR" \
  --scene chair \
  --condition corrupt_occluder \
  --subset-name seed_20260710 \
  --top-k 20 \
  --output-dir "$PR193_DIR" \
  --copy-pr19-ready-bundle \
  --write-markdown
```

`--exact-log-dir` must contain the PR19.2 exact files:

- `gaussian_identity_table.csv`
- `gaussian_lifecycle_events.csv`
- `view_gaussian_event_attribution.csv`
- `gaussian_support_summary.csv`
- `exact_gaussian_logging_summary.json`
- `exact_gaussian_logging_validation.json`
- `artifact_manifest.csv`

`--pr17-dir` must contain `clean_prior_normalized_rows.csv`.
`--pr18-dir` must contain `pr18_spillover_classification.csv`; transition
rows are used when `pr18_view_identity_transition.csv` is present.

## View Group Priority

The binding priority is:

1. `direct_corrupted` when PR17 marks `was_corrupted = true`.
2. `co_visible_collateral` when PR18 marks `spillover_class = co_visible_collateral`.
3. `clean_prior_demoted` for PR18 `clean_prior_false_positive` or
   `prior_demoted`, PR18 transition demotion, or the known train_013 pattern:
   raw false positive, normalized false positive false, and not corrupted.
4. `other_clean` for remaining non-corrupted views.

A corrupted view is never overwritten as collateral or clean-prior-demoted.

## Outputs

The output directory contains:

- `pr193_view_group_map.csv`
- `pr193_view_group_binding_summary.json`
- grouped exact tables
- exact direct/collateral overlap diagnostics
- train_013 exact control diagnostics
- artifact manifests
- `pr193_report.md`

Grouped identity and support tables explicitly separate unique-view support
counts from event-weighted counts. This avoids the ambiguity in earlier support
tables where group counts could be interpreted as either unique views or event
weights.

## PR19-Ready Bundle

With `--copy-pr19-ready-bundle`, PR19.3 writes:

```text
pr19_exact_input_bundle/
  exact_gaussian_logging/
    gaussian_identity_table.csv
    gaussian_lifecycle_events.csv
    view_gaussian_event_attribution.csv
    gaussian_support_summary.csv
    exact_gaussian_logging_summary.json
    exact_gaussian_logging_validation.json
    artifact_manifest.csv
  pr193_view_group_map.csv
  pr193_view_group_binding_summary.json
  pr193_pr19_exact_input_bundle_manifest.csv
```

The exact files use PR19-known filenames, but their rows include PR19.3
view-group bindings. PR19 input-resolution/scoring is not changed in PR19.3.

## Interpretation

PR19.3 succeeds when it shows whether direct corrupted views and PR18
co-visible collateral views support overlapping exact Gaussian IDs, and whether
train_013 remains a clean-prior-demoted control at exact Gaussian ID level.

This is still observation-only. It does not justify a defense or intervention
until PR19 exact-mode analysis is rerun and reviewed.
