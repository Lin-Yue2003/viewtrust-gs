# PR17 Clean-Prior Normalized Offline ViewTrust

PR17 adds an offline analysis layer that normalizes raw PR13 / PR16
ViewTrust risk by a per-scene clean prior. It is intended to separate stable
view identity prior from corruption-induced lift.

## Offline-only Guarantee

PR17 is offline analysis only.
It does not use corruption labels for scoring.
It does not implement a defense.
It does not reject views, suppress updates, reweight loss, or gate densification.
High normalized risk is post-hoc evidence, not proof of maliciousness.

## Inputs

PR17 consumes existing offline signal directories and their artifact manifests:

```text
offline_viewtrust_signals.csv
offline_viewtrust_artifact_manifest.csv
```

When `offline_viewtrust_artifact_manifest.csv` records `input_clean`,
`input_corrupt`, or `input_comparison` paths, PR17 resolves and reads those
paths. It does not assume copied input directories exist inside the offline
signal directory.

## Clean Prior

For each view, PR17 estimates `clean_prior_risk` from clean-side features.
Preferred source is the clean view influence table referenced by the artifact
manifest. If that is unavailable, PR17 falls back to clean-side columns already
present in `offline_viewtrust_signals.csv`.

The clean prior uses robust positive z-scores over clean loss, visibility,
birth, prune, and survival-anomaly features. Corruption labels are not part of
this computation.

## Formula

```text
delta_risk = raw_risk - clean_prior_risk
positive_delta_risk = max(0, delta_risk)
prior_suppressed_risk = raw_risk / (1 + clean_prior_risk)
rank_lift_score = clean_prior_rank - raw_rank

normalized_viewtrust_risk =
  positive_delta_risk
  + 0.25 * prior_suppressed_risk
  + 0.10 * max(0, rank_lift_score) / max(view_count - 1, 1)

normalized_consistency = 1 / (1 + normalized_viewtrust_risk)
```

The weights are configured in
`configs/offline_viewtrust_signal/default_pr17_clean_prior.json`.

## Command

```bash
python scripts/measure/analyze_clean_prior_normalized_viewtrust.py \
  --input-root outputs/reports \
  --plan-dir "$PR16_PLAN_ANALYSIS_DIR" \
  --output-dir "$PR17_DIR" \
  --scenes chair drums \
  --conditions corrupt_occluder corrupt_noise corrupt_mixed \
  --subset-names original seed_20260710 \
  --top-k 5 \
  --allow-missing \
  --write-markdown
```

## Outputs

```text
clean_prior_normalized_summary.json
clean_prior_normalized_rows.csv
clean_prior_normalized_rankings.csv
clean_prior_normalized_group_metrics.csv
clean_prior_normalized_ablation.csv
clean_prior_false_positive_reduction.csv
clean_prior_view_identity_diagnosis.csv
clean_prior_component_comparison.csv
clean_prior_missing_outputs.csv
clean_prior_report.md
artifact_manifest.csv
```

## Interpretation

PR17 compares raw top-k rankings against normalized top-k rankings. A useful
result reduces repeated false positives such as stable clean-prior views while
preserving most corrupted-view recall. It may also expose cases where raw risk
is mostly loss-only and cases where normalized risk better reflects
corruption-induced lift.

This is a prerequisite before any training-time intervention PR. It is not the
intervention.

## Known Limitations

- Clean priors are only as complete as the available clean-side features.
- Missing PR16 outputs are reported in `clean_prior_missing_outputs.csv`.
- `drums` original may remain missing until that PR16 cell is generated.
- The normalized score is transparent and conservative, not tuned with labels.
