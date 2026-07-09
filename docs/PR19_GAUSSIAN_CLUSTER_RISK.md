# PR19 Gaussian Cluster Risk

PR19 is offline observation only. It is not a defense, does not reject views,
does not reweight losses, does not suppress Gaussian updates, and does not gate
densification or pruning.

PR19 does not tune PR17 clean-prior normalization or PR18 co-visibility
spillover diagnosis. It consumes those outputs to ask a narrower question:
whether direct corrupted views and co-visible collateral views concentrate on
the same Gaussian lifecycle or support patterns, while clean-prior demoted views
such as `train_013` remain separated.

## Evidence Levels

PR19 reports one evidence level per scene/subset/condition:

- `exact_gaussian_id`: existing artifacts contain exact per-Gaussian IDs such
  as `gaussian_id`, `parent_id`, `child_gaussian_id`, `source_gaussian_id`,
  `gaussian_ids`, or `affected_gaussian_ids`.
- `aggregate_event_proxy`: exact IDs are unavailable, but aggregate lifecycle,
  view influence, visibility delta, or comparison rows are available.
- `unavailable`: required influence files cannot be resolved.

If only `aggregate_event_proxy` is available, PR19 supports
representation-level suspicion but does not prove exact Gaussian causal overlap.

## Inputs

PR19 reads:

- PR17 `clean_prior_normalized_rows.csv`
- PR18 `pr18_spillover_classification.csv`
- PR16 / PR13 offline signal directories under `outputs/reports`
- `offline_viewtrust_artifact_manifest.csv` to resolve input influence dirs
- `view_influence.csv`
- `view_lifecycle_attribution.csv`
- `view_iteration_events.csv`
- `view_influence_comparison.csv`

Corruption labels are used only for post-hoc grouping and evaluation. They are
not used to tune risk weights or compute cluster risk.

## View Groups

PR19 assigns each view to:

- `direct_corrupted`
- `co_visible_collateral`
- `clean_prior_demoted`
- `other_clean`

The `co_visible_collateral` group comes from PR18. The `clean_prior_demoted`
group captures views like `train_013` that had raw risk but were demoted by
clean-prior normalization.

## Risk Formula

Exact Gaussian mode:

```text
gaussian_cluster_risk =
  0.30 * source_concentration_score
+ 0.25 * corrupted_plus_collateral_ratio
+ 0.20 * lifecycle_instability_score
+ 0.15 * weak_support_score
+ 0.10 * visibility_delta_score
```

Aggregate event proxy mode:

```text
event_cluster_risk =
  0.30 * source_group_concentration_score
+ 0.25 * corrupted_plus_collateral_event_ratio
+ 0.20 * lifecycle_instability_score
+ 0.15 * clean_vs_corrupt_delta_score
+ 0.10 * visibility_delta_score
```

Lifecycle, weak-support, visibility, and delta components are robustly
normalized within each scene/subset/condition. High risk means a candidate
representation-risk region, not proof of maliciousness.

## Command

```bash
python scripts/measure/analyze_pr19_gaussian_cluster_risk.py \
  --input-root outputs/reports \
  --plan-dir "$PR16_PLAN_ANALYSIS_DIR" \
  --pr17-dir "$PR17_DIR" \
  --pr18-dir "$PR18_DIR" \
  --output-dir "$PR19_DIR" \
  --scenes chair drums \
  --conditions corrupt_occluder corrupt_noise corrupt_mixed \
  --subset-names original seed_20260710 \
  --top-k 20 \
  --allow-missing \
  --write-markdown
```

## Outputs

PR19 writes:

- `pr19_gaussian_cluster_risk_summary.json`
- `pr19_evidence_availability.csv`
- `pr19_view_group_map.csv`
- `pr19_cluster_risk_rows.csv`
- `pr19_cluster_risk_rankings.csv`
- `pr19_group_concentration_summary.csv`
- `pr19_direct_collateral_overlap.csv`
- `pr19_train013_control_summary.csv`
- `pr19_intervention_candidate_preview.csv`
- `pr19_missing_outputs.csv`
- `pr19_report.md`
- `artifact_manifest.csv`

The intervention preview is not executable intervention logic. Every row has
`do_not_apply_intervention = true`.

## Interpretation

PR19 is successful when it shows whether direct corrupted and collateral views
share top-ranked exact Gaussian clusters or aggregate event clusters, and
whether `train_013` remains a clean-prior control with low overlap.

If direct corrupted and collateral views do not overlap in cluster/event
patterns, that is a valid result: it suggests PR18 collateral may be mostly
camera-neighbor spillover rather than representation-level support overlap.

PR19 is a prerequisite for future trust-aware densification or update control,
but it does not implement either.
