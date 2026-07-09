# PR19.5 Exact Attribution Semantics Audit

PR19.5 audits whether PR19.3 / PR19.4 exact logs are semantically strong
enough for PR19 exact-mode conclusions or PR20 intervention planning.

It is offline observation only. It does not change training, rendering,
`third_party`, PR17 / PR18 scoring, PR19.3 view-group binding, PR19.4 support
filtering, PR19 scoring, or defense behavior.

## Why This Exists

PR19.4 showed that broad exact support can be degenerate: direct corrupted and
co-visible collateral views may share the same final alive Gaussian set simply
because many views observe or update many final Gaussians.

PR19.5 separates diagnostic signals from usable exact evidence:

- `birth`, `prune`, and `low_entropy` may support train_013 control.
- They are still not valid PR19 exact modes unless they also show nontrivial
  direct/collateral exact overlap.
- A mode is usable for PR19 exact mode only when it has non-degenerate
  direct/collateral exact Gaussian overlap.

## Command

```bash
python scripts/measure/audit_pr195_exact_attribution_semantics.py \
  --pr193-dir "$PR193_DIR" \
  --pr194-dir "$PR194_DIR" \
  --scene chair \
  --condition corrupt_occluder \
  --subset-name seed_20260710 \
  --output-dir "$PR195_DIR" \
  --write-markdown
```

## Outputs

- `pr195_attribution_semantics_summary.json`
- `pr195_support_mode_failure_analysis.csv`
- `pr195_event_type_group_distribution.csv`
- `pr195_view_group_event_distribution.csv`
- `pr195_high_event_semantics_audit.csv`
- `pr195_suspicious_alive_degeneracy_audit.csv`
- `pr195_birth_prune_semantics_audit.csv`
- `pr195_train013_semantics_audit.csv`
- `pr195_required_attribution_field_gap.csv`
- `pr195_pr20_readiness_assessment.csv`
- `pr195_next_step_recommendation.md`
- `pr195_missing_inputs.csv`
- `pr195_report.md`
- `artifact_manifest.csv`

## Field Gap Audit

PR19.5 checks whether current exact logs include attribution fields needed for
stronger interpretation:

- sparse render contribution fields such as `pixel_x`, `pixel_y`,
  `alpha_contribution`, and `splat_weight`
- residual-weighted fields such as `residual_value` and `residual_l1`
- gradient/update fields such as `gradient_norm`, `delta_opacity`,
  `delta_scale`, and `delta_sh`
- localization flags such as `target_region_flag` and `artifact_region_flag`

When those are missing, PR19.5 should set the corrected recommendation to
`none` and block intervention readiness.

## Expected Current Verdict

Current exact lifecycle replay logs provide stable Gaussian IDs, lifecycle
context, view-group binding, and event-weighted context. They do not yet prove
causal artifact localization.

The recommended next PR is to implement sparse per-view per-Gaussian render
contribution and residual-weighted attribution before any intervention.
