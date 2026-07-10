# PR20.1 Proxy Degeneracy Diagnosis

PR20.1 audits whether PR20.0 sparse residual attribution is informative or
degenerates into fixed view-level candidate pools with uniform weights.

It is observation-only. It does not implement defense, view rejection, update
suppression, loss reweighting, densification gating, exact render contribution,
or training/rendering behavior changes.

## Command

Single PR20.0 directory:

```bash
python scripts/measure/analyze_pr201_proxy_degeneracy.py \
  --pr200-dir "$PR200_DIR" \
  --scene chair \
  --condition corrupt_occluder \
  --subset-name seed_20260710 \
  --output-dir "$PR201_DIR" \
  --top-k 16 \
  --write-markdown
```

Aggregate:

```bash
python scripts/measure/analyze_pr201_proxy_degeneracy.py \
  --pr200-dir "$PR200_CHAIR_DIR" \
  --pr200-dir "$PR200_DRUMS_DIR" \
  --output-dir "$PR201_AGG_DIR" \
  --top-k 16 \
  --write-markdown
```

## Diagnostics

PR20.1 checks:

- whether each selected pixel in a view reuses the same candidate Gaussian set
- whether `splat_weight` and `alpha_contribution` are uniform
- whether direct/collateral overlap is explained by shared proxy candidate
  pools
- whether train_013 control is only candidate-pool separation

The expected current evidence quality is:

```text
approximate_projected_gaussian
```

The expected current attribution method is:

```text
view_event_weighted_gaussian_proxy
```

## Outputs

- `pr201_proxy_degeneracy_summary.json`
- `pr201_run_summary.csv`
- `pr201_pixel_candidate_reuse.csv`
- `pr201_view_candidate_pool.csv`
- `pr201_view_candidate_pool_overlap.csv`
- `pr201_candidate_weight_uniformity.csv`
- `pr201_group_candidate_pool_audit.csv`
- `pr201_direct_collateral_degeneracy.csv`
- `pr201_train013_proxy_control_audit.csv`
- `pr201_proxy_failure_cases.csv`
- `pr201_recommendations.json`
- `pr201_missing_inputs.csv`
- `pr201_report.md`
- `artifact_manifest.csv`

## Interpretation

If proxy degeneracy is confirmed, PR20.0 is still useful for:

- input/output pipeline validation
- residual selection sanity checks
- candidate-pool diagnostics
- motivating exact attribution

It is not safe for:

- causal Gaussian artifact localization
- view rejection
- densification gating
- training intervention
- claiming exact render contribution

Recommended next PR:

```text
PR21.0: gsplat feasibility and exact sparse pixel-to-Gaussian attribution replay
```
