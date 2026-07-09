# PR20.0 Sparse Render Attribution

PR20.0 adds an observation-only sparse residual and Gaussian attribution
pipeline. It is not a defense and does not reject views, reweight losses,
suppress updates, gate densification, or modify training/rendering behavior.

## Purpose

PR19.5 showed that current exact lifecycle logs provide event-weighted context
but not enough attribution semantics for intervention. PR20.0 starts closing
that gap by:

- selecting high-residual pixels from rendered/ground-truth image pairs
- attaching candidate Gaussian IDs to those pixels
- aggregating residual-weighted attribution by Gaussian and view group
- measuring direct/collateral residual overlap
- checking train_013 residual control behavior

## Attribution Method

The initial implementation avoids `third_party` changes. It uses existing
rendered/ground-truth images and PR19.3 view-Gaussian attribution rows. When
true per-pixel splat contributors are unavailable, output is labeled:

```text
evidence_quality = approximate_projected_gaussian
attribution_method = view_event_weighted_gaussian_proxy
```

This must not be interpreted as exact render contribution. Exact contribution
would require sparse renderer internals or another observation-only renderer
wrapper that exposes per-pixel top-k Gaussian contributors.

## Command

```bash
python scripts/measure/build_pr200_sparse_render_attribution.py \
  --pr193-dir "$PR193_DIR" \
  --pr195-dir "$PR195_DIR" \
  --run-dir "$RUN_DIR" \
  --data-root "$VIEWTRUST_DATA_ROOT" \
  --scene chair \
  --condition corrupt_occluder \
  --subset-name seed_20260710 \
  --output-dir "$PR200_DIR" \
  --top-pixels 512 \
  --top-gaussians-per-pixel 16 \
  --residual-metric l1 \
  --artifact-mask-mode top_residual \
  --write-markdown
```

## Outputs

- `pr200_sparse_render_attribution_summary.json`
- `pr200_selected_views.csv`
- `pr200_view_residual_summary.csv`
- `pr200_sparse_pixel_residuals.csv`
- `pr200_pixel_gaussian_contributions.csv`
- `pr200_gaussian_residual_attribution.csv`
- `pr200_view_group_residual_attribution.csv`
- `pr200_direct_collateral_residual_overlap.csv`
- `pr200_train013_residual_control.csv`
- `pr200_attribution_quality_audit.csv`
- `pr200_missing_inputs.csv`
- `pr200_report.md`
- `artifact_manifest.csv`

## Safety

The summary must keep:

```text
observation_only = true
training_intervention = false
defense_enabled = false
view_rejection_enabled = false
densification_gating_enabled = false
third_party_modified = false
training_behavior_modified = false
rendering_behavior_modified_for_training = false
pr20_ready_for_intervention = false
```

Even if residual overlap is found, PR20.0 remains evidence collection only. A
later PR must review evidence quality before any intervention is considered.
