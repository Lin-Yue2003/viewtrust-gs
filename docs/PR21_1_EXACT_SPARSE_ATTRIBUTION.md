# PR21.1 Exact Sparse Pixel-to-Gaussian Attribution Replay

PR21.1 is the first offline replay step that attempts to replace PR20 proxy
candidate pools with exact sparse contributors recovered from the installed
`gsplat` package. It remains observation-only and does not implement defense,
view rejection, loss reweighting, update suppression, densification gating,
training changes, rendering changes used by training, or `third_party` changes.

## Inputs

The CLI consumes:

```text
--run-dir   official 3DGS run root
--pr200-dir PR20 sparse residual/proxy attribution output
--pr210-dir PR21.0a gsplat feasibility output
--views     strict matched selected views
```

PR20 is used only to select sparse high-residual pixels and compare the old
proxy candidate pools against exact contributors. PR20 proxy rows are never
used as exact evidence.

PR21.0a is used to enforce strict selected-view camera matching. Every selected
view must have:

```text
strict_match = true
split_consistent = true
valid_for_exact_attribution = true
```

## CLI

```bash
python scripts/measure/run_pr211_exact_sparse_attribution.py \
  --run-dir "$RUN_DIR" \
  --pr200-dir "$PR200_DIR" \
  --pr210-dir "$PR210A_DIR" \
  --scene chair \
  --condition corrupt_occluder \
  --subset-name seed_20260710 \
  --iteration 700 \
  --split train \
  --views train_004 train_009 train_012 train_017 train_014 train_013 \
  --output-dir "$PR211_DIR" \
  --device cuda:0 \
  --top-pixels-per-view 128 \
  --max-contributors-per-pixel 16 \
  --write-markdown
```

## Replay Method

The server path attempts to:

1. Read official `point_cloud.ply`.
2. Convert official 3DGS parameters to gsplat tensors:
   - means from `x, y, z`
   - scales with `exp(scale_*)`
   - opacities with `sigmoid(opacity)`
   - rotations by quaternion normalization
   - colors from SH DC fields for replay compatibility
3. Build camera intrinsics/extrinsics from strict PR21.0a matched cameras.
4. Run `gsplat.rasterization(..., packed=True)`.
5. Audit returned metadata keys.
6. Resolve a valid transmittance tensor for `rasterize_to_indices_in_range`.
7. Use `rasterize_to_indices_in_range` when available to recover sparse
   contributor IDs for selected pixels.

If contributor IDs cannot be retrieved, PR21.1 writes blockers and
`exact_attribution_succeeded = false`. It does not fabricate exact rows and
does not silently fall back to PR20 proxy rows.

## Transmittance Resolution

PR21.1a fixes sparse contributor extraction for gsplat APIs that require an
explicit `transmittances` argument. Candidate sources are:

```text
metadata.transmittances
metadata.transmittance
metadata.T
metadata.final_T
metadata.render_transmittances
rasterization_output_1
```

A candidate is selected only if shape/device checks pass and a small dry-run
call to `rasterize_to_indices_in_range` succeeds with non-empty contributor
tensors. Render alpha is not blindly treated as transmittance. If no candidate
passes, PR21.1a writes a `gsplat_transmittance_resolution` blocker and keeps
exact rows empty.

The contributor API call uses the explicit gsplat 1.5.3-style argument set:

```text
range_start
range_end
transmittances
means2d
conics
opacities
image_width
image_height
tile_size
isect_offsets
flatten_ids
```

## Outputs

PR21.1 writes:

```text
pr211_exact_sparse_attribution_summary.json
pr211_input_readiness_audit.csv
pr211_checkpoint_activation_audit.csv
pr211_selected_pixels.csv
pr211_gsplat_metadata_audit.csv
pr211_gsplat_contributor_api_audit.csv
pr211_transmittance_audit.csv
pr211_gsplat_rasterization_output_audit.csv
pr211_exact_pixel_gaussian_contributions.csv
pr211_gaussian_residual_attribution_exact.csv
pr211_view_group_residual_attribution_exact.csv
pr211_direct_collateral_exact_overlap.csv
pr211_train013_exact_control.csv
pr211_exact_vs_proxy_comparison.csv
pr211_weight_nonuniformity_audit.csv
pr211_missing_fields.csv
pr211_blockers.csv
pr211_recommendations.json
pr211_report.md
artifact_manifest.csv
```

`ready_for_intervention` must remain `false`, even when exact attribution
succeeds.

## Local Smoke

The local smoke test does not require GPU or real `gsplat`. It injects
synthetic exact contributor rows to test aggregation, direct/collateral overlap,
train013 selected-pixel controls, exact-vs-proxy comparison, failure behavior,
and artifact manifests:

```bash
python scripts/smoke/pr211_exact_sparse_attribution_smoke_test.py
```

## Interpretation

If `exact_attribution_succeeded = true`, PR21.1 has recovered exact sparse
contributors for selected pixels and the next step is PR21.2 exact-vs-proxy
failure analysis. This still does not justify intervention.

If `exact_attribution_succeeded = false`, inspect `pr211_blockers.csv` and
`pr211_gsplat_metadata_audit.csv`; fix sparse contributor extraction before
comparison.

When exact sparse replay fails, exact-vs-proxy rows report
`exact unavailable due to failed sparse replay` instead of treating empty exact
rows as a real difference from PR20 proxy candidates. Direct/collateral and
train013 tables also mark exact evidence as unavailable rather than claiming no
overlap.
