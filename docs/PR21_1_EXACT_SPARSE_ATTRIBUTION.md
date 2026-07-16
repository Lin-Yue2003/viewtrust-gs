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
4. Run `gsplat.rasterization(..., packed=True)` for compatibility audit.
5. Audit returned metadata keys and high-level rasterization outputs.
6. Resolve whether a legacy high-level transmittance tensor exists for
   `rasterize_to_indices_in_range`.
7. Audit installed gsplat source to decide the contributor extraction path.
8. Run a source-verified internal loop with `packed=False` when possible:
   `transmittances = 1.0 - render_alphas[..., 0]`,
   `rasterize_to_indices_in_range(...)`, then `accumulate(...)` to update
   `render_alphas`.
9. Recover sparse contributor IDs for selected pixels without using PR20 proxy
   rows as exact evidence.

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

## Source-Guided Path Selection

PR21.1b adds runtime source audit and path decision artifacts:

```text
pr211_gsplat_source_audit.csv
pr211_contributor_path_decision.json
pr211_contributor_path_attempts.csv
```

The source audit inspects installed `gsplat` modules with `inspect.unwrap`,
records signatures for `rasterization`, `rasterize_to_indices_in_range`, and
`accumulate`, and searches package source for terms such as `render_alphas`,
`transmittances`, `isect_offsets`, and `flatten_ids`.

Candidate paths are recorded as:

```text
Path A: public rasterize_to_indices_in_range with a validated transmittance tensor
Path B: lower-level source-audited gsplat functions
Path C: exact contributor IDs from tile/intersection metadata plus footprint checks
Path D: source-level failure
```

If source evidence supports `render_alphas`, PR21.1b tests
`rasterization_output_1.squeeze(-1)` and
`1 - rasterization_output_1.squeeze(-1)`. A candidate is still selected only
after dry-run validation and contributor output checks.

If only pixel-level contributor IDs are recovered, PR21.1b labels rows as:

```text
evidence_quality = exact_sparse_contributor_id_only
attribution_method = gsplat_sparse_contributor_id_replay
```

Alpha, transmittance, and splat-weight fields remain empty in that mode.

## Source-Verified Internal Loop

PR21.1c uses the installed gsplat `_torch_impl.py` loop as implementation
evidence. The key finding is that `transmittances` is not a stable metadata key
or high-level rasterization output; it is computed inside the render loop as:

```text
transmittances = 1.0 - render_alphas[..., 0]
```

The PR21.1c replay therefore attempts `gsplat.rasterization(..., packed=False)`
for contributor recovery so the metadata shapes match the source assertions:

```text
means2d:       image_dims + (N, 2)
conics:        image_dims + (N, 3)
opacities:     image_dims + (N,)
colors:        image_dims + (N, channels)
isect_offsets: image_dims + (tile_height, tile_width)
render_alphas: image_dims + (H, W, 1)
transmittance: image_dims + (H, W)
```

PR21.1d removes the dependency on nerfacc for this ID-only path. The replay
collects contributor IDs immediately after `rasterize_to_indices_in_range`.
It first tries `gsplat.accumulate`; if that fails because `nerfacc_cuda` cannot
build or import, it falls back to a local source-verified alpha-only update that
uses the minimal gsplat formula:

```text
x = pixel_id % image_width
y = pixel_id // image_width
sigma = 0.5 * (c0 * dx * dx + c2 * dy * dy) + c1 * dx * dy
alpha = min(0.999, opacity * exp(-sigma))
acc_alpha = acc_alpha + (1 - acc_alpha) * alpha
```

The fallback updates `render_alphas` only to keep the internal loop's
transmittance state valid. It does not claim exact alpha, transmittance, splat
weight, color, or residual-weighted contribution values.

PR21.1e removes ambiguous multi-view `image_id` mapping from exact contributor
recovery. Instead of replaying all selected views in one camera batch and
assuming `image_id` maps to requested view order, it runs one selected view at a
time. In each one-view replay, only `image_id == 0` is accepted as exact
evidence and all accepted rows are mapped to the outer-loop `view_name`.
Nonzero image IDs and coordinate-transform candidates are written to
`pr211_per_view_replay_audit.csv` as diagnostics only.

If `packed=False` is unavailable or fails, the replay records a source-validated
packed attempt only if the shapes can be audited. It does not use proxy rows as
exact evidence. Successful PR21.1c/PR21.1d rows are intentionally ID-only:

```text
evidence_quality = exact_sparse_contributor_id_only
attribution_method = gsplat_internal_loop_contributor_id_replay
ready_for_intervention = false
```

Weighted alpha/transmittance/splat contribution is deferred until a later PR
source-verifies those scalar semantics.

## PR21.1f Drums Alignment Audit

PR21.1f diagnoses why drums remains excluded from PR21.2. It consumes PR20
selected-pixel proxy artifacts and PR21.1e drums per-view replay artifacts, then
writes an offline selected-pixel source alignment audit. It may reuse
`pr211_per_view_replay_audit.csv` for raw per-view contributor coverage and does
not rerun training or modify gsplat.

```bash
python scripts/measure/run_pr211f_drums_selected_pixel_alignment_audit.py \
  --run-dir "$RUN_DIR" \
  --pr200-dir "$PR200_DRUMS_DIR" \
  --pr211-dir "$PR211E_DRUMS_DIR" \
  --pr210-dir "$PR210A_DRUMS_DIR" \
  --scene drums \
  --condition corrupt_occluder \
  --subset-name seed_20260710 \
  --views train_004 train_009 train_012 train_017 train_007 train_013 \
  --output-dir "$PR211F_DRUMS_DIR" \
  --device cuda:0 \
  --top-pixels-per-view 128 \
  --max-contributors-per-pixel 16 \
  --write-markdown
```

The PR21.1f outputs are:

```text
pr211f_drums_selected_pixel_alignment_summary.json
pr211f_drums_pr20_selected_pixel_audit.csv
pr211f_drums_exact_replay_raw_pixel_coverage.csv
pr211f_drums_coordinate_convention_audit.csv
pr211f_drums_residual_source_alignment_audit.csv
pr211f_drums_top_residual_crosscheck.csv
pr211f_drums_alignment_diagnosis.csv
pr211f_drums_source_search_paths.csv
pr211f_drums_source_file_inventory.csv
pr211f_drums_selected_pixel_alignment_report.md
artifact_manifest.csv
```

Flip, swap, and neighborhood hits are diagnostic-only. PR21.1f does not emit
exact contributor rows, does not use PR20 proxy rows as exact rows, and keeps
`exact_evidence_allowed_for_drums = false` unless a future PR proves normal
coordinate-aligned exact evidence.

PR21.1f-a strengthens the drums diagnosis. If some selected views have raw
contributors and non-normal diagnostic hits while other selected views have no
raw contributors, the summary reports
`likely_failure_mode_overall = mixed_coordinate_candidate_and_no_raw_contributors`
instead of reducing the whole run to `exact_replay_has_no_raw_contributors`.
The summary also records the views with raw contributors, without raw
contributors, with normal hits, and with diagnostic hits.

PR21.1f-a also writes a source discovery audit. `pr211f_drums_source_search_paths.csv`
lists every searched root and candidate counts, while
`pr211f_drums_source_file_inventory.csv` lists likely render, GT, residual,
selected-pixel, config, and audit candidates. Candidate paths are discovery
evidence only; they do not verify that a file is the exact PR20 residual source.

## PR21.1g PR20 Selected-Pixel Provenance

PR21.1g traces whether the drums selected pixels in
`pr200_pixel_gaussian_contributions.csv` can be reproduced from PR20's own
residual CSV artifacts and source-code path. It is a provenance audit only.
It does not alter PR20 outputs, PR21 exact replay, training, rendering,
`third_party`, or installed packages.

```bash
python scripts/measure/run_pr211g_pr20_selected_pixel_provenance_audit.py \
  --pr200-dir "$PR200_DRUMS_DIR" \
  --pr211f-dir "$PR211FA_DRUMS_DIR" \
  --pr211-dir "$PR211E_DRUMS_DIR" \
  --run-dir "$PR200_DRUMS_RUN_ROOT" \
  --scene drums \
  --condition corrupt_occluder \
  --subset-name seed_20260710 \
  --views train_004 train_009 train_012 train_017 train_007 train_013 \
  --output-dir "$PR211G_DRUMS_DIR" \
  --top-pixels-per-view 128 \
  --max-contributors-per-pixel 16 \
  --write-markdown
```

PR21.1g writes:

```text
pr211g_pr20_selected_pixel_provenance_summary.json
pr211g_pr20_selected_from_proxy_contributions.csv
pr211g_pr20_residual_csv_schema_audit.csv
pr211g_pr20_residual_to_selected_reproduction.csv
pr211g_pr20_selected_pixel_membership_in_residual_csv.csv
pr211g_pr20_code_provenance_audit.csv
pr211g_pr20_code_provenance_summary.json
pr211g_pr20_pixel_set_hash_comparison.csv
pr211g_pr20_selected_pixel_provenance_diagnosis.csv
pr211g_pr20_selected_pixel_provenance_report.md
artifact_manifest.csv
```

Even if PR20 selected-pixel provenance is verified, drums remains excluded from
PR21.2 unless PR21 exact replay also validates normal-coordinate selected-pixel
hits. PR20 proxy rows are not exact contributor rows.

## PR21.3 Chair Exact Evidence Positioning

PR21.3 returns to the chair-only exact-evidence line and writes a
research-facing interpretation package. It summarizes what PR20/PR20.1 proxy
evidence claimed, what PR21.1e made possible, what PR21.2 found, and which
paper-safe claims are supported. The positioning is deliberately conservative:
ViewTrust-GS is framed as exact contributor attribution and trust-signal
validation, not as a defense or intervention method.

PR21.3 writes:

```text
pr213_chair_exact_evidence_positioning_summary.json
pr213_chair_claim_table.csv
pr213_chair_limitation_table.csv
pr213_chair_exact_evidence_positioning_report.md
pr213_paper_wording_snippets.md
pr213_next_step_decision_memo.md
artifact_manifest.csv
```

The recommended next step is PR21.2a ID namespace audit first, then PR21.4
exact contribution magnitude if the namespace is validated.

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
pr211_gsplat_source_audit.csv
pr211_internal_loop_shape_audit.csv
pr211_internal_loop_attempts.csv
pr211_accumulation_audit.csv
pr211_per_view_replay_audit.csv
pr211_contributor_path_decision.json
pr211_contributor_path_attempts.csv
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
python scripts/smoke/pr211f_drums_selected_pixel_alignment_smoke_test.py
python scripts/smoke/pr211g_pr20_selected_pixel_provenance_smoke_test.py
python scripts/smoke/pr213_chair_exact_evidence_positioning_smoke_test.py
```

## Interpretation

If `exact_attribution_succeeded = true`, PR21.1 has recovered exact sparse
contributors for selected pixels. If the evidence quality is
`exact_sparse_contributor_id_only`, the next step is PR21.2 ID-level
exact-vs-proxy comparison. If the evidence quality is
`exact_sparse_render_contribution`, PR21.2 may compare weighted attribution.
Neither case justifies intervention.

If `exact_attribution_succeeded = false`, inspect `pr211_blockers.csv` and
`pr211_gsplat_metadata_audit.csv`; fix sparse contributor extraction before
comparison.

When exact sparse replay fails, exact-vs-proxy rows report
`exact unavailable due to failed sparse replay` instead of treating empty exact
rows as a real difference from PR20 proxy candidates. Direct/collateral and
train013 tables also mark exact evidence as unavailable rather than claiming no
overlap.
