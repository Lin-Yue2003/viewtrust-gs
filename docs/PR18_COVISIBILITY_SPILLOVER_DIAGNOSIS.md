# PR18 Co-visibility Spillover Diagnosis

PR18 is offline observation only. It is not a defense, does not tune PR17
clean-prior normalized scores, and does not change training, rendering, loss
weighting, view selection, Gaussian updates, pruning, or densification.

The goal is to explain remaining clean views that enter normalized top-k after
PR17. A high-risk clean view is not automatically a false alarm: it may be a
camera-neighbor or Gaussian-support collateral view that shares geometry with
the corrupted subset.

## Inputs

PR18 consumes existing artifacts:

- PR16 plan directory with `pr16_subset_manifest.csv`
- PR17 output directory with `clean_prior_normalized_rows.csv`
- PR13 / PR16 offline signal directories under `outputs/reports`
- `offline_viewtrust_artifact_manifest.csv` paths when input files are not
  physically copied into the signal directory
- clean NeRF Synthetic `transforms_train.json`

Corruption labels are used only for post-hoc evaluation and grouping. They are
not used to compute camera distance, support overlap, normalized risk, or any
score.

## Camera Evidence

For each scene, PR18 loads training camera transforms and computes all pairwise:

- camera center distance
- rotation angle in degrees
- combined camera distance

Center and rotation distances are normalized by scene-level robust nearest
neighbor medians. A view is considered camera-neighbor evidence for spillover
when its nearest corrupted view is among its top `neighbor_k` camera neighbors,
or when the combined distance is below the median nearest-neighbor distance
times the configured factor.

## Index Evidence

NeRF Synthetic train views are ordered, so PR18 also reports nearest corrupted
index gap, whether a view lies between corrupted indices, and index neighbors.
This is auxiliary evidence only; camera geometry is the primary neighborhood
signal.

## Gaussian Support Evidence

PR18 attempts best-effort Gaussian-support overlap:

- `exact` when per-Gaussian IDs are available
- `proxy` when only view-level lifecycle or comparison vectors are available
- `unavailable` when the current artifacts do not expose enough support fields

Unavailable Gaussian overlap is reported as a limitation and does not fail the
analysis.

## Classification

Candidate views are raw or normalized false positives. PR18 assigns:

- `clean_prior_false_positive`: high clean-prior risk with small delta or
  demotion by PR17.
- `co_visible_collateral`: low clean prior, positive normalized lift, rank lift,
  and at least one co-visibility evidence source.
- `unexplained_false_positive`: normalized top-k clean view without clean-prior,
  camera, index, or Gaussian-support explanation.
- `prior_demoted`: raw false positive not retained by normalized top-k.

The classifier is diagnostic. It does not feed back into training or scoring.

## Command

```bash
python scripts/measure/analyze_pr18_covisibility_spillover.py \
  --data-root "$VIEWTRUST_DATA_ROOT" \
  --input-root outputs/reports \
  --plan-dir "$PR16_PLAN_ANALYSIS_DIR" \
  --pr17-dir "$PR17_DIR" \
  --output-dir "$PR18_DIR" \
  --scenes chair drums \
  --conditions corrupt_occluder corrupt_noise corrupt_mixed \
  --subset-names original seed_20260710 \
  --top-k 5 \
  --allow-missing \
  --write-markdown
```

## Outputs

PR18 writes:

- `pr18_covisibility_spillover_summary.json`
- `pr18_candidate_false_positive_diagnosis.csv`
- `pr18_camera_neighbor_table.csv`
- `pr18_view_pair_distance_table.csv`
- `pr18_gaussian_support_overlap.csv`
- `pr18_spillover_classification.csv`
- `pr18_condition_summary.csv`
- `pr18_view_identity_transition.csv`
- `pr18_missing_outputs.csv`
- `pr18_report.md`
- `artifact_manifest.csv`

## Interpretation

PR18 separates stable clean-prior false positives from corruption-sensitive
collateral views and unexplained false positives. Co-visible collateral views
should not necessarily be suppressed in future interventions: they may be
legitimate evidence that corrupted views affected nearby geometry.

PR18 is a prerequisite for later Gaussian-level risk analysis and any future
trust-aware densification gating, but it does not implement those interventions.
