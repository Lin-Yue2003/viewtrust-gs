# PR21.0 gsplat Feasibility and Official-Checkpoint Replay Harness

PR21.0 starts the post-PR20.1 exact-attribution phase by auditing whether an
official Gaussian Splatting run can be inspected or replayed with the installed
`gsplat` package. It is observation-only and does not implement exact sparse
attribution, defense, view rejection, update suppression, or densification
gating.

## Why This Exists

PR20.1 showed that the PR20.0 residual-weighted attribution path still behaves
like a view-event proxy. PR21.0 therefore asks a narrower question:

```text
Can the official 3DGS checkpoint, PLY schema, camera metadata, and selected
official renders be audited well enough to design exact sparse pixel-to-Gaussian
attribution in PR21.1?
```

This PR does not claim causal localization or exact render contribution.

## Inputs

The probe expects an existing official-run directory such as:

```text
outputs/baseline/chair_corrupt_occluder_seed_20260710_gaussian_splatting/pr16_chair_corrupt_occluder_seed_20260710_i700
```

Required artifacts are audited, not modified:

```text
trainer_output/point_cloud/iteration_700/point_cloud.ply
trainer_output/cameras.json
trainer_output/cfg_args
trainer_output/exposure.json
view_evaluation/render_models/train_test_model/train/ours_700/renders/
view_evaluation/render_models/train_test_model/train/ours_700/gt/
```

## Local-Safe Smoke

The local smoke test creates a tiny fake run directory with a fake PLY,
`cameras.json`, and PNG render/GT files. It does not import `torch` or `gsplat`
at module import time and does not require CUDA:

```bash
python scripts/smoke/pr210_gsplat_feasibility_smoke_test.py
```

## Server Probe

Run on the server after activating the working environment:

```bash
cd /trainingData/sage/yue/viewtrust-gs
deactivate 2>/dev/null || true
source scripts/env/activate_server_viewtrust_p0.sh
export VIEWTRUST_DATA_ROOT=/trainingData/sage/yue/viewtrust-data
unset PYTHONPATH

export PR200_CHAIR_RUN_ROOT=$(find outputs/baseline -type d \
  -name "pr16_chair_corrupt_occluder_seed_20260710_i700" \
  | grep "chair_corrupt_occluder_seed_20260710_gaussian_splatting" \
  | sort | tail -1)

export PR210_CHAIR_DIR=outputs/reports/pr210_gsplat_feasibility_chair_occluder_seed20260710_$(date +%Y%m%dT%H%M%S)

python scripts/measure/probe_pr210_gsplat_feasibility.py \
  --run-dir "$PR200_CHAIR_RUN_ROOT" \
  --scene chair \
  --condition corrupt_occluder \
  --subset-name seed_20260710 \
  --iteration 700 \
  --split train \
  --views train_004 train_009 train_012 train_017 train_014 train_013 \
  --output-dir "$PR210_CHAIR_DIR" \
  --device cuda:0 \
  --max-views 6 \
  --write-markdown
```

Drums uses the same command shape with the drums run root and selected views.

## Outputs

PR21.0 writes:

```text
pr210_gsplat_feasibility_summary.json
pr210_dependency_probe.json
pr210_run_artifact_audit.csv
pr210_ply_schema_audit.csv
pr210_camera_schema_audit.csv
pr210_selected_view_audit.csv
pr210_gsplat_api_audit.csv
pr210_checkpoint_conversion_audit.csv
pr210_render_replay_audit.csv
pr210_render_parity_metrics.csv
pr210_blockers.csv
pr210_recommendations.json
pr210_report.md
pr210_missing_inputs.csv
artifact_manifest.csv
```

The summary always keeps:

```text
observation_only = true
training_intervention = false
defense_enabled = false
view_rejection_enabled = false
densification_gating_enabled = false
third_party_modified = false
exact_sparse_attribution_ready = false
```

## Interpretation

`pr21_ready_for_exact_attribution` may be true only when the installed
environment exposes `gsplat`, the official PLY and camera schemas look
convertible, selected views have strict split-aware camera matches, and public
gsplat metadata/intersection APIs appear available. Even then,
`exact_sparse_attribution_ready` remains false in PR21.0 because the actual
sparse replay and per-pixel contributor extraction belong to PR21.1.

## Strict Selected-View Matching

PR21.0a makes selected-view matching strict and split-aware. A requested
training view such as `train_004` must match camera metadata normalized to
`train_004`. A camera named `test_004` is recorded only as diagnostic
suffix-match evidence and is not valid for exact attribution.

The selected-view audit includes:

```text
requested_split
requested_prefix
requested_index
matched_prefix
matched_index
strict_match
split_consistent
suffix_match_only
view_match_blocker
valid_for_exact_attribution
match_quality
```

If any selected view has `suffix_match_only = true` or
`view_match_blocker = true`, then:

```text
selected_view_matching_supported = false
pr21_ready_for_exact_attribution = false
```

`pr210_blockers.csv` records this as a `selected_view_matching` error because
PR21.1 exact sparse attribution requires the camera pose to correspond exactly
to the selected render/GT view.

If render replay is not implemented, PR21.0 writes explicit render replay
blockers and empty parity placeholders. It does not fake parity metrics.
