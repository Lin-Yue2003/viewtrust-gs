# PR16 Subset and Scene Bias Probe

PR16 diagnoses whether offline ViewTrust signal behavior is tied to a fixed
corrupted-view subset, fixed view identity, or the single chair scene. It adds
planning and analysis tooling only.

## Offline-only Guarantee

This PR16 report is offline observation only.
It is not a trust score used during training.
It is not a defense.
It is not a poison classifier.
It does not reject views, suppress updates, reweight loss, or gate densification.
Corruption labels are used only for evaluation summaries, not for scoring or ranking.

PR16 does not modify PR13 scoring, PR14 aggregation, PR15 analysis behavior,
training, rendering, or `third_party`.

## Scenes

PR16 supports:

```text
chair
drum
```

Use the scene name exactly as `drum`. Do not rename it to `drums`.

## Planner

The planner creates deterministic corrupted subset manifests and a condition
matrix. It does not run training.

```bash
python scripts/experiments/plan_pr16_subset_scene_bias_probe.py \
  --data-root "$VIEWTRUST_DATA_ROOT" \
  --output-dir "$PR16_PLAN_DIR" \
  --scenes chair drum \
  --conditions corrupt_occluder corrupt_noise corrupt_mixed \
  --subset-names original seed_20260708 seed_20260709 \
  --subset-seeds 20260708 20260709 \
  --corrupted-view-count 4 \
  --top-k 5 \
  --write-commands
```

Outputs:

```text
pr16_condition_matrix.csv
pr16_subset_manifest.csv
pr16_seed_reproducibility_summary.json
pr16_run_commands.sh
pr16_plan_report.md
artifact_manifest.csv
```

Seeded subsets are sampled only from discovered training views. The same seed
and train-view list produce the same subset hash. Different seed collisions are
reported as warnings.

`pr16_run_commands.sh` is executable in TODO-only mode by default. It prints
the scene, subset, condition, missing stage, and expected PR16 input directory
for each matrix cell, then exits successfully. Set
`PR16_EXECUTE_HEAVY_STAGES=1` only after reviewing the plan; the generated
script may prepare natural corruption data, but it still marks training, view
influence extraction, comparison, offline signal generation, and metadata
validation as TODO unless those stages have been made explicit.

## Analyzer

The analyzer consumes existing PR13 / PR14 / PR15-style offline outputs across
scene, subset, and condition.

```bash
python scripts/measure/analyze_pr16_subset_scene_bias.py \
  --input-root outputs/reports \
  --plan-dir "$PR16_PLAN_DIR" \
  --output-dir "$PR16_ANALYSIS_DIR" \
  --scenes chair drum \
  --conditions corrupt_occluder corrupt_noise corrupt_mixed \
  --subset-names original seed_20260708 seed_20260709 \
  --top-k 5 \
  --write-markdown
```

Outputs:

```text
pr16_bias_probe_summary.json
pr16_scene_subset_condition_results.csv
pr16_subset_bias_summary.csv
pr16_scene_bias_summary.csv
pr16_view_identity_bias_table.csv
pr16_repeated_false_positive_table.csv
pr16_component_comparison.csv
pr16_missing_outputs.csv
pr16_bias_probe_report.md
artifact_manifest.csv
```

The analyzer reports missing scene/subset/condition outputs without crashing
unless every output is missing.

## Interpretation

PR16 helps answer whether top-ranked views change when corrupted subsets change,
whether `train_013` remains high-risk when not corrupted, whether `train_014`
remains a repeated false positive, whether `full_signal` still improves over
`loss_only` and `lifecycle_only`, whether `corrupt_noise` remains
loss-dominated, and whether `drum` behaves similarly to `chair`.

Careful wording matters: use terms such as ranked highly, associated with
corruption status, offline signal, post-hoc evaluation, candidate high-impact
view, view identity bias, and subset bias.

## Local Smoke Test

```bash
python scripts/smoke/pr16_subset_scene_bias_smoke_test.py
```

The smoke test builds fake chair/drum outputs and checks deterministic planning,
scene coverage, view identity bias rows, repeated false positives, component
comparison rows, report wording, and artifact manifest self-validation.

## Server Validation

```bash
cd /trainingData/sage/yue/viewtrust-gs
source scripts/env/activate_server_viewtrust_p0.sh
export VIEWTRUST_DATA_ROOT=/trainingData/sage/yue/viewtrust-data
unset PYTHONPATH

bash scripts/checks/run_static_checks.sh
bash scripts/checks/run_mock_checks.sh
python scripts/smoke/pr16_subset_scene_bias_smoke_test.py

export PR16_PLAN_DIR=outputs/reports/pr16_subset_scene_bias_plan_$(date +%Y%m%dT%H%M%S)
python scripts/experiments/plan_pr16_subset_scene_bias_probe.py \
  --data-root "$VIEWTRUST_DATA_ROOT" \
  --output-dir "$PR16_PLAN_DIR" \
  --scenes chair drum \
  --conditions corrupt_occluder corrupt_noise corrupt_mixed \
  --subset-names original seed_20260708 seed_20260709 \
  --subset-seeds 20260708 20260709 \
  --corrupted-view-count 4 \
  --top-k 5 \
  --write-commands

export PR16_ANALYSIS_DIR=outputs/reports/pr16_subset_scene_bias_analysis_$(date +%Y%m%dT%H%M%S)
python scripts/measure/analyze_pr16_subset_scene_bias.py \
  --input-root outputs/reports \
  --plan-dir "$PR16_PLAN_DIR" \
  --output-dir "$PR16_ANALYSIS_DIR" \
  --scenes chair drum \
  --conditions corrupt_occluder corrupt_noise corrupt_mixed \
  --subset-names original seed_20260708 seed_20260709 \
  --top-k 5 \
  --write-markdown
```

## Recommended Next Decisions

If PR16 passes, expand to more scenes/classes or build a synthetic target-poison
benchmark. If PR16 shows view identity bias, add clean-prior normalization or
delta-risk analysis before any defense. If `full_signal` fails to beat
`loss_only` under subset changes, redesign lifecycle weighting before moving on.
