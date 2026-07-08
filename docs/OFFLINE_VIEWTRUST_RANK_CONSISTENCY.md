# PR15 Offline ViewTrust Rank Consistency

PR15 diagnoses existing PR13 / PR14.1 offline ViewTrust outputs across natural
corruption conditions. It does not recompute scores, change rankings, rerun
training, rerun rendering, or modify third-party code.

## Offline-only Guarantee

This PR15 report is offline observation only.
It is not a trust score used during training.
It is not a defense.
It is not a poison classifier.
It does not reject views, suppress updates, reweight loss, or gate densification.
Corruption labels are used only for evaluation summaries, not for scoring or ranking.

## What PR15 Diagnoses

PR15 answers whether the same views repeatedly rank highly across corruption
conditions, whether uncorrupted top-k false positives are fixed or
condition-specific, and whether `full_signal` appears to add diagnostic value
beyond `loss_only` and `lifecycle_only`.

The analysis is meant to decide whether the next step should be multi-seed
validation, corrupted-subset variation, cross-object validation, or signal
redesign.

## Inputs

PR15 consumes:

```text
offline_viewtrust_multi_condition_summary.json
offline_viewtrust_multi_condition_results.csv
offline_viewtrust_multi_condition_ablation.csv
offline_viewtrust_condition_ranking.csv
offline_viewtrust_failure_cases.csv
```

from a PR14.1 multi-condition directory, plus per-condition PR13 / PR14-input
directories such as:

```text
outputs/reports/offline_viewtrust_corrupt_occluder_pr14_input
outputs/reports/offline_viewtrust_corrupt_blur_pr14_input
outputs/reports/offline_viewtrust_corrupt_exposure_pr14_input
outputs/reports/offline_viewtrust_corrupt_color_shift_pr14_input
outputs/reports/offline_viewtrust_corrupt_noise_pr14_input
outputs/reports/offline_viewtrust_corrupt_mixed_pr14_input
```

The discovery order prefers `offline_viewtrust_<condition>_pr14_input`, then
newer lexicographic `offline_viewtrust_<condition>_pr13*` directories.

## Command

```bash
python scripts/measure/analyze_offline_viewtrust_rank_consistency.py \
  --multi-condition-dir "$PR14_FULL_DIR" \
  --input-root outputs/reports \
  --scene chair \
  --conditions corrupt_occluder corrupt_blur corrupt_exposure corrupt_color_shift corrupt_noise corrupt_mixed \
  --top-k 5 \
  --output-dir "$PR15_DIR" \
  --write-markdown
```

## Outputs

PR15 writes a new report directory containing:

```text
cross_condition_view_rank_table.csv
cross_condition_view_rank_summary.json
repeated_top_views.csv
false_positive_topk_views.csv
corrupted_view_rank_distribution.csv
component_win_table.csv
component_condition_summary.csv
component_gap_table.csv
rank_consistency_report.md
artifact_manifest.csv
```

The artifact manifest uses a two-pass write so its own row reports
`exists=true` and a nonzero size.

## Local-safe Smoke Test

```bash
python scripts/smoke/offline_viewtrust_rank_consistency_smoke_test.py
```

The smoke test creates fake PR14 and per-condition PR13 outputs under a
temporary directory. It checks repeated top views, false positives, corrupted
rank distribution, component diagnosis tables, offline-only summary fields,
report wording, and artifact manifest self-validation.

## Server Validation

```bash
cd /trainingData/sage/yue/viewtrust-gs
source scripts/env/activate_server_viewtrust_p0.sh
export VIEWTRUST_DATA_ROOT=/trainingData/sage/yue/viewtrust-data
unset PYTHONPATH

bash scripts/checks/run_static_checks.sh
bash scripts/checks/run_mock_checks.sh
python scripts/smoke/offline_viewtrust_rank_consistency_smoke_test.py

export PR14_FULL_DIR=$(find outputs/reports -maxdepth 1 -type d \( \
  -name "offline_viewtrust_multi_condition_pr141*" -o \
  -name "offline_viewtrust_multi_condition_pr14_full_retry*" -o \
  -name "offline_viewtrust_multi_condition_pr14_full*" \
\) | sort | tail -1)

export PR15_DIR=outputs/reports/offline_viewtrust_rank_consistency_pr15_$(date +%Y%m%dT%H%M%S)

python scripts/measure/analyze_offline_viewtrust_rank_consistency.py \
  --multi-condition-dir "$PR14_FULL_DIR" \
  --input-root outputs/reports \
  --scene chair \
  --conditions corrupt_occluder corrupt_blur corrupt_exposure corrupt_color_shift corrupt_noise corrupt_mixed \
  --top-k 5 \
  --output-dir "$PR15_DIR" \
  --write-markdown
```

## Limitations

PR15 remains limited to a single chair mini scene, a fixed corrupted-view
subset, a single seed, natural corruptions, and offline post-hoc evaluation. It
does not establish causality and does not validate malicious attacks.
