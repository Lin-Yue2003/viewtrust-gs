# Offline ViewTrust Multi-Condition Validation

PR14 aggregates PR13 offline ViewTrust signal outputs across natural corruption
conditions. It asks whether the offline candidate signal ranks corrupted views
highly across corruption types, rather than only on `corrupt_occluder`.

This is offline validation only. This is not a trust score used during
training. This is not a defense. This is not a poison classifier. Corruption
labels are used only for post-hoc evaluation.

PR14 does not retrain, render, change 3DGS training, modify `third_party`, gate
densification, reweight loss, suppress updates, or change rendering behavior.

## Conditions

Default conditions:

```text
corrupt_occluder
corrupt_blur
corrupt_exposure
corrupt_color_shift
corrupt_noise
corrupt_mixed
```

The default scene is `chair`, clean condition is `clean`, and `top-k` is `5`.

## Partial Validation

Use this when only some PR13 condition outputs exist:

```bash
python scripts/measure/aggregate_offline_viewtrust_results.py \
  --input-root outputs/reports \
  --output-dir outputs/reports/offline_viewtrust_multi_condition_pr14_partial_$(date +%Y%m%dT%H%M%S) \
  --scene chair \
  --clean-condition clean \
  --conditions corrupt_occluder corrupt_blur corrupt_exposure corrupt_color_shift corrupt_noise corrupt_mixed \
  --top-k 5 \
  --write-markdown
```

Missing conditions are recorded in the summary, results CSV, and failure cases
CSV. The command exits successfully unless `--require-all-conditions` is used.

## Strict Validation

After all PR13 condition outputs exist:

```bash
python scripts/measure/aggregate_offline_viewtrust_results.py \
  --input-root outputs/reports \
  --output-dir outputs/reports/offline_viewtrust_multi_condition_pr14_full_$(date +%Y%m%dT%H%M%S) \
  --scene chair \
  --clean-condition clean \
  --conditions corrupt_occluder corrupt_blur corrupt_exposure corrupt_color_shift corrupt_noise corrupt_mixed \
  --top-k 5 \
  --require-all-conditions \
  --write-markdown
```

Strict validation exits non-zero if any requested condition is missing or
invalid.

## Outputs

```text
offline_viewtrust_multi_condition_summary.json
offline_viewtrust_multi_condition_results.csv
offline_viewtrust_multi_condition_ablation.csv
offline_viewtrust_condition_ranking.csv
offline_viewtrust_failure_cases.csv
offline_viewtrust_multi_condition_report.md
offline_viewtrust_multi_condition_artifact_manifest.csv
```

The artifact manifest uses `relative_path,path,exists,file_type,size_bytes,required,artifact_group`
and is written with a two-pass self-validation strategy so its own row has
`exists=true` and a positive size.

## Failure Cases

PR14 records:

```text
missing_condition_output
invalid_condition_output
corrupted_view_not_in_top_k
top_ranked_uncorrupted_view
low_risk_gap
zero_corrupted_in_top_k
```

These are analysis diagnostics, not proof of detection or failure of a defense.

## Next Step

PR15 should build missing per-condition PR13 artifacts from existing PR12.1 view
influence outputs, then rerun PR14 strict validation. Later work should extend
this to multiple seeds before making any method claim.
