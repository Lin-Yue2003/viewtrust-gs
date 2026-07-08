# Offline ViewTrust Signals

PR13 builds offline candidate ViewTrust signals from split-correct PR12.1 view
influence tables. It does not change training, rendering, loss, optimization,
densification, pruning, or sampling behavior.

This is not a defense, not a poison classifier, and not a training-time trust
score. Corruption labels are used only after scoring for post-hoc evaluation.

## Inputs

PR13 expects existing PR12.1 outputs:

```text
<clean-view-influence-dir>/view_influence_summary.json
<clean-view-influence-dir>/view_influence.csv
<corrupt-view-influence-dir>/view_influence_summary.json
<corrupt-view-influence-dir>/view_influence.csv
<view-influence-comparison-dir>/view_influence_comparison_summary.json
<view-influence-comparison-dir>/view_influence_comparison.csv
```

## Command

```bash
python scripts/measure/build_offline_viewtrust_signals.py \
  --clean-view-influence-dir "$CLEAN_VIEW_INFLUENCE_DIR" \
  --corrupt-view-influence-dir "$CORRUPT_VIEW_INFLUENCE_DIR" \
  --view-influence-comparison-dir "$VIEW_INFLUENCE_COMPARE_DIR" \
  --output-dir outputs/reports/offline_viewtrust_corrupt_occluder_pr13_$(date +%Y%m%dT%H%M%S) \
  --write-markdown \
  --top-k 5
```

If `--signal-config` is omitted, the script uses
`configs/offline_viewtrust_signal/default_pr13_signal.json` and copies the
resolved config to `offline_viewtrust_config.json`.

## Signal Formula

Features are normalized with robust z-scores:

```text
robust_z(x) = (x - median(x)) / (1.4826 * MAD(x) + eps)
```

Risk-like components use the positive part of the robust z-score. The default
candidate signal is:

```text
offline_viewtrust_risk =
  0.20 * loss_component
+ 0.15 * visibility_component
+ 0.20 * birth_component
+ 0.25 * prune_component
+ 0.10 * survival_component
+ 0.10 * delta_component
```

`offline_viewtrust_consistency` is:

```text
1 / (1 + offline_viewtrust_risk)
```

Components:

```text
loss_component: robust high-loss signal
visibility_component: robust visibility-drop signal
birth_component: robust birth-rate signal
prune_component: robust prune-death-rate signal
survival_component: robust low birth-survival signal
delta_component: robust clean-vs-corrupt delta signal
lifecycle_component: average of birth, prune, and survival components
```

## Outputs

```text
offline_viewtrust_summary.json
offline_viewtrust_signals.csv
offline_viewtrust_rankings.csv
offline_viewtrust_group_metrics.csv
offline_viewtrust_signal_ablation.csv
offline_viewtrust_config.json
offline_viewtrust_report.md
offline_viewtrust_artifact_manifest.csv
```

## Interpretation

Use careful wording:

```text
offline ViewTrust signal design
candidate evidence signal
view-level lifecycle anomaly ranking
post-hoc analysis for later trust-aware training
```

Avoid claims such as poison detection, bad-view detection, defense success, or
automatic rejection. A high offline risk score does not prove maliciousness.

## Next Experiments

PR13 should be evaluated on multiple natural corruption conditions and multiple
seeds before making any method claim. PR14 should focus on multi-condition and
multi-seed offline signal validation.

PR14 adds `scripts/measure/aggregate_offline_viewtrust_results.py`, which
aggregates existing PR13 outputs across natural corruption conditions. It can
run in partial mode when only some conditions exist, or strict mode with
`--require-all-conditions` after all condition outputs are available.
