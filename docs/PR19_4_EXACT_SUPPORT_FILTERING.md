# PR19.4 Exact Support Filtering

PR19.4 is an offline diagnostic layer over PR19.3 grouped exact Gaussian logs.
It keeps PR19.3 broad support as a baseline, then computes stricter support
modes so broad visibility/update overlap is not mistaken for causal Gaussian
artifact creation.

It does not change training, rendering, `third_party`, PR17 / PR18 scoring,
PR19.3 view-group binding, PR19 scoring, or defense behavior. Corruption labels
are used only for grouping and evaluation, not for scoring.

## Command

```bash
python scripts/measure/analyze_pr194_exact_support_filters.py \
  --pr193-dir "$PR193_DIR" \
  --scene chair \
  --condition corrupt_occluder \
  --subset-name seed_20260710 \
  --output-dir "$PR194_DIR" \
  --support-modes broad birth prune high_event dominant_source low_entropy suspicious_alive \
  --event-percentile 95 \
  --dominant-source-threshold 0.5 \
  --low-entropy-threshold 0.35 \
  --min-event-count 3 \
  --write-markdown
```

## Support Modes

- `broad`: reproduces PR19.3 broad visibility/update/support behavior.
- `birth`: uses clone/split/unknown densification birth lifecycle events.
- `prune`: uses prune death lifecycle events.
- `high_event`: keeps only view-Gaussian event pairs above the configured event
  percentile and `--min-event-count`.
- `dominant_source`: keeps groups whose event fraction exceeds
  `--dominant-source-threshold`.
- `low_entropy`: keeps dominant sources for Gaussians whose normalized group
  entropy is below `--low-entropy-threshold`.
- `suspicious_alive`: keeps final alive Gaussians satisfying at least two
  concentration/lifecycle/event criteria.

All modes are reported side by side. PR19.4 should not hide broad support
degeneracy.

## Outputs

The output directory contains:

- `pr194_exact_support_filter_summary.json`
- `pr194_support_mode_comparison.csv`
- `pr194_filtered_gaussian_support_by_mode.csv`
- `pr194_direct_collateral_overlap_by_mode.csv`
- `pr194_train013_control_by_mode.csv`
- `pr194_gaussian_mode_membership.csv`
- `pr194_view_group_event_concentration.csv`
- `pr194_nontrivial_overlap_candidates.csv`
- `pr194_missing_inputs.csv`
- `pr194_report.md`
- `artifact_manifest.csv`

## Interpretation

`broad_overlap_degeneracy_detected = true` means broad exact support is too
coarse for scientific interpretation. It says direct corrupted and collateral
views share broad final Gaussian support, not that they causally created the
same suspicious Gaussian artifacts.

A non-broad mode supports nontrivial exact overlap only when direct and
collateral support both exist, overlap by exact Gaussian ID, and avoid the
near-perfect broad-overlap degeneracy.

If no non-broad mode is reliable, the correct conclusion is that current exact
lifecycle logs remain too broad for causal localization and that future work
needs finer per-pixel, gradient-weighted, or contribution-weighted attribution
before any intervention.
