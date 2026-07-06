# No-op Equivalence

PR9 compares an uninstrumented clean baseline run with an instrumented PR7+PR8
run.

This does not prove bitwise determinism. GPU training and official Gaussian
Splatting may vary slightly across runs. The intended claim is narrower:

```text
ViewTrust observation is designed as a no-op with respect to training decisions.
It does not change loss, optimizer behavior, rendering, densification criteria,
pruning criteria, opacity reset criteria, or camera sampling.
```

PR9 checks for successful runs, plausible Gaussian counts, observation
invariants, and gross runtime/count deviations.

## Run

```bash
python scripts/measure/compare_noop_runs.py \
  --baseline-run-dir "$BASELINE_RUN_DIR" \
  --observed-run-dir "$OBSERVED_RUN_DIR" \
  --output-dir outputs/reports/priority0_noop_$(date +%Y%m%dT%H%M%S) \
  --require-success \
  --require-observation-invariants \
  --write-markdown
```

Outputs:

```text
noop_equivalence_summary.json
noop_equivalence_report.md
noop_equivalence_metrics.csv
```

The observed run should include PR7 training events and PR8 Gaussian lifecycle
outputs. The baseline run should be a clean run without those flags.
