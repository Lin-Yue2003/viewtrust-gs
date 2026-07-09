# PR19.1 Exact Gaussian Lifecycle Logging

PR19.1 exists because PR19 could only use `aggregate_event_proxy` evidence on
the current server outputs. Aggregate event clusters are useful, but they cannot
prove exact Gaussian-level causal localization. PR19.1 introduces a sidecar
stable Gaussian identity tracker and validation tools so future runs can produce
`exact_gaussian_id` evidence.

PR19.1 is logging and instrumentation infrastructure only. It is not a defense,
does not reject views, does not reweight losses, does not suppress Gaussian
updates, and does not gate densification or pruning.

## Stable IDs

Stable Gaussian IDs are monotonic sidecar integers assigned once:

- Initial Gaussians receive IDs `0..N-1`.
- Clone births receive new IDs whose parent/root IDs come from the cloned
  parent.
- Split births receive new IDs whose parent/root IDs come from the split
  parent.
- Pruned IDs are marked dead and removed from the alive sidecar order.

Tensor row indices are not stable identities. PR19.1 records row indices only as
diagnostic metadata such as `row_index_before`, `row_index_after`, and
`final_row_index`.

## Evidence Quality

PR19.1 distinguishes:

- `exact`: clone/split/prune parent mapping is known.
- `partial`: new Gaussian count is known but parent mapping is unavailable,
  logged as `densify_birth_unknown`.
- `aggregate_event_proxy`: PR19 fallback evidence from view-level lifecycle
  tables. This must not be conflated with exact IDs.

## Output Files

When exact logging is enabled, the tracker writes:

- `gaussian_identity_table.csv`
- `gaussian_lifecycle_events.csv`
- `view_gaussian_event_attribution.csv`
- `gaussian_support_summary.csv`
- `exact_gaussian_logging_summary.json`
- `exact_gaussian_logging_validation.json`
- `artifact_manifest.csv`

The config file is
`configs/offline_viewtrust_signal/default_pr191_exact_gaussian_logging.json`,
and `enabled` defaults to `false`.

## Validation

Validate an exact log directory with:

```bash
python scripts/measure/validate_pr191_exact_gaussian_logging.py \
  --exact-log-dir "$EXACT_LOG_DIR" \
  --output-dir "$PR191_VALIDATE_DIR" \
  --write-markdown
```

The validation checks duplicate alive IDs, alive row count, parent/root
existence, prune references, required files, and `uses_row_index_as_stable_id =
false`.

## Current Integration Status

This PR adds tracker, schema, smoke, and validation infrastructure. It does not
modify `third_party` or official 3DGS training code. Real exact logging in 3DGS
training requires a follow-up integration patch that passes clone/split/prune
masks and current view context into the tracker. If that patch cannot be made
without modifying `third_party`, the blocker should be reported explicitly
rather than silently changing training behavior.

Future PR19 reruns should consume these exact logs to test whether direct
corrupted and co-visible collateral views share exact Gaussian clusters while
`train_013` remains a clean-prior control.
