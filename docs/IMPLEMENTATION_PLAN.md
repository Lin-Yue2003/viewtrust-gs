# ViewTrust-GS Implementation Plan

## 1. Current Repository Structure

At project start, the repository contains only the project guide and third-party source trees:

```text
viewtrust-gs/
  Agent_guide/
    constrain.md
  third_party/
    gaussian-splatting/
    gsplat/
```

The ViewTrust project scaffold, configs, scripts, docs, package modules, and tests do not yet exist.

## 2. Files to Create

This first task creates the observation-only scaffold:

```text
docs/LOCAL_DEVELOPMENT.md
docs/SERVER_ENVIRONMENT.md
docs/IMPLEMENTATION_PLAN.md
docs/TESTING_PROTOCOL.md

scripts/env/activate_server_viewtrust_p0.sh
scripts/env/check_server_environment.sh
scripts/smoke/gsplat_cuda_smoke_test.py
scripts/smoke/mock_cpu_smoke_test.py
scripts/checks/run_static_checks.sh
scripts/checks/run_mock_checks.sh
scripts/checks/run_server_checks.sh

configs/default.yaml
configs/local.example.yaml
configs/server.example.yaml

.env.example
.env.server.example

viewtrust/__init__.py
viewtrust/configs/
viewtrust/logging/
viewtrust/checks/
viewtrust/analysis/
viewtrust/utils/

tests/unit/
tests/mock/

outputs/.gitkeep
third_party/.gitkeep
```

Package directories will include `__init__.py` files so static syntax checks can import the scaffold cleanly.

## 3. Files to Modify

No training-related code will be modified in this first task.

The existing third-party training sources under `third_party/gaussian-splatting/` and `third_party/gsplat/` are treated as read-only for this scaffold pass.

## 4. Local Mac Testing Strategy

Local Mac checks are limited to LOCAL-SAFE work:

```text
python -m compileall viewtrust scripts
bash scripts/checks/run_static_checks.sh
bash scripts/checks/run_mock_checks.sh
```

The local mock smoke test must not import `gsplat`, require CUDA, or execute real 3DGS training. It validates config loading, relative path resolution, run ID generation, metadata JSON writing, output directory creation, and a placeholder logger schema.

## 5. Remote Server Testing Strategy

Remote server checks run only after the repository is synced to the Ubuntu GPU server and the server environment is activated:

```bash
source scripts/env/activate_server_viewtrust_p0.sh
bash scripts/checks/run_server_checks.sh
```

Server checks validate CUDA, `nvcc`, PyTorch CUDA availability, GPU inventory, `gsplat` import, and a minimal CUDA rasterization smoke test.

## 6. Path Policy

Core Python code must not hard-code machine-specific absolute paths, including:

```text
the known server project root
the known server environment prefix
Linux home-directory paths
local Mac user-directory paths
absolute dataset paths
absolute output paths
```

Core code must use relative defaults, command-line arguments, config files, or environment variables:

```text
VIEWTRUST_PROJECT_ROOT
VIEWTRUST_DATA_ROOT
VIEWTRUST_OUTPUT_ROOT
VIEWTRUST_THIRD_PARTY_ROOT
CUDA_HOME
CUDA_PATH
```

Server-specific absolute paths may appear only in:

```text
docs/SERVER_ENVIRONMENT.md
.env.server.example
configs/server.example.yaml
scripts/env/activate_server_viewtrust_p0.sh
scripts/env/check_server_environment.sh
```

## 7. Environment Activation Policy

Local development does not require CUDA activation. Local scripts infer the project root from their location and default to relative paths.

Server execution must use the micromamba environment at the known server location and must use the environment CUDA 12.x toolchain, not the old system CUDA 11.0 toolchain. CUDA headers may be provided under the environment target include directory as well as the top-level include directory.

## 8. GPU-Specific Import Isolation

GPU-specific imports such as `torch` CUDA calls and `gsplat` imports are isolated to server-only scripts or functions:

```text
scripts/env/check_server_environment.sh
scripts/smoke/gsplat_cuda_smoke_test.py
scripts/checks/run_server_checks.sh
```

CPU-only project modules and mock tests must not import `gsplat`.

## 9. LOCAL-SAFE

LOCAL-SAFE tasks may run on the local Mac:

```text
documentation checks
config parsing
relative path resolution
CPU-only mock smoke test
Python syntax compilation
schema placeholder validation
small output writes under ./outputs/mock_smoke_test/
```

## 10. SERVER-REQUIRED

SERVER-REQUIRED tasks must run on the remote Ubuntu GPU server:

```text
CUDA validation
nvcc validation
torch.cuda.is_available() validation
GPU enumeration
gsplat import validation
gsplat CUDA rasterization smoke test
real 3DGS training
Priority 0 GPU memory and timing measurement
```

## 11. Definition of Done for This First Task

This first task is done when:

```text
the scaffold files and directories exist
local documentation describes the two-tier workflow
server documentation records the known validated environment
configs use relative local defaults
server examples document server paths only in approved files
local static checks pass
local mock checks pass without CUDA or gsplat
server checks are documented but not required to pass on Mac
training-related code remains unchanged
```

## 12. Explicit Non-Modification Statement

Training behavior will not be modified in this first task.

This task does not implement ViewTrust scoring, a defense, densification gating, pruning changes, rendering changes, loss changes, optimizer changes, dataset sampling changes, or attack generation.

## Future Priority 0 Roadmap

After the scaffold is validated locally and on the server, future Priority 0 work can add observation-only logging around training execution. That future work must preserve training behavior and record metadata, timing, GPU memory, configuration snapshots, and view/iteration observations without changing optimization or rendering decisions.

## Priority 0 Logging Foundation

The next implementation step adds a local-safe Priority 0 logger that writes run
metadata, config snapshots, and JSONL observation events. This foundation is
intended to be called later by server-side training wrappers, but it does not
modify training code or import GPU-specific dependencies.

## PR Progression

```text
PR0: environment and scaffold
PR1: observed command validation
PR2: minimal dataset policy and NeRF Synthetic chair subset recipe
PR3: baseline clean training wrapper
PR4: training-internal Priority 0 logging
PR5: training dynamics logging from clean baseline runs
PR6: view-level clean metrics
PR7: training event and densification dynamics logging
PR8: Gaussian lifecycle logging
PR9: no-op equivalence and Priority 0 report
PR10: natural corruption condition generation
PR11: clean-vs-corrupt observation comparison
PR12: view-to-Gaussian influence attribution
PR13: offline ViewTrust signal design
PR14: multi-condition offline ViewTrust signal validation
PR15: cross-condition rank consistency and component diagnosis
PR16: subset and scene bias probe
```

PR1 validates external observation before any training-loop instrumentation. It
uses `scripts/measure/run_observed_command.py` to observe subprocesses from the
outside and record Priority 0 artifacts without modifying training behavior.

PR2 creates a minimal NeRF Synthetic chair subset preparation recipe. It is
still observation preparation only and does not modify training behavior.

PR3 creates a clean chair baseline training wrapper. It uses the observed
command infrastructure to run an external trainer and does not modify training
internals.

PR3 server validation completed for the clean chair mini subset with the
official Gaussian Splatting trainer at 500 iterations. Follow-up cleanup keeps
the work observation-only: document server-local trainer dependencies, validate
trainer CUDA submodule imports when requested, inspect observed baseline
artifacts, and keep prepared transforms compatible with the official reader.

PR5 Training Dynamics Logging extracts post-hoc global dynamics from completed
clean baseline runs. It writes derived CSV tables and a summary JSON from
existing observed artifacts and trainer outputs without rerunning training or
modifying trainer behavior.

PR5 does not include view-level metrics or Gaussian lifecycle tracking. Those
belong to PR6 and PR7.

PR6 View-level Clean Metrics renders train, test, and target views from a
successful clean baseline model and writes post-hoc per-view metric tables. PR6
does not include trust scores, natural corruption or poison conditions,
Gaussian lifecycle tracking, or densification event attribution.

PR6.1 fixes official Gaussian Splatting Blender rendering by requiring `--eval`
for train/test and target-as-test render commands. It also validates requested
split counts so the known bad state `train=25, test=0, target=0` cannot pass as
a successful clean mini chair evaluation.

PR7 Training Event and Densification Dynamics Logging adds an opt-in,
observation-only patch for the local official Gaussian Splatting trainer. It
records global iteration events, Gaussian count timeseries, and densification
schedule/trigger rows without changing loss, optimizer, sampling, rendering,
densification, pruning, opacity reset, or save behavior.

PR7 does not include per-Gaussian lifecycle IDs, parent-child split/clone
tracking, view attribution, trust scores, defenses, corruptions, or poisoning.
Those belong to later PRs.

PR7.1 fixes the server child-process environment for training event logging.
The clean baseline wrapper injects the project root into the child trainer
`PYTHONPATH`, preflights the observer import before training, exposes
`--training-event-strict`, and keeps observer failures visible. It remains
observation-only and does not change training behavior.

PR7.2 fixes training event data correctness. Visibility stats are computed from
the boolean visibility mask and normalized by current Gaussian count, observer
rows are sanity-checked, summaries distinguish `requested_iterations` from
`logged_iteration_count`, and `inspect_training_events.py --require-events`
fails on impossible scalar rows. It remains observation-only.

PR11.1 fixes training event scalar timing consistency for densification/pruning
iterations. `iteration_metrics.gaussian_count` now records the render-time
pre-densification count used by visibility and radii stats, while post-prune
counts remain in densification event rows. It remains observation-only.

PR8 Gaussian Lifecycle Logging adds non-trainable per-run Gaussian lifecycle
IDs, clone/split/prune observation hooks, final lifecycle tables, lifecycle
summary inspection, and invariant checks. It is a separate patch applied after
PR7 training events so an existing PR7-patched server checkout can be upgraded
without restoring the trainer. It remains strictly observation-only.

PR9 No-op Equivalence and Priority 0 Observation Report adds read-only
comparison and reporting scripts. It compares uninstrumented and PR7+PR8
observed clean runs with tolerance-based gross-deviation checks, then
consolidates Priority 0 artifacts into JSON, CSV, and Markdown reports. It does
not claim bitwise determinism and does not add training interventions.

PR10 Natural Corruption Condition Generation creates CPU-only tooling for
storage-conscious NeRF Synthetic chair corrupt conditions:
`corrupt_occluder`, `corrupt_blur`, `corrupt_exposure`,
`corrupt_color_shift`, `corrupt_noise`, and `corrupt_mixed`. It corrupts only
selected train views by default, keeps test and target views unchanged,
preserves extensionless official Gaussian Splatting transform paths, and writes
condition manifests, summaries, CSV tables, and preview grids. It does not add
poisoning, trust scores, defenses, or training-time interventions.

PR10.1 makes generated natural corruption conditions directly compatible with
the existing baseline wrapper by writing `manifest.json` with the
`viewtrust.nerf_synthetic_subset.manifest` schema.

PR11 Clean-vs-Corrupt Observation Comparison compares a clean observed run with
a natural corruption observed run using existing PR7 training events, PR8
Gaussian lifecycle logs, and optional PR6 view metrics. It writes JSON, CSV,
artifact manifest, and Markdown reports. It does not detect corruption,
classify trust, implement a ViewTrust score, implement a defense, or change
training behavior.

PR11.2 fixes corruption manifest linking in clean-vs-corrupt comparison
reports. The comparison script can resolve corruption summaries from run
metadata, explicit condition roots, or `--data-root --scene --corruption-condition`.

PR12 View-to-Gaussian Influence Attribution adds sampled view identity to
training event logs, source-view context to lifecycle birth/prune events, and
read-only builders for `view_influence.csv`,
`view_lifecycle_attribution.csv`, and `view_iteration_events.csv`. It is
temporal/source-view attribution only; it does not compute trust scores,
classify views, defend, suppress updates, gate densification, or change
training behavior.

PR12.1 fixes the training split protocol and scales the view influence report
builder. The baseline wrapper now passes official Gaussian Splatting `--eval`
by default so Blender test cameras remain held out, the training event
inspector can require train-only sampling, and the view influence builder
streams lifecycle rows while reporting runtime, timing, input-row, throughput,
observation-only, and split-aware summary fields. It remains observation-only
and does not change loss, optimization, rendering, densification, pruning,
trust scoring, defense, corruption, or poisoning behavior.

PR13 Offline ViewTrust Signal Design builds post-hoc candidate signal tables
from split-correct PR12.1 view influence outputs. It adds robust normalization,
interpretable lifecycle/loss/visibility/delta components, risk/consistency
rankings, corrupted-vs-uncorrupted evaluation summaries, ablation metrics, and
a Markdown report. Corruption labels are used only after scoring for
evaluation. PR13 does not implement a training-time trust score, defense,
poison classifier, loss reweighting, update suppression, densification gating,
or any training/rendering behavior change.

PR14 Multi-condition Offline ViewTrust Signal Validation aggregates existing
PR13 outputs across natural corruption conditions. It discovers the newest
valid offline signal directory per condition, writes per-condition results,
cross-condition ablation rows, condition rankings, failure-case diagnostics, a
multi-condition summary, a Markdown report, and a two-pass artifact manifest.
It supports partial validation when only some condition outputs exist and
strict validation with `--require-all-conditions`. PR14 does not retrain,
render, modify `third_party`, implement a defense, implement a training-time
trust score, reweight loss, suppress updates, or gate densification.

PR15 Cross-condition Rank Consistency and Component Diagnosis consumes existing
PR14.1 multi-condition outputs plus per-condition PR13 / PR14-input offline
signal directories. It writes repeated-top-view diagnostics, false-positive
top-k summaries, corrupted-view rank distributions, component win/gap tables,
a cross-condition summary, a Markdown report, and a two-pass artifact
manifest. PR15 is offline observation only: it does not change PR13 scoring,
PR14 aggregation metrics, training behavior, rendering behavior,
`third_party`, trust-aware training, defense behavior, update suppression,
loss reweighting, or densification gating.

PR16 Subset and Scene Bias Probe adds offline planning and analysis around
fixed corrupted-subset bias, view identity bias, and chair-vs-drum scene bias.
The planner writes deterministic corrupted subset manifests, seed
reproducibility summaries, condition matrices, and a command guide without
running heavy stages. The analyzer consumes existing PR13 / PR14 / PR15-style
outputs across scene, subset, and condition to write subset summaries, scene
summaries, view identity bias diagnostics, repeated false-positive tables,
component comparisons, a Markdown report, and a two-pass artifact manifest.
PR16 does not change PR13 scoring, PR14 aggregation, PR15 behavior, training,
rendering, `third_party`, trust-aware training, defense behavior, loss
reweighting, update suppression, or densification gating.
