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
PR17: clean-prior normalized offline ViewTrust signals
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

PR17 Clean-Prior Normalized Offline ViewTrust Signals adds a post-hoc
normalization layer over existing PR13 / PR16 offline signal outputs. It
estimates a per-view clean prior from clean-side view influence features or
clean columns recorded in existing artifacts, then writes raw-vs-normalized
rankings, group metrics, ablation metrics, false-positive reduction tables,
view-identity diagnostics, missing-output reports, a Markdown report, and a
two-pass artifact manifest. PR17 is offline analysis only: it does not change
raw PR13 scoring, PR14 aggregation, PR15 or PR16 behavior, training, rendering,
`third_party`, loss reweighting, view rejection, update suppression,
densification gating, or any defense behavior.

PR18 Co-visibility Spillover Diagnosis adds an offline explanation layer for
remaining PR17 normalized false positives. It uses clean training camera
transforms, PR16 corrupted-subset manifests, PR17 normalized rows, and
best-effort view/Gaussian support artifacts to separate stable clean-prior
false positives from co-visible collateral views and unexplained false
positives. PR18 does not tune PR17 scores, does not use corruption labels for
scoring, and does not change training, rendering, `third_party`, PR13 scoring,
PR14 aggregation, PR15 analysis, PR16 behavior, PR17 normalization, or any
defense/intervention behavior.

PR19 Gaussian Cluster Risk and Support Concentration adds an offline
Gaussian-level or lifecycle-event-level diagnosis after PR18. It resolves
existing view influence artifacts through offline manifests, detects whether
exact per-Gaussian IDs are available, falls back to aggregate event proxy
clusters when they are not, and ranks candidate representation-risk clusters.
PR19 writes evidence availability, view-group maps, cluster risk rows,
direct/collateral overlap diagnostics, train_013 control summaries, and a
preview-only intervention candidate table where every row remains marked
`do_not_apply_intervention = true`. PR19 does not change training, rendering,
`third_party`, PR13 scoring, PR14 aggregation, PR15 analysis, PR16 behavior,
PR17 normalization, PR18 diagnosis, or any defense/intervention behavior.

PR19.1 Exact Gaussian Lifecycle Attribution Logging adds a CPU-only sidecar
Gaussian identity tracker, exact log schemas, validation helpers, and local
smoke coverage so future runs can produce `exact_gaussian_id` evidence instead
of only `aggregate_event_proxy` clusters. Stable Gaussian IDs are monotonic
sidecar IDs and are explicitly distinct from mutable tensor row indices. This
PR does not modify `third_party` or official training behavior; real 3DGS
integration requires a later opt-in patch that passes clone/split/prune masks
and view context into the sidecar tracker without changing optimization.

PR19.2 Exact Gaussian Logging Runner Integration connects the PR19.1 exact
schema to `scripts/measure/build_view_influence_table.py`, the existing
ViewTrust runner that produces `view_influence.csv`,
`view_lifecycle_attribution.csv`, and `view_iteration_events.csv`. Exact
logging remains disabled by default and is enabled only with
`--enable-exact-gaussian-logging`; the runner replays existing real lifecycle
rows into PR19.1 exact log files without modifying `third_party` or training
behavior. PR19 exact-mode discovery can consume these exact logs when they are
stored under a view influence output directory.

PR19.3 Exact View-Group Binding adds a post-hoc offline binding step that
attaches PR17 / PR18 view-group semantics to PR19.2 exact Gaussian logs. It
writes grouped exact tables, direct/collateral exact overlap diagnostics,
train_013 exact control diagnostics, and an optional PR19-ready exact input
bundle under `pr19_exact_input_bundle/exact_gaussian_logging/`. PR19.3 does
not change PR19 scoring, PR17 normalization, PR18 diagnosis, training,
rendering, `third_party`, loss reweighting, update suppression, densification
gating, or defense behavior.

PR19.4 Exact Event-Level Support Filtering adds an offline diagnostic layer on
top of PR19.3 grouped exact logs. It compares broad PR19.3 support against
stricter support modes (`birth`, `prune`, `high_event`, `dominant_source`,
`low_entropy`, and `suspicious_alive`) to detect broad-overlap degeneracy and
identify whether any non-broad mode supports nontrivial direct/collateral exact
Gaussian overlap while preserving train_013 as a clean-prior control. PR19.4
does not change PR19.3 binding, PR19 scoring, PR17 / PR18 scoring, training,
rendering, `third_party`, or any defense/intervention behavior.

PR19.5 Exact Attribution Semantics Audit consumes PR19.3 and PR19.4 outputs to
decide whether current exact lifecycle logs are semantically strong enough for
PR19 exact-mode conclusions or PR20 intervention planning. It corrects PR19.4
recommendation semantics so diagnostic train_013-only modes cannot be
recommended as exact PR19 modes when nontrivial direct/collateral overlap is
absent. It writes support-mode failure analysis, event semantics audits,
required attribution field gaps, PR20 readiness criteria, and a next-step
recommendation. PR19.5 remains offline-only and does not change training,
rendering, `third_party`, scoring, or defense/intervention behavior.

PR20.0 Sparse Render-Contribution and Residual-Weighted Gaussian Attribution
Logging adds an observation-only sparse residual attribution pipeline. It
selects high-residual pixels from existing render/ground-truth image pairs and
uses PR19.3 view-Gaussian attribution rows as an explicitly approximate
Gaussian candidate source when exact per-pixel splat contributors are not
available. It writes sparse pixel residuals, residual-weighted proxy
pixel-Gaussian rows, Gaussian and view-group residual attribution summaries,
direct/collateral residual overlap, train_013 residual controls, and quality
audits. PR20.0 does not modify `third_party`, training behavior, rendering
behavior used by training, optimization, scoring, or any defense/intervention
behavior; `pr20_ready_for_intervention` remains false.

PR20.1 Proxy Degeneracy Diagnosis audits PR20.0 outputs to determine whether
the approximate proxy degenerates into fixed view-level candidate pools and
uniform per-pixel weights. It reports pixel candidate reuse, candidate pool
overlap, weight uniformity, direct/collateral proxy overlap degeneracy,
train_013 proxy-pool separation, failure cases, and recommendations for exact
sparse render contribution attribution. PR20.1 does not change PR20.0 output
generation, training, rendering, `third_party`, scoring, or any
defense/intervention behavior.

PR21.0 gsplat Feasibility and Official-Checkpoint Replay Harness adds an
observation-only audit for whether official 3DGS checkpoints, PLY schemas,
camera metadata, selected views, installed `gsplat` APIs, and conversion
metadata are sufficient to design exact sparse pixel-to-Gaussian attribution in
PR21.1. It writes dependency, artifact, PLY, camera, selected-view, gsplat API,
conversion, render-replay, blocker, recommendation, and manifest outputs. It
does not implement exact attribution, render replay parity claims, training
changes, rendering changes used by training, `third_party` changes, defense,
view rejection, update suppression, or densification gating.

PR21.0a tightens PR21.0 selected-view camera matching so requested train views
must match strict split-aware camera names. Numeric suffix-only matches such as
`train_004` to `test_004` are now recorded as `selected_view_matching` errors
and block PR21.1 readiness. PR21.0a remains observation-only and does not add
render replay, exact attribution, training changes, rendering changes,
`third_party` changes, or intervention behavior.

PR21.1 Exact Sparse Pixel-to-Gaussian Attribution Replay adds an offline
server-only gsplat replay path that attempts to recover actual sparse
pixel-level Gaussian contributor IDs for PR20 selected high-residual pixels. It
uses PR21.0a strict camera matching, audits checkpoint activation assumptions,
writes exact pixel-Gaussian rows only when contributor IDs are retrieved from
gsplat metadata/API, and compares exact rows against PR20 proxy candidates. If
exact IDs are unavailable, it writes blockers and does not fabricate exact
evidence. PR21.1 remains observation-only; `ready_for_intervention` stays
false and no training, rendering, `third_party`, defense, view rejection,
update suppression, or densification gating behavior is changed.

PR21.1a fixes sparse contributor extraction for gsplat APIs that require an
explicit `transmittances` tensor. It audits rasterization return tensors,
candidate transmittance sources, and contributor API compatibility before
calling `rasterize_to_indices_in_range` with the full gsplat 1.5.3-style
signature. If no valid transmittance source exists or selected-pixel filtering
fails, exact rows remain empty and blockers explain the failure. PR21.1a also
corrects failed-replay table wording so empty exact rows are not interpreted as
real exact/proxy differences or no-overlap evidence. It remains observation-only
and does not change training, rendering, `third_party`, or intervention
behavior.

PR21.1b adds source-guided contributor extraction. It audits installed gsplat
source and signatures at runtime, records contributor path decisions, and tests
source-supported squeezed / one-minus-alpha transmittance candidates before
calling `rasterize_to_indices_in_range`. It can produce strict
`exact_sparse_contributor_id_only` rows when pixel-level contributor IDs are
recovered but alpha/transmittance/splat weights are unavailable. It still never
uses PR20 proxy rows as exact evidence, keeps `ready_for_intervention = false`,
and does not modify training, rendering, `third_party`, installed site-packages,
or intervention behavior.

PR21.1c adds a source-verified gsplat internal-loop replay for contributor-ID
recovery. It reproduces the installed gsplat pattern
`transmittances = 1.0 - render_alphas[..., 0]`, calls
`rasterize_to_indices_in_range`, and updates `render_alphas` through
`accumulate` for offline selected-pixel replay. It prefers `packed=False`
metadata for source-compatible shapes, records shape/attempt audits, and labels
successful rows as `exact_sparse_contributor_id_only` with
`attribution_method = gsplat_internal_loop_contributor_id_replay`. PR21.1c does
not claim weighted alpha/transmittance/splat semantics, does not use proxy rows
as exact evidence, and remains observation-only with no training, rendering,
`third_party`, installed site-packages, defense, or intervention changes.

PR21.1d removes the nerfacc dependency from the internal-loop contributor-ID
path. If `gsplat.accumulate` fails because `nerfacc_cuda` cannot build or
import, the offline replay uses a source-verified pure-torch alpha-only update
to maintain `render_alphas` and the next-batch transmittance state. Contributor
IDs are collected before accumulation, but success still requires a valid alpha
state update and selected-pixel hits. Outputs remain
`exact_sparse_contributor_id_only`; PR21.1d does not claim exact weighted render
contribution and does not modify training, rendering, `third_party`, installed
gsplat, installed nerfacc, defense, or intervention behavior.

PR21.1e switches exact contributor-ID recovery to per-view internal-loop replay
to avoid ambiguous multi-view `image_id` mapping. Each selected view is replayed
as a one-view camera batch, only `image_id == 0` is accepted, and accepted rows
are mapped to the outer-loop view name. Coordinate flips, swaps, and
neighborhood matches are recorded as diagnostics in
`pr211_per_view_replay_audit.csv` but are not emitted as exact evidence.
PR21.1e remains ID-only and observation-only, with no training, rendering,
`third_party`, installed package, defense, or intervention changes.

PR21.2 adds a chair-only exact-vs-proxy contributor-ID comparison. It validates
PR21.1e chair exact contributor-ID-only evidence, compares PR20 proxy candidate
sets against exact sets at pixel, view, and group levels, and reassesses PR20.1
proxy degeneracy claims without treating proxy rows as exact evidence. Drums is
explicitly excluded until coordinate alignment is resolved. PR21.2 remains
observation-only and does not provide exact alpha/transmittance/splat weights or
any intervention mechanism.
