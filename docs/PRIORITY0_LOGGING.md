# Priority 0 Observation Logging

Priority 0 logging is observation-only infrastructure.

It may record:

```text
run metadata
config snapshots
event JSONL records
iteration observations
view observations
timing observations
GPU memory observations
```

It must not change:

```text
loss
optimizer
densification
pruning
rendering
dataset sampling
training control flow
```

## Local-Safe Logger

The reusable logger lives in:

```text
viewtrust/logging/schema.py
viewtrust/logging/writer.py
```

It uses only the Python standard library. It does not import CUDA, PyTorch, or
`gsplat`.

## Output Layout

A Priority 0 run directory contains:

```text
metadata.json
config_snapshot.json
events.jsonl
```

Each event uses:

```text
schema_name: viewtrust.priority0.event
schema_version: 1
run_id
event_type
created_at_utc
payload
```

Current event types are:

```text
run_start
run_end
config_snapshot
iteration_observation
view_observation
gpu_memory_observation
timing_observation
mock_observation
```

## Local Validation

Run:

```bash
bash scripts/checks/run_static_checks.sh
bash scripts/checks/run_mock_checks.sh
```

The mock checks write only under:

```text
./outputs/
```

## Server Validation

After syncing to the server:

```bash
source scripts/env/activate_server_viewtrust_p0.sh
bash scripts/checks/run_server_checks.sh
bash scripts/checks/run_mock_checks.sh
```

The current logger smoke test is CPU-only. Real GPU memory and timing records
will be added later as observation-only events around the training entrypoint.
