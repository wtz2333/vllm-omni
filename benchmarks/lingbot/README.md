# Interactive World Model Benchmark

This directory contains a model- and runtime-neutral benchmark core for
stateful interactive world models. The directory name reflects the first
integration target, but the benchmark does not import LingBot, vLLM-Omni, or
any model-specific control schema.

Model-specific loading and request translation belong in adapters outside this
directory.

## Core lifecycle

`interactive_world_model_benchmark.py` executes:

```text
adapter.initialize(workload)
  └── repeated sessions
        ├── adapter.create_session(...)
        ├── adapter.execute_tick(...) × N
        └── adapter.close_session(...)
adapter.shutdown()
```

Warmup sessions are excluded from headline statistics. Every session is
closed in a `finally` block, and `shutdown()` is attempted even when a tick or
contract validation fails.

## Adapter protocol

An adapter factory is loaded using `module.path:factory`. The factory receives
the JSON object from `--adapter-args` and synchronously returns an object with
these async methods:

```python
async def initialize(workload) -> Mapping[str, Any]: ...
async def create_session(session_id, workload) -> None: ...
async def execute_tick(session_id, tick_index, tick) -> Mapping[str, Any]: ...
async def close_session(session_id) -> None: ...
async def shutdown() -> None: ...
```

`execute_tick()` must return:

| Field | Type | Contract |
|---|---|---|
| `session_id` | string | Must equal the submitted session |
| `event_id` | non-negative integer | Must equal the workload tick |
| `tick_index` | non-negative integer | Must equal the submitted position |
| `finite` | boolean | Must explicitly be `true` |
| `output_units` | positive number | Frames, latent frames, actions, or another declared unit |
| `shape` | positive integer sequence, optional | Output tensor/media shape |
| `stage_durations` | mapping, optional | Non-negative seconds by stage |
| `peak_memory_mb` | non-negative number, optional | Adapter-defined request peak |
| `metadata` | mapping, optional | Serializable adapter evidence |

Controls are opaque mappings. The benchmark transports and records them but
does not interpret camera keys, poses, robot actions, or model-specific
schemas.

## Workload format

See `interactive_workload.example.json`. Important fields:

- `output_rate_hz` defines media/control time for RTF;
- `expected_output_units` validates per-tick output size;
- `label` enables generic phase summaries such as `first`, `fill`, or
  `rolling`;
- `metadata` can pin resolution, seed, model revision, or parallel layout
  without making those fields part of the core protocol.

## Run

```bash
python -m benchmarks.lingbot.interactive_world_model_benchmark \
  --adapter my_world_model_adapter:create_adapter \
  --adapter-args '{"model":"/models/world-model","parallel_size":2}' \
  --workload benchmarks/lingbot/interactive_workload.example.json \
  --warmup-sessions 1 \
  --measured-sessions 5 \
  --output-dir /tmp/world-model-benchmark
```

Outputs:

- `result.json`: raw tick observations, lifecycle timing, and summaries;
- `summary.md`: compact decision table.

## Metrics

- initialize, session-create, session-close, and shutdown latency;
- tick P50/P90/P95;
- first-tick and steady-tick latency;
- per-tick-index and per-label latency;
- output units/second and real-time factor;
- adapter-reported stage timing and peak memory;
- output shape and finite-value correctness gates.

The framework does not impose a universal quality metric because decoded
video, latents, actions, and state predictions require different references.
Adapters should place model-specific parity/quality evidence under
`metadata`, or run a separate accuracy benchmark.

