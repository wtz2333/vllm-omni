# Realtime AR-Diffusion sessions

> **Status:** experimental. This document describes the execution contract under
> `vllm_omni.experimental.ar_diffusion` and its public
> `WS /v1/realtime/world` transport mapping.

## Scope

Realtime AR-Diffusion serves world models that generate one autoregressive
latent block per request while retaining model state across requests. The
generic runtime owns identity, event ordering, session lifecycle, routing, and
KV allocation. A model adapter owns model-specific controls and conditioning.

LingBot World v2 is the first single-KV-branch integration. DreamZero uses the
same runner capability with two logical CFG branches.

## Identities

Four identifiers have distinct lifetimes:

| Identifier | Lifetime | Owner | Contract |
| --- | --- | --- | --- |
| `session_id` | One persistent world | Session manager | Selects runner KV and model-owned state. |
| `event_id` | One control/prompt update | Transport adapter | Unique and monotonically increasing within a session, including across reset. |
| `chunk_index` | One committed AR block position | Session | Contiguous from zero; reset starts again at zero. |
| `request_id` | One chunk execution | Session | Correlates the tick with its returned metadata. |

`AsyncOmni` keeps its normal UUID-suffixed engine routing ID. The tick
`request_id` remains inside the immutable AR-Diffusion snapshot and the
standard output metadata, while caller-visible `OmniRequestOutput.request_id`
continues to use AsyncOmni's existing external-ID contract. Protocol identity
therefore does not require a private entrypoint override.

## Request lifecycle

1. `ARDiffusionSession.accept_event()` validates and queues an immutable
   `ARDiffusionSessionEvent`. Prompt and opaque control tracks may be updated in
   the same event.
2. `ARDiffusionSession.next_chunk()` serializes chunk execution and calls
   `_build_tick_locked()`. The resulting `ARDiffusionTickRequest` is a snapshot
   of the current prompt, reduced controls, ordered event IDs, and chunk index.
3. `ARDiffusionOmniTickConsumer.execute_tick()` copies the stage sampling
   parameters, stores the typed tick in
   `sampling_params.extra_args["ar_diffusion_tick"]`, constructs the normal Omni
   prompt through its `prompt_provider`, and invokes `AsyncOmni.generate()`.
4. `AsyncOmni` allocates its normal unique engine routing ID and submits one
   `OmniDiffusionRequest` through the orchestrator.
5. `ARDiffusionModelRunner._request_session()` parses the typed tick.
   `_get_or_create_session()` obtains runner-owned paged KV, and
   `bind_ar_diffusion_state()` exposes it to the pipeline only for the duration
   of `execute_model()`.
6. The model pipeline validates the tick and generates exactly one latent
   block. LingBot interprets camera controls in
   `LingBotCameraControlReducer` and
   `LingBotWorldCausalDMDPipeline.forward()`; the generic runtime does not parse
   camera or action schemas.
7. The pipeline returns one `DiffusionOutput` with the generated payload and an
   `ARDiffusionChunkMetadata` mapping under `metadata["ar_diffusion"]`.
   `AsyncOmni` exposes it as
   `OmniRequestOutput.multimodal_output["metadata"]["ar_diffusion"]`.
8. The consumer parses the standard envelope. The session commits its reducer
   state, prompt, controls, event IDs, and next chunk index only when returned
   metadata exactly equals the submitted tick snapshot.

The metadata envelope is:

```json
{
  "metadata": {
    "ar_diffusion": {
      "session_id": "world-7",
      "request_id": "ar-world-7-3-...",
      "chunk_index": 3,
      "applied_event_ids": [10, 11]
    }
  }
}
```

## Transaction and failure semantics

Accepted events are not acknowledged as applied until the model output and
metadata are validated. Reducer `prepare()` is speculative; reducer `commit()`
and the session snapshot form one logical commit. A model, metadata, or reducer
failure leaves the chunk index and queued events uncommitted and transitions the
session to `FAILED`.

After a tick failure, worker state is closed because a forward may have
partially modified or already released KV. The same chunk is not retried in
place. A caller must explicitly reset the session, starting again at chunk
zero, or close it.

Explicit close follows this state machine:

```text
ACTIVE or FAILED
       |
       v
    CLOSING ---- worker cleanup succeeds ----> CLOSED
       |
       +------- worker cleanup fails --------> CLEANUP_FAILED
                                                   |
                                                   +-- close retry --> CLOSING
```

`CLEANUP_FAILED` is a local tombstone. The session manager retains it, rejects
creation of the same `session_id`, and rejects reset. Only an idempotent close
retry that succeeds removes the session. Transport disconnect uses the same
path.

## KV ownership and capacity

`ARDiffusionKVCacheSpec` is the pipeline-owned immutable geometry contract. It
declares:

- TP-local self-attention geometry, AR block size, sink, and recent window;
- logical KV branches and their worker-local slots;
- fixed-length cross-attention caches;
- scratch tokens required by an uncommitted forward;
- persistent model-owned CUDA bytes per resident session; and
- the requested resident-session capacity.

The runner computes:

```text
required(capacity) =
    scratch
  + managed_self_attention_pages(capacity)
  + capacity * cross_attention_bytes_per_session
  + capacity * model_owned_state_bytes_per_session
```

`gpu_memory_fraction` is a soft expansion budget:

```text
configured_budget = available_device_bytes * gpu_memory_fraction
effective_budget = max(configured_budget, required(1))
```

If `required(1)` exceeds actual available device memory, initialization fails.
Otherwise the runner always admits one viable session and uses the configured
fraction to select additional resident sessions up to the model-declared cap.
All model-owned reservations are deducted before the paged self-attention pool
is allocated.

LingBot reports a bounded two-block image-condition tensor: the first block
contains the initial image condition and the second is a reusable blank tail for
all later ticks. If stateful VAE output is enabled it additionally reserves the shipped decoder's
maximum-resolution causal cache with a conservative 4 GiB per-session upper
bound. The direct BF16 decoder retains 1,888,573,440 logical bytes; the full
worker execution path retained 3,728,824,320 storage bytes during validation.
DreamZero reports the measured 603 MiB per-session Wan VAE
causal-convolution state.

## Concurrency and routing limits

The current runner executes one request at a time:

- `max_num_seqs` must be one;
- request batching and diffusion step execution are rejected;
- one AR block is generated per engine request; and
- an AR-Diffusion stage must have exactly one replica.

The single-replica restriction is intentional. Multi-replica support requires
session-affine routing so every tick, reset, and close reaches the worker that
owns the session. Random or round-robin routing is not correct.

Multiple resident sessions may share one runner. Capacity selection accounts
for all persistent pools, and the runner performs LRU eviction through the same
model lifecycle hook used by explicit close.

## Model boundary

The generic runtime transports controls as opaque
`ARDiffusionControlInput(track, schema, data)` values. Model adapters validate
and reduce those values. This keeps target/velocity, key state, pose,
intrinsics, and other model-specific conventions out of the session protocol.

For LingBot World v2:

- `lingbot_world/actions.py` parses key-state and trajectory controls;
- `lingbot_world/camera.py` loads or constructs camera trajectories and Plücker embeddings;
- `lingbot_world/transformer.py` implements checkpoint-compatible causal attention; and
- `lingbot_world/pipeline.py` constructs conditioning, owns model session state, and
  produces one latent or pixel block plus the standard metadata envelope.

## Ulysses sequence parallelism

LingBot supports pure Ulysses sequence parallelism for both direct and
AR-Diffusion execution. Hidden tokens, camera features, token-expanded timestep
modulation, and RoPE tables are sharded together. Self-attention performs the
sequence-to-head all-to-all before reading or writing paged KV, while static
text K/V uses the same local head shard. Ring and AllGather-KV modes remain
unsupported for this model.

## Stateful streaming VAE decode

LingBot pixel output is opt-in through
`model_config.ar_diffusion_streaming_vae=true`. Latent output remains the
default capability and does not allocate decoder state. An enabled session may
choose `output_type="pt"`, `"np"`, or `"pil"`; it cannot switch between
latent and pixel output after its first committed tick.

`WanVAEStreamingDecodeState` owns one causal-convolution feature-cache list per
session. It is passed explicitly into `OmniAutoencoderKLWan.decode_streaming()`;
the VAE module's request-local `_feat_map` is never shared between worlds.
Decoder state fixes batch size, latent spatial shape, dtype, and device on the
first call. Reset, close, failed-forward cleanup, and LRU eviction all reach
the pipeline lifecycle hook, which clears and drops the state.

The shipped causal geometry emits:

- 9 pixel frames for the first three-latent-frame block (`1 + 4 + 4`);
- 12 pixel frames for every later block (`4 + 4 + 4`); and
- an open-ended sequence of later blocks while the session remains active.

Pixel chunks use the standard `payload.video` output envelope. The metadata
retains `ar_diffusion` identity and adds video FPS/shape plus
`streaming_video.pixel_frame_start` and `pixel_frame_count`. Stateful decode
supports `spatial_shard_height` and `spatial_shard_width` when
`vae_patch_parallel_size` matches the full DiT group. Each rank owns the
temporal cache for its spatial shard and participates in halo exchange; rank 0
assembles the emitted pixel chunk. Tile-parallel stateful decode remains
unsupported because independent tiles require separately routed session state.

## Non-goals

This contract does not currently provide:

- camera/action semantics shared by every world model;
- session migration or replication across workers;
- retry of an ambiguous partially executed chunk; or
- cross-stage KV transfer.

Those features can be layered on top without adding LingBot-specific identity
or lifecycle fields to the generic runtime.

## Required regression coverage

CPU contract tests cover:

- separation of tick correlation IDs from internal engine routing IDs;
- event ordering, metadata equality, and reducer fail-closed commit;
- close/disconnect failure tombstones and cleanup retry;
- capacity one under the default fraction for shipped LingBot geometry;
- capacity expansion under a tuned fraction;
- DreamZero model-owned state accounting;
- runner session reuse, reset, close, LRU eviction, and forward cleanup; and
- Wan streaming decoder state isolation, shape locking, cache accounting, and
  9/12-frame chunk geometry; and
- LingBot model registration, imports, camera/action reduction, one-block
  latent/pixel output metadata, opt-in validation, and lifecycle cleanup.

GPU validation must additionally exercise real model loading, action input,
prompt switching, rolling-window replacement, reset, close, finite output
latents/pixels, 9/12-frame chunk boundaries, assembled-video decode, and exact
metadata/request identity.
