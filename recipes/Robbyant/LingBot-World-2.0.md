# LingBot-World 2.0

> Offline and experimental realtime interactive world generation

## Summary

- Vendor: Robbyant
- Model: `robbyant/lingbot-world-v2-14b-causal-fast-diffusers`
- Task: image-conditioned interactive world generation
- Modes: offline trajectory replay, in-process ticks, and public realtime WebSocket
- Hardware validated: NVIDIA H200 and B200
- Maintainer: Community

The checkpoint is separately licensed under CC BY-NC-SA and restricted to
non-commercial use. The vLLM-Omni integration code remains Apache-2.0.

## Offline generation

The offline path consumes one source image and an action directory containing
`poses.npy` with shape `[frames, 4, 4]` and `intrinsics.npy` with shape
`[frames, 4]`.

```bash
python examples/offline_inference/diffusion/lingbot_world_v2.py \
  --prompt "The camera moves slowly forward through the scene." \
  --image /path/to/first_frame.png \
  --action-dir /path/to/actions/forward \
  --num-frames 81 \
  --output lingbot_world_v2.mp4
```

The current causal-fast checkpoint uses four DMD sampling steps. Raw frame
counts must be `9 + 12k`, up to 117 frames. Camera files may contain a
longer bounded source trajectory (the official example contains 269 frames);
the runtime consumes only the prefix needed by the request.

## Realtime in-process generation

The realtime example keeps AR-Diffusion KV and model-owned state across
requests. Each JSONL line describes the prompt and/or three latent-frame camera
actions applied at the next chunk boundary:

```json
{"event_id":1,"prompt":"A road through a forest","frames":[["j"],[],[]]}
{"event_id":2,"frames":[["w"],["w"],["w"]]}
{"event_id":3,"prompt":"The road enters a snowy valley","frames":[[],[],[]]}
```

Run:

```bash
python examples/offline_inference/diffusion/lingbot_world_v2_realtime.py \
  --image /path/to/first_frame.png \
  --events /path/to/events.jsonl \
  --output-dir /tmp/lingbot-realtime \
  --gpu-memory-fraction 0.6
```

The example writes one tensor and one metadata JSON file per chunk. It exercises
the same `ARDiffusionSessionManager -> ARDiffusionOmniTickConsumer ->
AsyncOmni -> ARDiffusionEngine` path used by the public transport.

## Public realtime WebSocket

Start the two-GPU realtime deployment:

```bash
vllm serve robbyant/lingbot-world-v2-14b-causal-fast-diffusers \
  --omni \
  --stage-configs-path vllm_omni/deploy/lingbot_world_realtime.yaml \
  --port 8000
```

`WS /v1/realtime/world` owns one persistent AR-Diffusion session. The browser
sends `session.control` messages containing held WASD/IJKL state, and the
server returns `video.chunk` metadata followed by binary JPEG/WebP frames. See
the [realtime Web UI](../../examples/online_serving/lingbot_world_realtime/README.md)
for the protocol and runnable frontend.

## Realtime identity and controls

- `session_id` identifies the persistent world and its worker-owned state.
- `event_id` identifies a prompt/control update and remains monotonic across reset.
- `chunk_index` is contiguous from zero and restarts from zero after reset.
- `request_id` correlates one chunk snapshot with its output metadata.
- AsyncOmni uses a separate UUID-suffixed internal engine routing ID.

The generic runtime transports controls as opaque snapshots. LingBot's adapter
accepts:

- `lingbot.camera_actions.v1` for per-latent-frame key states such as `w`, `a`,
  `s`, `d`, `i`, `j`, `k`, and `l`;
- `lingbot.camera_trajectory.v1` for explicit pose/intrinsics trajectories.

## Validation

Real-checkpoint validation uses 480x832 output, four DMD steps, and seed 42.
The exercised matrix includes:

- TP=1 and TP=2 execution;
- two interleaved resident sessions;
- action input and prompt switching;
- seven contiguous chunks crossing the sink plus recent rolling window;
- direct versus paged replay;
- CUDA Graph execution;
- reset, close, failure cleanup, and exact metadata matching; and
- nine-frame VAE decode.

Official `generate.py` accuracy and performance numbers, together with
generation artifacts, are recorded in the PR validation section for the exact
tested commit.

## Current limitations

- Only the 14B causal-fast checkpoint is supported.
- The public realtime server currently admits one active world session.
- AR-Diffusion stages currently require one replica because session-affine
  routing across replicas is not implemented.
- One AR block is generated per request and `max_num_seqs` must be one.
- Pure Ulysses SP and spatial-shard VAE parallel decode are supported. Ring,
  pipeline/CFG parallelism, HSDP, quantization, Cache-DiT, TeaCache,
  causal-pretrain, and the 1.3B checkpoint are not claimed.
- No AMD GPU, Ascend NPU, or Intel GPU support is claimed.

## References

- Checkpoint: <https://huggingface.co/robbyant/lingbot-world-v2-14b-causal-fast-diffusers>
- Official implementation: <https://github.com/robbyant/lingbot-world-v2>
- Offline example: [`examples/offline_inference/diffusion/lingbot_world_v2.py`](../../examples/offline_inference/diffusion/lingbot_world_v2.py)
- Realtime design: [`docs/design/feature/realtime_ar_diffusion.md`](../../docs/design/feature/realtime_ar_diffusion.md)
- Realtime Web UI: [`examples/online_serving/lingbot_world_realtime/`](../../examples/online_serving/lingbot_world_realtime/)
