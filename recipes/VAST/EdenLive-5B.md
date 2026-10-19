# EdenLive 5B few-step

EdenLive renders a reference video using a target first frame and a text prompt.
This experimental integration implements the released `dit-5b` checkpoint from
[`VAST-AI/EdenLive`](https://huggingface.co/VAST-AI/EdenLive) on the
`experimental.ar_diffusion` engine. It reuses Wan layers and the native Wan2.2
VAE, with Proxy-DINO conditioning and runner-owned paged KV.

The implementation follows
[EdenLive revision `8f5d16e`](https://github.com/VAST-AI-Research/EdenLive/tree/8f5d16e03910c1dd72632c96ccf9d44dc27670b8).
CPU tests cover tiny-model numerical agreement and runtime contracts. On one
H200, both official 117-frame samples and a 705-frame rollout with static
prompts and seed `2026092901` match the reference's normalized latents and raw
RGB frames exactly. AsyncOmni streaming, cancellation after the first chunk,
and a subsequent request in the same engine also pass. Throughput has not been
benchmarked.

## Scope

| Setting | Supported baseline |
| --- | --- |
| Checkpoint | 5B few-step, `dit-5b/` |
| Inputs | One RGB first frame, one complete reference clip, one static prompt |
| Execution | One CUDA device, one request at a time, eager BF16 |
| Attention | `CUDNN_ATTN`; the DINO encoder also selects cuDNN SDPA locally |
| Default output | 1280 × 704, 117 frames, 24 FPS |
| Sampling | Four stochastic flow steps, CFG scale 1 |
| AR state | Three latent frames per chunk; six sink + twelve recent frames |
| Streaming output | First chunk: 9 RGB frames; subsequent full chunks: 12 frames |

Frame counts must be `4*k+1`; the last latent chunk may contain fewer than three
frames. Resolution is fixed when the KV pool is created and must match
`model_config.ar_diffusion_height/width`, both positive multiples of 32.
Supply reference frames at the desired output frame rate: this pipeline selects
frames 0, 4, 8, … and does not resample the source timeline.

The initial integration does not implement 14B or bidirectional checkpoints,
live reference-frame ingestion, camera/game controls, dynamic prompts, LightTAE,
multi-device execution, quantization, CPU offload or compiled execution.
The complete reference clip is supplied once even when output streams.
The supplied deployments preserve the reference's attention kernel and FP32
timestep-to-sigma round trip. Other attention backends can change BF16 outputs
and require separate quality validation.

## Components

Only these paths in the released weight tree are needed:

```text
model_index.json
dit-5b/
vae-5b/
umt5/tokenizer/
umt5/text_encoder/
vision-encoder/backbone/
vision-encoder/projection.safetensors
```

The transformer checkpoint's input projection includes 1024 proxy channels in
addition to 48 latent channels. The JSON's `in_channels=48` describes the base
latent width. The loader builds the widened projection before loading weights.
The VAE's native `.pth` keys are converted in memory and checked strictly.

## Offline generation

Run the shared image-to-video example from the repository root. Use the
`cover.png` and `recording.mp4` from an
[official sample](https://github.com/VAST-AI-Research/EdenLive/tree/8f5d16e03910c1dd72632c96ccf9d44dc27670b8/examples/samples).

```bash
python examples/offline_inference/image_to_video/image_to_video.py \
  --model VAST-AI/EdenLive \
  --image /path/to/cover.png --video /path/to/recording.mp4 \
  --prompt "A cinematic rendering of the reference scene." \
  --num-frames 117 --num-inference-steps 4 --guidance-scale 1 \
  --height 704 --width 1280 --fps 24 --seed 2026092901 \
  --enforce-eager --output /path/to/edenlive.mp4
```

Model discovery selects `vllm_omni/deploy/edenlive_5b.yaml` automatically.
A local model directory with the same component tree can replace the model ID.
No EdenLive/FlashDreams Python installation or game frontend is required.

Python callers pass the same media through
`multi_modal_data={"image": first_frame, "video": reference_video}`.
Images accept PIL images, paths or HWC RGB uint8 arrays. Video accepts a local
file path or a sequence/array of HWC RGB uint8 frames. `output_type="latent"`
returns normalized `[1, 48, T, H/16, W/16]` latents under `payload.latents`.

## Serving and streaming

```bash
vllm serve VAST-AI/EdenLive --omni \
  --deploy-config vllm_omni/deploy/edenlive_5b.yaml
```

Use the standard video endpoint with both an image and a video reference.
For chunked output, select `vllm_omni/deploy/edenlive_5b_streaming.yaml` with a
streaming consumer. The deployment enables `step_execution` and
`streaming_output`; the runner owns session admission, KV allocation and cleanup.
Stopping or failing a request releases its KV and causal VAE decoder history.

The key cache stores **unrotated** keys. Every attention read applies RoPE by
slot within the fixed window. Four noisy probes use scratch pages, followed by
a separate clean-latent forward at timestep zero that commits history.
The target first latent is held at timestep zero and re-pinned after every
sampling transition in the first chunk.

## CPU verification

Requires the repository's development dependencies and compatible vLLM,
Diffusers and Transformers versions. These tests need no downloaded weights.

```bash
CUDA_VISIBLE_DEVICES='' pytest -q tests/diffusion/models/edenlive \
  -m 'core_model and cpu' --run-level core_model
```

The tests are included by the existing **Simple · Diffusion Models Test** job in
`.buildkite/cuda/test-ready.yml`.
