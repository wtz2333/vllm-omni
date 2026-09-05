# Echo-WM Flash offline generation

Generate a video with synchronized audio from a first-frame image, a prompt,
and an optional camera Action DSL. This example uses `EchoWMCausalPipeline`
through the standard `Omni` entrypoint and saves an MP4 with audio plus a JSON
file recording the request and generation time.

See the [Echo-WM recipe](../../../recipes/Echo-Team/Echo-WM.md) for setup,
supported features, and validation boundaries.

## Setup

Use a vLLM-Omni environment containing the Echo-WM integration. Image encoding
and audiovisual decoding additionally use the released JoyAI-Echo `ltx_core`
implementation. From the vLLM-Omni checkout:

```bash
export ECHOWM_SOURCE=/path/to/JoyAI-Echo
export PYTHONPATH="$PWD:$ECHOWM_SOURCE/echo_wm/ltx-core/src${PYTHONPATH:+:$PYTHONPATH}"
export ECHOWM_REFERENCE_ROOT="$ECHOWM_SOURCE/echo_wm"
```

The bundled `ltx-core` directory has no `pyproject.toml`; add its `src` directory
to `PYTHONPATH`. Do not install JoyAI-Echo's complete requirements into the
vLLM-Omni environment: those requirements pin a different PyTorch stack.

Download `echo-wm-flash.safetensors` from
[Echo-Team/Echo-WM](https://huggingface.co/Echo-Team/Echo-WM), and separately
obtain the upstream
[Gemma 3 text encoder](https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized).
For reproducing an existing LTX-2.3 run, use the same Gemma checkpoint and
tokenizer assets. `--gemma-path` accepts either an LTX Diffusers bundle root
(containing `text_encoder/` and `tokenizer/`) or a regular Gemma Hugging Face
model directory. The loader selects the corresponding component folders.
A generic Gemma checkpoint and the text encoder shipped in an LTX bundle can
contain different weights even when their architecture and prompt are equal;
use the original bundle when comparing against historical outputs.

The unquantized Gemma weights are required. PyAV must provide H.264 and AAC
encoders for preprocessing and MP4 output; no external FFmpeg executable is
used by the example.

## Run a case

The upstream repository provides `echo_wm/examples/wm_causal_cases/*/case.json`
with sibling `input.jpg` files:

```bash
python examples/offline_inference/echo_wm/echo_wm.py \
  --model /path/to/Echo-WM/echo-wm-flash.safetensors \
  --gemma-path /path/to/gemma-3 \
  --case-json "$ECHOWM_SOURCE/echo_wm/examples/wm_causal_cases/0079/case.json" \
  --num-frames 169 \
  --output outputs/echo_wm_case0079.mp4
```

Add `--dry-run` to resolve inputs without loading models or constructing Omni.
`--model` also accepts a directory containing `echo-wm-flash.safetensors`.
The explicit `EchoWMCausalPipeline` class selects the registered
`echo_wm.yaml` request-execution defaults, so a native checkpoint needs no HF
`config.json` or `model_index.json`. Chunk streaming remains opt-in through
`echo_wm_stepwise.yaml`.

The example defaults to **1280 × 704, 169 frames, 24 fps, BF16, four denoising
steps per block, and TORCH_SDPA**. A longer action string is truncated to the
requested duration; loading a case does not automatically generate its entire
trajectory. `--num-frames` must be `1 + 24*n`, for example 25, 49, 169, or 385.

## Provide your own inputs

```bash
python examples/offline_inference/echo_wm/echo_wm.py \
  --model /path/to/Echo-WM \
  --gemma-path /path/to/gemma-3 \
  --image /path/to/first_frame.jpg \
  --prompt "A camera moves through a forest; leaves rustle in the breeze." \
  --action "w-96,l-96" \
  --seed 42 --fov-deg 70 \
  --output outputs/echo_wm.mp4
```

Each action segment is `<keys>-<frames>`, with comma-separated segments.
`w/s` moves forward/back, `a/d` strafes, `i/k` pitches up/down, `j/l` turns
left/right, and `none` holds the camera. Keys can be combined (`wj-96`).
Omitting `--action` with direct inputs produces a stationary camera.

A custom case JSON accepts `prompt`, `action`, `seed`, `fov_deg`, and optionally
`image` relative to that JSON file. `--image`, `--prompt`, `--action`, `--seed`,
and `--fov-deg` override case values. The default image is `input.jpg` beside
the case JSON.

## Parallel options

Use `--tp 2` for tensor parallelism or `--sp 2` for Ulysses. `TP * SP` must
divide the eight UCPE heads. These are experimental configurations pending
complete GPU output-quality validation; two-rank CPU forward parity is covered
by the regression suite. SP replicates weights and shards tokens/cache heads;
TP shards transformer projections. The text encoder and media components still
need memory on each participating rank.

The example requests one complete output. It does not expose live camera
updates during generation. Ring, CFG parallelism, pipeline parallelism,
quantization, and cache acceleration are not supported by this integration.


Decoded stepwise output currently decodes the complete committed prefix for
each chunk, reloads the media components, and emits only the new frames and
audio samples. It does not keep persistent codec state or provide constant
per-chunk latency. Whole-request demo parity does not establish sample-level
parity for concatenated streaming media; use the default request path for
the validated full-video comparison.

The CPU reference-parity tests use `ECHOWM_REFERENCE_ROOT`; without it, only
those optional reference tests skip. The standalone configuration, loader,
request, media-contract, and Gloo parallel tests still run.
