# Echo-WM Flash Usage Guide

[Echo-WM Flash Preview](https://huggingface.co/Echo-Team/Echo-WM) generates
video and audio jointly from a first-frame image, a prompt, and camera actions.
Its LTX-2.3 audio-video transformer uses four DMD steps per causal block,
a persistent attention sink, a bounded FIFO cache, and UCPE camera conditioning.
This recipe covers the Flash checkpoint and the standard offline `Omni` API.

## Installing vLLM-Omni

Follow the [CUDA source installation guide](../../docs/getting_started/installation/gpu.md)
for a checkout containing `EchoWMCausalPipeline`. The currently documented
vLLM dependency is selected explicitly:

```bash
export VLLM_VERSION=0.28.0
uv pip install "vllm==$VLLM_VERSION" --torch-backend=auto
uv pip install -e .
```

The native Echo-WM media codecs use the optional upstream `ltx_core` source:

```bash
git clone https://github.com/jd-opensource/JoyAI-Echo.git /path/to/JoyAI-Echo
export ECHOWM_SOURCE=/path/to/JoyAI-Echo
export PYTHONPATH="$PWD:$ECHOWM_SOURCE/echo_wm/ltx-core/src${PYTHONPATH:+:$PYTHONPATH}"
```

There is no `pyproject.toml` in that bundled `ltx-core` directory. Add `src` to
`PYTHONPATH`; do not run `pip install` on the directory. Retain the vLLM-Omni
PyTorch stack rather than applying the upstream repository's full requirements.
Pillow, PyAV, Einops, Safetensors, Transformers, and SentencePiece are needed
for image preprocessing, Gemma text encoding, and audiovisual decoding. PyAV
must provide H.264 and AAC encoders.

Download the model and the separate unquantized Gemma encoder:

```bash
hf download Echo-Team/Echo-WM echo-wm-flash.safetensors --local-dir /path/to/Echo-WM
hf download google/gemma-3-12b-it-qat-q4_0-unquantized --local-dir /path/to/gemma-3
```

Access to Gemma requires acceptance of its model terms and Hugging Face
credentials. Echo-WM's checkpoint includes the transformer, text connectors,
video VAE, audio VAE, and vocoder; Gemma weights are loaded separately.

## Image and action to video with audio

```bash
python examples/offline_inference/echo_wm/echo_wm.py \
  --model /path/to/Echo-WM \
  --gemma-path /path/to/gemma-3 \
  --case-json "$ECHOWM_SOURCE/echo_wm/examples/wm_causal_cases/0079/case.json" \
  --height 704 --width 1280 --num-frames 169 --seed 42 \
  --dtype bfloat16 --attention-backend TORCH_SDPA \
  --output outputs/echo_wm.mp4
```

`--dry-run` resolves the configuration without starting the engine. The script
accepts either the safetensors file or its containing directory and resolves
the directory to `echo-wm-flash.safetensors`. Case input defaults to the sibling
`input.jpg`; individual image, prompt, action, seed, and FOV flags override
case values. The frame default remains 169 even when the case action is longer.

The script uses this API contract:

```python
from PIL import Image
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

omni = Omni(
    model="/path/to/Echo-WM/echo-wm-flash.safetensors",
    model_class_name="EchoWMCausalPipeline",
    model_config={
        "echo_wm_gemma_path": "/path/to/gemma-3",
        "echo_wm_height": 704,
        "echo_wm_width": 1280,
    },
    dtype="bfloat16",
    diffusion_attention_backend="TORCH_SDPA",
    enforce_eager=True,
)
try:
    with Image.open("first_frame.jpg") as source:
        image = source.convert("RGB")
    outputs = omni.generate(
        {"prompt": "A forest path with rustling leaves.", "multi_modal_data": {"image": image}},
        OmniDiffusionSamplingParams(
            height=704, width=1280, num_frames=169, fps=24,
            num_inference_steps=4, seed=42, output_type="pt",
            extra_args={"echo_wm_action": "w-96,l-96", "echo_wm_fov_deg": 70},
        ),
    )
finally:
    omni.close()
```

Decoded video is `outputs[0].images[0]`, an RGB `uint8` tensor `[T,H,W,3]`.
Audio is `outputs[0].multimodal_output["audio"]`, a float tensor `[C,S]`, with
sample rate in `audio_sample_rate`. The example reuses
`mux_video_audio_bytes` to write an MP4 containing both tracks.

## Parameters and support

| Parameter | Default | Contract |
|-----------|---------|----------|
| Height × width | 704 × 1280 | Positive multiples of 32; initialize the model and request at the same size |
| `--num-frames` | 169 | `1 + 24*n`, including the first frame |
| `--fps` | 24 | Camera trajectory and output playback rate |
| Denoising steps | 4 | Fixed by the distilled checkpoint |
| `--seed` | Case seed or 42 | Seed for the causal rollout |
| `--action` | Case action or stationary | Comma-separated `<keys>-<frames>` camera controls |
| `--fov-deg` | Case FOV or 70 | Horizontal camera field of view |
| `--dtype` | `bfloat16` | `float32` is also accepted and requires more memory |
| `--tp` / `--sp` | 1 / 1 | Experimental TP / Ulysses; product must divide 8 |

Use `w/s` for forward/back, `a/d` for strafe, `i/k` for pitch, `j/l` for yaw,
and `none` for stationary segments. Multiple keys can be combined.

## Deployment and validation scope

The integration targets NVIDIA CUDA. Full decoded output and GPU quality
validation are tracked separately from CPU mathematical parity. No minimum
GPU memory or performance claim is established by this recipe: memory depends
on dimensions, dtype, context caches, Gemma, and media codecs. Start with a
single GPU with sufficient capacity; use `--num-frames 25 --height 352 --width 640`
for a smaller smoke workload.

TP and Ulysses are implemented; the regression suite includes a real two-rank
Gloo SP comparison and a 22-latent-frame comparison against the official model
across FIFO eviction. Their GPU output quality must be checked for the chosen
configuration. Ring, CFG parallelism, pipeline parallelism, HSDP, quantization,
cache acceleration, and VAE parallelism are unsupported. Each request accepts
one image and one prompt. The step-execution adapter supports a fixed action
trajectory, not live camera updates or request batching.


Decoded stepwise output currently decodes the complete committed prefix for
each chunk, reloads the media components, and emits only the new frames and
audio samples. It does not keep persistent codec state or provide constant
per-chunk latency. Whole-request demo parity does not establish sample-level
parity for concatenated streaming media; use the default request path for
the validated full-video comparison.

## References

- [Runnable example and CLI details](../../examples/offline_inference/echo_wm/README.md)
- [Official Flash inference documentation](https://github.com/jd-opensource/JoyAI-Echo/blob/main/echo_wm/README_CAUSAL.md)
- [Model weights](https://huggingface.co/Echo-Team/Echo-WM)
