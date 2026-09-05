# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Generate an Echo-WM Flash video with audio from an image and camera actions."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path, help="Flash safetensors file or its containing directory.")
    parser.add_argument("--gemma-path", required=True, type=Path, help="Local Gemma 3 text encoder directory.")
    parser.add_argument(
        "--case-json", type=Path, help="Upstream case.json; CLI image/prompt/action overrides its fields."
    )
    parser.add_argument("--image", type=Path)
    parser.add_argument("--prompt")
    parser.add_argument("--action", help="Action DSL, e.g. 'w-96,l-96'; omit for a stationary camera.")
    parser.add_argument("--output", type=Path, default=Path("echo_wm.mp4"))
    parser.add_argument("--height", type=int, default=704)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument(
        "--num-frames", type=int, default=169, help="1 + 24*n frames; defaults to 169 even for longer cases."
    )
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--seed", type=int, help="Overrides case seed; otherwise 42.")
    parser.add_argument("--fov-deg", type=float, help="Overrides case FOV; otherwise 70 degrees.")
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--tensor-parallel-size", "--tp", type=int, default=1)
    parser.add_argument("--ulysses-degree", "--sp", type=int, default=1)
    parser.add_argument("--attention-backend", default="TORCH_SDPA", help="Diffusion attention backend.")
    parser.add_argument("--dry-run", action="store_true", help="Resolve and validate inputs without constructing Omni.")
    args = parser.parse_args()
    try:
        return resolve_args(args)
    except (OSError, ValueError, TypeError) as exc:
        parser.error(str(exc))


def resolve_args(args: argparse.Namespace) -> argparse.Namespace:
    case = {}
    if args.case_json is not None:
        args.case_json = args.case_json.expanduser().resolve()
        case = json.loads(args.case_json.read_text())
        if not isinstance(case, dict):
            raise ValueError("--case-json must contain a JSON object")
    if args.image is None and args.case_json is not None:
        args.image = args.case_json.parent / case.get("image", "input.jpg")
    if args.image is None:
        raise ValueError("Provide --image or --case-json")
    args.image = args.image.expanduser().resolve()
    if not args.image.is_file():
        raise ValueError(f"Image does not exist: {args.image}")
    args.prompt = args.prompt if args.prompt is not None else case.get("prompt", "")
    if not isinstance(args.prompt, str) or not args.prompt.strip():
        raise ValueError("Provide a nonempty --prompt or case prompt")
    args.action = args.action if args.action is not None else case.get("action")
    if args.action is not None and not isinstance(args.action, str):
        raise ValueError("Action must be a DSL string")
    args.seed = args.seed if args.seed is not None else int(case.get("seed", 42))
    args.fov_deg = args.fov_deg if args.fov_deg is not None else float(case.get("fov_deg", 70.0))
    args.model = args.model.expanduser().resolve()
    if args.model.is_dir():
        args.model = args.model / "echo-wm-flash.safetensors"
    if not args.model.is_file():
        raise ValueError(f"Flash checkpoint does not exist: {args.model}")
    args.gemma_path = args.gemma_path.expanduser().resolve()
    if not args.gemma_path.is_dir():
        raise ValueError(f"Gemma directory does not exist: {args.gemma_path}")
    args.output = args.output.expanduser().resolve()
    if args.output.suffix.lower() != ".mp4":
        raise ValueError("--output must end in .mp4")
    if args.height <= 0 or args.width <= 0 or args.height % 32 or args.width % 32:
        raise ValueError("Height and width must be positive multiples of 32")
    if args.num_frames < 25 or (args.num_frames - 1) % 24:
        raise ValueError("--num-frames must be 1 + 24*n with n >= 1 (25, 49, ..., 169, ...)")
    if not math.isfinite(args.fps) or args.fps <= 0:
        raise ValueError("--fps must be positive and finite")
    if not math.isfinite(args.fov_deg) or not 0 < args.fov_deg < 180:
        raise ValueError("--fov-deg must lie between 0 and 180 degrees")
    if args.tensor_parallel_size < 1 or args.ulysses_degree < 1:
        raise ValueError("TP and SP degrees must be positive")
    if 8 % (args.tensor_parallel_size * args.ulysses_degree):
        raise ValueError("TP * SP must divide the 8 UCPE attention heads")
    return args


def save_output(output, path: Path, *, fps: float) -> dict:
    import numpy as np
    import torch

    from vllm_omni.diffusion.utils.media_utils import mux_video_audio_bytes

    if not output.images or len(output.images) != 1:
        raise ValueError("Expected exactly one decoded video in OmniRequestOutput.images")
    video = output.images[0]
    if isinstance(video, torch.Tensor):
        video = video.detach().cpu().numpy()
    video = np.asarray(video)
    if video.ndim != 4 or video.shape[-1] != 3 or video.dtype != np.uint8:
        raise ValueError("Echo-WM video output must be uint8 RGB [T, H, W, 3]")
    media = output.multimodal_output or {}
    audio = media.get("audio")
    sample_rate = media.get("audio_sample_rate")
    if audio is None or sample_rate is None:
        raise ValueError("Echo-WM output is missing audio or its sample rate")
    if isinstance(audio, torch.Tensor):
        audio = audio.detach().cpu().float().numpy()
    audio = np.asarray(audio, dtype=np.float32)
    # The upstream MP4 writer converts to signed PCM16 before AAC encoding.
    # Mirror that encoding boundary without changing the pipeline's waveform.
    audio_for_mux = (np.clip(audio, -1, 1) * 32767).astype(np.int16).astype(np.float32) / 32768
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(mux_video_audio_bytes(video, audio_for_mux, fps=fps, audio_sample_rate=int(sample_rate), crf="23"))
    return {"video_shape": list(video.shape), "audio_shape": list(audio.shape), "audio_sample_rate": int(sample_rate)}


def main() -> None:
    args = parse_args()
    from vllm_omni.diffusion.models.echo_wm.actions import parse_action_string

    if args.action is not None:
        parse_action_string(args.action)
    configuration = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    print(json.dumps(configuration, indent=2))
    if args.dry_run:
        return

    from PIL import Image

    from vllm_omni.entrypoints.omni import Omni
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    with Image.open(args.image) as source:
        image = source.convert("RGB")
    prompt = {"prompt": args.prompt, "multi_modal_data": {"image": image}}
    sampling_params = OmniDiffusionSamplingParams(
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        fps=args.fps,
        num_inference_steps=4,
        seed=args.seed,
        output_type="pt",
        extra_args={"echo_wm_action": args.action, "echo_wm_fov_deg": args.fov_deg},
    )
    omni = Omni(
        model=str(args.model),
        model_class_name="EchoWMCausalPipeline",
        dtype=args.dtype,
        model_config={
            "echo_wm_gemma_path": str(args.gemma_path),
            "echo_wm_height": args.height,
            "echo_wm_width": args.width,
        },
        tensor_parallel_size=args.tensor_parallel_size,
        ulysses_degree=args.ulysses_degree,
        diffusion_attention_backend=args.attention_backend,
        enforce_eager=True,
    )
    try:
        import torch

        measure_memory = (
            args.tensor_parallel_size == 1
            and args.ulysses_degree == 1
            and torch.cuda.is_initialized()
            and torch.cuda.memory_allocated() > 0
        )
        if measure_memory:
            torch.accelerator.reset_peak_memory_stats()
        start = time.perf_counter()
        outputs = omni.generate(prompt, sampling_params)
        generation_seconds = time.perf_counter() - start
        peak_allocated_gib = torch.accelerator.max_memory_allocated() / 2**30 if measure_memory else None
        if len(outputs) != 1:
            raise ValueError(f"Expected one generated result, received {len(outputs)}")
        media = save_output(outputs[0], args.output, fps=args.fps)
    finally:
        omni.close()
    metadata = {
        "configuration": configuration,
        "generation_seconds": generation_seconds,
        "peak_allocated_gib": peak_allocated_gib,
        **media,
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved video with audio to {args.output} ({generation_seconds:.2f} s generation)")


if __name__ == "__main__":
    main()
