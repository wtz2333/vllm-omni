# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run LingBot-World 2.0 as one stepwise generate() over many latent chunks.

This example issues a single AsyncOmni request with step execution and streaming
output. It is not a public HTTP or WebSocket protocol. Each streamed output is
one three-latent-frame AR block plus identity metadata. The current
implementation supports at most ten chunks per request because the image
condition is bounded to 117 pixel frames.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

_MODEL = "robbyant/lingbot-world-v2-14b-causal-fast-diffusers"
_FRAMES_PER_BLOCK = 3
_TEMPORAL_COMPRESSION = 4
_MAX_CHUNKS = 10


def _num_frames_for_chunks(num_chunks: int) -> int:
    latent_frames = num_chunks * _FRAMES_PER_BLOCK
    return (latent_frames - 1) * _TEMPORAL_COMPRESSION + 1


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one stepwise LingBot-World 2.0 generate() over N latent chunks.")
    parser.add_argument("--model", default=_MODEL, help="Hugging Face model ID or local checkpoint path.")
    parser.add_argument("--image", required=True, help="Initial RGB image.")
    parser.add_argument("--prompt", required=True, help="Scene prompt for the whole request.")
    parser.add_argument(
        "--events",
        help="JSONL file with one optional frames list per AR block. Defaults to idle camera chunks.",
    )
    parser.add_argument("--chunks", type=int, default=3, help="Number of AR blocks when --events is omitted.")
    parser.add_argument("--output-dir", required=True, help="Directory for chunk latents and metadata.")
    parser.add_argument("--request-id", default="lingbot-world-stepwise", help="Stable request/session identifier.")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.1)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--enforce-eager", action="store_true")
    return parser.parse_args(argv)


def _load_script(path: Path | None, num_chunks: int) -> list[list[list[str]]]:
    if path is None:
        if num_chunks <= 0 or num_chunks > _MAX_CHUNKS:
            raise ValueError(f"--chunks must be between 1 and {_MAX_CHUNKS}.")
        return [[[] for _ in range(_FRAMES_PER_BLOCK)] for _ in range(num_chunks)]
    script: list[list[list[str]]] = []
    for line_number, raw_line in enumerate(path.read_text().splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"events line {line_number} is not valid JSON: {exc.msg}.") from None
        if not isinstance(value, dict):
            raise ValueError(f"events line {line_number} must be a JSON object.")
        frames = value.get("frames")
        if frames is None:
            frames = [[] for _ in range(_FRAMES_PER_BLOCK)]
        if not isinstance(frames, list) or len(frames) != _FRAMES_PER_BLOCK:
            raise ValueError(f"events line {line_number} frames must contain exactly three lists.")
        for frame in frames:
            if not isinstance(frame, list) or any(not isinstance(action, str) for action in frame):
                raise ValueError(f"events line {line_number} frames must contain only action strings.")
        script.append(frames)
    if not script:
        raise ValueError("events file must contain at least one event.")
    if len(script) > _MAX_CHUNKS:
        raise ValueError(f"events file must contain at most {_MAX_CHUNKS} chunks.")
    return script


def _validate_args(args: argparse.Namespace) -> tuple[Path, Path | None, Path]:
    image = Path(args.image).expanduser().resolve()
    events = Path(args.events).expanduser().resolve() if args.events else None
    output_dir = Path(args.output_dir).expanduser().resolve()
    if not image.is_file():
        raise ValueError("--image must point to an existing file.")
    if events is not None and not events.is_file():
        raise ValueError("--events must point to an existing JSONL file.")
    if not args.prompt.strip():
        raise ValueError("--prompt must contain non-whitespace text.")
    if args.height <= 0 or args.width <= 0 or args.height % 16 or args.width % 16:
        raise ValueError("--height and --width must be positive multiples of 16.")
    if args.tensor_parallel_size <= 0:
        raise ValueError("--tensor-parallel-size must be positive.")
    if not math.isfinite(args.gpu_memory_fraction) or not 0 < args.gpu_memory_fraction <= 1:
        raise ValueError("--gpu-memory-fraction must be in (0, 1].")
    return image, events, output_dir


def _chunk_metadata(output: Any) -> dict[str, Any]:
    multimodal = getattr(output, "multimodal_output", None) or {}
    metadata = multimodal.get("metadata") if isinstance(multimodal, dict) else None
    if not isinstance(metadata, dict) or not isinstance(metadata.get("ar_diffusion"), dict):
        raise RuntimeError("Expected ar_diffusion metadata on each streamed LingBot chunk.")
    return dict(metadata["ar_diffusion"])


async def run(argv: Sequence[str] | None = None) -> Path:
    args = parse_args(argv)
    image, events_path, output_dir = _validate_args(args)
    script = _load_script(events_path, args.chunks)
    num_frames = _num_frames_for_chunks(len(script))
    output_dir.mkdir(parents=True, exist_ok=True)

    import torch

    from vllm_omni.entrypoints.async_omni import AsyncOmni
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    engine = AsyncOmni(
        model=args.model,
        engine_backend="vllm_omni.experimental.ar_diffusion.engine.ARDiffusionEngine",
        enforce_eager=args.enforce_eager,
        tensor_parallel_size=args.tensor_parallel_size,
        max_num_seqs=1,
        step_execution=True,
        diffusion_streaming_output=True,
        model_config={
            "ar_diffusion_height": args.height,
            "ar_diffusion_width": args.width,
            "ar_diffusion_kv_config": {
                "gpu_memory_fraction": args.gpu_memory_fraction,
                "warmup_cudagraph": True,
            },
        },
    )
    sampling = OmniDiffusionSamplingParams(
        height=args.height,
        width=args.width,
        num_frames=num_frames,
        num_inference_steps=4,
        max_sequence_length=512,
        seed=args.seed,
        output_type="latent",
        extra_args={
            "flow_shift": 5.0,
            "camera_action_script": script,
        },
    )
    prompt = {
        "prompt": args.prompt,
        "multi_modal_data": {"image": str(image)},
    }
    measurements: list[dict[str, Any]] = []
    chunk_index = 0
    try:
        async for output in engine.generate(prompt, sampling, request_id=args.request_id):
            images = getattr(output, "images", None)
            if not images:
                continue
            if len(images) != 1 or not isinstance(images[0], torch.Tensor):
                raise RuntimeError("Expected one latent tensor from each stepwise LingBot chunk.")
            latent = images[0].detach().float().cpu()
            metadata = _chunk_metadata(output)
            torch.save(latent, output_dir / f"chunk_{chunk_index:03d}.pt")
            (output_dir / f"chunk_{chunk_index:03d}.json").write_text(
                json.dumps(metadata, indent=2, sort_keys=True) + "\n"
            )
            measurements.append(
                {
                    "chunk_index": chunk_index,
                    "shape": list(latent.shape),
                    "finite": bool(torch.isfinite(latent).all()),
                    "metadata": metadata,
                    "finished": bool(getattr(output, "finished", False)),
                }
            )
            print(json.dumps(measurements[-1], sort_keys=True), flush=True)
            chunk_index += 1
    finally:
        engine.shutdown()

    if chunk_index != len(script):
        raise RuntimeError(f"Expected {len(script)} streamed chunks, got {chunk_index}.")
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps({"chunks": measurements}, indent=2, sort_keys=True) + "\n")
    return summary_path


def main(argv: Sequence[str] | None = None) -> Path:
    return asyncio.run(run(argv))


if __name__ == "__main__":
    main()
