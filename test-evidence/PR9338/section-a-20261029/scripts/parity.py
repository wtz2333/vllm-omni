# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from torch.nn.attention import SDPBackend, sdpa_kernel

PROMPTS = [
    "A forest beside a clear river, photorealistic, daylight, gentle camera motion.",
    "A forest beside a clear river, photorealistic, daylight. Birds fly above the river.",
    "A forest beside a clear river, photorealistic, daylight. The camera approaches a wooden bridge.",
]


def condition(args, index):
    if args.timeline == "long":
        event = min(19, (index - 3) // 2)
        if index < 3:
            return PROMPTS[0], None
        return PROMPTS[1 if event % 2 == 0 else 2], "pan_left" if event % 2 == 0 else None
    return PROMPTS[index % len(PROMPTS)], "pan_left" if index % 2 else None


def save_output(args, role, index, latent, video):
    array = video.numpy()
    digest = hashlib.sha256(array.tobytes()).hexdigest()
    record = {
        "video_sha256": digest,
        "shape": list(array.shape),
        "frame_mean": array.mean(axis=(1, 2, 3)).tolist(),
        "frame_near_white_fraction": (array >= 250).mean(axis=(1, 2, 3)).tolist(),
    }
    tensors = {"latents": latent}
    if not args.compact:
        tensors["video"] = video
    save_file(tensors, str(args.output / f"{role}-{index:04d}.safetensors"))
    (args.output / f"{role}-{index:04d}.json").write_text(json.dumps(record) + "\n")
    if len(array):
        args.media_file.write(args.muxer.mux_video_frames(array))


def reference(args):
    from configs import config as reference_config
    from hydra import compose, initialize_config_dir
    from hymm.pipeline.causal_ar import CPTCausalInferencePipeline
    from omegaconf import OmegaConf

    from vllm_omni.diffusion.distributed.utils import get_local_device

    device = get_local_device()
    root = Path(os.environ["HY_NOMAS_REFERENCE_MODEL"])
    with initialize_config_dir(config_dir=str(Path(reference_config.__file__).parent), version_base="1.3"):
        config = compose(config_name="config", overrides=["+experiment=inference"])
    OmegaConf.set_struct(config, False)
    config.dit_weight = str(root / "DiT/distilled/diffusion_pytorch_model_ema_0_995.safetensors")
    config.model.vae.checkpoint = str(root / "VAE/PixelUMM_FT/diffusion_pytorch_model.safetensors")
    config.model.vae.parallel_mode = "none"
    config.model.text_encoder.model_name = os.environ.get(
        "HY_NOMAS_REFERENCE_TEXT_ENCODER", "Qwen/Qwen2.5-VL-7B-Instruct"
    )
    config.model.dit.seg_ar_attention_backend = "flex"
    config.parallelism.sp_size = 1
    config.parallelism.shard_text_encoder = False
    rollout = args.reference_rollout
    if rollout == "auto":
        rollout = "seg_ar" if args.history == "recent" else "chunk_ar"
    config.inference.rollout_type = rollout
    config.inference.inference_chunk_size = 16 if rollout == "chunk_ar" else 0
    config.inference.interactive_noise_max_seconds = args.capacity / 6
    config.inference.interactive_noise_hard_max_seconds = args.capacity / 6
    config.inference.sliding_window = {
        "enabled": args.history == "recent",
        "sampling": "recent",
        "past_frames": args.budget,
    }
    pipe = CPTCausalInferencePipeline(reference_config._flatten_cfg(config), device, config.dit_weight).eval()
    native_noise = pipe._fixed_initial_noise_timeline
    noise_generator = torch.Generator(device="cpu").manual_seed(args.seed)
    cache_generator = torch.Generator(device="cpu").manual_seed(args.seed ^ 0x4E4F4D4153)

    def incremental_noise(**kwargs):
        native_noise(**kwargs)
        value = torch.cat(
            [
                torch.randn(
                    (kwargs["batch_size"], pipe.in_channels, 1, kwargs["latent_h"], kwargs["latent_w"]),
                    generator=noise_generator,
                    dtype=pipe.target_dtype,
                )
                for _ in range(kwargs["latent_frames"])
            ],
            dim=2,
        )
        return value.to(pipe.device), cache_generator

    if args.noise == "incremental_v1":
        pipe._fixed_initial_noise_timeline = incremental_noise
    latents, lengths, commits = [], [], []
    decode = pipe._decode_latents
    forward = pipe.transformer.forward_with_cache

    def record_decode(value, num_frames):
        latents.append(value.detach().cpu().contiguous())
        return decode(value, num_frames)

    def record_forward(*values, **kwargs):
        if kwargs["update_cache"]:
            lengths.append(kwargs["hidden_states"].shape[2])
            commits.append(
                {
                    "history_in": int(kwargs["cache_valid_len"]),
                    "cache_text": bool(kwargs["cache_txt"]),
                    "position": int(kwargs["global_frame_counter"]),
                }
            )
            save_file(
                {"commit_input": kwargs["hidden_states"].detach().cpu().contiguous()},
                str(args.output / f"reference-commit-{len(commits) - 1:04d}.safetensors"),
            )
        return forward(*values, **kwargs)

    pipe._decode_latents = record_decode
    pipe.transformer.forward_with_cache = record_forward
    decoder = pipe.vae.create_streaming_decode_session(to_cpu=True, retain_history=False)
    records = []
    with sdpa_kernel(
        [SDPBackend.CUDNN_ATTENTION, SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]
    ):
        for index in range(args.turns):
            start = len(lengths)
            pipe.inference(
                prompts=[condition(args, index)[0]],
                num_frames=args.frames,
                height=args.height,
                width=args.width,
                guidance_scale=1.0,
                num_inference_steps=args.steps,
                flow_shift=5.0,
                seed=args.seed,
                camera_trajectory=condition(args, index)[1],
                preserve_kv_cache=True,
            )
            assert len(latents) == index + 1
            latent = latents[-1]
            parts, left = [], 0
            for chunk_index, length in enumerate(lengths[start:]):
                right = left + length
                decoder.final_segment = index + 1 == args.turns and chunk_index + 1 == len(lengths[start:])
                pixels = decoder.push(
                    pipe._prepare_vae_decode_latents(latent[:, :, left:right].to(device)),
                    pixel_frames=min(args.frames, right * 4) - left * 4,
                )
                if pixels.numel():
                    parts.append(pixels)
                left = right
            assert left == latent.shape[2]
            if index + 1 == args.turns:
                tail = decoder.finish()
                if tail.numel():
                    parts.append(tail)
            pixels = torch.cat(parts, 2)
            video = pipe._postprocess_vae_video(pixels)[0].permute(1, 2, 3, 0)
            video = video.float().mul(255).round().to(torch.uint8).contiguous()
            save_output(args, "reference", index, latent, video)
            records.append({"turn": index, "chunks": lengths[start:], "history_tokens": pipe.cache_valid_len})
            print(json.dumps(records[-1]), flush=True)
    (args.output / "reference.json").write_text(
        json.dumps(
            {
                "noise": args.noise,
                "scope": "native oracle" if args.noise == "native" else "controlled noise inputs",
                "records": records,
                "commits": commits,
            },
            indent=2,
        )
        + "\n"
    )


def candidate(args):
    # Preserve the validated runtime initialization order.
    # isort: off
    from vllm.config import LoadConfig
    from vllm.utils.torch_utils import set_default_torch_dtype

    from vllm_omni.diffusion.data import OmniDiffusionConfig
    from vllm_omni.diffusion.distributed.utils import get_local_device
    from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
    from vllm_omni.diffusion.models.hy_nomas.pipeline_hy_nomas import HyNomasPipeline
    from vllm_omni.diffusion.request import OmniDiffusionRequest
    from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
    from vllm_omni.experimental.ar_diffusion.kv_cache.config import ARDiffusionKVConfig
    from vllm_omni.experimental.ar_diffusion.kv_cache.manager import ARDiffusionKVCache
    from vllm_omni.experimental.ar_diffusion.kv_cache.state import ARDiffusionKVState
    from vllm_omni.experimental.ar_diffusion.tick_protocol import ARDiffusionControlInput, ARDiffusionTickRequest
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams
    from vllm_omni.platforms import current_omni_platform
    # isort: on

    device = get_local_device()
    config = OmniDiffusionConfig(
        model=os.environ["HY_NOMAS_MODEL"],
        dtype=torch.bfloat16,
        distributed_executor_backend="mp",
        enforce_eager=True,
        custom_pipeline_args={
            "ar_height": args.height,
            "ar_width": args.width,
            "ar_max_latent_frames": args.capacity,
            "ar_max_segments": max(16, args.turns),
            "ar_history_mode": args.history,
            "ar_history_latent_frames": args.budget,
        },
    )
    with set_default_torch_dtype(torch.bfloat16):
        pipe = HyNomasPipeline(od_config=config)
    pipe.load_weights(DiffusersPipelineLoader(LoadConfig(), config).get_all_weights(pipe))
    spec = pipe.ar_diffusion_kv_cache_spec()
    cache = ARDiffusionKVCache(
        ARDiffusionKVConfig(enable=True, chunk_size=spec.tokens_per_frame, window_chunks=1, warmup_cudagraph=False),
        num_layers=spec.num_layers,
        num_kv_heads=spec.num_kv_heads,
        head_size=spec.head_size,
        dtype=torch.bfloat16,
        block_size=spec.segmented.block_size,
        max_model_len=spec.segmented.initial_staging_tokens,
        available_bytes=current_omni_platform.mem_get_info()[0],
        kv_branches=spec.kv_branches,
        session_capacity=1,
        device=device,
        segmented=spec.segmented,
        model_owned_state_bytes_per_session=spec.model_owned_state_bytes_per_session,
    )
    owner = ARDiffusionKVState(cache, "parity", {"cond": cache.begin_request("parity")}, num_layers=spec.num_layers)
    latents, commits = [], []
    post_decode = pipe.post_decode
    forward = pipe.transformer.forward_with_cache

    def record_decode(state, **kwargs):
        latents.append(state.latents.detach().cpu().contiguous())
        return post_decode(state, **kwargs)

    def record_forward(*values, **kwargs):
        if kwargs["update_cache"]:
            commits.append(
                {
                    "history_in": int(kwargs["cache_valid_len"]),
                    "cache_text": bool(kwargs["cache_txt"]),
                    "position": int(kwargs["global_frame_counter"]),
                }
            )
            save_file(
                {"commit_input": kwargs["hidden_states"].detach().cpu().contiguous()},
                str(args.output / f"candidate-commit-{len(commits) - 1:04d}.safetensors"),
            )
        return forward(*values, **kwargs)

    pipe.post_decode = record_decode
    pipe.transformer.forward_with_cache = record_forward
    records = []
    try:
        for index in range(args.turns):
            prompt, camera = condition(args, index)
            tick = ARDiffusionTickRequest(
                "parity",
                f"turn-{index}",
                index,
                prompt=prompt,
                close_session=index + 1 == args.turns,
                controls=(ARDiffusionControlInput("camera", "hy_nomas.camera.v1", {"camera_trajectory": camera}),),
            )
            params = OmniDiffusionSamplingParams(
                height=args.height,
                width=args.width,
                num_frames=args.frames,
                seed=args.seed,
                num_inference_steps=args.steps,
                guidance_scale=1,
                extra_args=(
                    tick.to_extra_args()
                    if args.frames <= 89
                    else {
                        "session_id": "parity",
                        "close_session": index + 1 == args.turns,
                        **({"camera_trajectory": camera} if camera is not None else {}),
                    }
                ),
            )
            start = len(latents)
            with pipe.bind_ar_diffusion_state("parity", owner):
                output = pipe(
                    DiffusionRequestBatch(
                        [OmniDiffusionRequest(prompt=tick.prompt, sampling_params=params, request_id=tick.request_id)]
                    )
                )
            latent = torch.cat(latents[start:], 2)
            video = torch.from_numpy(output.output["payload"]["video"].copy())
            save_output(args, "candidate", index, latent, video)
            records.append({"turn": index, "metadata": output.output["metadata"]["hy_nomas"]})
            print(json.dumps(records[-1]), flush=True)
    finally:
        owner.close()
        pipe.close_ar_diffusion_session("parity")
    (args.output / "candidate.json").write_text(json.dumps({"records": records, "commits": commits}, indent=2) + "\n")


def compare(args):
    records = []
    for index in range(args.turns):
        expected = load_file(str(args.output / f"reference-{index:04d}.safetensors"))
        actual = load_file(str(args.output / f"candidate-{index:04d}.safetensors"))
        row = {"turn": index}
        for key in ("latents", "video"):
            if key == "video" and args.compact:
                first = json.loads((args.output / f"reference-{index:04d}.json").read_text())
                second = json.loads((args.output / f"candidate-{index:04d}.json").read_text())
                equal = first["shape"] == second["shape"] and first["video_sha256"] == second["video_sha256"]
                row[key] = {"exact": equal, "max_abs": 0 if equal else None, "mae": 0 if equal else None}
                continue
            assert actual[key].shape == expected[key].shape
            difference = (actual[key].float() - expected[key].float()).abs()
            row[key] = {
                "max_abs": float(difference.max()),
                "mae": float(difference.mean()),
                "different_elements": int(torch.count_nonzero(difference)),
                "exact": not bool(torch.count_nonzero(difference)),
            }
        records.append(row)
    result = {"history": args.history, "noise": args.noise, "records": records}
    (args.output / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    assert all(row[key]["exact"] for row in records for key in ("latents", "video")), result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", choices=("reference", "candidate", "compare"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--history", choices=("full", "recent"), default="full")
    parser.add_argument("--noise", choices=("native", "incremental_v1"), default="native")
    parser.add_argument("--budget", type=int, default=18)
    parser.add_argument("--capacity", type=int, default=120)
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--frames", type=int, default=49)
    parser.add_argument("--turns", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeline", choices=("cycle", "long"), default="cycle")
    parser.add_argument("--compact", action="store_true")
    parser.add_argument("--reference-rollout", choices=("auto", "seg_ar", "chunk_ar"), default="auto")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        if args.worker == "compare":
            compare(args)
        else:
            from vllm_omni.diffusion.utils.media_utils import FragmentedMP4Muxer

            with (args.output / f"{args.worker}.mp4").open("wb") as args.media_file:
                args.muxer = FragmentedMP4Muxer(width=args.width, height=args.height, fps=24)
                try:
                    {"reference": reference, "candidate": candidate}[args.worker](args)
                finally:
                    args.media_file.write(args.muxer.close())
