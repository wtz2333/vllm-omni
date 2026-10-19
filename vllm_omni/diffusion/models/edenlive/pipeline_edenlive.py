# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""EdenLive 5B few-step reference-video rendering on experimental AR-Diffusion."""

import math
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, ClassVar

import ftfy
import numpy as np
import regex as re
import torch
import torch.nn.functional as F
from diffusers.utils.torch_utils import randn_tensor
from PIL import Image
from safetensors.torch import load_file
from torch import nn
from transformers import AutoModel, AutoTokenizer, UMT5EncoderModel
from vllm.model_executor.models.utils import AutoWeightsLoader

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.models.edenlive.conditioning import ProxyDinoEncoder, projection_weights
from vllm_omni.diffusion.models.edenlive.sampling import TIMESTEPS, step
from vllm_omni.diffusion.models.edenlive.transformer import (
    CHUNK_FRAMES,
    KV_BRANCH,
    SINK_FRAMES,
    WINDOW_FRAMES,
    EdenLiveTransformer,
)
from vllm_omni.diffusion.models.edenlive.weights import (
    component_config,
    component_file,
    decoder_state_bytes,
    load_vae,
)
from vllm_omni.diffusion.models.interface import (
    ReferenceVideoDecodeSpec,
    SupportImageInput,
    SupportsComponentDiscovery,
    SupportsStepExecution,
)
from vllm_omni.diffusion.models.progress_bar import ProgressBarMixin
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.utils.video_decode import decode_path_video_frames
from vllm_omni.diffusion.worker.input_batch import InputBatch
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.diffusion.worker.utils import StepRequestState
from vllm_omni.experimental.ar_diffusion.capability import ARDiffusionKVBranchSpec, ARDiffusionKVCacheSpec
from vllm_omni.experimental.ar_diffusion.kv_cache.state import ARDiffusionKVState
from vllm_omni.experimental.ar_diffusion.streaming_decode import StreamingDecodeState, WanStreamingDecoder
from vllm_omni.inputs.data import OmniDiffusionSamplingParams


def rgb_image(value: Any) -> Image.Image:
    if isinstance(value, list) and len(value) == 1:
        value = value[0]
    if isinstance(value, str | Path):
        with Image.open(value) as image:
            return image.convert("RGB")
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray) and value.dtype == np.uint8 and value.ndim == 3 and value.shape[-1] == 3:
        return Image.fromarray(value)
    raise ValueError("EdenLive image/frame must be a PIL image, image path, or HWC RGB uint8 array")


def first_frame_pixels(value: Any, height: int, width: int) -> torch.Tensor:
    """Match the reference's tensor-bilinear resize followed by a center crop."""
    pixels = torch.from_numpy(np.array(rgb_image(value), copy=True)).permute(2, 0, 1).float().unsqueeze(0)
    scale = max(height / pixels.shape[2], width / pixels.shape[3])
    pixels = F.interpolate(
        pixels,
        size=(math.ceil(pixels.shape[2] * scale), math.ceil(pixels.shape[3] * scale)),
        mode="bilinear",
        align_corners=False,
    )
    top, left = (pixels.shape[2] - height) // 2, (pixels.shape[3] - width) // 2
    return (pixels[:, :, top : top + height, left : left + width] / 127.5 - 1).unsqueeze(2)


def proxy_frames(value: Any, num_frames: int, *, height: int | None = None, width: int | None = None) -> torch.Tensor:
    """Select decoded frames 0,4,8,... without changing their timeline."""
    if isinstance(value, Mapping):
        value = value.get("frames", value.get("data", value.get("video")))
    if isinstance(value, str | Path):
        value = decode_path_video_frames(value, max_frames=num_frames)
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    if not isinstance(value, list | tuple | np.ndarray) or len(value) < num_frames:
        raise ValueError(f"EdenLive needs at least {num_frames} decoded reference video frames")
    frames = []
    for index in range(0, num_frames, 4):
        frame = rgb_image(value[index])
        if height is not None and width is not None and frame.size != (width, height):
            source_w, source_h = frame.size
            crop_w = min(source_w, round(source_h * width / height))
            crop_h = min(source_h, round(source_w * height / width))
            left, top = (source_w - crop_w) // 2, (source_h - crop_h) // 2
            frame = frame.crop((left, top, left + crop_w, top + crop_h)).resize(
                (width, height), Image.Resampling.BILINEAR
            )
        frames.append(np.array(frame, copy=True))
    if len({frame.shape for frame in frames}) != 1:
        raise ValueError("EdenLive reference frames must have the same size")
    return torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2)


def validate_sampling(sampling: OmniDiffusionSamplingParams, height: int, width: int) -> int:
    requested_size = (
        height if sampling.height is None else sampling.height,
        width if sampling.width is None else sampling.width,
    )
    if requested_size != (height, width):
        raise ValueError(f"EdenLive request size must match the AR cache geometry {height}x{width}")
    if sampling.num_inference_steps not in (None, 4):
        raise ValueError("EdenLive 5B few-step requires num_inference_steps=4")
    if sampling.timesteps is not None or sampling.sigmas is not None:
        raise ValueError("EdenLive 5B uses its fixed four-step schedule")
    if sampling.max_sequence_length not in (None, 512):
        raise ValueError("EdenLive 5B uses max_sequence_length=512")
    if sampling.extra_args.get("flow_shift") not in (None, 5, 5.0):
        raise ValueError("EdenLive 5B uses the released shift-5 schedule")
    if sampling.guidance_scale is not None and sampling.guidance_scale != 1.0:
        raise ValueError("EdenLive 5B few-step requires guidance_scale=1.0")
    if sampling.num_outputs_per_prompt != 1:
        raise ValueError("EdenLive currently supports one output per request")
    if sampling.output_type not in (None, "pil", "np", "pt", "latent"):
        raise ValueError("Unsupported EdenLive output_type")
    frames = sampling.num_frames if sampling.num_frames is not None else 117
    if not isinstance(frames, int) or isinstance(frames, bool) or frames < 1 or (frames - 1) % 4:
        raise ValueError("EdenLive num_frames must be positive and equal to 4*k+1")
    if sampling.extra_args.get("ar_diffusion_tick") is not None:
        raise ValueError("EdenLive accepts a reference-video rollout, not control-only AR ticks")
    return frames


class EdenLivePipeline(
    nn.Module, SupportImageInput, SupportsComponentDiscovery, SupportsStepExecution, ProgressBarMixin
):
    """One reference clip per request, with four probes and a clean KV commit per chunk."""

    _dit_modules: ClassVar[list[str]] = ["transformer"]
    _encoder_modules: ClassVar[list[str]] = ["text_encoder", "proxy_encoder"]
    _vae_modules: ClassVar[list[str]] = ["vae"]
    dummy_run_num_frames: ClassVar[int] = 0
    supports_step_execution: ClassVar[bool] = True
    supports_chunk_step_grouping: ClassVar[bool] = True

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = "") -> None:
        super().__init__()
        del prefix
        self.od_config = od_config
        parallel = od_config.parallel_config
        if parallel.world_size != 1 or parallel.use_hsdp or parallel.text_encoder_tp_size != 1:
            raise ValueError("EdenLive 5B currently supports single-device inference")
        if not od_config.enforce_eager:
            raise ValueError("EdenLive 5B currently requires enforce_eager=True")
        if od_config.quantization_config is not None:
            raise ValueError("EdenLive 5B currently supports unquantized weights")
        if od_config.enable_cpu_offload or od_config.enable_layerwise_offload:
            raise ValueError("EdenLive 5B currently requires resident model components")
        if od_config.step_execution and not od_config.streaming_output:
            raise ValueError("EdenLive step execution requires streaming_output=True")
        engine = str(od_config.engine_backend or "")
        if engine != "vllm_omni.experimental.ar_diffusion.engine.ARDiffusionEngine":
            raise ValueError("EdenLive requires the edenlive_5b deploy config and ARDiffusionEngine")
        self.device = get_local_device()
        self.height = int(od_config.model_config.get("ar_diffusion_height", 704))
        self.width = int(od_config.model_config.get("ar_diffusion_width", 1280))
        if min(self.height, self.width) <= 0 or self.height % 32 or self.width % 32:
            raise ValueError("EdenLive dimensions must be positive multiples of 32")
        model, revision, dtype = od_config.model, od_config.revision, od_config.dtype
        config = component_config(model, "dit-5b", revision)
        if config.get("in_channels") != 48 or config.get("out_channels") != 48:
            raise ValueError("Expected the released EdenLive dit-5b few-step configuration")
        self.transformer = EdenLiveTransformer(config).to(dtype=dtype)
        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=model,
                subfolder="dit-5b",
                revision=revision,
                prefix="transformer.",
                fall_back_to_pt=False,
            )
        ]
        # Avoid direct-to-CUDA Hub loads; placement stays in the platform allocator.
        with torch.device("cpu"):
            self.tokenizer = AutoTokenizer.from_pretrained(model, subfolder="umt5/tokenizer", revision=revision)
            self.text_encoder = UMT5EncoderModel.from_pretrained(
                model, subfolder="umt5/text_encoder", revision=revision, torch_dtype=dtype
            ).eval()
            backbone = AutoModel.from_pretrained(
                model, subfolder="vision-encoder/backbone", revision=revision, torch_dtype=torch.float32
            ).eval()
            self.proxy_encoder = ProxyDinoEncoder(backbone).eval()
            projection = projection_weights(
                load_file(component_file(model, "vision-encoder/projection.safetensors", revision)),
                set(self.proxy_encoder.state_dict()),
            )
            self.proxy_encoder.load_state_dict(projection, strict=True)
            self.vae = load_vae(model, dtype, revision)
        self.text_encoder.to(self.device)
        self.proxy_encoder.to(self.device)
        self.vae.to(self.device)
        self.decoder = WanStreamingDecoder(self.vae)
        self._decode_states: dict[str, StreamingDecodeState] = {}
        self._ar_state: ARDiffusionKVState | None = None

    @staticmethod
    def reference_video_decode_spec(*, num_frames=None, extra_args=None) -> ReferenceVideoDecodeSpec:
        return ReferenceVideoDecodeSpec(max_frames=num_frames or 117, keep="first")

    def ar_diffusion_kv_cache_spec(self) -> ARDiffusionKVCacheSpec:
        config = self.transformer.config
        latent_area = (self.height // 16) * (self.width // 16)
        itemsize = torch.empty((), dtype=self.transformer.dtype).element_size()
        # Retained decoder history, first frame, text, and current proxy/latent chunk.
        owned = decoder_state_bytes(self.vae, self.height, self.width)
        owned += (48 * latent_area + 512 * config.text_dim + CHUNK_FRAMES * (48 + 1024) * latent_area) * itemsize
        return ARDiffusionKVCacheSpec(
            num_layers=config.num_layers,
            num_kv_heads=config.num_attention_heads,
            head_size=config.attention_head_dim,
            tokens_per_frame=latent_area // 4,
            frames_per_block=CHUNK_FRAMES,
            window_frames=WINDOW_FRAMES - SINK_FRAMES,
            sink_frames=SINK_FRAMES,
            kv_branches=(ARDiffusionKVBranchSpec(KV_BRANCH, 0),),
            session_capacity=1,
            model_owned_state_bytes_per_session=owned,
        )

    @contextmanager
    def bind_ar_diffusion_state(self, session_id: str, state: ARDiffusionKVState) -> Iterator[None]:
        if self._ar_state is not None or state.session_id != session_id:
            raise RuntimeError("EdenLive AR state is already bound or belongs to another session")
        spec = self.ar_diffusion_kv_cache_spec()
        if state.kv_cache.spec.reset_at_boundary or (
            state.kv_cache.spec.sliding_window,
            state.kv_cache.spec.sink_chunks,
        ) != (
            spec.window_frames * spec.tokens_per_frame,
            SINK_FRAMES,
        ):
            raise ValueError("EdenLive requires a 12-frame recent window and 6 sink frames")
        self._ar_state = state
        try:
            yield
        finally:
            self._ar_state = None

    def reset_ar_diffusion_session(self, session_id: str) -> None:
        self.close_ar_diffusion_session(session_id)

    def close_ar_diffusion_session(self, session_id: str) -> None:
        decode_state = self._decode_states.pop(session_id, None)
        if decode_state is not None:
            self.decoder.release(decode_state)

    def _bound_state(self) -> ARDiffusionKVState:
        if self._ar_state is None:
            raise RuntimeError("EdenLive requires a bound ARDiffusionModelRunner session")
        return self._ar_state

    def _encode_prompt(self, prompt: str) -> torch.Tensor:
        import html

        prompt = re.sub(r"\s+", " ", html.unescape(html.unescape(ftfy.fix_text(prompt)))).strip()
        inputs = self.tokenizer(prompt, padding="max_length", max_length=512, truncation=True, return_tensors="pt")
        mask = inputs.attention_mask.to(self.device)
        encoded = self.text_encoder(inputs.input_ids.to(self.device), mask).last_hidden_state
        return encoded.masked_fill(~mask.bool().unsqueeze(-1), 0).to(self.transformer.dtype)

    def _latent_stats(self) -> tuple[torch.Tensor, torch.Tensor]:
        mean = torch.tensor(self.vae.config.latents_mean, device=self.device, dtype=self.vae.dtype).view(1, -1, 1, 1, 1)
        inv_std = 1 / torch.tensor(self.vae.config.latents_std, device=self.device, dtype=self.vae.dtype).view(
            1, -1, 1, 1, 1
        )
        return mean, inv_std

    @torch.no_grad()
    def prepare_encode(self, state: StepRequestState, **kwargs: Any) -> StepRequestState:
        ar = self._bound_state()
        if ar.adapter(KV_BRANCH).absolute_num_computed_tokens:
            raise ValueError("EdenLive starts a new rollout at frame zero; reset the AR session first")
        frames = validate_sampling(state.sampling, self.height, self.width)
        prompt = state.prompt
        if not isinstance(prompt, dict) or not isinstance(prompt.get("prompt"), str):
            raise ValueError("EdenLive requires text plus multi_modal_data.image and multi_modal_data.video")
        media = prompt.get("multi_modal_data", {})
        if not isinstance(media, Mapping):
            raise ValueError("EdenLive multi_modal_data must contain image and video references")
        pixels = first_frame_pixels(media.get("image"), self.height, self.width)
        proxy = proxy_frames(media.get("video"), frames, height=self.height, width=self.width).float() / 127.5 - 1
        generator = state.sampling.generator
        if not isinstance(generator, torch.Generator):
            generator = torch.Generator(device=self.device).manual_seed(state.sampling.seed or 0)
            state.sampling.generator = generator
        state.prompt_embeds = self._encode_prompt(prompt["prompt"])
        latent = self.vae.encode(pixels.to(self.device, self.vae.dtype)).latent_dist.mode()
        mean, inv_std = self._latent_stats()
        condition = (latent - mean) * inv_std
        latent_frames = (frames - 1) // 4 + 1
        # The reference draws the whole initial trajectory before any stochastic step.
        noise = randn_tensor(
            (1, 48, latent_frames, self.height // 16, self.width // 16),
            generator=generator,
            device=self.device,
            dtype=self.transformer.dtype,
        ).cpu()
        state.chunk_index = 0
        state.total_chunks = (latent_frames + CHUNK_FRAMES - 1) // CHUNK_FRAMES
        state.chunk_num_steps = 4
        state.extra = {"proxy": proxy, "condition": condition, "noise": noise, "session_id": ar.session_id}
        self._prepare_chunk(state)
        return state

    def _pin_first_frame(self, state: StepRequestState) -> None:
        if state.chunk_index == 0:
            state.latents[:, :, :1] = state.extra["condition"]

    def _prepare_chunk(self, state: StepRequestState) -> None:
        start = state.chunk_index * CHUNK_FRAMES
        state.latents = state.extra["noise"][:, :, start : start + CHUNK_FRAMES].to(self.device, copy=True)
        self._pin_first_frame(state)
        frames = (state.extra["proxy"][start : start + CHUNK_FRAMES].to(self.device) + 1) * 0.5
        state.extra["control"] = self.proxy_encoder(
            frames, latent_size=(self.height // 16, self.width // 16), dtype=self.transformer.dtype
        )
        state.step_index = state.step_in_chunk = 0
        # The reference round-trips configured timesteps through FP32 sigmas
        # on the execution device; CUDA division changes the model input by an ULP.
        state.timesteps = torch.tensor(TIMESTEPS, device=self.device, dtype=torch.float32) / 1000 * 1000

    @torch.no_grad()
    def denoise_step(
        self, input_batch: InputBatch | None, *, states: Sequence[StepRequestState] | None = None, **kwargs: Any
    ):
        if states is None or len(states) != 1:
            raise ValueError("EdenLive supports one request at a time")
        state = states[0]
        self._pin_first_frame(state)
        timestep = state.current_timestep.expand(1)
        if state.chunk_index == 0:
            timestep = timestep[:, None].expand(1, state.latents.shape[2]).clone()
            timestep[:, 0] = 0
        return self.transformer(
            state.latents,
            state.extra["control"],
            timestep,
            state.prompt_embeds,
            state=self._bound_state(),
            start_frame=state.chunk_index * CHUNK_FRAMES,
        )

    def step_scheduler(self, state: StepRequestState, noise_pred: torch.Tensor, **kwargs: Any) -> None:
        state.latents = step(state.latents, noise_pred, state.step_in_chunk, state.sampling.generator)
        self._pin_first_frame(state)
        state.step_in_chunk += 1
        state.step_index = state.step_in_chunk

    @torch.no_grad()
    def post_decode(self, state: StepRequestState, **kwargs: Any) -> DiffusionOutput:
        ar = self._bound_state()
        chunk_index = state.chunk_index
        self.transformer(
            state.latents,
            state.extra["control"],
            torch.zeros(1, device=self.device, dtype=torch.float32),
            state.prompt_embeds,
            state=ar,
            start_frame=chunk_index * CHUNK_FRAMES,
            commit=True,
        )
        if state.sampling.output_type == "latent":
            payload = {"latents": state.latents.cpu()}
        else:
            session_id = state.extra["session_id"]
            if session_id not in self._decode_states:
                self._decode_states[session_id] = self.decoder.new_decode_state(session_id)
            mean, inv_std = self._latent_stats()
            pixels = self.decoder.decode_chunk(
                state.latents.to(self.vae.dtype) / inv_std + mean, self._decode_states[session_id]
            )
            payload = {
                "video": ((pixels[0].permute(1, 2, 3, 0).float() + 1) * 127.5)
                .round()
                .clamp(0, 255)
                .byte()
                .cpu()
                .numpy()
            }
        state.chunk_index += 1
        finished = state.request_denoise_completed
        output = DiffusionOutput(
            output={"payload": payload}, chunk_index=chunk_index, total_chunks=state.total_chunks, finished=finished
        )
        if not finished:
            self._prepare_chunk(state)
        return output

    @torch.no_grad()
    def forward(self, req: OmniDiffusionRequest | DiffusionRequestBatch) -> DiffusionOutput:
        if isinstance(req, DiffusionRequestBatch):
            if len(req.requests) != 1:
                raise ValueError("EdenLive supports one request at a time")
            req = req.requests[0]
        ar = self._bound_state()
        ar.reset()
        self.close_ar_diffusion_session(ar.session_id)
        state = StepRequestState(request_id=req.request_id, prompt=req.prompt, sampling=req.sampling_params)
        try:
            self.prepare_encode(state)
            outputs = []
            with self.progress_bar(total=state.total_chunks * 4) as progress:
                while not state.request_denoise_completed:
                    for _ in range(4):
                        prediction = self.denoise_step(None, states=[state])
                        self.step_scheduler(state, prediction)
                        progress.update()
                    outputs.append(self.post_decode(state).output["payload"])
            if state.sampling.output_type == "latent":
                payload = {"latents": torch.cat([item["latents"] for item in outputs], dim=2)}
            else:
                payload = {"video": np.concatenate([item["video"] for item in outputs], axis=0)}
            return DiffusionOutput(output={"payload": payload})
        finally:
            self.close_ar_diffusion_session(ar.session_id)
            ar.reset()

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        return AutoWeightsLoader(self).load_weights(weights)


def get_edenlive_post_process_func(od_config: OmniDiffusionConfig):
    def post_process(output, **kwargs):
        return output

    return post_process
