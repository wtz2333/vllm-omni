# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Echo-WM Flash causal audio-video pipeline.

Implements the 4-step DMD causal rollout of Echo-WM Flash on top of the
Echo-WM transformer port: an image-sink commit, an audio-prefix denoise, then
3-latent-frame audio-video chunks, each denoised with four probe forwards and
one clean commit forward, with bounded sink+FIFO caches carrying history.

Image conditioning and audiovisual decoding use the optional released
``ltx_core`` VAE implementations. Precomputed image/text tensors and latent
output remain available for precision verification. Causal history uses
request-owned bounded windows; mixed audio/video paged KV is not supported.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, ClassVar

import torch
from safetensors import safe_open
from torch import nn
from vllm.logger import init_logger

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.models.echo_wm.attention import upstream_sdpa_kernel
from vllm_omni.diffusion.models.echo_wm.causal_cache import (
    DEFAULT_CAUSAL_TIMESTEPS,
    EchoWMCacheConfig,
    build_audio_positions,
    build_video_positions,
    causal_audio_blocks,
    causal_audio_frames,
    causal_video_blocks,
    compute_cross_slices,
    make_cross_rope_template,
    make_split_rope,
    resolve_causal_sigmas,
)
from vllm_omni.diffusion.models.echo_wm.text_stack import EchoWMTextStack
from vllm_omni.diffusion.models.echo_wm.transformer import EchoWMTransformer3DModel, EchoWMUCPEConfig
from vllm_omni.diffusion.models.interface import SupportsComponentDiscovery, SupportsStepExecution
from vllm_omni.diffusion.models.progress_bar import ProgressBarMixin
from vllm_omni.diffusion.request import OmniDiffusionRequest, resolve_video_num_frames
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.diffusion.worker.utils import StepRequestState

logger = init_logger(__name__)

_DEFAULT_WIDTH = 1280
_DEFAULT_HEIGHT = 704
_DEFAULT_FPS = 24.0
_ECHOWM_DMD_STEPS = 4


def _read_checkpoint_config(checkpoint_path: str) -> dict[str, Any]:
    with safe_open(checkpoint_path, framework="pt") as handle:
        metadata = handle.metadata() or {}
    config = metadata.get("config")
    if config is None:
        raise ValueError(f"{checkpoint_path} does not embed an Echo-WM model config")
    return json.loads(config)


@dataclass
class _EchoWMRequestInputs:
    height: int
    width: int
    num_frames: int
    fps: float
    seed: int
    timesteps: tuple[int, ...]
    cache_config: EchoWMCacheConfig
    output_type: str
    image_latent_tokens: torch.Tensor  # (1, patches_per_frame, 128) clean first frame
    video_context: torch.Tensor
    audio_context: torch.Tensor
    ucpe_viewmats: torch.Tensor  # (1, latent_frames, 4, 4) fp32
    ucpe_Ks: torch.Tensor  # noqa: N815  (1, latent_frames, 3, 3) fp32


class EchoWMCausalPipeline(
    nn.Module,
    SupportsComponentDiscovery,
    SupportsStepExecution,
    ProgressBarMixin,
):
    """Causal 4-step DMD rollout for Echo-WM Flash."""

    supports_step_execution: ClassVar[bool] = True
    support_image_input: ClassVar[bool] = True
    default_num_inference_steps: ClassVar[int] = _ECHOWM_DMD_STEPS
    _dit_modules = ["transformer"]
    _encoder_modules = ["text_stack"]
    _vae_modules: tuple[str, ...] = ()
    dummy_run_num_frames: ClassVar[int] = 0

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = "") -> None:
        super().__init__()
        self.od_config = od_config
        self.model_path = od_config.model
        model_config = dict(getattr(od_config, "model_config", {}) or {})
        self.model_config = model_config
        parallel = getattr(od_config, "parallel_config", None)
        if getattr(parallel, "pipeline_parallel_size", 1) > 1:
            raise ValueError("Echo-WM does not support pipeline parallelism yet")
        if getattr(parallel, "ring_degree", 1) > 1:
            raise ValueError("Echo-WM supports Ulysses sequence parallelism, not Ring attention")
        if getattr(parallel, "cfg_parallel_size", 1) > 1:
            raise ValueError("Echo-WM Flash has distilled guidance and does not support CFG parallelism")
        if getattr(parallel, "use_hsdp", False):
            raise ValueError("Echo-WM does not support HSDP yet")
        if getattr(od_config, "quantization_config", None) is not None:
            raise ValueError("Echo-WM does not support quantization yet")

        checkpoint_config = _read_checkpoint_config(self.model_path)
        self.checkpoint_config = checkpoint_config
        height = int(model_config.get("echo_wm_height", _DEFAULT_HEIGHT))
        width = int(model_config.get("echo_wm_width", _DEFAULT_WIDTH))
        self.ucpe_config = EchoWMUCPEConfig(
            enabled=True,
            attn_dim=1024,
            num_heads=8,
            patches_x=width // 32,
            patches_y=height // 32,
            image_width=width,
            image_height=height,
        )
        self.transformer = EchoWMTransformer3DModel.from_config(
            checkpoint_config.get("transformer", {}),
            ucpe=self.ucpe_config,
            prefix=f"{prefix}transformer." if prefix else "transformer.",
        )
        transformer_cfg = checkpoint_config.get("transformer", {})
        self.text_stack = EchoWMTextStack(
            video_dim=self.transformer.inner_dim,
            audio_dim=self.transformer.audio_inner_dim,
            connector_num_layers=int(transformer_cfg.get("connector_num_layers", 8)),
            video_heads=int(transformer_cfg.get("connector_num_attention_heads", 32)),
            video_head_dim=int(transformer_cfg.get("connector_attention_head_dim", 128)),
            audio_heads=int(transformer_cfg.get("audio_connector_num_attention_heads", 32)),
            audio_head_dim=int(transformer_cfg.get("audio_connector_attention_head_dim", 64)),
            num_registers=int(transformer_cfg.get("connector_num_learnable_registers", 128)),
            rope_max_pos=int(transformer_cfg.get("connector_positional_embedding_max_pos", [4096])[0]),
            apply_gated_attention=bool(transformer_cfg.get("connector_apply_gated_attention", True)),
        )
        # No diffusers component layout: weights load straight from the single
        # Echo-WM safetensors file (see load_weights).
        self.weights_sources = []

        # Optional in-process Gemma encoder (precision runs use fixture
        # embeddings produced by the reference implementation instead).
        self._media = None
        self._gemma = None
        self._gemma_tokenizer = None
        self._gemma_path = model_config.get("echo_wm_gemma_path")
        self.device = self._resolve_device()
        self.dtype = getattr(od_config, "dtype", torch.get_default_dtype())

    @staticmethod
    def _resolve_device() -> torch.device:
        try:
            from vllm_omni.diffusion.distributed.utils import get_local_device

            return get_local_device()
        except Exception:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------ weights

    def load_weights(self, weights) -> set[str]:
        """Load transformer and text-stack weights from the single checkpoint."""
        from safetensors import safe_open as _open

        covered: set[str] = set()
        with _open(self.model_path, framework="pt") as handle:
            transformer_keys = []
            text_keys = []
            for name in handle.keys():
                if name.startswith("model.diffusion_model.") and "embeddings_connector" not in name:
                    transformer_keys.append(name)
                elif (
                    name.startswith("model.diffusion_model.")
                    and "embeddings_connector" in name
                    or name.startswith("text_embedding_projection.")
                ):
                    text_keys.append(name)
            transformer_weights = ((name, handle.get_tensor(name)) for name in transformer_keys)
            loaded = self.transformer.load_weights(transformer_weights)
            covered.update(f"transformer.{name}" for name, _ in self.transformer.named_parameters() if name in loaded)
            text_weights = ((name, handle.get_tensor(name)) for name in text_keys)
            loaded = self.text_stack.load_weights(text_weights)
            covered.update(f"text_stack.{name}" for name, _ in self.text_stack.named_parameters() if name in loaded)
        del weights  # the framework iterator is empty for single-file checkpoints
        return covered

    # ------------------------------------------------------------------ text

    def _encode_prompt(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
        if self._gemma is None:
            if not self._gemma_path:
                raise ValueError(
                    "Echo-WM text encoding requires model_config.echo_wm_gemma_path; "
                    "otherwise pass fixture embeddings via sampling.extra_args.echo_wm_prompt_embeds"
                )
            from pathlib import Path

            from transformers import AutoTokenizer, Gemma3ForConditionalGeneration

            # LTX bundles store Gemma weights and tokenizer assets separately.
            # A regular Gemma HF directory keeps both at its root.
            gemma_root = Path(self._gemma_path)
            model_root = gemma_root / "text_encoder" if (gemma_root / "text_encoder").is_dir() else gemma_root
            tokenizer_root = gemma_root / "tokenizer" if (gemma_root / "tokenizer").is_dir() else gemma_root
            self._gemma_tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_root), local_files_only=True)
            self._gemma_tokenizer.padding_side = "left"
            if self._gemma_tokenizer.pad_token is None:
                self._gemma_tokenizer.pad_token = self._gemma_tokenizer.eos_token
            self._gemma = (
                Gemma3ForConditionalGeneration.from_pretrained(str(model_root), dtype=self.dtype, local_files_only=True)
                .to(self.device)
                .eval()
            )
        encoded = self._gemma_tokenizer(
            prompt.strip(),
            padding="max_length",
            max_length=1024,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoded.input_ids.to(self.device)
        attention_mask = encoded.attention_mask.to(self.device)
        with torch.inference_mode(), upstream_sdpa_kernel(self.device):
            hidden_states = self._gemma.model(
                input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True
            ).hidden_states
        stacked = torch.stack(hidden_states, dim=-1)
        with torch.inference_mode():
            video_context, audio_context = self.text_stack(stacked, attention_mask)
        return video_context, audio_context

    def _media_adapter(self):
        from vllm_omni.diffusion.models.echo_wm.media import EchoWMMediaAdapter

        if self._media is None:
            self._media = EchoWMMediaAdapter(self.model_path, device=self.device, dtype=self.dtype)
        self._media.device = self.device
        self._media.dtype = self.dtype
        return self._media

    def _encode_image(self, image, *, height: int, width: int) -> torch.Tensor:
        return self._media_adapter().encode_image(image, height=height, width=width)

    def _decode_outputs(self, video, audio, *, inputs: _EchoWMRequestInputs, generator) -> dict[str, Any]:
        decoded = self._media_adapter().decode(
            video,
            audio,
            height=inputs.height,
            width=inputs.width,
            generator=generator,
        )
        return {
            "payload": {"video": decoded["video"], "audio": decoded["audio"]},
            "metadata": {
                "video": {"fps": inputs.fps},
                "audio": {"sample_rate": decoded["sample_rate"]},
            },
        }

    # ------------------------------------------------------------------ request parsing

    def _parse_request(self, req: DiffusionRequestBatch) -> _EchoWMRequestInputs:
        if req.num_reqs != 1 or len(req.prompts) != 1:
            raise ValueError("Echo-WM supports a single prompt request, not request batching.")
        sampling = req.sampling_params
        if int(sampling.num_outputs_per_prompt or 1) != 1:
            raise ValueError("Echo-WM requires num_outputs_per_prompt=1.")
        extra_args = dict(getattr(sampling, "extra_args", None) or {})

        prompt_value = req.prompts[0]
        prompt = (
            prompt_value
            if isinstance(prompt_value, str)
            else (prompt_value.get("prompt") if isinstance(prompt_value, dict) else None)
        )
        height = int(
            sampling.height
            if getattr(sampling, "height", None) is not None
            else extra_args.get("echo_wm_height", self.ucpe_config.image_height)
        )
        width = int(
            sampling.width
            if getattr(sampling, "width", None) is not None
            else extra_args.get("echo_wm_width", self.ucpe_config.image_width)
        )
        if height <= 0 or width <= 0 or height % 32 or width % 32:
            raise ValueError("Echo-WM height and width must be positive multiples of 32")
        if (height, width) != (self.ucpe_config.image_height, self.ucpe_config.image_width):
            raise ValueError(
                "Echo-WM request dimensions must match model_config.echo_wm_height/echo_wm_width "
                f"({self.ucpe_config.image_height}x{self.ucpe_config.image_width}), got {height}x{width}"
            )
        num_frames = int(
            resolve_video_num_frames(
                sampling.num_frames,
                default_num_frames=int(extra_args.get("echo_wm_num_frames", 241)),
                is_dummy_run=False,
            )
        )
        fps_value = getattr(sampling, "frame_rate", None)
        if fps_value is None:
            fps_value = getattr(sampling, "fps", None)
        fps = float(fps_value if fps_value is not None else extra_args.get("echo_wm_fps", _DEFAULT_FPS))
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("Echo-WM fps must be positive")
        seed_value = getattr(sampling, "seed", None)
        seed = int(seed_value if seed_value is not None else extra_args.get("echo_wm_seed", 42))
        self.device = self.transformer.patchify_proj.weight.device
        self.dtype = self.transformer.patchify_proj.weight.dtype
        steps = int(sampling.num_inference_steps if sampling.num_inference_steps is not None else _ECHOWM_DMD_STEPS)
        if steps != _ECHOWM_DMD_STEPS:
            raise ValueError(f"Echo-WM Flash requires num_inference_steps={_ECHOWM_DMD_STEPS}, got {steps}")

        latent_frames = (num_frames - 1) // 8 + 1
        if num_frames != (latent_frames - 1) * 8 + 1:
            raise ValueError("echo_wm num_frames must be 1 + 8*n")
        cache_config = EchoWMCacheConfig(
            video_local_attn_size=int(extra_args.get("echo_wm_video_local_attn_size", 19)),
            video_sink_size=int(extra_args.get("echo_wm_video_sink_size", 7)),
        )
        cache_config.validate()
        causal_video_blocks(latent_frames, cache_config.video_chunk_size)

        output_type = getattr(sampling, "output_type", "latent")
        image_tokens = extra_args.get("echo_wm_image_latent")
        if image_tokens is None:
            media = prompt_value.get("multi_modal_data") if isinstance(prompt_value, dict) else None
            image = media.get("image") if isinstance(media, dict) else None
            if isinstance(image, (list, tuple)):
                if len(image) != 1:
                    raise ValueError("Echo-WM requires exactly one conditioning image")
                image = image[0]
            if image is None:
                raise ValueError("Echo-WM requires multi_modal_data.image or extra_args.echo_wm_image_latent")
            image_tokens = self._encode_image(image, height=height, width=width)
        if not isinstance(image_tokens, torch.Tensor) or image_tokens.ndim != 3:
            raise ValueError("echo_wm_image_latent must be a tensor of shape (1, patches_per_frame, 128)")
        patches_per_frame = (height // 32) * (width // 32)
        channels = self.transformer.config.in_channels
        if image_tokens.shape != (1, patches_per_frame, channels):
            raise ValueError(
                f"echo_wm_image_latent must have shape (1, {patches_per_frame}, {channels}), "
                f"got {tuple(image_tokens.shape)}"
            )

        embeddings = extra_args.get("echo_wm_prompt_embeds")
        if embeddings is not None:
            video_context, audio_context = embeddings
        elif isinstance(prompt, str) and prompt.strip():
            video_context, audio_context = self._encode_prompt(prompt)
        else:
            raise ValueError(
                "Echo-WM requires either sampling.extra_args.echo_wm_prompt_embeds "
                "(video/audio context tensors) or a text prompt with echo_wm_gemma_path configured"
            )

        action = extra_args.get("echo_wm_action")
        if action is None or isinstance(action, str):
            from vllm_omni.diffusion.models.echo_wm.actions import build_action_condition

            action = build_action_condition(
                action if action is not None else f"none-{num_frames}",
                num_frames=num_frames,
                width=width,
                height=height,
                fps=fps,
                fov_deg=float(extra_args.get("echo_wm_fov_deg", 70.0)),
                device=self.transformer.patchify_proj.weight.device,
            )
        if not isinstance(action, dict) or "ucpe_viewmats" not in action or "ucpe_Ks" not in action:
            raise ValueError(
                "Echo-WM requires sampling.extra_args.echo_wm_action: a dict with fp32 "
                "ucpe_viewmats (1, latent_frames, 4, 4) and ucpe_Ks (1, latent_frames, 3, 3)"
            )
        viewmats = torch.as_tensor(action["ucpe_viewmats"])
        ks = torch.as_tensor(action["ucpe_Ks"])
        if tuple(viewmats.shape) != (1, latent_frames, 4, 4) or tuple(ks.shape) != (1, latent_frames, 3, 3):
            raise ValueError(
                f"echo_wm_action arrays must cover {latent_frames} latent frames, got "
                f"{tuple(viewmats.shape)} / {tuple(ks.shape)}"
            )
        device = self.transformer.patchify_proj.weight.device
        # The reference patchifier presents the clean image in token-major
        # storage (stride 1 along tokens). BF16 GEMM changes rounding when a
        # contiguous feature-major copy selects a different kernel.
        image_tokens = image_tokens.to(device=device, dtype=self.dtype)
        image_tokens = image_tokens.transpose(1, 2).contiguous().transpose(1, 2)
        timesteps = tuple(extra_args.get("echo_wm_timesteps", DEFAULT_CAUSAL_TIMESTEPS))
        if len(timesteps) != _ECHOWM_DMD_STEPS:
            raise ValueError("Echo-WM Flash requires exactly four echo_wm_timesteps")
        resolve_causal_sigmas(timesteps)
        return _EchoWMRequestInputs(
            height=height,
            width=width,
            num_frames=num_frames,
            fps=fps,
            seed=seed,
            timesteps=timesteps,
            cache_config=cache_config,
            output_type=output_type,
            image_latent_tokens=image_tokens,
            video_context=video_context.to(device=device, dtype=self.dtype),
            audio_context=audio_context.to(device=device, dtype=self.dtype),
            ucpe_viewmats=viewmats.to(device=device, dtype=torch.float32),
            ucpe_Ks=ks.to(device=device, dtype=torch.float32),
        )

    # ------------------------------------------------------------------ session state

    def _init_session(self, inputs: _EchoWMRequestInputs):
        """Allocate caches, RoPE templates, and the initial noise buffers."""
        device = self.transformer.patchify_proj.weight.device
        patches_per_frame = (inputs.height // 32) * (inputs.width // 32)
        latent_frames = (inputs.num_frames - 1) // 8 + 1
        audio_frames = causal_audio_frames(latent_frames, inputs.cache_config.video_chunk_size)
        caches = self.transformer.allocate_caches(
            batch_size=1,
            patches_per_frame=patches_per_frame,
            text_seq_len=inputs.video_context.shape[1],
            cache_config=inputs.cache_config,
            device=device,
            dtype=self.dtype,
        )
        # VideoLatentTools stores video coordinates in the latent dtype before
        # constructing RoPE; audio coordinates deliberately stay in FP32.
        video_positions = build_video_positions(
            latent_frames, inputs.height, inputs.width, fps=inputs.fps, device=device
        ).to(self.dtype)
        audio_positions = build_audio_positions(audio_frames, device=device)
        video_window = inputs.cache_config.video_local_attn_size * patches_per_frame
        video_rope = make_split_rope(
            video_positions[:, :, :video_window],
            dim=self.transformer.inner_dim,
            num_heads=self.transformer.config.num_attention_heads,
            max_pos=list(self.transformer.config.positional_embedding_max_pos),
            out_dtype=self.dtype,
            device=device,
        )
        audio_rope = make_split_rope(
            audio_positions[:, :, : inputs.cache_config.audio_local_attn_size],
            dim=self.transformer.audio_inner_dim,
            num_heads=self.transformer.config.audio_num_attention_heads,
            max_pos=list(self.transformer.config.audio_positional_embedding_max_pos),
            out_dtype=self.dtype,
            device=device,
        )
        cross_max_pos = max(
            self.transformer.config.positional_embedding_max_pos[0],
            self.transformer.config.audio_positional_embedding_max_pos[0],
        )
        video_cross_rope = make_cross_rope_template(
            video_positions[:, :, :video_window],
            dim=self.transformer.config.audio_cross_attention_dim,
            num_heads=self.transformer.config.audio_num_attention_heads,
            max_pos=cross_max_pos,
            out_dtype=self.dtype,
            device=device,
        )
        audio_cross_rope = make_cross_rope_template(
            audio_positions[:, :, : inputs.cache_config.audio_local_attn_size],
            dim=self.transformer.config.audio_cross_attention_dim,
            num_heads=self.transformer.config.audio_num_attention_heads,
            max_pos=cross_max_pos,
            out_dtype=self.dtype,
            device=device,
        )
        a2v_slices, v2a_slices = compute_cross_slices(latent_frames, patches_per_frame, inputs.cache_config)
        for layer in caches:
            layer.video_rope = video_rope
            layer.audio_rope = audio_rope
            layer.video_cross_rope = video_cross_rope
            layer.audio_cross_rope = audio_cross_rope
            layer.a2v_q_slices = a2v_slices
            layer.v2a_q_slices = v2a_slices
            layer.ucpe_full_viewmats = inputs.ucpe_viewmats
            layer.ucpe_full_Ks = inputs.ucpe_Ks

        sigmas = resolve_causal_sigmas(list(inputs.timesteps))
        video_blocks = causal_video_blocks(latent_frames, inputs.cache_config.video_chunk_size)
        audio_blocks = causal_audio_blocks(latent_frames, inputs.cache_config.video_chunk_size)

        # Noise draw order is pinned by the reference rollout: initial video,
        # then audio, then the per-step re-noising draws in block order.
        generator = torch.Generator(device=device).manual_seed(inputs.seed)
        total_video_tokens = latent_frames * patches_per_frame
        channels = self.transformer.config.in_channels
        initial_video = torch.randn(
            (1, total_video_tokens, channels), generator=generator, device=device, dtype=self.dtype
        )
        initial_audio = torch.randn((1, audio_frames, channels), generator=generator, device=device, dtype=self.dtype)
        # Reference VAE noise starts after the initial video/audio draws;
        # rollout re-noising has its own independent generator state.
        decode_generator = torch.Generator(device=device)
        decode_generator.set_state(generator.get_state())
        return SimpleNamespace(
            decode_generator=decode_generator,
            decode_generator_state=decode_generator.get_state(),
            decoded_video_frames=0,
            decoded_audio_samples=0,
            caches=caches,
            sigmas=sigmas,
            video_blocks=video_blocks,
            audio_blocks=audio_blocks,
            patches_per_frame=patches_per_frame,
            initial_video=initial_video,
            initial_audio=initial_audio,
            generator=generator,
            video_output=None,
            audio_output=None,
        )

    # ------------------------------------------------------------------ rollout math

    def _forward_block(
        self,
        session,
        inputs: _EchoWMRequestInputs,
        video_tokens: torch.Tensor | None,
        audio_tokens: torch.Tensor | None,
        video_sigma: float,
        audio_sigma: float | None,
        video_start: int,
        audio_start: int,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """One transformer forward; velocities converted to x0 in fp32."""
        ppf = session.patches_per_frame
        if video_tokens is not None:
            video_frames_end = (video_start + video_tokens.shape[1]) // ppf
        else:
            video_frames_end = video_start // ppf
        viewmats = inputs.ucpe_viewmats[:, video_start // ppf : video_frames_end] if video_tokens is not None else None
        ks = inputs.ucpe_Ks[:, video_start // ppf : video_frames_end] if video_tokens is not None else None
        with torch.inference_mode():
            velocity_video, velocity_audio = self.transformer(
                video_tokens=video_tokens,
                audio_tokens=audio_tokens,
                video_sigma=video_sigma,
                audio_sigma=audio_sigma,
                video_context=inputs.video_context,
                audio_context=inputs.audio_context,
                caches=session.caches,
                video_token_start=video_start,
                audio_token_start=audio_start,
                ucpe_viewmats=viewmats,
                ucpe_Ks=ks,
                patches_per_frame=ppf,
            )
        video_x0 = audio_x0 = None
        if velocity_video is not None and video_tokens is not None:
            sigma_tensor = torch.as_tensor(video_sigma, device=video_tokens.device, dtype=video_tokens.dtype).float()
            video_x0 = (video_tokens.float() - velocity_video.float() * sigma_tensor).to(video_tokens.dtype)
        if velocity_audio is not None and audio_tokens is not None:
            sigma = audio_sigma if audio_sigma is not None else video_sigma
            sigma_tensor = torch.as_tensor(sigma, device=audio_tokens.device, dtype=audio_tokens.dtype).float()
            audio_x0 = (audio_tokens.float() - velocity_audio.float() * sigma_tensor).to(audio_tokens.dtype)
        return video_x0, audio_x0

    @staticmethod
    def _advance(denoised: torch.Tensor, next_sigma: float, generator: torch.Generator) -> torch.Tensor:
        noise = torch.randn(denoised.shape, generator=generator, device=denoised.device, dtype=denoised.dtype)
        return (1 - next_sigma) * denoised + next_sigma * noise

    def _prepare_outputs(self, session, inputs: _EchoWMRequestInputs) -> None:
        latent_frames = (inputs.num_frames - 1) // 8 + 1
        audio_frames = causal_audio_frames(latent_frames, inputs.cache_config.video_chunk_size)
        channels = session.initial_video.shape[-1]
        session.video_output = torch.zeros(
            (1, latent_frames * session.patches_per_frame, channels),
            device=session.initial_video.device,
            dtype=self.dtype,
        )
        session.audio_output = torch.zeros(
            (1, audio_frames, channels), device=session.initial_audio.device, dtype=self.dtype
        )
        session.video_output[:, : session.patches_per_frame] = inputs.image_latent_tokens

    # ------------------------------------------------------------------ request mode

    def forward(self, req: DiffusionRequestBatch) -> DiffusionOutput:
        inputs = self._parse_request(req)
        session = self._init_session(inputs)
        self._prepare_outputs(session, inputs)
        ppf = session.patches_per_frame
        sigmas = session.sigmas

        with torch.inference_mode():
            # Image-sink commit: the clean first frame enters the video caches.
            self._forward_block(session, inputs, inputs.image_latent_tokens, None, 0.0, None, 0, 0)
            # Audio prefix: 4 steps with the clean image at sigma 0.
            audio_start, audio_end = session.audio_blocks[0]
            audio_sample = session.initial_audio[:, audio_start:audio_end]
            for step, sigma in enumerate(sigmas):
                _, denoised_audio = self._forward_block(
                    session, inputs, inputs.image_latent_tokens, audio_sample, 0.0, sigma, 0, audio_start
                )
                audio_sample = (
                    denoised_audio
                    if step == len(sigmas) - 1
                    else self._advance(denoised_audio, sigmas[step + 1], session.generator)
                )
            session.audio_output[:, audio_start:audio_end] = audio_sample
            # Clean commit for the prefix.
            self._forward_block(session, inputs, inputs.image_latent_tokens, audio_sample, 0.0, 0.0, 0, audio_start)

            # Audio-video blocks.
            for video_block, audio_block in zip(session.video_blocks[1:], session.audio_blocks[1:], strict=True):
                video_start, video_end = video_block
                audio_start, audio_end = audio_block
                video_sample = session.initial_video[:, video_start * ppf : video_end * ppf]
                audio_sample = session.initial_audio[:, audio_start:audio_end]
                for step, sigma in enumerate(sigmas):
                    denoised_video, denoised_audio = self._forward_block(
                        session,
                        inputs,
                        video_sample,
                        audio_sample,
                        sigma,
                        sigma,
                        video_start * ppf,
                        audio_start,
                    )
                    if step == len(sigmas) - 1:
                        video_sample, audio_sample = denoised_video, denoised_audio
                    else:
                        video_sample = self._advance(denoised_video, sigmas[step + 1], session.generator)
                        audio_sample = self._advance(denoised_audio, sigmas[step + 1], session.generator)
                session.video_output[:, video_start * ppf : video_end * ppf] = video_sample
                session.audio_output[:, audio_start:audio_end] = audio_sample
                # Clean refresh commit.
                self._forward_block(
                    session, inputs, video_sample, audio_sample, 0.0, 0.0, video_start * ppf, audio_start
                )

        if inputs.output_type == "latent":
            output = {
                "payload": {
                    "latents": {
                        "video": session.video_output,
                        "audio": session.audio_output,
                        "patches_per_frame": ppf,
                        "latent_frames": session.video_output.shape[1] // ppf,
                        "audio_frames": session.audio_output.shape[1],
                    }
                },
                "metadata": {},
            }
        else:
            output = self._decode_outputs(
                session.video_output,
                session.audio_output,
                inputs=inputs,
                generator=session.decode_generator,
            )
        output["metadata"]["echo_wm"] = {"seed": inputs.seed, "timesteps": list(inputs.timesteps)}
        return DiffusionOutput(output=output)

    # ------------------------------------------------------------------ stepwise execution

    def prepare_encode(self, state: StepRequestState, **kwargs) -> StepRequestState:
        """Validate the request, initialize the session, and prime the image sink."""
        del kwargs
        req = DiffusionRequestBatch(
            requests=[
                OmniDiffusionRequest(
                    prompt=state.prompt,
                    sampling_params=state.sampling,
                    request_id=state.request_id,
                )
            ]
        )
        inputs = self._parse_request(req)
        session = self._init_session(inputs)
        self._prepare_outputs(session, inputs)
        # The image-sink commit writes the clean first frame into the caches;
        # it is not a denoising step and produces no chunk.
        self._forward_block(session, inputs, inputs.image_latent_tokens, None, 0.0, None, 0, 0)
        state.extra = {
            "echo_wm_inputs": inputs,
            "echo_wm_session": session,
        }
        state.total_chunks = len(session.video_blocks)  # prefix block + AV blocks
        state.chunk_num_steps = _ECHOWM_DMD_STEPS
        state.chunk_index = 0
        state.step_in_chunk = 0
        state.step_index = 0
        state.timesteps = torch.tensor(
            inputs.timesteps * state.total_chunks,
            device=inputs.image_latent_tokens.device,
            dtype=torch.float32,
        )
        self._prepare_next_chunk(state)
        return state

    def _prepare_next_chunk(self, state: StepRequestState) -> None:
        session: SimpleNamespace = state.extra["echo_wm_session"]
        inputs: _EchoWMRequestInputs = state.extra["echo_wm_inputs"]
        ppf = session.patches_per_frame
        video_block = session.video_blocks[state.chunk_index]
        audio_block = session.audio_blocks[state.chunk_index]
        video_start, video_end = video_block
        audio_start, audio_end = audio_block
        if state.chunk_index == 0:
            video_sample = inputs.image_latent_tokens
        else:
            video_sample = session.initial_video[:, video_start * ppf : video_end * ppf]
        audio_sample = session.initial_audio[:, audio_start:audio_end]
        # The runner gathers and scatters state.latents after every step. Keep
        # both modalities in that authoritative tensor, splitting at this offset.
        session.current_video_length = video_sample.shape[1]
        state.latents = torch.cat((video_sample, audio_sample), dim=1)
        state.step_in_chunk = 0
        session.current_video_start = video_start * ppf
        session.current_audio_start = audio_start

    def denoise_step(self, input_batch, *, states=None, **kwargs):
        del input_batch, kwargs
        if states is None or len(states) != 1:
            raise ValueError("Echo-WM step execution requires exactly one request")
        state = states[0]
        session: SimpleNamespace = state.extra["echo_wm_session"]
        inputs: _EchoWMRequestInputs = state.extra["echo_wm_inputs"]
        sigma = session.sigmas[state.step_in_chunk]
        if state.chunk_index == 0:
            video_sigma = 0.0
            audio_sigma = sigma
        else:
            video_sigma = audio_sigma = sigma
        video_x0, audio_x0 = self._forward_block(
            session,
            inputs,
            state.latents[:, : session.current_video_length],
            state.latents[:, session.current_video_length :],
            video_sigma,
            audio_sigma,
            session.current_video_start,
            session.current_audio_start,
        )
        state.extra["echo_wm_step_result"] = (video_x0, audio_x0, sigma)
        return None

    def step_scheduler(self, state: StepRequestState, noise_pred, **kwargs) -> None:
        del noise_pred, kwargs
        session: SimpleNamespace = state.extra["echo_wm_session"]
        video_x0, audio_x0, _ = state.extra.pop("echo_wm_step_result")
        step = state.step_in_chunk
        if step == len(session.sigmas) - 1:
            video_sample, audio_sample = video_x0, audio_x0
        else:
            next_sigma = session.sigmas[step + 1]
            video_sample = (
                video_x0 if state.chunk_index == 0 else self._advance(video_x0, next_sigma, session.generator)
            )
            audio_sample = self._advance(audio_x0, next_sigma, session.generator)
        state.latents = torch.cat((video_sample, audio_sample), dim=1)
        state.step_in_chunk += 1
        state.step_index += 1
        if (
            not getattr(self.od_config, "streaming_output", False)
            and state.chunk_denoise_completed
            and state.chunk_index + 1 < state.total_chunks
        ):
            self._commit_chunk(state)

    def _commit_chunk(self, state: StepRequestState) -> tuple[dict[str, torch.Tensor], int]:
        """Commit clean latents and advance the model-owned chunk boundary."""
        session: SimpleNamespace = state.extra["echo_wm_session"]
        inputs: _EchoWMRequestInputs = state.extra["echo_wm_inputs"]
        ppf = session.patches_per_frame
        video_start, video_end = session.video_blocks[state.chunk_index]
        audio_start, audio_end = session.audio_blocks[state.chunk_index]
        video_sample = state.latents[:, : session.current_video_length]
        audio_sample = state.latents[:, session.current_video_length :]
        session.video_output[:, video_start * ppf : video_end * ppf] = video_sample
        session.audio_output[:, audio_start:audio_end] = audio_sample
        self._forward_block(
            session,
            inputs,
            video_sample,
            audio_sample,
            0.0,
            0.0,
            video_start * ppf,
            audio_start,
        )
        chunk_payload = {"video": video_sample, "audio": audio_sample}
        completed_chunk = state.chunk_index
        state.chunk_index += 1
        if not state.request_denoise_completed:
            self._prepare_next_chunk(state)
        return chunk_payload, completed_chunk

    def post_decode(self, state: StepRequestState, **kwargs) -> DiffusionOutput:
        del kwargs
        session: SimpleNamespace = state.extra["echo_wm_session"]
        inputs: _EchoWMRequestInputs = state.extra["echo_wm_inputs"]
        chunk_payload, completed_chunk = self._commit_chunk(state)
        streaming = getattr(self.od_config, "streaming_output", False)
        payload = (
            chunk_payload
            if streaming
            else {
                "video": session.video_output,
                "audio": session.audio_output,
                "patches_per_frame": session.patches_per_frame,
                "latent_frames": session.video_output.shape[1] // session.patches_per_frame,
                "audio_frames": session.audio_output.shape[1],
            }
        )
        if inputs.output_type == "latent":
            output = {"payload": {"latents": payload}, "metadata": {}}
        else:
            video_end = session.video_blocks[completed_chunk][1] * session.patches_per_frame
            audio_end = session.audio_blocks[completed_chunk][1]
            # Decode the causal prefix so later chunks retain VAE context, then
            # emit only newly generated media. Keep VAE randomness request-local.
            session.decode_generator.set_state(session.decode_generator_state)
            output = self._decode_outputs(
                session.video_output[:, :video_end],
                session.audio_output[:, :audio_end],
                inputs=inputs,
                generator=session.decode_generator,
            )
            if streaming:
                video = output["payload"]["video"]
                audio = output["payload"]["audio"]
                output["payload"]["video"] = video[session.decoded_video_frames :]
                output["payload"]["audio"] = audio[..., session.decoded_audio_samples :]
                session.decoded_video_frames = video.shape[0]
                session.decoded_audio_samples = audio.shape[-1]
        output["metadata"]["echo_wm"] = {
            "chunk_index": completed_chunk,
            "total_chunks": state.total_chunks,
            "seed": inputs.seed,
        }
        return DiffusionOutput(
            output=output,
            chunk_index=completed_chunk,
            total_chunks=state.total_chunks,
            finished=state.request_denoise_completed,
        )
