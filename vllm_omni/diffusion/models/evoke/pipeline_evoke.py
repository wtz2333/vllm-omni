# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Evoke causal-diffusion pipeline and checkpoint layout handling.

Evoke performs a short diffusion rollout for each causal video chunk and feeds
the generated latent history into the next chunk.  Its denoiser is Helios
lineage code, but its public checkpoint layout and CFG-free three-step recipe
are model-specific; this module owns those differences.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from diffusers import AutoencoderKLWan
from torch import nn
from transformers import AutoConfig, AutoTokenizer, UMT5EncoderModel

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.models.evoke.evoke_transformer import EvokeTransformer3DModel
from vllm_omni.diffusion.models.helios.pipeline_helios import (
    HeliosPipeline,
    HeliosScheduler,
    load_json_config,
    load_transformer_config,
)

EVOKE_MODEL_ID = "AlayaLab/Evoke"
EVOKE_TURBO_MODEL_ID = "AlayaLab/Evoke-Turbo"
EVOKE_OUTPUT_HEIGHT = 384
EVOKE_OUTPUT_WIDTH = 640
EVOKE_OUTPUT_FRAMES = 73
EVOKE_DEFAULT_STEPS = 3
EVOKE_DEFAULT_GUIDANCE = 1.0
EVOKE_DEFAULT_MAX_SEQUENCE_LENGTH = 512
EVOKE_BASE_SUBFOLDER = "evoke-base"
EVOKE_STAGE3_TRANSFORMER_SUBFOLDER = "evoke/stage3_post_distillation/transformer"


@dataclass(frozen=True)
class EvokeLayout:
    base_model: str
    base_subfolder: str
    transformer_model: str
    transformer_subfolder: str | None
    variant: str


def _join_subfolder(prefix: str, child: str) -> str:
    return f"{prefix.rstrip('/')}/{child}" if prefix else child


def _local_json(model: str, subfolder: str | None, filename: str) -> dict[str, Any]:
    path = Path(model) / (subfolder or "") / filename
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as file:
        value = json.load(file)
    return value if isinstance(value, dict) else {}


def resolve_evoke_layout(model: str) -> EvokeLayout:
    """Resolve public HF IDs and assembled local Evoke/Turbo trees."""

    normalized = str(model).rstrip("/")
    if normalized.lower() == EVOKE_TURBO_MODEL_ID.lower():
        return EvokeLayout(EVOKE_MODEL_ID, EVOKE_BASE_SUBFOLDER, normalized, None, "turbo")
    if normalized.lower() == EVOKE_MODEL_ID.lower():
        return EvokeLayout(
            normalized,
            EVOKE_BASE_SUBFOLDER,
            normalized,
            EVOKE_STAGE3_TRANSFORMER_SUBFOLDER,
            "evoke",
        )

    root = Path(normalized)
    if not root.is_dir():
        raise ValueError(f"{model!r} is not an Evoke checkpoint or local directory.")
    if (root / "evoke-turbo/transformer/config.json").is_file():
        return EvokeLayout(normalized, EVOKE_BASE_SUBFOLDER, normalized, "evoke-turbo/transformer", "turbo")
    if (root / f"{EVOKE_STAGE3_TRANSFORMER_SUBFOLDER}/config.json").is_file():
        return EvokeLayout(
            normalized,
            EVOKE_BASE_SUBFOLDER,
            normalized,
            EVOKE_STAGE3_TRANSFORMER_SUBFOLDER,
            "evoke",
        )
    if (root / "diffusion_pytorch_model.safetensors.index.json").is_file():
        base_model = normalized if (root / f"{EVOKE_BASE_SUBFOLDER}/vae/config.json").is_file() else EVOKE_MODEL_ID
        return EvokeLayout(base_model, EVOKE_BASE_SUBFOLDER, normalized, None, "turbo")
    if (root / "transformer/config.json").is_file():
        return EvokeLayout(normalized, "", normalized, "transformer", "evoke")
    raise ValueError(
        f"{model!r} does not contain an Evoke transformer config. Expected a nested "
        "stage3_post_distillation/transformer or Turbo root layout."
    )


class EvokePipeline(HeliosPipeline):
    """Evoke causal chunk rollout using the vLLM-native Helios-family core."""

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = "") -> None:
        # The parent pipeline assumes a conventional Diffusers root. Evoke
        # splits its shared components from the transformer, so construct the
        # same runtime components against the resolved layout here.
        nn.Module.__init__(self)
        self.od_config = od_config
        self.device = get_local_device()
        dtype = getattr(od_config, "dtype", torch.bfloat16)
        layout = resolve_evoke_layout(od_config.model)
        self.evoke_layout = layout
        local_base = os.path.isdir(layout.base_model)
        local_transformer = os.path.isdir(layout.transformer_model)

        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=layout.transformer_model,
                subfolder=layout.transformer_subfolder,
                revision=od_config.revision,
                prefix="transformer.",
                fall_back_to_pt=False,
            )
        ]

        base_tokenizer = _join_subfolder(layout.base_subfolder, "tokenizer")
        base_text_encoder = _join_subfolder(layout.base_subfolder, "text_encoder")
        base_vae = _join_subfolder(layout.base_subfolder, "vae")
        base_scheduler = _join_subfolder(layout.base_subfolder, "scheduler")

        self.tokenizer = AutoTokenizer.from_pretrained(
            layout.base_model, subfolder=base_tokenizer, local_files_only=local_base
        )
        text_config = AutoConfig.from_pretrained(
            layout.base_model, subfolder=base_text_encoder, local_files_only=local_base
        )
        text_config.tie_word_embeddings = True
        self.text_encoder = UMT5EncoderModel.from_pretrained(
            layout.base_model,
            subfolder=base_text_encoder,
            config=text_config,
            torch_dtype=dtype,
            local_files_only=local_base,
        ).to(self.device)
        self.vae = AutoencoderKLWan.from_pretrained(
            layout.base_model, subfolder=base_vae, torch_dtype=torch.float32, local_files_only=local_base
        ).to(self.device)

        if layout.transformer_subfolder is None and not local_transformer:
            from vllm.transformers_utils.config import get_hf_file_to_dict

            transformer_config = (
                get_hf_file_to_dict("config.json", layout.transformer_model, revision=od_config.revision) or {}
            )
        else:
            transformer_config = load_transformer_config(
                layout.transformer_model,
                layout.transformer_subfolder or "",
                local_files_only=local_transformer,
            )
        if not transformer_config:
            transformer_config = _local_json(layout.transformer_model, layout.transformer_subfolder, "config.json")
        if not transformer_config:
            raise FileNotFoundError("Evoke transformer/config.json was not found.")

        transformer_kwargs: dict[str, Any] = {}
        config_keys = (
            "patch_size",
            "num_attention_heads",
            "attention_head_dim",
            "in_channels",
            "out_channels",
            "text_dim",
            "freq_dim",
            "ffn_dim",
            "num_layers",
            "cross_attn_norm",
            "qk_norm",
            "eps",
            "added_kv_proj_dim",
            "rope_dim",
            "rope_theta",
            "guidance_cross_attn",
            "zero_history_timestep",
            "has_multi_term_memory_patch",
            "is_amplify_history",
            "history_scale_mode",
            "use_raw_sink_frames",
        )
        for key in config_keys:
            if key in transformer_config:
                value = transformer_config[key]
                if key in {"patch_size", "rope_dim"} and isinstance(value, list):
                    value = tuple(value)
                transformer_kwargs[key] = value
        self.transformer = EvokeTransformer3DModel(
            quant_config=od_config.quantization_config,
            **transformer_kwargs,
        )

        scheduler_config = load_json_config(
            layout.base_model, base_scheduler, "scheduler_config.json", local_files_only=local_base
        )
        if not scheduler_config:
            scheduler_config = _local_json(layout.base_model, base_scheduler, "scheduler_config.json")
        scheduler_kwargs = {
            key: scheduler_config[key]
            for key in (
                "num_train_timesteps",
                "shift",
                "stages",
                "stage_range",
                "gamma",
                "thresholding",
                "prediction_type",
                "solver_order",
                "predict_x0",
                "solver_type",
                "lower_order_final",
                "disable_corrector",
                "use_flow_sigmas",
                "scheduler_type",
                "use_dynamic_shifting",
                "time_shift_type",
            )
            if key in scheduler_config
        }
        # The checkpoint ships a stage-1 scheduler (stages=1, dynamic
        # shifting enabled). Evoke's released post-distilled/Turbo recipe uses
        # a three-level pyramid and applies the per-stage shift in the rollout;
        # construct the pyramid scheduler exactly as the official inference
        # script does to avoid missing ``ori_start_sigmas[1/2]`` and double
        # time-shifting.
        scheduler_kwargs.update(
            stages=3,
            stage_range=[0.0, 1 / 3, 2 / 3, 1.0],
            shift=1.0,
            use_dynamic_shifting=False,
        )
        self.scheduler = HeliosScheduler(**scheduler_kwargs)
        self.is_distilled = scheduler_kwargs.get("scheduler_type") == "dmd"
        self.vae_scale_factor_temporal = self.vae.config.scale_factor_temporal
        self.vae_scale_factor_spatial = self.vae.config.scale_factor_spatial
        self._guidance_scale = None
        self._num_timesteps = None
        self._current_timestep = None
        self.setup_diffusion_pipeline_profiler(
            enable_diffusion_pipeline_profiler=od_config.enable_diffusion_pipeline_profiler
        )

    def forward(self, req, *args: Any, **kwargs: Any) -> DiffusionOutput:
        sampling = req.sampling_params
        extra = dict(getattr(sampling, "extra_args", {}) or {})
        # The shipped post-distilled and Turbo checkpoints are trained for
        # exactly three CFG-free steps. Explicit request values still win.
        kwargs.setdefault("height", EVOKE_OUTPUT_HEIGHT)
        kwargs.setdefault("width", EVOKE_OUTPUT_WIDTH)
        kwargs.setdefault("frame_num", EVOKE_OUTPUT_FRAMES)
        kwargs.setdefault("num_inference_steps", EVOKE_DEFAULT_STEPS)
        kwargs.setdefault("guidance_scale", EVOKE_DEFAULT_GUIDANCE)
        kwargs.setdefault("is_enable_stage2", True)
        kwargs.setdefault("pyramid_num_stages", 3)
        kwargs.setdefault("pyramid_num_inference_steps_list", [1, 1, 1])
        kwargs.setdefault("history_sizes", [16, 2, 1])
        kwargs.setdefault("num_latent_frames_per_chunk", 9)
        if sampling.max_sequence_length is None:
            sampling.max_sequence_length = int(extra.get("max_sequence_length", EVOKE_DEFAULT_MAX_SEQUENCE_LENGTH))
        attention_kwargs = dict(extra.get("attention_kwargs", {}) or {})
        for key in ("sink_latents", "nearby_sink_latents"):
            if key in extra:
                attention_kwargs[key] = extra[key]
        extra["attention_kwargs"] = attention_kwargs
        sampling.extra_args = extra
        return super().forward(req, *args, **kwargs)


class EvokeTurboPipeline(EvokePipeline):
    """Registry alias for the merged Evoke-Turbo transformer checkpoint."""


def get_evoke_pre_process_func(od_config: OmniDiffusionConfig):
    del od_config

    def pre_process_func(request):
        sampling = request.sampling_params
        extra = dict(getattr(sampling, "extra_args", {}) or {})
        if sampling.max_sequence_length is None:
            sampling.max_sequence_length = int(extra.get("max_sequence_length", EVOKE_DEFAULT_MAX_SEQUENCE_LENGTH))
        sampling.extra_args = extra
        return request

    return pre_process_func


def get_evoke_post_process_func(od_config: OmniDiffusionConfig):
    del od_config

    def post_process_func(output: torch.Tensor, output_type: str = "np"):
        if output_type == "latent":
            return output
        from diffusers.video_processor import VideoProcessor

        return VideoProcessor(vae_scale_factor=8).postprocess_video(output, output_type=output_type)

    return post_process_func
