# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU contract tests for Evoke/Turbo discovery and checkpoint layouts."""

import json

import pytest

from vllm_omni.diffusion.data import resolve_model_class_name
from vllm_omni.diffusion.model_metadata import get_diffusion_model_metadata
from vllm_omni.diffusion.models.evoke.pipeline_evoke import (
    EVOKE_MODEL_ID,
    EVOKE_TURBO_MODEL_ID,
    resolve_evoke_layout,
)
from vllm_omni.diffusion.models.helios.pipeline_helios import _raw_sink_kwargs
from vllm_omni.diffusion.registry import _DIFFUSION_MODELS

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def _write(path, value="{}"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def test_public_ids_resolve_to_distinct_pipeline_aliases():
    assert resolve_model_class_name(EVOKE_MODEL_ID) == "EvokePipeline"
    assert resolve_model_class_name(EVOKE_TURBO_MODEL_ID) == "EvokeTurboPipeline"


def test_nested_evoke_layout(tmp_path):
    _write(tmp_path / "evoke-base/vae/config.json")
    _write(tmp_path / "evoke/stage3_post_distillation/transformer/config.json")

    layout = resolve_evoke_layout(str(tmp_path))

    assert layout.base_subfolder == "evoke-base"
    assert layout.transformer_subfolder == "evoke/stage3_post_distillation/transformer"
    assert layout.variant == "evoke"


def test_turbo_root_layout_reuses_public_base(tmp_path):
    _write(tmp_path / "config.json")
    _write(tmp_path / "diffusion_pytorch_model.safetensors.index.json", json.dumps({"weight_map": {}}))

    layout = resolve_evoke_layout(str(tmp_path))

    assert layout.base_model == EVOKE_MODEL_ID
    assert layout.transformer_model == str(tmp_path)
    assert layout.transformer_subfolder is None
    assert layout.variant == "turbo"


def test_local_nested_config_resolves_evoke_pipeline(tmp_path):
    _write(tmp_path / "evoke-base/vae/config.json")
    _write(
        tmp_path / "evoke/stage3_post_distillation/transformer/config.json",
        json.dumps({"_class_name": "EvokeTransformer3DModel"}),
    )

    assert resolve_model_class_name(str(tmp_path)) == "EvokePipeline"


@pytest.mark.parametrize("model_class", ["EvokePipeline", "EvokeTurboPipeline"])
def test_registry_and_output_metadata(model_class):
    assert _DIFFUSION_MODELS[model_class] == ("evoke", "pipeline_evoke", model_class)
    metadata = get_diffusion_model_metadata(model_class)
    assert metadata.final_output_type == "video"
    assert metadata.max_multimodal_image_inputs == 1


def test_sink_kwargs_are_forwarded_without_mutating_attention_options():
    sink = object()
    options = {"sink_latents": sink, "dropout": 0.0}
    forwarded = _raw_sink_kwargs(options)
    assert forwarded == {"sink_latents": sink}
    assert options == {"sink_latents": sink, "dropout": 0.0}
