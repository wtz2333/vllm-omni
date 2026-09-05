# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Resolve native Echo-WM checkpoints before model workers or HF lookups."""

from pathlib import Path

import pytest

from vllm_omni.config.stage_config import _DEPLOY_DIR
from vllm_omni.entrypoints.utils import load_and_resolve_stage_configs

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def test_explicit_echo_wm_class_resolves_native_file(tmp_path, monkeypatch):
    # Deliberately avoid an Echo-WM basename: the explicit architecture, not
    # a filename heuristic or parent config.json, must select the pipeline.
    checkpoint = tmp_path / "student.safetensors"
    checkpoint.write_bytes(b"discovery does not load tensor weights")

    def unexpected_probe(*_args, **_kwargs):
        pytest.fail("Explicit native diffusion class must bypass early HF config discovery")

    monkeypatch.setattr("vllm_omni.entrypoints.utils.get_config", unexpected_probe)
    monkeypatch.setattr("vllm_omni.entrypoints.utils.get_diffusion_model_index", unexpected_probe)
    monkeypatch.setattr("vllm_omni.config.config_factory.get_config", unexpected_probe)
    monkeypatch.setattr("vllm_omni.config.config_factory.get_hf_file_to_dict", unexpected_probe)
    config_path, stages, _ = load_and_resolve_stage_configs(
        str(checkpoint),
        {"model_class_name": "EchoWMCausalPipeline", "model_config": {"echo_wm_gemma_path": "/models/gemma"}},
        trust_remote_code=False,
    )
    assert Path(config_path).name == "echo_wm.yaml"
    assert len(stages) == 1
    stage = stages[0]
    assert stage.engine_args.model_class_name == "EchoWMCausalPipeline"
    assert stage.engine_args.model_config.echo_wm_gemma_path == "/models/gemma"
    assert stage.engine_args.step_execution is False
    assert stage.engine_args.streaming_output is False
    assert stage.engine_args.diffusion_attention_config.default == "TORCH_SDPA"
    assert stage.default_sampling_params.num_inference_steps == 4


def test_explicit_stepwise_deploy_overrides_echo_wm_default(tmp_path):
    checkpoint = tmp_path / "student.safetensors"
    checkpoint.write_bytes(b"discovery does not load tensor weights")
    stepwise = str(_DEPLOY_DIR / "echo_wm_stepwise.yaml")
    config_path, stages, _ = load_and_resolve_stage_configs(
        str(checkpoint),
        {"model_class_name": "EchoWMCausalPipeline"},
        trust_remote_code=False,
        deploy_config_path=stepwise,
    )
    assert config_path == stepwise
    assert len(stages) == 1
    assert stages[0].engine_args.step_execution is True
    assert stages[0].engine_args.streaming_output is True
    assert stages[0].engine_args.diffusion_attention_config.default == "TORCH_SDPA"


@pytest.mark.parametrize("parallel_arg", ["tensor_parallel_size", "ulysses_degree"])
def test_echo_wm_default_preserves_parallel_overrides(tmp_path, parallel_arg):
    checkpoint = tmp_path / "student.safetensors"
    checkpoint.write_bytes(b"discovery does not load tensor weights")
    _, stages, _ = load_and_resolve_stage_configs(
        str(checkpoint),
        {"model_class_name": "EchoWMCausalPipeline", parallel_arg: 2},
        trust_remote_code=False,
    )
    assert stages[0].engine_args.parallel_config[parallel_arg] == 2
    # Inherit the caller's visible devices; a fixed "0" would hide the second
    # GPU from otherwise valid TP=2/SP=2 configurations.
    assert stages[0].runtime.get("devices") is None


@pytest.mark.parametrize("parallel_arg", [None, "tensor_parallel_size", "ulysses_degree"])
def test_native_file_stage_config_enriches_without_hf_probes(tmp_path, monkeypatch, parallel_arg):
    from types import SimpleNamespace

    import torch
    from omegaconf import OmegaConf

    from vllm_omni.diffusion.data import OmniDiffusionConfig, resolve_model_class_name
    from vllm_omni.diffusion.inline_stage_diffusion_client import InlineStageDiffusionClient
    from vllm_omni.diffusion.utils.hf_utils import get_diffusion_model_index
    from vllm_omni.engine.async_omni_engine import AsyncOmniEngine

    checkpoint = tmp_path / "student.safetensors"
    checkpoint.write_bytes(b"configuration discovery does not deserialize weights")
    overrides = {
        "model_class_name": "EchoWMCausalPipeline",
        "diffusion_attention_backend": "TORCH_SDPA",
        "dtype": "float32",
        "model_config": {
            "echo_wm_gemma_path": "/models/gemma",
            "echo_wm_height": 352,
            "echo_wm_width": 640,
        },
    }
    if parallel_arg:
        overrides[parallel_arg] = 2
    _, stages, _ = load_and_resolve_stage_configs(str(checkpoint), overrides, trust_remote_code=False)

    def unexpected_probe(*_args, **_kwargs):
        pytest.fail("Native file configuration must not be interpreted as a Hub repository")

    monkeypatch.setattr("vllm.transformers_utils.config.get_hf_file_to_dict", unexpected_probe)
    monkeypatch.setattr("vllm_omni.diffusion.utils.hf_utils.get_hf_file_to_dict", unexpected_probe)
    engine_args = OmegaConf.to_container(stages[0].engine_args, resolve=True)
    config = OmniDiffusionConfig.from_kwargs(model=str(checkpoint), **engine_args)
    InlineStageDiffusionClient._enrich_config(SimpleNamespace(od_config=config))
    assert config.model == str(checkpoint)
    assert config.model_class_name == "EchoWMCausalPipeline"
    assert config.supports_multimodal_inputs
    assert config.max_multimodal_image_inputs == 1
    assert config.parallel_config.world_size == (2 if parallel_arg else 1)
    assert config.dtype == torch.float32
    assert config.model_config == overrides["model_config"]
    assert get_diffusion_model_index(str(checkpoint)) is None
    assert resolve_model_class_name(str(checkpoint)) is None

    engine = SimpleNamespace(model=str(checkpoint), stage_configs=stages, _diffusion_od_config_view=None)
    view = AsyncOmniEngine.get_diffusion_od_config(engine)
    assert view.model_class_name == "EchoWMCausalPipeline"
    assert view.supports_multimodal_inputs and view.max_multimodal_image_inputs == 1


@pytest.mark.parametrize("model_class", [None, "WanPipeline", "UnknownPipeline"])
def test_native_file_rejects_undeclared_pipeline(tmp_path, model_class):
    from vllm_omni.diffusion.data import OmniDiffusionConfig

    checkpoint = tmp_path / "student.safetensors"
    checkpoint.touch()
    config = OmniDiffusionConfig(model=str(checkpoint), model_class_name=model_class)
    with pytest.raises(ValueError, match="supports native single-file loading"):
        config.enrich_config()
