# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Resolve native Echo-WM checkpoints through the structured config API."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from vllm_omni.config.resolver import resolve_omni_config
from vllm_omni.config.stage_config import _DEPLOY_DIR
from vllm_omni.diffusion.data import OmniDiffusionConfig, resolve_model_class_name
from vllm_omni.engine.omni_engine_base import OmniEngineBase

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def _resolve(checkpoint: Path, overrides: dict, deploy_config_path: str | None = None):
    return resolve_omni_config(
        str(checkpoint),
        trust_remote_code=False,
        deploy_config_path=deploy_config_path,
        cli_overrides=overrides,
        stage_overrides=None,
        strategy_config_path=None,
    )


def test_explicit_echo_wm_class_resolves_native_file_without_hf_discovery(tmp_path, mocker):
    checkpoint = tmp_path / "student.safetensors"
    checkpoint.write_bytes(b"discovery does not load tensor weights")
    mocker.patch(
        "vllm_omni.config.resolver.StageConfigFactory.create_from_model",
        side_effect=AssertionError("native Echo-WM files must use the generic native path"),
    )

    resolved = _resolve(
        checkpoint,
        {
            "model_class_name": "EchoWMCausalPipeline",
            "model_config": {"echo_wm_gemma_path": "/models/gemma"},
            "diffusion_attention_backend": "TORCH_SDPA",
        },
    )

    assert resolved.config_path is None
    assert len(resolved.stage_configs) == 1
    stage = resolved.stage_configs[0]
    assert stage.diffusion_config.model_class_name == "EchoWMCausalPipeline"
    assert stage.diffusion_config.model_config == {"echo_wm_gemma_path": "/models/gemma"}
    assert stage.diffusion_config.streaming_output is False
    assert stage.diffusion_config.diffusion_attention_config.default.backend == "TORCH_SDPA"


def test_explicit_stepwise_deploy_overrides_echo_wm_defaults(tmp_path):
    checkpoint = tmp_path / "student.safetensors"
    checkpoint.write_bytes(b"discovery does not load tensor weights")
    stepwise = str(_DEPLOY_DIR / "echo_wm_stepwise.yaml")

    resolved = _resolve(checkpoint, {"model_class_name": "EchoWMCausalPipeline"}, stepwise)

    assert resolved.config_path == stepwise
    stage = resolved.stage_configs[0]
    assert stage.diffusion_config.model_class_name == "EchoWMCausalPipeline"
    assert stage.diffusion_config.step_execution is True
    assert stage.diffusion_config.streaming_output is True
    assert stage.diffusion_config.diffusion_attention_config.default.backend == "TORCH_SDPA"
    assert stage.model_config.default_sampling_params["num_inference_steps"] == 4


@pytest.mark.parametrize("parallel_arg", ["tensor_parallel_size", "ulysses_degree"])
def test_echo_wm_native_defaults_preserve_parallel_overrides(tmp_path, parallel_arg):
    checkpoint = tmp_path / "student.safetensors"
    checkpoint.write_bytes(b"discovery does not load tensor weights")

    resolved = _resolve(
        checkpoint,
        {"model_class_name": "EchoWMCausalPipeline", parallel_arg: 2},
    )

    parallel_config = resolved.stage_configs[0].parallel_config
    assert getattr(parallel_config, parallel_arg) == 2


def test_native_file_enriches_without_hf_probes(tmp_path, monkeypatch):
    checkpoint = tmp_path / "student.safetensors"
    checkpoint.write_bytes(b"configuration discovery does not deserialize weights")

    def unexpected_probe(*_args, **_kwargs):
        pytest.fail("native file configuration must not be interpreted as a Hub repository")

    monkeypatch.setattr("vllm.transformers_utils.config.get_hf_file_to_dict", unexpected_probe)
    monkeypatch.setattr("vllm_omni.diffusion.utils.hf_utils.get_hf_file_to_dict", unexpected_probe)
    config = OmniDiffusionConfig.from_kwargs(
        model=str(checkpoint),
        model_class_name="EchoWMCausalPipeline",
        dtype="float32",
        model_config={"echo_wm_height": 352, "echo_wm_width": 640},
    )
    config.enrich_config()

    assert config.model_class_name == "EchoWMCausalPipeline"
    assert config.supports_multimodal_inputs
    assert config.max_multimodal_image_inputs == 1
    assert config.model_config == {"echo_wm_height": 352, "echo_wm_width": 640}
    assert resolve_model_class_name(str(checkpoint)) is None

    resolved = _resolve(checkpoint, {"model_class_name": "EchoWMCausalPipeline"})
    engine = SimpleNamespace(
        model=str(checkpoint),
        stage_configs=resolved.stage_configs,
        pipeline_config=resolved.pipeline_config,
        _diffusion_od_config_view=None,
    )
    view = OmniEngineBase.get_diffusion_od_config(engine)
    assert view.model_class_name == "EchoWMCausalPipeline"
    assert view.supports_multimodal_inputs and view.max_multimodal_image_inputs == 1


@pytest.mark.parametrize("model_class", [None, "WanPipeline", "UnknownPipeline"])
def test_native_file_rejects_undeclared_pipeline(tmp_path, model_class):
    checkpoint = tmp_path / "student.safetensors"
    checkpoint.touch()
    config = OmniDiffusionConfig(model=str(checkpoint), model_class_name=model_class)
    with pytest.raises(ValueError, match="supports native single-file loading"):
        config.enrich_config()
