# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest

from tests.diffusion.models.lingbot_world.test_pipeline_lingbot_world import (
    _load_pipeline_module,
    _od_config,
    _pipeline,
    _RecordingTransformer,
)
from tests.diffusion.models.lingbot_world.test_pipeline_lingbot_world import (
    _stub_pipeline_dependencies as _stub_pipeline_dependencies,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


@pytest.mark.parametrize("policy", [None, True, "cache_dit", {}, []])
def test_unknown_kv_policy_fails_before_component_loading(policy) -> None:
    config = _od_config()
    config.model_config["lingbot_kv_policy"] = policy
    with pytest.raises(ValueError, match="lingbot_kv_policy"):
        _pipeline(_load_pipeline_module(), od_config=config)


@pytest.mark.parametrize("eager,reuse", [(False, False), (True, True)])
def test_framefold_rejects_incompatible_execution_modes(eager, reuse) -> None:
    config = _od_config(enforce_eager=eager)
    config.model_config.update(lingbot_kv_policy="framefold", lingbot_reuse_last_step_kv=reuse)
    with pytest.raises(ValueError, match="FrameFold requires"):
        _pipeline(_load_pipeline_module(), od_config=config)


def test_framefold_configuration_preserves_fifo_default_and_reserves_workspace(monkeypatch) -> None:
    module = _load_pipeline_module()

    def factory(*args, **kwargs):
        transformer = _RecordingTransformer()
        transformer.config.sliding_window_num_frames = 18
        transformer.config.sink_size = 9
        return transformer

    monkeypatch.setattr(module.CausalLingBotWorldTransformer3DModel, "from_config", factory)
    default = _pipeline(module)
    config = _od_config()
    config.model_config["lingbot_kv_policy"] = "framefold"
    candidate = _pipeline(module, od_config=config)
    assert getattr(default.transformer, "framefold_config", None) is None
    assert candidate.transformer.framefold_config.budget_frames == 18
    assert candidate.transformer.framefold_config.sink_frames == 9
    first, second = default.ar_diffusion_kv_cache_spec(), candidate.ar_diffusion_kv_cache_spec()
    assert first.window_frames == second.window_frames == 9
    assert second.model_owned_state_bytes_per_session > first.model_owned_state_bytes_per_session


def test_framefold_rejects_budget_that_cannot_protect_history() -> None:
    config = _od_config()
    config.model_config["lingbot_kv_policy"] = "framefold"
    with pytest.raises(ValueError, match="FrameFold budget"):
        _pipeline(_load_pipeline_module(), od_config=config)


@pytest.mark.parametrize("flag", ["lingbot_share_first_block_kv", "lingbot_npu_leased_post_attention"])
def test_framefold_rejects_incompatible_cache_modes(flag) -> None:
    config = _od_config()
    config.model_config.update(lingbot_kv_policy="framefold")
    config.model_config[flag] = True
    with pytest.raises(ValueError, match="FrameFold requires"):
        _pipeline(_load_pipeline_module(), od_config=config)
