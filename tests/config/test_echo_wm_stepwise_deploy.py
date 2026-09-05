# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EchoWM deployment values must reach the stage consumed by the engine."""

import pytest

from tests.helpers.stage_config import get_deploy_config_path
from vllm_omni.config.stage_config import load_deploy_config, merge_pipeline_deploy
from vllm_omni.model_executor.models.echo_wm.pipeline import ECHO_WM_PIPELINE

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_echo_wm_deploy_enables_streaming_and_preserves_model_config():
    deploy = load_deploy_config(get_deploy_config_path("echo_wm_stepwise.yaml"))
    assert len(deploy.stages) == 1
    stages = merge_pipeline_deploy(ECHO_WM_PIPELINE, deploy, {})
    assert len(stages) == 1
    args = stages[0].yaml_engine_args
    assert args["model_class_name"] == "EchoWMCausalPipeline"
    assert args["step_execution"] is True
    assert args["streaming_output"] is True
    assert args["max_num_seqs"] == 1
    assert args["model_config"] == {"echo_wm_height": 704, "echo_wm_width": 1280}
    assert stages[0].yaml_extras["default_sampling_params"]["num_inference_steps"] == 4
