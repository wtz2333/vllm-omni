# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Echo-WM single-stage diffusion topology.

Serving resolves a deploy config through this registration, so the stepwise
stage cannot be launched with ``--deploy-config`` until the pipeline name is
registered here. Like LingBot-World, there is deliberately no
``default_deploy_config_name``: Echo-WM checkpoints are matched by name and
stepwise serving stays opt-in through
``--deploy-config vllm_omni/deploy/echo_wm_stepwise.yaml``.
"""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

ECHO_WM_PIPELINE = PipelineConfig(
    model_type="echo_wm",
    model_arch="EchoWMCausalPipeline",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="diffusion",
            execution_type=StageExecutionType.DIFFUSION,
            input_sources=(),
            final_output=True,
            final_output_type="video",
            model_arch="EchoWMCausalPipeline",
        ),
    ),
)
