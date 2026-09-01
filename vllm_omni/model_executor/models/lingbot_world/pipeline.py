# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""LingBot-World 2.0 single-stage diffusion topology.

Serving resolves a deploy config through this registration, so the stepwise
stage cannot be launched with ``--deploy-config`` until the pipeline name is
registered here.
"""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

LINGBOT_WORLD_PIPELINE = PipelineConfig(
    model_type="lingbot_world",
    default_deploy_config_name="lingbot_world_v2_stepwise.yaml",
    model_arch="LingBotWorldCausalDMDPipeline",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="diffusion",
            execution_type=StageExecutionType.DIFFUSION,
            input_sources=(),
            final_output=True,
            final_output_type="video",
            model_arch="LingBotWorldCausalDMDPipeline",
        ),
    ),
)
