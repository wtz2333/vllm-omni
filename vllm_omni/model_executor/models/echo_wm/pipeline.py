# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Echo-WM request execution by default; chunk streaming is opt-in."""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

ECHO_WM_PIPELINE = PipelineConfig(
    model_type="echo_wm",
    model_arch="EchoWMCausalPipeline",
    default_deploy_config_name="echo_wm.yaml",
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
