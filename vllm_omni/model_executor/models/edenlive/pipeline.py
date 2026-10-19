# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""EdenLive's default single-device AR-Diffusion topology."""

from vllm_omni.config.stage_config import PipelineConfig, StageExecutionType, StagePipelineConfig

EDENLIVE_PIPELINE = PipelineConfig(
    model_type="edenlive",
    model_arch="EdenLivePipeline",
    diffusers_class_name="EdenLivePipeline",
    default_deploy_config_name="edenlive_5b.yaml",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="diffusion",
            execution_type=StageExecutionType.DIFFUSION,
            input_sources=(),
            final_output=True,
            final_output_type="video",
            model_arch="EdenLivePipeline",
        ),
    ),
)
