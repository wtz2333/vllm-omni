# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Released EdenLive 5B few-step defaults for shared video entry points."""

from collections.abc import Mapping
from typing import Any

from vllm_omni.model_extras.video_generation import VideoGenerationDefaults


def get_edenlive_video_generation_defaults(extra_body: Mapping[str, Any] | None = None) -> VideoGenerationDefaults:
    return VideoGenerationDefaults(
        width=1280,
        height=704,
        num_frames=117,
        num_inference_steps=4,
        fps=24,
        guidance_scale=1.0,
        dimension_multiple=32,
        output="edenlive.mp4",
    )


def edenlive_transformer_config_subfolder(*, model: str | None, revision: str | None = None) -> str:
    return "dit-5b"
