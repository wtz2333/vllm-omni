# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

from vllm_omni.model_extras.helios import HELIOS_EXTRA_BODY_PARAMS

# Evoke's causal rollout consumes the same image/video conditioning fields as
# Helios plus its raw sink-frame and camera metadata when supplied by a caller.
EVOKE_EXTRA_BODY_PARAMS = HELIOS_EXTRA_BODY_PARAMS | frozenset(
    {
        "sink_latents",
        "nearby_sink_latents",
        "nearby_sink_indices",
        "camera_poses",
        "camera_intrinsics",
        "action",
        "depth_backend",
        "max_sequence_length",
    }
)
EVOKE_EXTRA_OUTPUT_PARAMS: frozenset[str] = frozenset()
