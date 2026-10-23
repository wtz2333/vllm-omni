# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""vLLM-native adapter for the Evoke diffusion transformer.

Evoke's denoiser has the Helios transformer parameter layout.  The model-specific
behavior is the causal rollout pipeline and raw sink-frame inputs, so the
adapter reuses the tested TP/SP implementation and only exposes Evoke's config
flag.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from vllm_omni.diffusion.models.helios.helios_transformer import HeliosTransformer3DModel

if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization.base_config import QuantizationConfig


class EvokeTransformer3DModel(HeliosTransformer3DModel):
    """Evoke ``EvokeTransformer3DModel`` with vLLM-native execution."""

    def __init__(
        self,
        *,
        use_raw_sink_frames: bool = True,
        quant_config: QuantizationConfig | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            use_raw_sink_frames=use_raw_sink_frames,
            quant_config=quant_config,
            **kwargs,
        )
