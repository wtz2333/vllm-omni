# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch
from torch import nn

from tests.helpers.mark import hardware_test
from vllm_omni.diffusion.offloader.base import OffloadConfig, OffloadStrategy
from vllm_omni.diffusion.offloader.sequential_backend import ModelLevelOffloadBackend
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]


@hardware_test(res={"cuda": "H200"}, num_cards=1)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_unselected_dit_is_resident_before_direct_projection() -> None:
    device = current_omni_platform.get_torch_device(0)
    pipeline = nn.Module()
    pipeline.transformer = nn.Sequential(nn.Linear(4, 4))
    pipeline.text_encoder = nn.Linear(4, 4)
    backend = ModelLevelOffloadBackend(
        OffloadConfig(
            strategy=OffloadStrategy.MODEL_LEVEL,
            components=frozenset({"text_encoder"}),
            pin_cpu_memory=False,
        ),
        device,
    )

    backend.enable(pipeline)
    try:
        assert all(parameter.device == device for parameter in pipeline.transformer.parameters())
        projected = pipeline.transformer[0](torch.ones(1, 4, device=device))
        assert projected.shape == (1, 4)
        assert torch.isfinite(projected).all()
    finally:
        backend.disable()
