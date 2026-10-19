# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU-only diffusion dispatch for EdenLive numerical tests."""

import json

import pytest
import torch

from vllm_omni.diffusion.config import set_current_diffusion_config
from vllm_omni.diffusion.data import AttentionConfig, AttentionSpec, OmniDiffusionConfig
from vllm_omni.diffusion.forward_context import set_forward_context


@pytest.fixture
def cpu_config(tmp_path, monkeypatch):
    from vllm_omni.diffusion.attention import selector
    from vllm_omni.diffusion.attention.backends.sdpa import SDPABackend

    # Host platform detection can report CUDA even with every device hidden.
    # Retain the real Attention layer and SDPA implementation, bypass only discovery.
    monkeypatch.setattr(selector, "_cached_get_backend_cls", lambda *args, **kwargs: SDPABackend)
    (tmp_path / "model_index.json").write_text(json.dumps({"_class_name": "EdenLivePipeline"}))
    config = OmniDiffusionConfig(
        model=str(tmp_path),
        model_class_name="EdenLivePipeline",
        dtype=torch.float32,
        enforce_eager=True,
        diffusion_attention_config=AttentionConfig(default=AttentionSpec(backend="TORCH_SDPA")),
    )
    with set_current_diffusion_config(config), set_forward_context(omni_diffusion_config=config):
        yield config
