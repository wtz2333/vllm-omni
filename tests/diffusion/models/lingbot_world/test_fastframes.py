# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest
import torch

from tests.diffusion.models.lingbot_world import test_lingbot_world_attention as attention_tests
from tests.diffusion.models.lingbot_world.test_lingbot_world_transformer import _cache, _tiny_model

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def test_from_config_accepts_lingbot_1_3b_checkpoint() -> None:
    module = attention_tests._load_module()
    config = {
        "num_attention_heads": 12,
        "num_layers": 30,
        "ffn_dim": 8960,
        "sink_size": 6,
        "num_frames_per_block": 4,
    }
    with torch.device("meta"):
        model = module.CausalLingBotWorldTransformer3DModel.from_config(config)
    assert model.dim == 1536
    assert len(model.blocks) == 30
    assert (model.config.sink_size, model.config.num_frames_per_block) == (6, 4)


@torch.inference_mode()
def test_fastframes_capture_preserves_flow_and_records_canonical_kv() -> None:
    module = attention_tests._load_module()
    model = _tiny_model(module, num_layers=2).eval()
    inputs = (
        torch.randn(1, 36, 1, 4, 4),
        torch.tensor([625.0]),
        torch.randn(1, 3, 6),
        torch.randn(1, 6 * 8 * 8, 1, 4, 4),
    )
    normalized_keys = [attention_tests._record_outputs(block.self_attn.norm_k) for block in model.blocks]
    projections = [attention_tests._record_outputs(block.self_attn.qkv) for block in model.blocks]
    head_inputs: list[tuple[torch.Tensor, ...]] = []
    model.head.register_forward_pre_hook(lambda _module, args: head_inputs.append(args))
    expected = model(*inputs, cache=_cache(module, model), start_frame=0, update_cache=False)
    keys, values = (torch.empty(2, 4, 2, 2) for _ in range(2))
    capture = SimpleNamespace(kv=(keys, values), capture_pixels=True, pixel_inputs=None, rope=None)
    actual = model(*inputs, cache=_cache(module, model), start_frame=0, update_cache=False, fastframes_capture=capture)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for index, (key, value) in enumerate(zip(keys, values, strict=True)):
        torch.testing.assert_close(key.flatten(1), normalized_keys[index][-1][0], rtol=0, atol=0)
        raw_value = projections[index][-1][0].chunk(3, dim=-1)[2]
        torch.testing.assert_close(value.flatten(1), raw_value, rtol=0, atol=0)
    assert capture.pixel_inputs is not None
    features, modulation = capture.pixel_inputs
    assert features is head_inputs[-1][0]
    torch.testing.assert_close(modulation, head_inputs[-1][1].unsqueeze(2), rtol=0, atol=0)
    assert capture.rope is not None
