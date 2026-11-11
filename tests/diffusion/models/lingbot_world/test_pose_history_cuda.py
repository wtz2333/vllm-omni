# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch

from tests.helpers.mark import hardware_test
from vllm_omni.diffusion.models.lingbot_world.pose_history import paged_history_attention
from vllm_omni.experimental.ar_diffusion.kv_cache.paged_attention import ARDiffusionPagedLayerInputs

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]


@hardware_test(res={"cuda": "H200"}, num_cards=1)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_paged_pose_reference_matches_dense_cpu() -> None:
    torch.manual_seed(13)
    shape = (64, 4, 64)
    base = ARDiffusionPagedLayerInputs(
        layer_idx=torch.tensor(0),
        key_pool=torch.randn(shape, dtype=torch.bfloat16),
        value_pool=torch.randn(shape, dtype=torch.bfloat16),
        video_slots=torch.arange(32, 48, dtype=torch.int64),
        block_size=16,
        seq_len=48,
        action_slots=torch.empty(0, dtype=torch.int64),
        block_table=torch.tensor([[0, 1, 2, 3]], dtype=torch.int32),
        seq_lens=torch.tensor([48], dtype=torch.int32),
        query_start_loc=torch.tensor([0, 16], dtype=torch.int32),
        max_query_len=16,
        max_seq_len=64,
    )
    gpu = base._replace(
        key_pool=base.key_pool.cuda(),
        value_pool=base.value_pool.cuda(),
        video_slots=base.video_slots.cuda(),
        action_slots=base.action_slots.cuda(),
        block_table=base.block_table.cuda(),
        seq_lens=base.seq_lens.cuda(),
        query_start_loc=base.query_start_loc.cuda(),
    )
    query = torch.randn(16, 4, 64, dtype=torch.bfloat16)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    ref_key = torch.randn(1, 20, 4, 64, dtype=torch.bfloat16)
    ref_value = torch.randn_like(ref_key)
    bounds = torch.full((1, 16, 1, 1), 0.15)
    bounds[:, :4] = 0
    expected = paged_history_attention(base, query, key, value, ref_key, ref_value, bounds, softmax_scale=0.125)
    actual = paged_history_attention(
        gpu,
        query.cuda(),
        key.cuda(),
        value.cuda(),
        ref_key.cuda(),
        ref_value.cuda(),
        bounds.cuda(),
        softmax_scale=0.125,
    )
    torch.accelerator.synchronize()
    torch.testing.assert_close(actual.cpu(), expected, atol=0.02, rtol=0.04)
    torch.testing.assert_close(gpu.key_pool[32:48].cpu(), key)
