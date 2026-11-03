# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch
from diffusers.utils.torch_utils import randn_tensor
from torch import nn

from tests.diffusion.models.lingbot_world import test_lingbot_world_attention as attention_tests
from tests.diffusion.models.lingbot_world.test_lingbot_world_transformer import _tiny_model
from tests.diffusion.models.wan2_2.conftest import noop_progress_bar
from vllm_omni.diffusion.layers.rope import RotaryEmbeddingWan
from vllm_omni.diffusion.models.lingbot_world.dmd_block import ARBlockContext, LingBotDMDBlockRunner
from vllm_omni.diffusion.models.lingbot_world.pipeline import LingBotWorldCausalDMDPipeline
from vllm_omni.experimental.ar_diffusion.capability import ARDiffusionKVBranchSpec
from vllm_omni.experimental.ar_diffusion.kv_cache import ARDiffusionKVCache, ARDiffusionKVConfig
from vllm_omni.experimental.ar_diffusion.kv_cache.paged_attention import ar_diffusion_paged_attention
from vllm_omni.experimental.ar_diffusion.kv_cache.state import ARDiffusionKVState

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def test_packed_window_attention_masks_partial_page(monkeypatch):
    module = attention_tests._load_module()
    monkeypatch.setattr(module, "ar_diffusion_paged_attention", ar_diffusion_paged_attention)
    generator = torch.Generator().manual_seed(7)
    query = torch.randn(1, 5, 2, 4, generator=generator)
    key = torch.randn(1, 9, 2, 4, generator=generator)
    value = torch.randn(1, 9, 2, 4, generator=generator)
    actual = module._paged_window_attention(query, key, value, block_size=16, softmax_scale=0.5)
    expected = torch.nn.functional.scaled_dot_product_attention(
        query.transpose(1, 2),
        key.transpose(1, 2),
        value.transpose(1, 2),
        scale=0.5,
    ).transpose(1, 2)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


@pytest.fixture(autouse=True)
def _native_rope_backend(monkeypatch):
    # Wan 的实际 CPU 实现接收带 head 广播维度的 cos/sin。
    monkeypatch.setattr(
        RotaryEmbeddingWan,
        "dispatch_forward",
        lambda self: lambda x, cos, sin: self.forward_native(x, cos.unsqueeze(-2), sin.unsqueeze(-2)),
    )


def _model():
    module = attention_tests._load_module()
    model = _tiny_model(module, num_layers=2, sliding_window_num_frames=6, attention_head_dim=4)
    model.framefold_config = module.FrameFoldConfig(budget_frames=6, sink_frames=1, recent_frames=1, pooling_factor=2)
    for block in model.blocks:
        block.self_attn.rotary_embedding = RotaryEmbeddingWan(is_neox_style=False, half_head_dim=True)
    return model


def _allocate(model, **kwargs):
    return model.allocate_cache(
        batch_size=1,
        latent_height=4,
        latent_width=4,
        device=torch.device("cpu"),
        dtype=torch.float32,
        **kwargs,
    )


@pytest.fixture
def packed_state():
    kv = ARDiffusionKVCache(
        ARDiffusionKVConfig(enable=True, chunk_size=4, window_chunks=5, sink_chunks=1, gpu_memory_fraction=1.0),
        num_layers=2,
        num_kv_heads=2,
        head_size=4,
        dtype=torch.float32,
        block_size=16,
        max_model_len=128,
        available_bytes=1 << 20,
        kv_branches=(ARDiffusionKVBranchSpec("main", 0),),
        session_capacity=1,
        frames_per_block=1,
        device=torch.device("cpu"),
    )
    state = ARDiffusionKVState(kv, "test", {"main": kv.begin_request("test")}, num_layers=2)
    yield state
    state.close()


def test_real_rope_attention_matches_between_contiguous_and_packed_history(packed_state):
    torch.manual_seed(12)
    model = _model().eval()
    direct = _allocate(model)
    packed = _allocate(model, kv_buffers=packed_state.reserve_packed_history("main", 24))
    with torch.inference_mode():
        for frame in range(16):
            hidden = torch.randn(1, 36, 1, 4, 4)
            text = torch.randn(1, 3, 6)
            camera = torch.randn(1, 384, 1, 4, 4)
            for update in (False, False, True):
                args = dict(
                    hidden_states=hidden,
                    timestep=torch.tensor([0.0 if update else 1.0]),
                    encoder_hidden_states=text,
                    camera_hidden_states=camera,
                    start_frame=frame,
                    update_cache=update,
                )
                expected = model(cache=direct, **args)
                actual = model(cache=packed, **args)
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            packed_state.commit_packed_context("main", 4)
            for first, second in zip(direct.self_attention, packed.self_attention):
                assert first.spans == second.spans and sum(second.spans) == frame + 1
                assert second.absolute_end == (frame + 1) * 4
    assert [layer.merge_count for layer in packed.self_attention] == [10, 10]


def test_no_merge_attention_matches_fifo_with_real_rope():
    torch.manual_seed(5)
    model = _model().eval()
    folded = _allocate(model)
    config = model.framefold_config
    model.framefold_config = None
    fifo = _allocate(model)
    model.framefold_config = config
    with torch.inference_mode():
        for frame in range(6):
            args = dict(
                hidden_states=torch.randn(1, 36, 1, 4, 4),
                timestep=torch.tensor([0.0]),
                encoder_hidden_states=torch.randn(1, 3, 6),
                camera_hidden_states=torch.randn(1, 384, 1, 4, 4),
                start_frame=frame,
                update_cache=True,
            )
            torch.testing.assert_close(model(cache=folded, **args), model(cache=fifo, **args), atol=0, rtol=0)
    assert all(layer.merge_count == 0 for layer in folded.self_attention)


def test_dmd_uses_packed_history_and_commits_only_clean_forward(packed_state, monkeypatch):
    model = _model().eval()
    cache = _allocate(model, kv_buffers=packed_state.reserve_packed_history("main", 24))
    runner = LingBotDMDBlockRunner(model, device=torch.device("cpu"), enforce_eager=True)
    monkeypatch.setattr(
        "vllm_omni.diffusion.models.lingbot_world.dmd_block.set_forward_context_denoise_step_idx", lambda step: None
    )
    condition = torch.randn(1, 34, 1, 4, 4)
    latents = torch.randn(1, 2, 1, 4, 4)
    camera, prompt = torch.randn(1, 384, 1, 4, 4), torch.randn(1, 3, 6)
    ar = ARBlockContext(packed_state, cache.cross_attention, "main", framefold_cache=cache)
    with torch.inference_mode():
        for step in range(4):
            flow = runner.probe_step(
                current_latents=latents,
                condition=condition,
                camera=camera,
                prompt_embeds=prompt,
                cache=None,
                ar=ar,
                start_frame=0,
                timestep_value=900 - step * 200,
                step_index=step,
            )
            assert flow.shape == latents.shape
            assert all(layer.spans == [] and layer.query_history is None for layer in cache.self_attention)
        runner.commit_block_kv(
            latents=latents,
            condition=condition,
            camera=camera,
            prompt_embeds=prompt,
            cache=None,
            ar=ar,
            start_frame=0,
        )
    assert packed_state._committed["main"] == 4
    assert all(layer.spans == [1] and layer.query_history is not None for layer in cache.self_attention)


@pytest.mark.parametrize("eager,reuse,first_block_bytes", [(False, False, 0), (True, True, 0), (True, False, 64)])
def test_dmd_rejects_modes_without_clean_eager_commit(eager, reuse, first_block_bytes):
    with pytest.raises(ValueError, match="FrameFold"):
        LingBotDMDBlockRunner(
            _model(),
            device=torch.device("cpu"),
            enforce_eager=eager,
            reuse_last_step_kv=reuse,
            first_block_kv_cpu_limit_bytes=first_block_bytes,
        )


def test_pipeline_reuses_and_releases_framefold_session_metadata(packed_state):
    pipeline = object.__new__(LingBotWorldCausalDMDPipeline)
    nn.Module.__init__(pipeline)
    pipeline.transformer = _model()
    pipeline._framefold_sessions = {}
    pipeline._ar_diffusion_kv_state = None
    pipeline._ar_sessions = {}
    pipeline._streaming_decode_states = {}
    pipeline._ar_height = pipeline._ar_width = 32
    pipeline.vae_scale_factor_spatial = 8
    pipeline.device = torch.device("cpu")
    with pipeline.bind_ar_diffusion_state("test", packed_state):
        first = pipeline._ar_block_context([None, None]).framefold_cache
        assert pipeline._ar_block_context([None, None]).framefold_cache is first
        pipeline.reset_ar_diffusion_session("test")
        packed_state.reset()
        second = pipeline._ar_block_context([None, None]).framefold_cache
        assert second is not first
        assert all(layer.spans == [] for layer in second.self_attention)
        pipeline.close_ar_diffusion_session("test")
        assert pipeline._framefold_sessions == {}


def test_dmd_complete_blocks_match_stepwise_sampling_across_merges(packed_state, monkeypatch):
    torch.manual_seed(8)
    model = _model().eval()
    direct = _allocate(model)
    packed = _allocate(model, kv_buffers=packed_state.reserve_packed_history("main", 24))
    runner = LingBotDMDBlockRunner(model, device=torch.device("cpu"), enforce_eager=True)
    monkeypatch.setattr(
        "vllm_omni.diffusion.models.lingbot_world.dmd_block.set_forward_context_denoise_step_idx", lambda step: None
    )
    direct_generator = torch.Generator().manual_seed(123)
    step_generator = torch.Generator().manual_seed(123)
    schedule = ((1000.0, 1.0), (800.0, 0.8), (500.0, 0.5), (200.0, 0.2))
    with torch.inference_mode(), noop_progress_bar() as progress:
        for frame in range(9):
            condition, camera = torch.randn(1, 34, 1, 4, 4), torch.randn(1, 384, 1, 4, 4)
            prompt = torch.randn(1, 3, 6)
            expected = runner.generate_block(
                condition=condition,
                camera=camera,
                prompt_embeds=prompt,
                cache=direct,
                ar=None,
                start_frame=frame,
                schedule=schedule,
                generator=direct_generator,
                progress_bar=progress,
            )
            ar = ARBlockContext(packed_state, packed.cross_attention, "main", framefold_cache=packed)
            latents = randn_tensor((1, 2, 1, 4, 4), generator=step_generator, device=torch.device("cpu"))
            for step, (timestep, sigma) in enumerate(schedule):
                flow = runner.probe_step(
                    current_latents=latents,
                    condition=condition,
                    camera=camera,
                    prompt_embeds=prompt,
                    cache=None,
                    ar=ar,
                    start_frame=frame,
                    timestep_value=timestep,
                    step_index=step,
                )
                latents = runner.apply_transition(
                    latents,
                    flow,
                    sigma,
                    next_sigma=schedule[step + 1][1] if step < 3 else None,
                    generator=step_generator,
                )
            runner.commit_block_kv(
                latents=latents,
                condition=condition,
                camera=camera,
                prompt_embeds=prompt,
                cache=None,
                ar=ar,
                start_frame=frame,
            )
            torch.testing.assert_close(latents, expected, atol=0, rtol=0)
            assert torch.equal(direct_generator.get_state(), step_generator.get_state())
    assert all(layer.merge_count == 3 for layer in packed.self_attention)
