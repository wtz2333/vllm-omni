# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU numerical and lifecycle contracts; no pretrained weights or GPU required."""

import numpy as np
import pytest
import torch
from diffusers.models.transformers.transformer_wan import WanTransformer3DModel

from vllm_omni.diffusion.models.edenlive.conditioning import projection_weights
from vllm_omni.diffusion.models.edenlive.pipeline_edenlive import first_frame_pixels, proxy_frames, validate_sampling
from vllm_omni.diffusion.models.edenlive.sampling import TIMESTEPS, step
from vllm_omni.diffusion.models.edenlive.transformer import EdenLiveTransformer, prepare_attention_context
from vllm_omni.experimental.ar_diffusion.capability import ARDiffusionKVBranchSpec
from vllm_omni.experimental.ar_diffusion.kv_cache import ARDiffusionKVCache, ARDiffusionKVConfig
from vllm_omni.experimental.ar_diffusion.kv_cache.state import ARDiffusionKVState
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

TINY_CONFIG = dict(
    in_channels=4,
    out_channels=4,
    num_layers=2,
    num_attention_heads=2,
    attention_head_dim=8,
    text_dim=16,
    freq_dim=8,
    ffn_dim=32,
    patch_size=(1, 2, 2),
    eps=1e-6,
    cross_attn_norm=True,
    rope_max_seq_len=32,
    image_dim=None,
    added_kv_proj_dim=None,
)


def make_state(*, tokens_per_frame=4, num_layers=2, dtype=torch.float32, block_size=16, name="test"):
    cache = ARDiffusionKVCache(
        ARDiffusionKVConfig(enable=True, chunk_size=tokens_per_frame, window_chunks=12, sink_chunks=6),
        num_layers=num_layers,
        num_kv_heads=2,
        head_size=8,
        dtype=dtype,
        block_size=block_size,
        max_model_len=4096,
        available_bytes=1 << 22,
        kv_branches=(ARDiffusionKVBranchSpec("main", 0),),
        session_capacity=1,
        frames_per_block=3,
        device=torch.device("cpu"),
    )
    return ARDiffusionKVState(cache, name, {"main": cache.begin_request(name)}, num_layers=num_layers)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("per_frame_timestep", [True, False])
def test_first_chunk_matches_diffusers_wan(cpu_config, dtype, per_frame_timestep):
    torch.manual_seed(5)
    reference = WanTransformer3DModel(**{**TINY_CONFIG, "in_channels": 10}).to(dtype=dtype).eval()
    model = EdenLiveTransformer(TINY_CONFIG, control_dim=6).to(dtype=dtype).eval()
    model.load_state_dict(reference.state_dict(), strict=True)
    # The released adapter casts the entire loaded DiT to its runtime dtype.
    latents = torch.randn(1, 4, 3, 4, 4, dtype=dtype)
    control = torch.randn(1, 6, 3, 4, 4, dtype=dtype)
    prompt = torch.randn(1, 7, 16, dtype=dtype)
    timestep = torch.tensor([[0.0, 937.5, 937.5]]) if per_frame_timestep else torch.tensor([937.5])
    reference_time_shapes, actual_time_shapes = [], []
    reference.condition_embedder.time_embedder.register_forward_pre_hook(
        lambda module, args: reference_time_shapes.append(args[0].shape)
    )
    model.condition_embedder.time_embedder.register_forward_pre_hook(
        lambda module, args: actual_time_shapes.append(args[0].shape)
    )
    state = make_state(dtype=dtype)
    try:
        with torch.no_grad():
            expected = reference(
                torch.cat((latents, control), dim=1),
                timestep.repeat_interleave(4, dim=1) if per_frame_timestep else timestep,
                prompt,
            ).sample
            actual = model(latents, control, timestep, prompt, state=state, start_frame=0)
        # BF16 projection kernels depend on GEMM shape even for repeated values.
        assert actual_time_shapes == reference_time_shapes
        torch.testing.assert_close(
            actual,
            expected,
            rtol=2e-5 if dtype == torch.float32 else 2e-2,
            atol=2e-5 if dtype == torch.float32 else 2e-2,
        )
        assert state.adapter("main").absolute_num_computed_tokens == 0
    finally:
        state.close()


@pytest.mark.parametrize("tokens_per_frame", [1, 4, 5, 16])
def test_exact_sink_window_survives_probes_eviction_and_short_tail(tokens_per_frame):
    state = make_state(tokens_per_frame=tokens_per_frame, num_layers=1)
    pool = state.kv_cache.key_cache(0).flatten(0, 1)
    try:
        # More than three complete window lengths exercises storage compaction too.
        for start in range(0, 63, 3):
            current = torch.arange(start * tokens_per_frame, (start + 3) * tokens_per_frame).float()
            history = list(range(min(start, 6) * tokens_per_frame))
            history += list(range(max(6, start - 9) * tokens_per_frame, start * tokens_per_frame)) if start > 6 else []
            for commit in (False, False, True):
                context = prepare_attention_context(
                    state,
                    start_frame=start,
                    frames=3,
                    tokens_per_frame=tokens_per_frame,
                    commit=commit,
                    device=torch.device("cpu"),
                )
                pool[context.write_slots] = current[:, None, None]
                actual = pool[context.read_slots, 0, 0]
                torch.testing.assert_close(actual, torch.tensor(history + current.tolist()))
                assert state.adapter("main").absolute_num_computed_tokens == start * tokens_per_frame
                if commit:
                    state.commit_paged_context("main")
        context = prepare_attention_context(
            state,
            start_frame=63,
            frames=1,
            tokens_per_frame=tokens_per_frame,
            commit=False,
            device=torch.device("cpu"),
        )
        pool[context.write_slots] = 63
        assert context.read_slots.numel() == 16 * tokens_per_frame
        state.reset()
        assert state.adapter("main").absolute_num_computed_tokens == 0
        assert not state.kv_cache.window_block_ids(state.adapter("main"))
    finally:
        state.close()


def test_raw_keys_are_rotated_by_window_slot_on_every_read(cpu_config):
    from vllm_omni.diffusion.models.edenlive.transformer import attend

    torch.manual_seed(17)
    model = EdenLiveTransformer(TINY_CONFIG, control_dim=6).eval()
    block = model.blocks[0]
    state = make_state(tokens_per_frame=1, num_layers=1)
    clean_keys, clean_values = [], []
    try:
        with torch.no_grad():
            for start in range(0, 27, 3):
                hidden = torch.randn(1, 3, 16)
                context = prepare_attention_context(
                    state,
                    start_frame=start,
                    frames=3,
                    tokens_per_frame=1,
                    commit=True,
                    device=torch.device("cpu"),
                )
                key = block.attn1.norm_k(block.attn1.to_k(hidden)).reshape(1, 3, 2, 8)
                value = block.attn1.to_v(hidden).reshape(1, 3, 2, 8)
                clean_keys.append(key)
                clean_values.append(value)
                keys, values = torch.cat(clean_keys, dim=1), torch.cat(clean_values, dim=1)
                if keys.shape[1] > 18:
                    keys = torch.cat((keys[:, :6], keys[:, -12:]), dim=1)
                    values = torch.cat((values[:, :6], values[:, -12:]), dim=1)
                rotary = model.rope(torch.empty(1, 1, keys.shape[1], 2, 2))
                query = block.attn1.norm_q(block.attn1.to_q(hidden)).reshape(1, 3, 2, 8)

                def rotate(tensor, cosine, sine):
                    pairs = torch.view_as_complex(tensor.float().reshape(*tensor.shape[:-1], -1, 2))
                    phase = torch.complex(cosine[..., ::2], sine[..., 1::2])
                    return torch.view_as_real(pairs * phase).flatten(-2)

                rq = rotate(query, rotary[0][:, -3:], rotary[1][:, -3:])
                rk = rotate(keys, *rotary)
                scores = torch.einsum("bqhd,bkhd->bhqk", rq, rk) / 8**0.5
                expected = torch.einsum("bhqk,bkhd->bqhd", scores.softmax(-1), values).flatten(2)
                expected = block.attn1.to_out[0](expected)
                actual = attend(block.attn1, block.self_attention, hidden, context=context, rotary=rotary)
                torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
                pool = state.kv_cache.key_cache(0).flatten(0, 1)
                torch.testing.assert_close(pool[context.write_slots], key[0])
                state.commit_paged_context("main")
    finally:
        state.close()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_stochastic_schedule_preserves_reference_rounding(dtype, monkeypatch):
    from vllm_omni.diffusion.models.edenlive import sampling

    # With zero re-noising, the full four-step reference trajectory has fixed values.
    monkeypatch.setattr(sampling, "randn_tensor", lambda shape, **kwargs: torch.zeros(shape, dtype=kwargs["dtype"]))
    current = torch.ones(1, 1, 1, 1, 1, dtype=dtype)
    generator = torch.Generator().manual_seed(0)
    sigmas = torch.tensor([1, 15 / 16, 5 / 6, 5 / 8], dtype=dtype)
    expected = current.clone()
    for index, sigma in enumerate(sigmas):
        expected = expected - sigma * torch.full_like(expected, 0.125)
        if index < 3:
            expected = expected * (1 - sigmas[index + 1])
        current = step(current, torch.full_like(current, 0.125), index, generator)
        torch.testing.assert_close(current, expected, rtol=0, atol=0)
    assert TIMESTEPS[1] == 937.5
    assert current.dtype == dtype


def test_proxy_frames_use_decoded_time_axis():
    frames = np.stack([np.full((8, 8, 3), index, dtype=np.uint8) for index in range(21)])
    selected = proxy_frames(frames, 21)
    assert selected[:, 0, 0, 0].tolist() == [0, 4, 8, 12, 16, 20]
    with pytest.raises(ValueError, match="at least 25"):
        proxy_frames(frames, 25)


def test_first_frame_center_crop_and_pixel_normalization():
    image = np.zeros((4, 8, 3), dtype=np.uint8)
    image[:, 2:6] = 255
    pixels = first_frame_pixels(image, 4, 4)
    assert pixels.shape == (1, 3, 1, 4, 4)
    torch.testing.assert_close(pixels, torch.ones_like(pixels), rtol=0, atol=0)
    resized = first_frame_pixels(image, 8, 8)
    assert resized.dtype == torch.float32
    assert resized.shape == (1, 3, 1, 8, 8)
    assert resized.min() >= -1 and resized.max() <= 1


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"num_inference_steps": 5}, "four|4"),
        ({"guidance_scale": 5.0}, "guidance_scale"),
        ({"num_frames": 12}, "4\\*k\\+1"),
        ({"num_outputs_per_prompt": 2}, "one output"),
        ({"height": 480}, "cache geometry"),
        ({"height": 0}, "cache geometry"),
        ({"max_sequence_length": 256}, "max_sequence_length"),
        ({"sigmas": [1.0, 0.5, 0.0]}, "fixed four-step"),
        ({"extra_args": {"flow_shift": 3.0}}, "shift-5"),
    ],
)
def test_invalid_sampling_is_rejected(overrides, match):
    with pytest.raises(ValueError, match=match):
        validate_sampling(OmniDiffusionSamplingParams(**overrides), 704, 1280)


def test_projection_loading_is_strict():
    weights = {
        f"model.module.encoder.{part}.{kind}": torch.ones(1)
        for part in ("global_projection", "patch_projection")
        for kind in ("weight", "bias")
    }
    assert len(projection_weights(weights)) == 4
    weights.pop("model.module.encoder.patch_projection.weight")
    with pytest.raises(ValueError, match="Incomplete"):
        projection_weights(weights)


def test_projection_checkpoint_preserves_frozen_backbone_weights():
    weights = {
        f"encoder.{part}.{kind}": torch.ones(1)
        for part in ("global_projection", "patch_projection")
        for kind in ("weight", "bias")
    }
    weights["encoder.backbone.embedding.weight"] = torch.tensor([3.0])
    converted = projection_weights(weights, {"backbone.model.embedding.weight"})
    torch.testing.assert_close(converted["backbone.model.embedding.weight"], torch.tensor([3.0]))
