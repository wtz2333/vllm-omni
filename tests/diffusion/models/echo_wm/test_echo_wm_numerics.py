# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Dtype-sensitive contracts of the released distilled checkpoint."""

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.models.echo_wm.causal_cache import resolve_causal_sigmas
from vllm_omni.diffusion.models.echo_wm.pipeline import EchoWMCausalPipeline
from vllm_omni.diffusion.models.echo_wm.transformer import EchoWMTransformer3DModel

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("sigma", resolve_causal_sigmas())
def test_timestep_preserves_reference_dtype_rounding(dtype, sigma):
    recorded = []

    def adaln(timesteps, *, hidden_dtype):
        recorded.append(timesteps)
        assert hidden_dtype == dtype
        return timesteps[:, None], timesteps[:, None]

    model = SimpleNamespace(timestep_scale_multiplier=1000)
    EchoWMTransformer3DModel._prepare_timesteps(model, sigma, adaln, 3, dtype, torch.device("cpu"))
    expected = torch.full((3,), sigma, dtype=dtype) * 1000
    assert recorded[0].dtype == expected.dtype
    torch.testing.assert_close(recorded[0], expected, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_velocity_to_x0_uses_quantized_modality_timestep(dtype):
    sigma = resolve_causal_sigmas()[1]
    tokens = torch.linspace(1, 3, 8, dtype=dtype).view(1, 2, 4)
    velocity = torch.full_like(tokens, 3.25)
    model = SimpleNamespace(transformer=lambda **kwargs: (velocity, velocity))
    session = SimpleNamespace(patches_per_frame=2, caches=[])
    inputs = SimpleNamespace(
        ucpe_viewmats=torch.eye(4)[None, None], ucpe_Ks=torch.eye(3)[None, None], video_context=None, audio_context=None
    )
    video, audio = EchoWMCausalPipeline._forward_block(model, session, inputs, tokens, tokens, sigma, sigma, 0, 0)
    timestep = torch.full(tokens.shape[:2], sigma, dtype=dtype).unsqueeze(-1)
    expected = (tokens.float() - velocity.float() * timestep.float()).to(dtype)
    torch.testing.assert_close(video, expected, rtol=0, atol=0)
    torch.testing.assert_close(audio, expected, rtol=0, atol=0)
    if dtype == torch.bfloat16:
        old = (tokens.float() - velocity.float() * sigma).to(dtype)
        assert not torch.equal(expected, old), "The regression must distinguish the former FP32-sigma path"


@pytest.mark.parametrize(
    "parallel, message",
    [
        ({"ring_degree": 2}, "Ring"),
        ({"cfg_parallel_size": 2}, "CFG"),
        ({"use_hsdp": True}, "HSDP"),
    ],
)
def test_unsupported_parallel_modes_fail_before_loading(parallel, message):
    config = SimpleNamespace(
        model="not-opened.safetensors", model_config={}, parallel_config=SimpleNamespace(**parallel)
    )
    with pytest.raises(ValueError, match=message):
        EchoWMCausalPipeline(od_config=config)


def test_bf16_session_rope_matches_reference_video_latent_tools(monkeypatch):
    import os
    from pathlib import Path

    root = Path(os.environ.get("ECHOWM_REFERENCE_ROOT", ""))
    if not (root / "ltx-core/src").is_dir():
        pytest.skip("Echo-WM reference repository unavailable")
    monkeypatch.syspath_prepend(str(root / "ltx-core/src"))
    from ltx_core.components.patchifiers import VideoLatentPatchifier
    from ltx_core.model.transformer.rope import LTXRopeType, generate_freq_grid_np, precompute_freqs_cis
    from ltx_core.tools import VideoLatentTools
    from ltx_core.types import VideoLatentShape

    from vllm_omni.diffusion.models.echo_wm.causal_cache import EchoWMCacheConfig

    reference_tools = VideoLatentTools(VideoLatentPatchifier(1), VideoLatentShape(1, 16, 7, 2, 4), 24.0)
    reference_positions = reference_tools.create_initial_state(torch.device("cpu"), torch.bfloat16).positions
    expected = precompute_freqs_cis(
        indices_grid=reference_positions,
        dim=16,
        out_dtype=torch.bfloat16,
        theta=10000.0,
        max_pos=[20, 16, 16],
        num_attention_heads=2,
        use_middle_indices_grid=True,
        rope_type=LTXRopeType.SPLIT,
        freq_grid_generator=generate_freq_grid_np,
    )
    cache = SimpleNamespace()
    config = SimpleNamespace(
        num_attention_heads=2,
        audio_num_attention_heads=2,
        positional_embedding_max_pos=[20, 16, 16],
        audio_positional_embedding_max_pos=[20],
        audio_cross_attention_dim=8,
        in_channels=16,
    )
    transformer = SimpleNamespace(
        patchify_proj=SimpleNamespace(weight=torch.empty(1, dtype=torch.bfloat16)),
        config=config,
        inner_dim=16,
        audio_inner_dim=8,
        allocate_caches=lambda **kwargs: [cache],
    )
    pipeline = SimpleNamespace(transformer=transformer, dtype=torch.bfloat16)
    inputs = SimpleNamespace(
        height=64,
        width=128,
        num_frames=49,
        fps=24.0,
        seed=42,
        cache_config=EchoWMCacheConfig(),
        video_context=torch.zeros(1, 6, 16),
        ucpe_viewmats=torch.eye(4).repeat(1, 7, 1, 1),
        ucpe_Ks=torch.eye(3).repeat(1, 7, 1, 1),
        timesteps=(1000, 750, 500, 250),
    )
    EchoWMCausalPipeline._init_session(pipeline, inputs)
    torch.testing.assert_close(cache.video_rope[0], expected[0], rtol=0, atol=0)
    torch.testing.assert_close(cache.video_rope[1], expected[1], rtol=0, atol=0)


def test_audio_timestamps_match_reference_across_long_rollout(monkeypatch):
    import os
    from pathlib import Path

    root = Path(os.environ.get("ECHOWM_REFERENCE_ROOT", ""))
    if not (root / "ltx-core/src").is_dir():
        pytest.skip("Echo-WM reference repository unavailable")
    monkeypatch.syspath_prepend(str(root / "ltx-core/src"))
    from ltx_core.components.patchifiers import AudioPatchifier
    from ltx_core.tools import AudioLatentTools
    from ltx_core.types import AudioLatentShape

    from vllm_omni.diffusion.models.echo_wm.causal_cache import build_audio_positions

    tools = AudioLatentTools(AudioPatchifier(1), AudioLatentShape(1, 8, 177, 16))
    expected = tools.create_initial_state(torch.device("cpu"), torch.bfloat16).positions
    actual = build_audio_positions(177)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_ucpe_geometry_survives_model_dtype_conversion():
    from vllm_omni.diffusion.models.echo_wm.ucpe import PropeDotProductAttention

    prope = PropeDotProductAttention(128, 40, 22, 1280, 704)
    original = {name: tensor.clone() for name, tensor in prope.named_buffers()}
    prope.bfloat16()
    for name, tensor in prope.named_buffers():
        assert tensor.dtype == torch.float32
        torch.testing.assert_close(tensor, original[name], rtol=0, atol=0)
    prope.half()
    for name, tensor in prope.named_buffers():
        torch.testing.assert_close(tensor, original[name], rtol=0, atol=0)


def test_ucpe_geometry_is_independent_of_loader_default_dtype():
    from vllm_omni.diffusion.models.echo_wm.ucpe import PropeDotProductAttention

    expected = PropeDotProductAttention(128, 40, 22, 1280, 704)
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        actual = PropeDotProductAttention(128, 40, 22, 1280, 704)
    finally:
        torch.set_default_dtype(previous)
    for name, tensor in actual.named_buffers():
        assert tensor.dtype == torch.float32
        torch.testing.assert_close(tensor, dict(expected.named_buffers())[name], rtol=0, atol=0)


def test_ucpe_geometry_survives_meta_materialization():
    from vllm_omni.diffusion.models.echo_wm.ucpe import PropeDotProductAttention

    expected = PropeDotProductAttention(128, 40, 22, 1280, 704)
    with torch.device("meta"):
        actual = PropeDotProductAttention(128, 40, 22, 1280, 704)
    actual.to("meta").to_empty(device="cpu")
    for name, tensor in actual.named_buffers():
        assert tensor.dtype == torch.float32
        torch.testing.assert_close(tensor, dict(expected.named_buffers())[name], rtol=0, atol=0)
