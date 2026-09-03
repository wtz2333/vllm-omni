# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Echo-WM transformer tests.

Layout, sigma and weight-mapping contracts always run. Forward-parity tests
additionally execute the Echo-WM reference implementation
(``/data/wtz2333/WorldModel/JoyAI-Echo/echo_wm``) in-process and compare the
two implementations bit-for-bit on a tiny random model; they skip when the
reference repository is absent (e.g. CI).
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import os

import pytest
import torch

from vllm_omni.diffusion.models.echo_wm import (
    DEFAULT_CAUSAL_TIMESTEPS,
    EchoWMCacheConfig,
    EchoWMTransformer3DModel,
    EchoWMUCPEConfig,
    causal_audio_blocks,
    causal_audio_frames,
    causal_video_blocks,
    resolve_causal_sigmas,
)
from vllm_omni.diffusion.models.echo_wm.causal_cache import (
    build_audio_positions,
    build_video_positions,
    compute_cross_slices,
    make_cross_rope_template,
    make_split_rope,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]

_ECHO_ROOT = Path("/data/wtz2333/WorldModel/JoyAI-Echo/echo_wm")


@pytest.fixture(autouse=True)
def _init_distributed():
    """Minimal world_size=1 TP group required to build the parallel linears."""
    from vllm.distributed.parallel_state import (
        cleanup_dist_env_and_memory,
        init_distributed_environment,
        initialize_model_parallel,
    )

    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29517")
    init_distributed_environment(
        world_size=1,
        rank=0,
        local_rank=0,
        distributed_init_method="env://",
    )
    initialize_model_parallel()
    yield
    cleanup_dist_env_and_memory()


@pytest.fixture(autouse=True)
def _force_default_gemm(monkeypatch):
    """Force CPU-compatible GEMM dispatch for CPU test tensors."""
    from vllm.model_executor.layers.utils import default_unquantized_gemm

    monkeypatch.setattr(
        "vllm.model_executor.layers.linear.dispatch_unquantized_gemm",
        lambda: default_unquantized_gemm,
    )


@pytest.fixture(autouse=True)
def _force_torch_sdpa():
    """Pin TORCH_SDPA so CPU tests do not pick CUDA-only backends (FA3)."""
    from vllm_omni.diffusion.config import set_current_diffusion_config
    from vllm_omni.diffusion.data import AttentionConfig

    od_config = SimpleNamespace(
        diffusion_attention_config=AttentionConfig(default="TORCH_SDPA"),
        parallel_config=SimpleNamespace(ring_degree=1),
    )
    with set_current_diffusion_config(od_config):
        yield

# ---------------------------------------------------------------------------
# Layout / schedule contracts (always run).
# ---------------------------------------------------------------------------


def test_causal_sigmas_match_reference_schedule() -> None:
    # Values produced by the reference LTX2Scheduler at the anchor token count
    # for timesteps (1000, 750, 500, 250).
    assert DEFAULT_CAUSAL_TIMESTEPS == (1000, 750, 500, 250)
    sigmas = resolve_causal_sigmas()
    expected = [0.9999999403953552, 0.9626806378364563, 0.8965547680854797, 0.7473050355911255]
    assert sigmas == pytest.approx(expected, rel=1e-6, abs=1e-9)


def test_block_layout_for_reference_geometry() -> None:
    # 31 latent frames (=241 pixel frames): image sink + ten 3-frame chunks.
    assert causal_video_blocks(31) == [(0, 1)] + [(start, start + 3) for start in range(1, 31, 3)]
    assert causal_audio_frames(31) == 2 + 10 * 25
    assert causal_audio_blocks(31) == [(0, 2)] + [(start, start + 25) for start in range(2, 252, 25)]


def test_cache_config_audio_alignment() -> None:
    config = EchoWMCacheConfig()
    config.validate()
    assert config.audio_local_attn_size == 152
    assert config.audio_sink_size == 52
    with pytest.raises(ValueError, match="1 \\+ n \\* video_chunk_size"):
        EchoWMCacheConfig(video_local_attn_size=20).validate()


def test_positions_causal_first_frame_fix() -> None:
    positions = build_video_positions(num_frames=4, height=64, width=64, fps=24.0)
    assert positions.shape == (1, 3, 4 * 4, 2)
    temporal = positions[0, 0].view(4, 4, 2)
    # causal_fix shifts the whole temporal axis by (1 - 8) and clamps:
    # frame 0 spans pixel frames [0, 1), frame 1 spans [1, 9).
    assert temporal[0, 0].tolist() == pytest.approx([0.0, 1.0 / 24.0], rel=1e-6)
    assert temporal[1, 0].tolist() == pytest.approx([1.0 / 24.0, 9.0 / 24.0], rel=1e-6)
    assert temporal[3, 0].tolist() == pytest.approx([17.0 / 24.0, 25.0 / 24.0], rel=1e-6)
    # Spatial axes are pixel coordinates.
    assert positions[0, 1, 0].tolist() == [0.0, 32.0]
    audio = build_audio_positions(num_frames=5)
    assert audio.shape == (1, 1, 5, 2)
    # Latent frame f covers mel [4f + 1 - 4, ...): f=1 -> start 0.01 s.
    assert audio[0, 0, 0, 0].item() == pytest.approx(0.0, abs=1e-9)
    assert audio[0, 0, 1, 0].item() == pytest.approx(160 / 16000, rel=1e-6)


def test_cross_slices_pin_reference_layout() -> None:
    config = EchoWMCacheConfig()
    a2v, v2a = compute_cross_slices(video_frames=7, patches_per_frame=8, cache=config)
    # First AV block (video 1..4, audio 2..27): the video query occupies the
    # tail of the 19-frame window template.
    assert a2v[(2, 27)] == ((4 - 3) * 8, 4 * 8)
    assert v2a[(8, 32)] == (min(27, 152) - 25, min(27, 152))
    # A late block is capped at the window edge.
    a2v_late, v2a_late = compute_cross_slices(video_frames=31, patches_per_frame=8, cache=config)
    assert a2v_late[(227, 252)] == ((19 - 3) * 8, 19 * 8)
    assert v2a_late[(28 * 8, 31 * 8)] == (152 - 25, 152)


def test_checkpoint_contract_rejects_drift() -> None:
    with pytest.raises(ValueError, match="checkpoint contract"):
        EchoWMTransformer3DModel.from_config({"num_layers": 24})
    assert EchoWMTransformer3DModel.from_config({}).config.num_layers == 48


# ---------------------------------------------------------------------------
# Weight mapping (always run).
# ---------------------------------------------------------------------------


def _tiny_model() -> EchoWMTransformer3DModel:
    torch.manual_seed(11)
    return EchoWMTransformer3DModel(
        num_layers=2,
        num_attention_heads=2,
        attention_head_dim=8,
        in_channels=16,
        out_channels=16,
        audio_num_attention_heads=2,
        audio_attention_head_dim=4,
        audio_in_channels=16,
        audio_out_channels=16,
        cross_attention_dim=16,
        audio_cross_attention_dim=8,
        positional_embedding_max_pos=(20, 16, 16),
        audio_positional_embedding_max_pos=(20,),
        ucpe=EchoWMUCPEConfig(
            attn_dim=16,
            num_heads=2,
            patches_x=4,
            patches_y=2,
            image_width=64,
            image_height=32,
        ),
    )


def _tiny_checkpoint(model: EchoWMTransformer3DModel) -> dict[str, torch.Tensor]:
    """Reference-layout checkpoint names (block-level ucpe, separate q/k/v)."""
    weights: dict[str, torch.Tensor] = {}
    next_value = 1

    def add(name: str, shape) -> None:
        nonlocal next_value
        weights[name] = torch.full(shape, next_value, dtype=torch.float32)
        next_value += 1

    for name, param in model.named_parameters():
        if ".to_qkv." in name:
            shard = param.shape[0] // 3
            for projection in ("q", "k", "v"):
                add(name.replace(".to_qkv.", f".to_{projection}."), (shard, *param.shape[1:]))
            continue
        ref_name = name.replace(".ucpe.", ".").replace(".norm_q.", ".q_norm.").replace(".norm_k.", ".k_norm.")
        add(ref_name, param.shape)
    return weights


def test_load_weights_maps_fused_qkv_and_ucpe_and_rejects_unknown() -> None:
    model = _tiny_model()
    checkpoint = _tiny_checkpoint(model)
    covered = model.load_weights(list(checkpoint.items()))
    # Packed QKV tensors report both the checkpoint name and the fused name.
    assert set(checkpoint) <= covered
    assert "transformer_blocks.0.attn1.to_qkv.weight" in covered

    # The fused QKV weight now holds the checkpoint shards in order.
    fused = dict(model.named_parameters())["transformer_blocks.0.attn1.to_qkv.weight"]
    shard_rows = fused.shape[0] // 3
    assert torch.equal(fused[:shard_rows], checkpoint["transformer_blocks.0.attn1.to_q.weight"])
    assert torch.equal(fused[shard_rows : 2 * shard_rows], checkpoint["transformer_blocks.0.attn1.to_k.weight"])
    assert torch.equal(fused[-shard_rows:], checkpoint["transformer_blocks.0.attn1.to_v.weight"])
    # The nested UCPE branch received the block-level checkpoint tensors.
    ucpe_weight = dict(model.named_parameters())["transformer_blocks.0.ucpe.ucpe_q_proj.weight"]
    assert torch.equal(ucpe_weight, checkpoint["transformer_blocks.0.ucpe_q_proj.weight"])

    with pytest.raises(KeyError, match="unknown Echo-WM transformer weight"):
        model.load_weights([("model.diffusion_model.not_a_weight", torch.zeros(1))])

    # Prefixes from the reference single-file checkpoint are stripped.
    model2 = _tiny_model()
    prefixed = {f"model.diffusion_model.{k}": v for k, v in _tiny_checkpoint(model2).items()}
    covered2 = model2.load_weights(list(prefixed.items()))
    assert set(prefixed) <= covered2


# ---------------------------------------------------------------------------
# Reference parity (skipped when the Echo-WM repository is absent).
# ---------------------------------------------------------------------------


def _reference_modules():
    if not (_ECHO_ROOT / "ltx-core" / "src").exists():
        pytest.skip("Echo-WM reference repository not available")
    for package in ("ltx-core/src", "ltx-causal/src"):
        sys.path.insert(0, str(_ECHO_ROOT / package))
    torch.manual_seed(3)

    from ltx_core.guidance.perturbations import BatchedPerturbationConfig
    from ltx_core.model.transformer.attention import AttentionFunction
    from ltx_core.model.transformer.model import LTXModel, LTXModelType
    from ltx_core.model.transformer.rope import LTXRopeType
    from ltx_core.model.transformer.transformer import ActionBlockConfig
    from ltx_core.model.transformer.modality import Modality

    return SimpleNamespace(
        BatchedPerturbationConfig=BatchedPerturbationConfig,
        AttentionFunction=AttentionFunction,
        LTXModel=LTXModel,
        LTXModelType=LTXModelType,
        LTXRopeType=LTXRopeType,
        ActionBlockConfig=ActionBlockConfig,
        Modality=Modality,
    )


def _tiny_reference(ref):
    model = ref.LTXModel(
        model_type=ref.LTXModelType.AudioVideo,
        num_attention_heads=2,
        attention_head_dim=8,
        in_channels=16,
        out_channels=16,
        num_layers=2,
        cross_attention_dim=16,
        norm_eps=1e-6,
        attention_type=ref.AttentionFunction.PYTORCH,
        positional_embedding_theta=10000.0,
        positional_embedding_max_pos=[20, 16, 16],
        timestep_scale_multiplier=1000,
        use_middle_indices_grid=True,
        audio_num_attention_heads=2,
        audio_attention_head_dim=4,
        audio_in_channels=16,
        audio_out_channels=16,
        audio_cross_attention_dim=8,
        audio_positional_embedding_max_pos=[20],
        av_ca_timestep_scale_multiplier=1000.0,
        rope_type=ref.LTXRopeType.SPLIT,
        double_precision_rope=True,
        apply_gated_attention=True,
        cross_attention_adaln=True,
    )
    model.enable_action_conditioning(
        ref.ActionBlockConfig(
            enabled=True,
            block_indices=[0, 1],
            ucpe=True,
            ucpe_attn_dim=16,
            ucpe_num_heads=2,
            ucpe_patches_x=4,
            ucpe_patches_y=2,
            ucpe_image_width=64,
            ucpe_image_height=32,
            ucpe_freq_base=100.0,
        )
    )
    # The reference allocates its modulation tables with torch.empty (they are
    # meant to be overwritten by checkpoint loading); re-initialize everything
    # deterministically so the tiny parity model holds finite values.
    with torch.no_grad():
        for param in model.parameters():
            if param.dtype.is_floating_point:
                param.normal_(0.0, 0.02)
    return model


def _copy_reference_weights(reference, model: EchoWMTransformer3DModel) -> None:
    ref_state = dict(reference.named_parameters())
    with torch.no_grad():
        for name, param in model.named_parameters():
            if ".to_qkv." in name:
                prefix, suffix = name.split(".to_qkv.", 1)
                param.copy_(torch.cat(
                    [ref_state[f"{prefix}.to_q.{suffix}"], ref_state[f"{prefix}.to_k.{suffix}"], ref_state[f"{prefix}.to_v.{suffix}"]],
                    dim=0,
                ))
                continue
            ref_name = name.replace(".ucpe.", ".").replace(".norm_q.", ".q_norm.").replace(".norm_k.", ".k_norm.")
            source = ref_state[ref_name]
            assert source.shape == param.shape, (ref_name, source.shape, param.shape)
            param.copy_(source.detach())


def _se3_cameras(frames: int, batch: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    angles = torch.linspace(0.05, 0.3, frames * batch).view(batch, frames)
    viewmats = torch.zeros(batch, frames, 4, 4)
    viewmats[..., 3, 3] = 1.0
    for i in range(batch):
        for f in range(frames):
            a = angles[i, f]
            viewmats[i, f, :3, :3] = torch.tensor(
                [[torch.cos(a), -torch.sin(a), 0.0], [torch.sin(a), torch.cos(a), 0.0], [0.0, 0.0, 1.0]]
            )
            viewmats[i, f, :3, 3] = torch.tensor([0.1 * f, 0.0, 0.05])
    ks = torch.zeros(batch, frames, 3, 3)
    ks[..., 0, 0] = 40.0
    ks[..., 1, 1] = 40.0
    ks[..., 0, 2] = 32.0
    ks[..., 1, 2] = 16.0
    ks[..., 2, 2] = 1.0
    return viewmats, ks


@pytest.fixture()
def setup():
    ref = _reference_modules()
    reference = _tiny_reference(ref).eval()
    model = _tiny_model().eval()
    _copy_reference_weights(reference, model)

    ppf = 8
    video_frames = 7
    audio_frames = causal_audio_frames(video_frames)
    cache_config = EchoWMCacheConfig()
    text_len = 6
    device = torch.device("cpu")
    dtype = torch.float32

    video_positions = build_video_positions(video_frames, height=64, width=128, fps=24.0, device=device)
    audio_positions = build_audio_positions(audio_frames, device=device)
    video_context = torch.randn(1, text_len, 16)
    audio_context = torch.randn(1, text_len, 8)
    viewmats, ks = _se3_cameras(video_frames)
    action = {"ucpe_viewmats": viewmats, "ucpe_Ks": ks}

    # Reference caches and bounded-window configuration.
    wrapper = SimpleNamespace(
        cache=cache_config,
        patches_per_frame=ppf,
        model=reference,
    )
    from ltx_causal.cache import configure_bounded_caches  # noqa: PLC0415 (reference-only import)

    ref_caches = reference.init_av_kv_caches(
        batch_size=1,
        max_video_tokens=video_frames * ppf,
        max_audio_tokens=audio_frames,
        text_seq_len=text_len,
        device=device,
        dtype=dtype,
        video_local_attn_tokens=cache_config.video_local_attn_size * ppf,
        video_sink_tokens=cache_config.video_sink_size * ppf,
        video_ucpe_local_attn_tokens=cache_config.video_local_attn_size * ppf,
        video_ucpe_sink_tokens=cache_config.video_sink_size * ppf,
        audio_local_attn_tokens=cache_config.audio_local_attn_size,
        audio_sink_tokens=cache_config.audio_sink_size,
    )
    configure_bounded_caches(wrapper, ref_caches, video_positions, audio_positions, action, dtype)

    # Port caches: templates and slice maps mirror configure_bounded_caches.
    caches = model.allocate_caches(
        batch_size=1,
        patches_per_frame=ppf,
        text_seq_len=text_len,
        cache_config=cache_config,
        device=device,
        dtype=dtype,
    )
    video_window = cache_config.video_local_attn_size * ppf
    video_rope = make_split_rope(
        video_positions[:, :, :video_window],
        dim=model.inner_dim,
        num_heads=model.config.num_attention_heads,
        max_pos=list(model.config.positional_embedding_max_pos),
        out_dtype=dtype,
        device=device,
    )
    audio_rope = make_split_rope(
        audio_positions[:, :, : cache_config.audio_local_attn_size],
        dim=model.audio_inner_dim,
        num_heads=model.config.audio_num_attention_heads,
        max_pos=list(model.config.audio_positional_embedding_max_pos),
        out_dtype=dtype,
        device=device,
    )
    video_cross_rope = make_cross_rope_template(
        video_positions[:, :, :video_window],
        dim=model.config.audio_cross_attention_dim,
        num_heads=model.config.audio_num_attention_heads,
        max_pos=20,
        out_dtype=dtype,
        device=device,
    )
    audio_cross_rope = make_cross_rope_template(
        audio_positions[:, :, : cache_config.audio_local_attn_size],
        dim=model.config.audio_cross_attention_dim,
        num_heads=model.config.audio_num_attention_heads,
        max_pos=20,
        out_dtype=dtype,
        device=device,
    )
    a2v_q_slices, v2a_q_slices = compute_cross_slices(video_frames, ppf, cache_config)
    for layer in caches:
        layer.video_rope = video_rope
        layer.audio_rope = audio_rope
        layer.video_cross_rope = video_cross_rope
        layer.audio_cross_rope = audio_cross_rope
        layer.a2v_q_slices = a2v_q_slices
        layer.v2a_q_slices = v2a_q_slices
        layer.ucpe_full_viewmats = viewmats
        layer.ucpe_full_Ks = ks
        layer.ucpe_bounded = True

    return SimpleNamespace(
        ref=ref,
        reference=reference,
        model=model,
        ref_caches=ref_caches,
        caches=caches,
        ppf=ppf,
        video_positions=video_positions,
        audio_positions=audio_positions,
        video_context=video_context,
        audio_context=audio_context,
        action=action,
        video_frames=video_frames,
        audio_frames=audio_frames,
        cache_config=cache_config,
    )


def _reference_forward(setup, video_tokens, audio_tokens, video_sigma, audio_sigma, video_start, audio_start):
    ref = setup.ref
    ppf = setup.ppf
    video = ref.Modality(
        latent=video_tokens,
        sigma=torch.ones(video_tokens.shape[0]),
        timesteps=torch.full(video_tokens.shape[:2], video_sigma),
        positions=setup.video_positions[:, :, video_start : video_start + video_tokens.shape[1]],
        context=setup.video_context,
        context_mask=None,
    )
    audio = None
    if audio_tokens is not None:
        audio = ref.Modality(
            latent=audio_tokens,
            sigma=torch.ones(audio_tokens.shape[0]),
            timesteps=torch.full(audio_tokens.shape[:2], audio_sigma),
            positions=setup.audio_positions[:, :, audio_start : audio_start + audio_tokens.shape[1]],
            context=setup.audio_context,
            context_mask=None,
        )
    sliced_action = {
        key: value[:, video_start // ppf : (video_start + video_tokens.shape[1]) // ppf]
        if key in {"ucpe_viewmats", "ucpe_Ks"}
        else value
        for key, value in setup.action.items()
    }
    with torch.inference_mode():
        return setup.reference(
            video=video,
            audio=audio,
            perturbations=ref.BatchedPerturbationConfig.empty(video_tokens.shape[0]),
            action_cond=sliced_action,
            kv_caches=setup.ref_caches,
            current_video_token_start=video_start,
            current_audio_token_start=audio_start,
        )


def _port_forward(setup, video_tokens, audio_tokens, video_sigma, audio_sigma, video_start, audio_start):
    ppf = setup.ppf
    sliced_action = {
        key: value[:, video_start // ppf : (video_start + video_tokens.shape[1]) // ppf]
        if key in {"ucpe_viewmats", "ucpe_Ks"}
        else value
        for key, value in setup.action.items()
    }
    with torch.inference_mode():
        return setup.model(
            video_tokens=video_tokens,
            audio_tokens=audio_tokens,
            video_sigma=video_sigma,
            audio_sigma=audio_sigma,
            video_context=setup.video_context,
            audio_context=setup.audio_context,
            caches=setup.caches,
            video_token_start=video_start,
            audio_token_start=audio_start,
            ucpe_viewmats=sliced_action["ucpe_viewmats"],
            ucpe_Ks=sliced_action["ucpe_Ks"],
            patches_per_frame=ppf,
        )


def test_rope_templates_match_reference(setup) -> None:
    from ltx_core.model.transformer.rope import precompute_freqs_cis  # noqa: PLC0415

    window = setup.cache_config.video_local_attn_size * setup.ppf
    reference = precompute_freqs_cis(
        setup.video_positions[:, :, :window],
        dim=setup.model.inner_dim,
        out_dtype=torch.float32,
        theta=10000.0,
        max_pos=[20, 16, 16],
        use_middle_indices_grid=True,
        num_attention_heads=2,
        rope_type=setup.ref.LTXRopeType.SPLIT,
        freq_grid_generator=__import__(
            "ltx_core.model.transformer.rope", fromlist=["generate_freq_grid_np"]
        ).generate_freq_grid_np,
    )
    mine = make_split_rope(
        setup.video_positions[:, :, :window],
        dim=setup.model.inner_dim,
        num_heads=2,
        max_pos=[20, 16, 16],
        out_dtype=torch.float32,
    )
    torch.testing.assert_close(mine[0], reference[0], rtol=0, atol=0)
    torch.testing.assert_close(mine[1], reference[1], rtol=0, atol=0)


def test_tiny_forward_parity_image_sink_and_two_blocks(setup) -> None:
    ppf = setup.ppf
    clean_image = torch.randn(1, ppf, 16)
    block_video = torch.randn(1, 3 * ppf, 16)
    audio_prefix = torch.randn(1, 2, 16)
    block_audio = torch.randn(1, 25, 16)

    # F1: image-sink commit (video only, sigma 0).
    ref_vx, ref_ax = _reference_forward(setup, clean_image, None, 0.0, 0.0, 0, 0)
    port_vx, port_ax = _port_forward(setup, clean_image, None, 0.0, 0.0, 0, 0)
    assert ref_ax is None and port_ax is None
    torch.testing.assert_close(port_vx, ref_vx, rtol=1e-5, atol=1e-5)

    # F2: audio-prefix denoise (video clean at sigma 0, audio at sigma).
    ref_vx, ref_ax = _reference_forward(setup, clean_image, audio_prefix, 0.0, 0.9, 0, 0)
    port_vx, port_ax = _port_forward(setup, clean_image, audio_prefix, 0.0, 0.9, 0, 0)
    torch.testing.assert_close(port_vx, ref_vx, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(port_ax, ref_ax, rtol=1e-5, atol=1e-5)

    # F3: first AV block.
    ref_vx, ref_ax = _reference_forward(setup, block_video, block_audio, 0.9, 0.9, ppf, 2)
    port_vx, port_ax = _port_forward(setup, block_video, block_audio, 0.9, 0.9, ppf, 2)
    torch.testing.assert_close(port_vx, ref_vx, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(port_ax, ref_ax, rtol=1e-5, atol=1e-5)

    # F4: second AV block (window starts evicting the FIFO tail).
    block_video_2 = torch.randn(1, 3 * ppf, 16)
    block_audio_2 = torch.randn(1, 25, 16)
    ref_vx, ref_ax = _reference_forward(setup, block_video_2, block_audio_2, 0.5, 0.5, 4 * ppf, 27)
    port_vx, port_ax = _port_forward(setup, block_video_2, block_audio_2, 0.5, 0.5, 4 * ppf, 27)
    torch.testing.assert_close(port_vx, ref_vx, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(port_ax, ref_ax, rtol=1e-5, atol=1e-5)

    # The transactional cache semantics must also match: rerun F3's range to
    # replace it in place (denoise-step retry), then compare cache contents.
    ref_vx, _ = _reference_forward(setup, block_video, block_audio, 0.2, 0.2, ppf, 2)
    port_vx, _ = _port_forward(setup, block_video, block_audio, 0.2, 0.2, ppf, 2)
    torch.testing.assert_close(port_vx, ref_vx, rtol=1e-5, atol=1e-5)
    ref_cache = setup.ref_caches[0]["video_self"]
    port_cache = setup.caches[0].video_self
    torch.testing.assert_close(port_cache.positions[: port_cache.length], ref_cache["positions"][: ref_cache["length"]])
    assert port_cache.length == ref_cache["length"]


def test_positions_match_reference_patchifier(setup) -> None:
    from ltx_core.components.patchifiers import AudioPatchifier, VideoLatentPatchifier  # noqa: PLC0415
    from ltx_core.types import AudioLatentShape, VideoLatentShape  # noqa: PLC0415
    from ltx_core.tools import get_pixel_coords  # noqa: PLC0415

    patchifier = VideoLatentPatchifier(1)
    # (batch, channels, frames, height, width): 7 latent frames of a 2x4 grid.
    shape = VideoLatentShape(1, 128, 7, 2, 4)
    latent_coords = patchifier.get_patch_grid_bounds(shape, device=torch.device("cpu"))
    reference = get_pixel_coords(latent_coords, (8, 32, 32), causal_fix=True).float()
    reference[:, 0, ...] = reference[:, 0, ...] / 24.0
    mine = build_video_positions(7, height=64, width=128, fps=24.0)
    torch.testing.assert_close(mine, reference, rtol=0, atol=0)

    audio_patchifier = AudioPatchifier(1)
    audio_shape = AudioLatentShape(1, 8, 52, 16)  # (batch, channels, frames, mel_bins)
    audio_reference = audio_patchifier.get_patch_grid_bounds(audio_shape, device=torch.device("cpu"))
    audio_mine = build_audio_positions(52)
    torch.testing.assert_close(audio_mine, audio_reference, rtol=0, atol=1e-7)
