# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import sys
from dataclasses import replace
from types import ModuleType

import pytest
import torch

from tests.diffusion.ar_diffusion.test_capability_runner import CapablePipeline, lingbot_like_spec, make_runner
from tests.diffusion.models.lingbot_world.test_lingbot_world_attention import _load_module
from vllm_omni.experimental.ar_diffusion.capability import ARDiffusionKVBranchSpec
from vllm_omni.experimental.ar_diffusion.kv_cache import manager as kv_manager
from vllm_omni.experimental.ar_diffusion.kv_cache.config import ARDiffusionKVConfig, QVGConfig
from vllm_omni.experimental.ar_diffusion.kv_cache.manager import ARDiffusionKVCache
from vllm_omni.experimental.ar_diffusion.kv_cache.qvg import QVGCodec, QVGHistory, QVGLayerContext, QVGTensor
from vllm_omni.experimental.ar_diffusion.kv_cache.state import ARDiffusionKVState

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cache, pytest.mark.cpu]


@pytest.fixture
def qvg_primitives(monkeypatch):
    # CPU 用例只替代外部 GPU primitives；配置、codec 适配和历史管理执行正式实现。
    names = (
        "quant_videogen",
        "quant_videogen.kmeans",
        "quant_videogen.real",
        "quant_videogen.kmeans.kmeans_euclid",
        "quant_videogen.real.accumulate",
        "quant_videogen.real.quant_pack",
        "quant_videogen.real.smooth",
    )
    modules = {name: ModuleType(name) for name in names}
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    def kmeans(x, n_clusters, max_iters, init_centroids):
        assert max_iters > 0 and init_centroids.shape == (x.shape[0], n_clusters, x.shape[-1])
        ids = torch.zeros(x.shape[:2], dtype=torch.long)
        centroids = x.mean(dim=1, keepdim=True).expand(-1, n_clusters, -1).clone()
        return ids, centroids, None, 1

    def subtract(x, ids, centroids):
        return x - centroids.gather(2, ids.long().unsqueeze(-1).expand_as(x))

    def pack(x, block_size, num_bits, scale_precision, pack_output_int8, asymmetric):
        assert pack_output_int8 and not asymmetric and scale_precision is torch.float8_e4m3fn
        # 测试输入每个 head 内为常量，残差应当精确为零。
        assert torch.count_nonzero(x) == 0
        return (
            torch.zeros((*x.shape[:-1], x.shape[-1] * num_bits // 8), dtype=torch.uint8),
            torch.zeros((*x.shape[:-1], x.shape[-1] // block_size), dtype=scale_precision),
            None,
        )

    def unpack(centroids_list, cluster_ids_list, residual_quant, scales, **kwargs):
        assert kwargs["PACK_INPUT_INT8"] and kwargs["CLUSTER_ID_INT8"]
        shape = (*cluster_ids_list[0].shape, centroids_list[0].shape[-1])
        out = torch.zeros(shape, dtype=kwargs["output_dtype"])
        for centroids, ids in zip(centroids_list, cluster_ids_list):
            out += centroids.gather(2, ids.long().unsqueeze(-1).expand(shape))
        return out

    setattr(modules["quant_videogen.kmeans.kmeans_euclid"], "batch_kmeans_Euclid", kmeans)

    def iteration(x, squared_norm, centroids):
        ids, updated, _, _ = kmeans(x, centroids.shape[1], 1, centroids)
        sizes = torch.zeros(centroids.shape[:2], dtype=torch.int32)
        sizes[:, 0] = x.shape[1]
        return updated, ids, sizes

    setattr(modules["quant_videogen.kmeans.kmeans_euclid"], "_euclid_iter", iteration)
    setattr(modules["quant_videogen.real.accumulate"], "nstage_accum", unpack)
    setattr(modules["quant_videogen.real.quant_pack"], "quant_pack", pack)
    setattr(modules["quant_videogen.real.smooth"], "minus_centroid", subtract)


@pytest.fixture
def codec(qvg_primitives):
    return QVGCodec(QVGConfig(num_centroids=2, group_size=16))


def make_history(codec, *, num_layers=2, agree=bool):
    workspace = (torch.empty(1, 112, 2, 64), torch.empty(1, 112, 2, 64))
    return QVGHistory(
        codec,
        num_layers=num_layers,
        tokens_per_frame=16,
        window_tokens=96,
        sink_tokens=16,
        workspace=workspace,
        agree=agree,
    )


def read_chunk(history, start, value, *, commit=True):
    outputs = []
    for index, context in enumerate(history.prepare(32, commit)):
        key = torch.full((1, 32, 2, 64), float(value + 10 * index))
        visible = context.read(start, key, key + 100)
        outputs.append(tuple(tensor.clone() for tensor in visible))
    return outputs


@pytest.mark.parametrize("bits", [2, 4])
@pytest.mark.parametrize("stages", [1, 4])
def test_codec_uses_packed_storage_and_preserves_sampling_rng(qvg_primitives, bits, stages):
    codec = QVGCodec(QVGConfig(bits=bits, num_stages=stages, num_centroids=2, group_size=16))
    key = torch.tensor([3.0, 7.0]).view(1, 2, 1).expand(48, 2, 64).clone()
    before = torch.get_rng_state().clone()
    packed = codec.encode(key)
    assert torch.equal(torch.get_rng_state(), before)
    assert packed.residual.dtype is torch.uint8
    assert packed.residual.numel() == key.numel() * bits // 8
    assert len(packed.centroids) == stages
    assert all(ids.dtype is torch.uint8 for ids in packed.assignments)
    assert packed.nbytes < key.numel() * key.element_size()
    torch.testing.assert_close(codec.decode(packed, key.dtype), key, rtol=0, atol=0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"method": "fp8"},
        {"bits": 3},
        {"bits": True},
        {"num_stages": 0},
        {"num_stages": 5},
        {"group_size": 8},
        {"num_centroids": 257},
        {"kmeans_max_iters": 0},
        {"kmeans_min_iters": 0},
        {"kmeans_check_interval": 0},
        {"kmeans_tolerance": -0.1},
        {"kmeans_tolerance": float("nan")},
        {"kmeans_tolerance": float("inf")},
        {"kmeans_tolerance": True},
        {"keep_recent_chunks": -1},
        {"seed": -1},
    ],
)
def test_invalid_compression_fails_before_allocation(kwargs):
    with pytest.raises(ValueError, match="QVG|compression"):
        ARDiffusionKVConfig(compression=kwargs)


def test_history_compression_has_an_independent_config():
    assert ARDiffusionKVConfig().compression is None


def test_kmeans_stops_on_centroid_movement_and_checks_only_at_intervals(qvg_primitives, mocker):
    codec = QVGCodec(QVGConfig(num_centroids=2, kmeans_min_iters=4, kmeans_check_interval=4))
    check = mocker.spy(codec, "_centroids_converged")
    iterate = mocker.spy(codec, "_iterate")
    tensor = torch.ones(2, 48, 64)
    ids, centroids, _, iterations = codec._kmeans(tensor, 2, 100, torch.ones(2, 2, 64))
    assert iterations == 4 and iterate.call_count == 4 and check.call_count == 1
    assert torch.equal(centroids, torch.ones_like(centroids))
    assert ids.shape == (2, 48)


def test_kmeans_honors_hard_budget_when_not_converged(qvg_primitives, monkeypatch, mocker):
    codec = QVGCodec(QVGConfig(num_centroids=2, kmeans_tolerance=1e-8, kmeans_check_interval=4))

    def moving(tensor, squared_norm, centroids):
        return centroids + 1, torch.zeros(tensor.shape[:2], dtype=torch.long), torch.ones(centroids.shape[:2])

    monkeypatch.setattr(codec, "_iterate", moving)
    check = mocker.spy(codec, "_centroids_converged")
    _, centers, _, iterations = codec._kmeans(torch.ones(2, 48, 64), 2, 11, torch.ones(2, 2, 64))
    assert iterations == 11 and check.call_count == 2
    assert torch.equal(centers, torch.full_like(centers, 12))


def test_kmeans_fixed_budget_does_not_read_convergence_scalars(qvg_primitives, mocker):
    codec = QVGCodec(QVGConfig(num_centroids=2, kmeans_tolerance=0))
    check = mocker.spy(codec, "_centroids_converged")
    _, _, _, iterations = codec._kmeans(torch.ones(2, 48, 64), 2, 7, torch.ones(2, 2, 64))
    assert iterations == 7 and check.call_count == 0
    config = ARDiffusionKVConfig(compression={"bits": 2, "group_size": 32})
    assert config.compression == QVGConfig(bits=2, group_size=32)
    for field in ("reuse_history_staging", "reset_at_boundary"):
        with pytest.raises(ValueError, match="QVG"):
            ARDiffusionKVConfig(compression={}, **{field: True})


def test_probes_leave_committed_history_unchanged(codec):
    history = make_history(codec)
    read_chunk(history, 0, 1)
    assert history.nbytes == 0
    assert history.commit() == 32
    before = history.nbytes
    for probe in (8, 9, 10):
        outputs = read_chunk(history, 32, probe, commit=False)
        assert history.nbytes == before
        assert torch.all(outputs[0][0][:, :32] == 1)
        assert torch.all(outputs[0][0][:, 32:] == probe)
        assert history.commit() == 0
    read_chunk(history, 32, 2)
    assert history.commit() == 32


def test_sliding_history_keeps_exact_sink_tail_and_absolute_positions(codec):
    history = make_history(codec)
    for index in range(10):
        read_chunk(history, index * 32, index)
        history.commit()
    outputs = read_chunk(history, 320, 99, commit=False)
    expected = torch.tensor([0.0] * 16 + [8.0] * 32 + [9.0] * 32 + [99.0] * 32)
    torch.testing.assert_close(outputs[0][0][0, :, 0, 0], expected, rtol=0, atol=0)
    torch.testing.assert_close(outputs[1][1][0, :, 0, 0], expected + 110, rtol=0, atol=0)
    with pytest.raises(ValueError, match="absolute token offset"):
        history.prepare(32, False)[0].read(0, torch.ones(1, 32, 2, 64), torch.ones(1, 32, 2, 64))


def test_completed_chunks_are_compressed_once_and_recent_chunk_stays_float(codec, mocker):
    history = make_history(codec)
    encode = mocker.spy(codec, "encode")
    for index in range(6):
        read_chunk(history, index * 32, index)
        history.commit()
    assert encode.call_count == 4 * 2 * 2
    assert isinstance(history._chunks[0].layers[0][0], torch.Tensor)
    assert isinstance(history._chunks[-1].layers[0][0], torch.Tensor)
    assert any(isinstance(chunk.layers[0][0], QVGTensor) for chunk in history._chunks)
    read_chunk(history, 192, 20, commit=False)
    assert encode.call_count == 16


@pytest.mark.parametrize("sink_frames", [0, 1, 9])
@pytest.mark.parametrize("keep_recent", [0, 1, 2])
def test_three_frame_chunks_preserve_default_and_partial_sink_geometry(codec, sink_frames, keep_recent):
    tokens = (sink_frames + 9) * 16
    history = QVGHistory(
        QVGCodec(replace(codec.config, keep_recent_chunks=keep_recent)),
        num_layers=1,
        tokens_per_frame=16,
        window_tokens=9 * 16,
        sink_tokens=sink_frames * 16,
        workspace=(torch.empty(1, tokens, 2, 64), torch.empty(1, tokens, 2, 64)),
    )
    for index in range(9):
        current = torch.full((1, 48, 2, 64), float(index))
        history.prepare(48, True)[0].read(index * 48, current, current + 100)
        history.commit()
    current = torch.full((1, 48, 2, 64), 99.0)
    key, value = history.prepare(48, False)[0].read(432, current, current + 100)
    expected = torch.tensor(
        [float(token // 48) for token in range(sink_frames * 16)] + [7.0] * 48 + [8.0] * 48 + [99.0] * 48
    )
    torch.testing.assert_close(key[0, :, 0, 0], expected, rtol=0, atol=0)
    torch.testing.assert_close(value[0, :, 0, 0], expected + 100, rtol=0, atol=0)


@pytest.mark.parametrize("remote_failure", [False, True])
def test_failed_compression_keeps_previous_history(codec, monkeypatch, remote_failure):
    history = make_history(codec)
    for index in range(2):
        read_chunk(history, index * 32, index)
        history.commit()
    previous, previous_bytes = history._chunks, history.nbytes
    read_chunk(history, 64, 2)
    if remote_failure:
        history.agree = lambda successful: False
    else:

        def fail(tensor):
            raise ValueError("injected encoder failure")

        monkeypatch.setattr(codec, "encode", fail)
    with pytest.raises(RuntimeError, match="previous history was retained"):
        history.commit()
    assert history._chunks is previous
    assert history.nbytes == previous_bytes
    read_chunk(history, 64, 3, commit=False)


def test_incomplete_clean_forward_cannot_publish_or_be_replaced(codec):
    history = make_history(codec)
    context = history.prepare(32, True)[0]
    context.read(0, torch.ones(1, 32, 2, 64), torch.ones(1, 32, 2, 64))
    with pytest.raises(RuntimeError, match="missing layers"):
        history.commit()
    with pytest.raises(RuntimeError, match="must be committed"):
        history.prepare(32, False)
    history.close()
    assert history.nbytes == 0
    with pytest.raises(RuntimeError, match="closed"):
        history.prepare(32, False)


def make_cache(codec, monkeypatch):
    monkeypatch.setattr("vllm_omni.experimental.ar_diffusion.kv_cache.manager.QVGCodec", lambda config: codec)
    return ARDiffusionKVCache(
        ARDiffusionKVConfig(
            enable=True,
            chunk_size=16,
            window_chunks=6,
            sink_chunks=1,
            compression=codec.config,
        ),
        num_layers=2,
        num_kv_heads=2,
        head_size=64,
        dtype=torch.float32,
        block_size=16,
        max_model_len=1024,
        available_bytes=1 << 26,
        kv_branches=(ARDiffusionKVBranchSpec("positive", 0), ARDiffusionKVBranchSpec("negative", 1)),
        session_capacity=2,
        frames_per_block=2,
        cross_attention_lengths={"text": 8},
        device=torch.device("cpu"),
    )


def test_manager_uses_one_workspace_without_floating_point_history_pools(codec, monkeypatch, mocker):
    allocate = mocker.spy(kv_manager, "allocate_kv_pool_with_views")
    cache = make_cache(codec, monkeypatch)
    allocate.assert_not_called()
    assert cache._k_pools == cache._v_pools == cache._kv_pools == []
    assert cache.history_staging == []
    assert cache.qvg_workspace is not None
    assert cache.qvg_workspace[0].shape == (1, 112, 2, 64)
    assert cache.history_staging_reserved_bytes >= sum(t.numel() * t.element_size() for t in cache.qvg_workspace)


def test_sessions_and_cfg_branches_reset_and_close_independently(codec, monkeypatch):
    cache = make_cache(codec, monkeypatch)

    def session(name):
        adapters = {branch.name: cache.begin_request(f"{name}::{branch.name}") for branch in cache.kv_branches}
        return ARDiffusionKVState(cache, name, adapters, num_layers=2)

    first, second = session("first"), session("second")
    for state, branch, value in ((first, "positive", 3), (first, "negative", 7), (second, "positive", 11)):
        for layer in state.get_kv_caches(branch, 32, True):
            layer.read(0, torch.full((1, 32, 2, 64), float(value)), torch.full((1, 32, 2, 64), float(value)))
        state.commit_paged_context(branch)
    first.populate_cross_attention("positive", "text", [(torch.ones(1, 8, 2, 64), torch.ones(1, 8, 2, 64))] * 2)
    first.reset(keep_cross_attention={"text"})
    assert first._compressed == {}
    assert first.is_cross_attention_populated("positive", "text")
    context = second.get_kv_caches("positive", 32, False)[0]
    key, _ = context.read(32, torch.zeros(1, 32, 2, 64), torch.zeros(1, 32, 2, 64))
    assert torch.all(key[:, :32] == 11)
    second.close()
    assert second._compressed == {}
    first.close()
    assert not cache._adapters and not cache._cross_sessions


def test_cancellation_releases_committed_and_pending_qvg_history(codec, monkeypatch):
    cache = make_cache(codec, monkeypatch)
    state = ARDiffusionKVState(
        cache,
        "cancelled",
        {branch.name: cache.begin_request(f"cancelled::{branch.name}") for branch in cache.kv_branches},
        num_layers=2,
    )
    for context in state.get_kv_caches("positive", 32, True):
        context.read(0, torch.ones(1, 32, 2, 64), torch.ones(1, 32, 2, 64))
    state.commit_paged_context("positive")
    state.get_kv_caches("positive", 32, True)[0].read(32, torch.ones(1, 32, 2, 64), torch.ones(1, 32, 2, 64))
    history = state._compressed["positive"]
    assert history.nbytes and history._active is not None
    pending = history._active
    pipeline = CapablePipeline(lingbot_like_spec())
    runner = make_runner(pipeline, preallocate=False)
    runner.kv_cache = cache
    runner._ar_diffusion_capability = pipeline
    runner._sessions["cancelled"] = state
    runner.release_step_requests(["cancelled"])
    assert history.nbytes == 0 and history._active is None
    assert not pending.layers
    assert not runner._sessions and not cache._adapters
    assert pipeline.closes == ["cancelled"]


@pytest.mark.parametrize("failure", ["capability", "eager", "device", "attention"])
def test_runner_rejects_unsupported_compression_before_allocating(failure):
    spec = lingbot_like_spec()
    if failure != "capability":
        spec = replace(spec, supported_history_compression=("qvg",))
    runner = make_runner(CapablePipeline(spec), preallocate=False)
    runner.ar_diffusion_kv_config = ARDiffusionKVConfig(enable=True, compression={})
    if failure == "eager":
        runner.od_config.enforce_eager = False
    if failure == "attention":
        runner.device = torch.device("cuda")
        runner.od_config.diffusion_kv_cache_dtype = "fp8"
    with pytest.raises(ValueError, match="QVG|pipeline"):
        runner._preallocate_kv_cache(available_bytes=1 << 26)
    assert runner.kv_cache is None


@torch.inference_mode()
def test_lingbot_attention_reads_qvg_history_and_preserves_linear_quantization(codec, monkeypatch):
    module = _load_module()
    monkeypatch.setattr(module, "QVGLayerContext", QVGLayerContext)
    from vllm_omni.experimental.ar_diffusion.kv_cache.paged_attention import contiguous_paged_attention

    monkeypatch.setattr(module, "contiguous_paged_attention", contiguous_paged_attention)
    quant_config = object()
    attention = module.LingBotSelfAttention(dim=128, num_heads=2, quant_config=quant_config)
    assert attention.qkv.quant_config is quant_config
    assert attention.o.quant_config is quant_config
    attention.qkv.weight.copy_(torch.eye(128).repeat(3, 1))
    attention.qkv.bias.zero_()
    attention.o.weight.copy_(torch.eye(128))
    attention.o.bias.zero_()
    attention.norm_q = torch.nn.Identity()
    attention.norm_k = torch.nn.Identity()
    native = module.allocate_lingbot_cache(
        batch_size=1,
        num_layers=1,
        max_tokens=112,
        num_local_heads=2,
        head_dim=64,
        device=torch.device("cpu"),
        dtype=torch.float32,
    ).self_attention[0]
    history = make_history(codec, num_layers=1)
    for index in range(6):
        hidden = torch.full((1, 32, 128), float(index + 1))
        expected = attention(hidden, cache=native, current_start=index * 32, sink_tokens=16, update_cache=True)
        actual = attention(
            hidden,
            cache=history.prepare(32, True)[0],
            current_start=index * 32,
            sink_tokens=16,
            update_cache=True,
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        history.commit()


@torch.inference_mode()
def test_pre_rope_history_restores_absolute_positions_across_eviction(codec, monkeypatch):
    from vllm_omni.diffusion.layers.rope import RotaryEmbeddingWan
    from vllm_omni.experimental.ar_diffusion.kv_cache.paged_attention import contiguous_paged_attention

    module = _load_module()
    monkeypatch.setattr(module, "QVGLayerContext", QVGLayerContext)
    monkeypatch.setattr(module, "contiguous_paged_attention", contiguous_paged_attention)
    rotary = RotaryEmbeddingWan(half_head_dim=True)

    class CPURotary(torch.nn.Module):
        def forward(self, value, cosine, sine):
            return rotary.forward_native(value, cosine, sine)

    attention = module.LingBotSelfAttention(dim=128, num_heads=2, quant_config=object())
    attention.rotary_embedding = CPURotary()
    attention.qkv.weight.copy_(torch.eye(128).repeat(3, 1))
    attention.qkv.bias.zero_()
    attention.o.weight.copy_(torch.eye(128))
    attention.o.bias.zero_()
    attention.norm_q = torch.nn.Identity()
    attention.norm_k = torch.nn.Identity()
    native = module.allocate_lingbot_cache(
        batch_size=1,
        num_layers=1,
        max_tokens=160,
        num_local_heads=2,
        head_dim=64,
        device=torch.device("cpu"),
        dtype=torch.float32,
    ).self_attention[0]
    history = QVGHistory(
        codec,
        num_layers=1,
        tokens_per_frame=16,
        window_tokens=144,
        sink_tokens=16,
        workspace=(torch.empty(1, 160, 2, 64), torch.empty(1, 160, 2, 64)),
    )

    def positions(indices):
        phase = indices.float().unsqueeze(-1) * torch.linspace(0.001, 0.03, 32)
        return phase.cos(), phase.sin()

    for index in range(9):
        start = index * 48
        hidden = torch.full((1, 48, 128), float(index + 1))
        current_rope = positions(torch.arange(start, start + 48))
        expected = attention(hidden, cache=native, current_start=start, sink_tokens=16, rotary_emb=current_rope)
        context = history.prepare(48, True)[0]
        ranges = context.token_ranges()
        indices = torch.cat([torch.arange(left, right) for left, right in ranges])
        actual = attention(
            hidden,
            cache=context,
            current_start=start,
            sink_tokens=16,
            rotary_emb=current_rope,
            key_rotary_emb=positions(indices),
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert torch.equal(context.forward_ctx.layers[0][0], hidden[0].unflatten(-1, (2, 64)))
        history.commit()
        with pytest.raises(RuntimeError, match="retired"):
            context.token_ranges()
    history.close()
