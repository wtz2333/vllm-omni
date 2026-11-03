# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch

from vllm_omni.experimental.ar_diffusion.capability import ARDiffusionKVBranchSpec
from vllm_omni.experimental.ar_diffusion.kv_cache import ARDiffusionKVCache, ARDiffusionKVConfig
from vllm_omni.experimental.ar_diffusion.kv_cache.state import ARDiffusionKVState

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


@pytest.fixture
def kv_cache():
    return ARDiffusionKVCache(
        ARDiffusionKVConfig(enable=True, chunk_size=6, window_chunks=5, sink_chunks=1, gpu_memory_fraction=1.0),
        num_layers=2,
        num_kv_heads=2,
        head_size=4,
        dtype=torch.float32,
        block_size=16,
        max_model_len=128,
        available_bytes=1 << 20,
        kv_branches=(ARDiffusionKVBranchSpec("main", 0),),
        session_capacity=2,
        frames_per_block=1,
        device=torch.device("cpu"),
    )


def _state(kv, name):
    return ARDiffusionKVState(kv, name, {"main": kv.begin_request(name)}, num_layers=2)


def test_packed_reservation_aliases_pools_and_isolates_sessions(kv_cache):
    first, second = _state(kv_cache, "first"), _state(kv_cache, "second")
    a, b = first.reserve_packed_history("main", 36), second.reserve_packed_history("main", 36)
    assert not set(a[0][2].tolist()) & set(b[0][2].tolist())
    for index, (key, value, slots) in enumerate(a):
        assert key.data_ptr() == kv_cache.key_cache(index).data_ptr()
        key.index_fill_(0, slots, 7)
        value.index_fill_(0, slots, 11)
        torch.testing.assert_close(kv_cache.key_cache(index).flatten(0, 1)[slots], torch.full((36, 2, 4), 7.0))
    first.close()
    second.close()


def test_packed_commit_retains_reservation_beyond_fifo_window_and_releases_pages(kv_cache):
    free = kv_cache.manager.block_pool.get_num_free_blocks()
    state = _state(kv_cache, "rollout")
    buffers = state.reserve_packed_history("main", 36)
    for _ in range(40):
        state.commit_packed_context("main", 6)
        assert torch.equal(state.reserve_packed_history("main", 36)[0][2], buffers[0][2])
    assert state._committed["main"] == 240
    state.close()
    assert kv_cache.manager.block_pool.get_num_free_blocks() == free
    with pytest.raises(RuntimeError, match="closed"):
        state.reserve_packed_history("main", 36)


def test_reset_releases_packed_reservation_and_restarts_timeline(kv_cache):
    state = _state(kv_cache, "reset")
    state.reserve_packed_history("main", 36)
    state.commit_packed_context("main", 6)
    occupied = kv_cache.manager.block_pool.get_num_free_blocks()
    state.reset()
    assert state._committed["main"] == 0 and state._packed_slots == {}
    state.reserve_packed_history("main", 36)
    assert kv_cache.manager.block_pool.get_num_free_blocks() == occupied
    state.close()


def test_packed_and_window_contexts_cannot_mix(kv_cache):
    state = _state(kv_cache, "packed")
    state.reserve_packed_history("main", 36)
    with pytest.raises(RuntimeError, match="frame-window"):
        state.get_kv_caches("main", seq_len=6)
    with pytest.raises(ValueError, match="capacity"):
        state.reserve_packed_history("main", 30)
    state.close()
    state = _state(kv_cache, "window")
    state.get_kv_caches("main", seq_len=6)
    with pytest.raises(RuntimeError, match="mix"):
        state.reserve_packed_history("main", 36)
    state.close()


@pytest.mark.parametrize("count", [0, -1, 37])
def test_packed_reservation_rejects_out_of_budget_counts(kv_cache, count):
    state = _state(kv_cache, "invalid")
    with pytest.raises(ValueError, match="capacity"):
        state.reserve_packed_history("main", count)
    state.close()


def test_runner_failure_releases_packed_pages_before_reusing_session_id():
    from tests.diffusion.ar_diffusion.test_capability_runner import CapablePipeline, lingbot_like_spec, make_runner

    pipeline = CapablePipeline(lingbot_like_spec())
    runner = make_runner(pipeline)
    free = runner.kv_cache.manager.block_pool.get_num_free_blocks()
    with pytest.raises(RuntimeError, match="injected"):
        with runner._bound_ar_session("failed", description="packed CPU regression", synchronize=False):
            pipeline.bound_state.reserve_packed_history("main", 6 * 16)
            raise RuntimeError("injected clean forward failure")
    assert pipeline.closes == ["failed"]
    assert "failed" not in runner._sessions
    assert runner.kv_cache.manager.block_pool.get_num_free_blocks() == free
    with runner._bound_ar_session("failed", description="fresh packed session", synchronize=False):
        fresh = pipeline.bound_state
        assert fresh._committed["main"] == 0
        fresh.reserve_packed_history("main", 6 * 16)
    runner.close_session("failed")
    assert runner.kv_cache.manager.block_pool.get_num_free_blocks() == free


def test_packed_history_rejects_segmented_capability(kv_cache, monkeypatch):
    state = _state(kv_cache, "segmented")
    monkeypatch.setattr(kv_cache, "segmented_spec", object())
    with pytest.raises(RuntimeError, match="Segmented KV"):
        state.reserve_packed_history("main", 36)
    state.close()
