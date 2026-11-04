# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch

from vllm_omni.diffusion.cache.framefold import FrameFoldConfig, FrameFoldLayerCache, fit_value_merge

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cache, pytest.mark.cpu]


def _identity(value):
    return value


def _rotate(grid, start):
    frames, spatial, _, dim = grid.shape
    positions = torch.arange(start, start + frames) if isinstance(start, int) else torch.tensor(start)
    phase = positions.to(grid.dtype)[:, None] * 0.7
    phase = phase + torch.arange(spatial, dtype=grid.dtype)[None, :] * 0.1
    cosine, sine = phase.cos()[:, :, None, None], phase.sin()[:, :, None, None]
    pairs = grid.unflatten(-1, (dim // 2, 2))
    real, imag = pairs.unbind(-1)
    return torch.stack((real * cosine - imag * sine, real * sine + imag * cosine), -1).flatten(-2)


def _cache(*, dtype=torch.float32):
    config = FrameFoldConfig(budget_frames=6, sink_frames=1, recent_frames=1, pooling_factor=2)
    shape = (36, 2, 4)
    return FrameFoldLayerCache(
        config,
        torch.full(shape, float("nan"), dtype=dtype),
        torch.full(shape, float("nan"), dtype=dtype),
        height=2,
        width=3,
        rope=lambda **kwargs: None,
    )


def _prepare(cache, query, key, value):
    return cache.prepare(query, key, value, current_start=cache.absolute_end, rotate=_rotate, reduce=_identity)


def test_fit_matches_independent_conditional_response_least_squares():
    generator = torch.Generator().manual_seed(31)
    q = torch.randn(3, 2, 4, generator=generator)
    k = torch.randn(2, 6, 2, 4, generator=generator)
    v = torch.randn(2, 6, 2, 4, generator=generator)
    merged = torch.randn(2, 3, 2, 4, generator=generator)
    actual_alpha, actual_score = fit_value_merge(q, k, v, merged, v[:, :3], v[:, 3:], _identity)
    for candidate in range(2):
        qh = q.double().permute(1, 0, 2)
        original = (qh @ k[candidate].double().permute(1, 2, 0) / 2).softmax(-1)
        target = original @ v[candidate].double().permute(1, 0, 2)
        weights = (qh @ merged[candidate].double().permute(1, 2, 0) / 2).softmax(-1)
        left = weights @ v[candidate, :3].double().permute(1, 0, 2)
        right = weights @ v[candidate, 3:].double().permute(1, 0, 2)
        delta = left - right
        alpha = (((target - right) * delta).sum() / delta.square().sum()).clamp(0, 1)
        score = (target - (right + alpha * delta)).norm(dim=-1).sum()
        torch.testing.assert_close(actual_alpha[candidate].double(), alpha, atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(actual_score[candidate].double(), score, atol=2e-6, rtol=2e-6)


def test_degenerate_fit_has_zero_coefficient_and_finite_response():
    q = torch.ones(2, 2, 4)
    k = torch.zeros(1, 4, 2, 4)
    v = torch.ones_like(k)
    alpha, distortion = fit_value_merge(q, k, v, k[:, :2], v[:, :2], v[:, 2:], _identity)
    assert torch.equal(alpha, torch.zeros(1))
    assert torch.equal(distortion, torch.zeros(1))


def test_head_shards_reduce_fitting_statistics_before_scoring():
    generator = torch.Generator().manual_seed(16)
    query = torch.randn(3, 2, 4, generator=generator)
    key = torch.randn(2, 6, 2, 4, generator=generator)
    value = torch.randn(2, 6, 2, 4, generator=generator)
    merged = torch.randn(2, 3, 2, 4, generator=generator)
    qh = query.permute(1, 0, 2)
    original = ((qh @ key.permute(0, 2, 3, 1)) / 2).softmax(-1) @ value.permute(0, 2, 1, 3)
    weights = ((qh @ merged.permute(0, 2, 3, 1)) / 2).softmax(-1)
    base = weights @ value[:, 3:].permute(0, 2, 1, 3)
    delta = weights @ (value[:, :3] - value[:, 3:]).permute(0, 2, 1, 3)
    stats = torch.stack((((original - base) * delta).sum((2, 3)), delta.square().sum((2, 3))), -1)
    total = stats.sum(1)
    alpha = (total[:, 0] / total[:, 1]).clamp(0, 1)
    errors = (original - base - alpha[:, None, None, None] * delta).norm(dim=-1).sum(-1)
    for head in range(2):
        calls: list[bool] = []

        def reduce(partial):
            expected = stats[:, head] if not calls else errors[:, head]
            torch.testing.assert_close(partial, expected, atol=2e-6, rtol=2e-6)
            other = stats[:, 1 - head] if not calls else errors[:, 1 - head]
            calls.append(True)
            return partial + other

        actual_alpha, score = fit_value_merge(
            query[:, head : head + 1],
            key[:, :, head : head + 1],
            value[:, :, head : head + 1],
            merged[:, :, head : head + 1],
            value[:, :3, head : head + 1],
            value[:, 3:, head : head + 1],
            reduce,
        )
        assert len(calls) == 2
        torch.testing.assert_close(actual_alpha, alpha, atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(score, errors.sum(1), atol=2e-6, rtol=2e-6)


def test_non_finite_fit_fails():
    q = torch.full((2, 1, 2), float("nan"))
    k = torch.ones(1, 4, 1, 2)
    with pytest.raises(RuntimeError, match="non-finite"):
        fit_value_merge(q, k, k, k[:, :2], k[:, :2], k[:, 2:], _identity)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_repeated_merges_preserve_sink_recent_and_span_weighted_pre_rope_keys(dtype):
    cache = _cache(dtype=dtype)
    for frame in range(30):
        key = torch.full((6, 2, 4), float(frame), dtype=dtype)
        q, encoded, next_key, value, spans = _prepare(cache, key + 1, key, key * 0.25)
        assert len(spans) <= 6
        assert torch.isfinite(encoded).all()
        cache.commit(q, next_key, value, spans)
        assert sum(cache.spans) == frame + 1
        assert cache.spans[0] == cache.spans[-1] == 1
        assert cache.key[0, 0, 0] == 0
        assert cache.key[(len(spans) - 1) * 6, 0, 0] == frame
        offset = 0
        for index, span in enumerate(cache.spans):
            expected = offset + (span - 1) / 2
            torch.testing.assert_close(cache.key[index * 6, 0, 0].float(), torch.tensor(expected), atol=0.15, rtol=0)
            offset += span
    assert cache.merge_count == 24
    assert cache.query_history.shape == (1, 2, 2, 4)


def test_noisy_probes_reuse_plan_and_leave_committed_storage_unchanged():
    cache = _cache()
    generator = torch.Generator().manual_seed(7)
    for _ in range(6):
        key = torch.randn(6, 2, 4, generator=generator)
        q, _, next_key, value, spans = _prepare(cache, key, key, key)
        cache.commit(q, next_key, value, spans)
    snapshot = (cache.key.clone(), cache.value.clone(), cache.spans.copy(), cache.query_history.clone())
    first = torch.randn(6, 2, 4, generator=generator)
    q, encoded, next_key, value, spans = _prepare(cache, first, first, first)
    plan = cache._plan.copy()
    second = _prepare(cache, first + 9, first + 3, first - 2)
    assert cache._plan == plan and len(plan) == 1
    assert cache.spans == snapshot[2] and cache.absolute_end == 36
    torch.testing.assert_close(cache.key, snapshot[0])
    torch.testing.assert_close(cache.value, snapshot[1])
    torch.testing.assert_close(cache.query_history, snapshot[3])
    torch.testing.assert_close(second[1][:-6], encoded[:-6])
    cache.commit(q, next_key, value, spans)
    assert cache.absolute_end == 42 and cache.merge_count == 1


def test_queries_and_unmerged_keys_keep_absolute_positions_after_merges():
    cache = _cache()
    generator = torch.Generator().manual_seed(81)
    previous_key = None
    for frame in range(30):
        query, key, incoming_value = [torch.randn(6, 2, 4, generator=generator) for _ in range(3)]
        q, encoded, canonical, value, spans = _prepare(cache, query, key, incoming_value)
        torch.testing.assert_close(q, _rotate(query.unsqueeze(0), frame)[0], atol=0, rtol=0)
        torch.testing.assert_close(encoded[-6:], _rotate(key.unsqueeze(0), frame)[0], atol=0, rtol=0)
        if previous_key is not None:
            torch.testing.assert_close(
                encoded[-12:-6], _rotate(previous_key.unsqueeze(0), frame - 1)[0], atol=0, rtol=0
            )
        position = 0
        for index, span in enumerate(spans):
            expected = _rotate(canonical[index * 6 : (index + 1) * 6].unsqueeze(0), position)[0]
            torch.testing.assert_close(encoded[index * 6 : (index + 1) * 6], expected, atol=0, rtol=0)
            position += span
        cache.commit(q, canonical, value, spans)
        previous_key = key.clone()
    assert cache.merge_count == 24


def test_keys_are_merged_before_rope_and_encoded_at_absolute_left_positions():
    cache = _cache()
    for frame in range(6):
        key = torch.full((6, 2, 4), frame + 1.0)
        q, _, next_key, value, spans = _prepare(cache, key, key, key)
        cache.commit(q, next_key, value, spans)
    previous = cache.key.unflatten(0, (6, 6)).clone()
    incoming = torch.ones(6, 2, 4)
    _, encoded, pre_rope, _, spans = _prepare(cache, incoming, incoming, incoming)
    index, beta, _ = cache._plan[0]
    expected = beta * previous[index] + (1 - beta) * previous[index + 1]
    torch.testing.assert_close(pre_rope.unflatten(0, (6, 6))[index], expected)
    positions = []
    frame = 0
    for span in spans:
        positions.append(frame)
        frame += span
    torch.testing.assert_close(encoded, _rotate(pre_rope.unflatten(0, (6, 6)), positions).flatten(0, 1))
    wrong = beta * _rotate(previous[index : index + 1], index) + (1 - beta) * _rotate(
        previous[index + 1 : index + 2], index + 1
    )
    assert not torch.allclose(_rotate(expected.unsqueeze(0), index), wrong)


@pytest.mark.parametrize(
    "kwargs", [{"budget_frames": True}, {"sink_frames": -1}, {"recent_frames": 0}, {"pooling_factor": 0}]
)
def test_invalid_configuration_fails(kwargs):
    config = {"budget_frames": 6, "sink_frames": 1}
    config.update(kwargs)
    with pytest.raises(ValueError, match="FrameFold"):
        FrameFoldConfig(**config)


def test_budget_and_non_contiguous_block_fail():
    with pytest.raises(ValueError, match="budget"):
        FrameFoldConfig(12, 9).validate_block(3)
    cache = _cache()
    value = torch.ones(6, 2, 4)
    with pytest.raises(ValueError, match="contiguous"):
        cache.prepare(value, value, value, current_start=6, rotate=_rotate, reduce=_identity)
