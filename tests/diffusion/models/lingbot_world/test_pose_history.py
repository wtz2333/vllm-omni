# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch

from vllm_omni.diffusion.models.lingbot_world.pose_history import (
    PoseHistory,
    bounded_reference_output,
    paged_history_attention,
    rotation_grid,
)
from vllm_omni.experimental.ar_diffusion.kv_cache.paged_attention import ARDiffusionPagedLayerInputs

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def _camera(frames: int = 3) -> tuple[torch.Tensor, torch.Tensor]:
    poses = torch.eye(4).repeat(frames, 1, 1)
    intrinsics = torch.tensor([[500.0, 500.0, 416.0, 240.0]]).repeat(frames, 1)
    return poses, intrinsics


def _observe(history: PoseHistory, count: int) -> None:
    poses, intrinsics = _camera()
    for chunk in range(count):
        history.observe(
            chunk_index=chunk,
            start_frame=chunk * 3,
            latents=torch.full((1, 1, 3, 4, 6), float(chunk)),
            poses=poses,
            intrinsics=intrinsics,
        )


def test_identity_rotation_uses_pixel_centers_and_preserves_the_latent() -> None:
    poses, intrinsics = _camera()
    grid, valid = rotation_grid(poses, intrinsics, poses, intrinsics, height=4, width=6)
    source = torch.arange(72.0).reshape(3, 1, 4, 6)
    result = torch.nn.functional.grid_sample(source, grid, padding_mode="border", align_corners=False)
    assert valid.all()
    torch.testing.assert_close(result, source, rtol=0, atol=1e-5)


def test_rotation_grid_uses_target_to_source_direction() -> None:
    source, intrinsics = _camera(1)
    target = source.clone()
    angle = torch.tensor(0.25)
    target[0, 0, 0] = angle.cos()
    target[0, 0, 2] = angle.sin()
    target[0, 2, 0] = -angle.sin()
    target[0, 2, 2] = angle.cos()
    grid, valid = rotation_grid(source, intrinsics, target, intrinsics, height=8, width=12)
    assert grid[0, 4, 6, 0] > (2 * 6.5 / 12 - 1)
    assert not valid.all()


def test_history_only_reads_committed_chunks_outside_the_recent_window() -> None:
    history = PoseHistory(sink_frames=3, recent_frames=6)
    _observe(history, 10)
    poses, intrinsics = _camera()
    prepared, reason = history.prepare(start_frame=30, poses=poses, intrinsics=intrinsics, device=torch.device("cpu"))
    assert reason == "hit"
    assert prepared is not None
    assert prepared.source_chunk == 6
    torch.testing.assert_close(prepared.latents, torch.full((1, 1, 3, 4, 6), 6.0))
    assert prepared.token_indices.numel() == 18
    assert prepared.query_bounds.shape == (1, 18, 1, 1)
    assert prepared.query_bounds.max().item() == pytest.approx(0.15)


def test_translation_is_not_treated_as_a_pure_rotation() -> None:
    history = PoseHistory(sink_frames=3, recent_frames=6)
    _observe(history, 10)
    poses, intrinsics = _camera()
    poses[:, 0, 3] = 0.01
    source, _ = _camera()
    with pytest.raises(ValueError, match="matching camera centers"):
        rotation_grid(source, intrinsics, poses, intrinsics, height=4, width=6)
    prepared, reason = history.prepare(start_frame=30, poses=poses, intrinsics=intrinsics, device=torch.device("cpu"))
    assert prepared is None
    assert reason == "camera_translation_or_insufficient_overlap"


def test_history_is_bounded_and_requires_ordered_commits() -> None:
    history = PoseHistory(sink_frames=3, recent_frames=6)
    _observe(history, 24)
    assert len(history.entries) == 8
    assert history.entries[0].chunk_index == 2
    assert sum(history._bytes(entry) for entry in history.entries) <= 32 << 20
    poses, intrinsics = _camera()
    with pytest.raises(ValueError, match="committed in order"):
        history.observe(
            chunk_index=25,
            start_frame=75,
            latents=torch.zeros(1, 1, 3, 4, 6),
            poses=poses,
            intrinsics=intrinsics,
        )
    history.close()
    assert not history.entries


def test_reference_fusion_respects_mask_and_relative_change_bound() -> None:
    native = torch.ones(1, 2, 1, 2)
    reference = torch.full_like(native, 101)
    lse = torch.zeros(1, 2)
    bounds = torch.tensor([0.0, 0.15]).reshape(1, 2, 1, 1)
    result = bounded_reference_output(native, reference, lse, lse, bounds)
    torch.testing.assert_close(result[:, 0], native[:, 0])
    assert torch.linalg.vector_norm(result[:, 1] - native[:, 1]) <= 0.150001 * torch.linalg.vector_norm(native[:, 1])


def test_paged_reference_attention_matches_joint_attention_without_an_active_cap() -> None:
    torch.manual_seed(17)
    query = torch.randn(2, 2, 4)
    key = torch.randn(2, 2, 4) * 0.1
    value = 1 + torch.randn(2, 2, 4) * 0.01
    reference_key = torch.randn(1, 3, 2, 4) * 0.1
    reference_value = 1 + torch.randn(1, 3, 2, 4) * 0.01
    inputs = ARDiffusionPagedLayerInputs(
        layer_idx=torch.tensor(0),
        key_pool=torch.randn(4, 2, 4) * 0.1,
        value_pool=1 + torch.randn(4, 2, 4) * 0.01,
        video_slots=torch.tensor([2, 3]),
        block_size=2,
        seq_len=4,
        action_slots=torch.empty(0, dtype=torch.int64),
        block_table=torch.tensor([[0, 1]]),
        query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
        seq_lens=torch.tensor([4]),
        max_query_len=2,
        max_seq_len=4,
    )
    output = paged_history_attention(
        inputs, query, key, value, reference_key, reference_value, torch.ones(1, 2, 1, 1), softmax_scale=0.5
    )
    all_key = torch.cat((inputs.key_pool, reference_key[0]), dim=0)
    all_value = torch.cat((inputs.value_pool, reference_value[0]), dim=0)
    logits = torch.einsum("qhd,khd->hqk", query, all_key) * 0.5
    expected = torch.einsum("hqk,khd->qhd", logits.softmax(-1), all_value)
    torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(inputs.key_pool[2:], key)
