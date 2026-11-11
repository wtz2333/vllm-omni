# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class _HistoryEntry:
    chunk_index: int
    start_frame: int
    latents: torch.Tensor
    poses: torch.Tensor
    intrinsics: torch.Tensor


@dataclass(frozen=True)
class PreparedPoseHistory:
    latents: torch.Tensor
    token_indices: torch.Tensor
    query_bounds: torch.Tensor
    source_chunk: int
    minimum_overlap: float


@dataclass(frozen=True)
class PoseReferenceKV:
    layers: list[tuple[torch.Tensor, torch.Tensor]]
    query_bounds: torch.Tensor


def _validate_camera(poses: torch.Tensor, intrinsics: torch.Tensor, frames: int) -> None:
    if frames <= 0 or poses.shape != (frames, 4, 4) or intrinsics.shape != (frames, 4):
        raise ValueError("Pose history requires one C2W pose and four intrinsics per latent frame.")
    if not torch.isfinite(poses).all() or not torch.isfinite(intrinsics).all():
        raise ValueError("Pose history camera values must be finite.")
    if (intrinsics[:, :2] <= 0).any():
        raise ValueError("Pose history focal lengths must be positive.")


def rotation_grid(
    source_poses: torch.Tensor,
    source_intrinsics: torch.Tensor,
    target_poses: torch.Tensor,
    target_intrinsics: torch.Tensor,
    *,
    height: int,
    width: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map each target latent cell to a source cell for a shared camera center."""
    frames = source_poses.shape[0]
    _validate_camera(source_poses, source_intrinsics, frames)
    _validate_camera(target_poses, target_intrinsics, frames)
    if height <= 0 or width <= 0:
        raise ValueError("Pose history latent dimensions must be positive.")
    source_poses = source_poses.to(dtype=torch.float32, device="cpu")
    target_poses = target_poses.to(dtype=torch.float32, device="cpu")
    if (torch.linalg.vector_norm(source_poses[:, :3, 3] - target_poses[:, :3, 3], dim=1) > 1e-5).any():
        raise ValueError("Pose history rotation projection requires matching camera centers.")
    source_k = source_intrinsics.to(dtype=torch.float32, device="cpu").clone()
    target_k = target_intrinsics.to(dtype=torch.float32, device="cpu").clone()
    for intrinsics in (source_k, target_k):
        intrinsics[:, (0, 2)] *= width / 832
        intrinsics[:, (1, 3)] *= height / 480

    y, x = torch.meshgrid(
        torch.arange(height, dtype=torch.float32) + 0.5,
        torch.arange(width, dtype=torch.float32) + 0.5,
        indexing="ij",
    )
    fx, fy, cx, cy = target_k.unbind(dim=1)
    rays = torch.stack(
        (
            (x[None] - cx[:, None, None]) / fx[:, None, None],
            (y[None] - cy[:, None, None]) / fy[:, None, None],
            torch.ones((frames, height, width), dtype=torch.float32),
        ),
        dim=-1,
    )
    relative_rotation = source_poses[:, :3, :3].transpose(1, 2) @ target_poses[:, :3, :3]
    source_rays = torch.einsum("fij,fhwj->fhwi", relative_rotation, rays)
    z = source_rays[..., 2]
    denominator = z.clamp_min(1e-8)
    fx, fy, cx, cy = source_k.unbind(dim=1)
    u = source_rays[..., 0] / denominator * fx[:, None, None] + cx[:, None, None]
    v = source_rays[..., 1] / denominator * fy[:, None, None] + cy[:, None, None]
    valid = (z > 0) & (u >= 0.5) & (u <= width - 0.5) & (v >= 0.5) & (v <= height - 0.5)
    grid = torch.stack((2 * u / width - 1, 2 * v / height - 1), dim=-1)
    return grid, valid


class PoseHistory:
    """Bounded request-local clean latents for same-center camera revisits."""

    def __init__(self, *, sink_frames: int, recent_frames: int) -> None:
        if sink_frames < 0 or recent_frames < 0:
            raise ValueError("Pose history window lengths must be non-negative.")
        self.sink_frames = sink_frames
        self.recent_frames = recent_frames
        self.entries: list[_HistoryEntry] = []
        self.next_chunk_index = 0
        self.capacity = 8
        self.max_bytes = 32 << 20

    @staticmethod
    def _bytes(entry: _HistoryEntry) -> int:
        return sum(t.numel() * t.element_size() for t in (entry.latents, entry.poses, entry.intrinsics))

    @staticmethod
    def _angle(first: _HistoryEntry, second: _HistoryEntry) -> float:
        delta = first.poses[-1, :3, :3].T @ second.poses[-1, :3, :3]
        return float(torch.acos(((torch.trace(delta) - 1) / 2).clamp(-1, 1)))

    def _eviction_index(self) -> int:
        # Keep the first archived view and discard a redundant later view first.
        def priority(index: int) -> tuple[float, int]:
            item = self.entries[index]
            similar = [
                self._angle(item, other)
                for other_index, other in enumerate(self.entries)
                if other_index != index
                and torch.linalg.vector_norm(item.poses[-1, :3, 3] - other.poses[-1, :3, 3]) <= 1e-5
            ]
            return (min(similar, default=float("inf")), item.chunk_index)

        return min(range(1, len(self.entries)), key=priority)

    def observe(
        self,
        *,
        chunk_index: int,
        start_frame: int,
        latents: torch.Tensor,
        poses: torch.Tensor,
        intrinsics: torch.Tensor,
    ) -> None:
        if chunk_index != self.next_chunk_index:
            raise ValueError("Pose history chunks must be committed in order.")
        if latents.ndim != 5 or latents.shape[0] != 1 or latents.shape[2] <= 0:
            raise ValueError("Pose history requires latents shaped [1, channels, frames, height, width].")
        frames = latents.shape[2]
        if start_frame != chunk_index * frames:
            raise ValueError("Pose history start frame does not match its chunk index.")
        _validate_camera(poses, intrinsics, frames)
        if chunk_index % 2 == 0 and start_frame >= self.sink_frames:
            entry = _HistoryEntry(
                chunk_index=chunk_index,
                start_frame=start_frame,
                latents=latents.detach(),
                poses=poses.detach().to(device="cpu", dtype=torch.float32, copy=True),
                intrinsics=intrinsics.detach().to(device="cpu", dtype=torch.float32, copy=True),
            )
            if self._bytes(entry) > self.max_bytes:
                raise MemoryError("One pose history entry exceeds the 32 MiB history budget.")
            self.entries.append(entry)
            while len(self.entries) > self.capacity or sum(map(self._bytes, self.entries)) > self.max_bytes:
                if len(self.entries) == 1:
                    raise MemoryError("Pose history cannot fit its first entry within the budget.")
                self.entries.pop(self._eviction_index())
            for index, candidate in enumerate(self.entries):
                if candidate is entry:
                    self.entries[index] = _HistoryEntry(
                        chunk_index=entry.chunk_index,
                        start_frame=entry.start_frame,
                        latents=entry.latents.to(device="cpu", copy=True),
                        poses=entry.poses,
                        intrinsics=entry.intrinsics,
                    )
                    break
        self.next_chunk_index += 1

    def prepare(
        self,
        *,
        start_frame: int,
        poses: torch.Tensor,
        intrinsics: torch.Tensor,
        device: torch.device,
    ) -> tuple[PreparedPoseHistory | None, str]:
        frames = poses.shape[0]
        _validate_camera(poses, intrinsics, frames)
        poses = poses.detach().to(device="cpu", dtype=torch.float32)
        intrinsics = intrinsics.detach().to(device="cpu", dtype=torch.float32)
        eligible = [entry for entry in self.entries if entry.start_frame + frames <= start_frame - self.recent_frames]
        if not eligible:
            return None, "no_eligible_history"
        candidates = []
        for entry in eligible:
            translation = torch.linalg.vector_norm(entry.poses[:, :3, 3] - poses[:, :3, 3], dim=1)
            if translation.max().item() > 1e-5:
                continue
            height, width = entry.latents.shape[-2:]
            if height % 2 or width % 2:
                raise ValueError("Pose history requires latent dimensions divisible by the DiT spatial patch.")
            grid, valid = rotation_grid(entry.poses, entry.intrinsics, poses, intrinsics, height=height, width=width)
            token_mask = F.avg_pool2d(valid[:, None].float(), 2, 2)[:, 0] >= 1.0
            overlap = token_mask.flatten(1).float().mean(dim=1)
            if overlap.min().item() >= 0.3:
                candidates.append((float(overlap.min()), entry.chunk_index, entry, grid, token_mask))
        if not candidates:
            return None, "camera_translation_or_insufficient_overlap"
        overlap, _, entry, grid, token_mask = max(candidates, key=lambda value: (value[0], value[1]))
        token_count = token_mask.shape[1] * token_mask.shape[2]
        selected = []
        for frame_index, mask in enumerate(token_mask):
            indices = torch.nonzero(mask.flatten(), as_tuple=False).flatten()
            if indices.numel() > 810:
                positions = torch.linspace(0, indices.numel() - 1, 810).round().long()
                indices = indices[positions]
            selected.append(indices + frame_index * token_count)
        token_indices = torch.cat(selected).to(device=device)
        if token_indices.numel() == 0:
            return None, "no_visible_reference_tokens"
        source = entry.latents.to(device=device)
        source = source.permute(0, 2, 1, 3, 4).flatten(0, 1)
        warped = F.grid_sample(
            source, grid.to(device=device), mode="bilinear", padding_mode="border", align_corners=False
        )
        warped = warped.unflatten(0, (1, frames)).permute(0, 2, 1, 3, 4).contiguous()
        progress = torch.arange(1, frames + 1, dtype=torch.float32) / frames
        strength = 0.15 * (3 * progress.square() - 2 * progress.pow(3))
        query_bounds = (token_mask.float() * strength[:, None, None]).reshape(1, -1, 1, 1).to(device=device)
        return PreparedPoseHistory(warped, token_indices, query_bounds, entry.chunk_index, overlap), "hit"

    def close(self) -> None:
        self.entries.clear()


def bounded_reference_output(
    native: torch.Tensor,
    reference: torch.Tensor,
    native_lse: torch.Tensor,
    reference_lse: torch.Tensor,
    bounds: torch.Tensor,
) -> torch.Tensor:
    """Fuse one reference with a per-query weight and relative-change bound."""
    original_dtype = native.dtype
    native = native.float()
    difference = reference.float() - native
    alpha = torch.sigmoid(reference_lse.float() - native_lse.float()).T[None, :, :, None]
    native_norm = torch.linalg.vector_norm(native, dim=-1, keepdim=True)
    difference_norm = torch.linalg.vector_norm(difference, dim=-1, keepdim=True)
    alpha = torch.minimum(torch.minimum(alpha, bounds), bounds * native_norm / (difference_norm + 1e-12))
    return (native + alpha * difference).to(dtype=original_dtype)


def paged_history_attention(
    inputs,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    reference_key: torch.Tensor,
    reference_value: torch.Tensor,
    bounds: torch.Tensor,
    *,
    softmax_scale: float,
) -> torch.Tensor:
    """Write the current page and combine native and reference attention in eager mode."""
    if query.ndim != 3 or reference_key.ndim != 4 or reference_key.shape[0] != 1:
        raise ValueError("Pose history attention requires one request and one reference batch.")
    if reference_key.shape != reference_value.shape or reference_key.shape[2:] != query.shape[1:]:
        raise ValueError("Pose history reference K/V geometry does not match the query.")
    if bounds.shape != (1, query.shape[0], 1, 1):
        raise ValueError("Pose history bounds do not match the current query.")
    inputs.key_pool[inputs.video_slots] = key.to(dtype=inputs.key_pool.dtype)
    inputs.value_pool[inputs.video_slots] = value.to(dtype=inputs.value_pool.dtype)
    key_cache = inputs.key_pool.unflatten(0, (-1, inputs.block_size))
    value_cache = inputs.value_pool.unflatten(0, (-1, inputs.block_size))
    if query.device.type == "cpu":
        if inputs.seq_lens.numel() != 1:
            raise ValueError("Pose history attention supports one request.")
        length = int(inputs.seq_lens[0].item())
        positions = torch.arange(length, device=query.device)
        blocks = inputs.block_table[0, torch.div(positions, inputs.block_size, rounding_mode="floor")].long()
        offsets = positions % inputs.block_size
        native_key = key_cache[blocks, offsets]
        native_value = value_cache[blocks, offsets]

        def dense(k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            logits = torch.einsum("qhd,khd->hqk", query.float(), k.float()) * softmax_scale
            output = torch.einsum("hqk,khd->qhd", logits.softmax(-1).to(v.dtype), v)
            return output, logits.logsumexp(-1)

        native, native_lse = dense(native_key, native_value)
        reference, reference_lse = dense(reference_key[0], reference_value[0])
    elif query.device.type == "cuda" and torch.version.hip is None:
        from vllm.vllm_flash_attn import flash_attn_varlen_func

        from vllm_omni.experimental.ar_diffusion.kv_cache.paged_attention import _resolve_fa_version

        fa_version = _resolve_fa_version(query.shape[-1])
        native, native_lse = flash_attn_varlen_func(
            q=query,
            k=key_cache,
            v=value_cache,
            cu_seqlens_q=inputs.query_start_loc,
            max_seqlen_q=inputs.max_query_len,
            seqused_k=inputs.seq_lens,
            max_seqlen_k=inputs.max_seq_len,
            softmax_scale=softmax_scale,
            causal=False,
            block_table=inputs.block_table,
            return_softmax_lse=True,
            fa_version=fa_version,
        )
        reference_length = reference_key.shape[1]
        reference_locations = torch.tensor([0, reference_length], dtype=torch.int32, device=query.device)
        reference, reference_lse = flash_attn_varlen_func(
            q=query,
            k=reference_key[0].contiguous(),
            v=reference_value[0].contiguous(),
            cu_seqlens_q=inputs.query_start_loc,
            cu_seqlens_k=reference_locations,
            max_seqlen_q=inputs.max_query_len,
            max_seqlen_k=reference_length,
            softmax_scale=softmax_scale,
            causal=False,
            return_softmax_lse=True,
            fa_version=fa_version,
        )
    else:
        raise NotImplementedError("Pose history attention currently supports CUDA and CPU tests only.")
    return bounded_reference_output(native[None], reference[None], native_lse, reference_lse, bounds)[0]
