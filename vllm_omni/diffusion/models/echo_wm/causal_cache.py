# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded causal cache layout for Echo-WM audio-video inference.

Ported from the Echo-WM reference rollout (``ltx-causal``): the video latent
timeline is generated in 3-latent-frame chunks after a 1-frame image sink, the
audio latent timeline in a 2-frame prefix plus 25-frame chunks aligned to the
video chunks, and every windowed attention keeps a bounded ``sink + recent
FIFO`` KV window whose RoPE is rebased onto a fixed window template.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch

__all__ = [
    "CAUSAL_VIDEO_CHUNK_SIZE",
    "AUDIO_PREFIX_FRAMES",
    "AUDIO_FRAMES_PER_VIDEO_BLOCK",
    "DEFAULT_CAUSAL_TIMESTEPS",
    "EchoWMCacheConfig",
    "EchoWMKVWindow",
    "EchoWMTextKV",
    "EchoWMLayerCaches",
    "resolve_causal_sigmas",
    "causal_video_blocks",
    "causal_audio_blocks",
    "causal_audio_frames",
    "build_video_positions",
    "build_audio_positions",
    "make_split_rope",
    "make_cross_rope_template",
    "compute_cross_slices",
]

CAUSAL_VIDEO_CHUNK_SIZE = 3
AUDIO_PREFIX_FRAMES = 2
AUDIO_FRAMES_PER_VIDEO_BLOCK = 25
DEFAULT_CAUSAL_TIMESTEPS = (1000, 750, 500, 250)

# LTX-2 sigma schedule anchors (see the reference ``LTX2Scheduler``).
_BASE_SHIFT_ANCHOR = 1024
_MAX_SHIFT_ANCHOR = 4096
_MAX_SHIFT = 2.05
_BASE_SHIFT = 0.95
_SIGMA_TERMINAL = 0.1


def _ltx2_sigma_schedule(steps: int = 1000, default_number_of_tokens: int = _MAX_SHIFT_ANCHOR) -> torch.Tensor:
    sigmas = torch.linspace(1.0, 0.0, steps + 1)
    mm = (_MAX_SHIFT - _BASE_SHIFT) / (_MAX_SHIFT_ANCHOR - _BASE_SHIFT_ANCHOR)
    b = _BASE_SHIFT - mm * _BASE_SHIFT_ANCHOR
    sigma_shift = default_number_of_tokens * mm + b
    sigmas = torch.where(
        sigmas != 0,
        math.exp(sigma_shift) / (math.exp(sigma_shift) + (1 / sigmas - 1) ** 1),
        0,
    )
    non_zero = sigmas[sigmas != 0]
    one_minus_z = 1.0 - non_zero
    scale = one_minus_z[-1] / (1.0 - _SIGMA_TERMINAL)
    stretched = 1.0 - (one_minus_z / scale)
    sigmas[sigmas != 0] = stretched
    return sigmas


def resolve_causal_sigmas(
    timesteps: tuple[int, ...] | list[int] = DEFAULT_CAUSAL_TIMESTEPS,
    *,
    num_train_timesteps: int = 1000,
) -> list[float]:
    """Map distilled student timesteps to model sigmas (no appended zero)."""
    if not timesteps:
        raise ValueError("at least one causal timestep is required")
    schedule = _ltx2_sigma_schedule(num_train_timesteps)
    result = []
    for timestep in timesteps:
        index = num_train_timesteps - int(timestep)
        if not 0 <= index < len(schedule):
            raise ValueError(f"causal timestep {timestep} is outside [0, {num_train_timesteps}]")
        result.append(float(schedule[index]))
    for current, following in zip(result, result[1:], strict=False):
        if current <= following:
            raise ValueError(f"causal sigmas must be strictly descending, got {result}")
    return result


def _causal_block_count(video_frames: int, chunk_size: int) -> int:
    if chunk_size != CAUSAL_VIDEO_CHUNK_SIZE:
        raise ValueError(f"Echo-WM Flash requires video_chunk_size={CAUSAL_VIDEO_CHUNK_SIZE}, got {chunk_size}")
    if video_frames < 1 or (video_frames - 1) % chunk_size:
        raise ValueError(f"latent video length must be 1 + n * chunk_size, got {video_frames}")
    return (video_frames - 1) // chunk_size


def causal_video_blocks(
    video_frames: int,
    chunk_size: int = CAUSAL_VIDEO_CHUNK_SIZE,
) -> list[tuple[int, int]]:
    """``[0, 1]`` (image sink) followed by fixed-size causal chunks."""
    _causal_block_count(video_frames, chunk_size)
    return [(0, 1), *[(start, start + chunk_size) for start in range(1, video_frames, chunk_size)]]


def causal_audio_frames(
    video_frames: int,
    chunk_size: int = CAUSAL_VIDEO_CHUNK_SIZE,
) -> int:
    """Map video latent frames to the aligned audio latent layout."""
    return AUDIO_PREFIX_FRAMES + _causal_block_count(video_frames, chunk_size) * AUDIO_FRAMES_PER_VIDEO_BLOCK


def causal_audio_blocks(
    video_frames: int,
    chunk_size: int = CAUSAL_VIDEO_CHUNK_SIZE,
) -> list[tuple[int, int]]:
    """Audio blocks paired 1:1 with :func:`causal_video_blocks`."""
    total = causal_audio_frames(video_frames, chunk_size)
    return [
        (0, AUDIO_PREFIX_FRAMES),
        *[
            (start, min(start + AUDIO_FRAMES_PER_VIDEO_BLOCK, total))
            for start in range(AUDIO_PREFIX_FRAMES, total, AUDIO_FRAMES_PER_VIDEO_BLOCK)
        ],
    ]


@dataclass(frozen=True)
class EchoWMCacheConfig:
    """Video cache sizes in latent-frame units, with aligned audio sizes."""

    video_local_attn_size: int = 19
    video_sink_size: int = 7
    video_chunk_size: int = CAUSAL_VIDEO_CHUNK_SIZE

    @property
    def audio_local_attn_size(self) -> int:
        return causal_audio_frames(self.video_local_attn_size, self.video_chunk_size)

    @property
    def audio_sink_size(self) -> int:
        return causal_audio_frames(self.video_sink_size, self.video_chunk_size)

    def validate(self) -> None:
        if self.video_chunk_size != CAUSAL_VIDEO_CHUNK_SIZE:
            raise ValueError(f"Echo-WM Flash requires video_chunk_size={CAUSAL_VIDEO_CHUNK_SIZE}")
        if not 0 < self.video_sink_size < self.video_local_attn_size:
            raise ValueError("expected 0 < video_sink_size < video_local_attn_size")
        if self.video_chunk_size > self.video_local_attn_size - self.video_sink_size:
            raise ValueError("video_chunk_size must fit in the FIFO portion of the cache")
        for name, size in (
            ("video_local_attn_size", self.video_local_attn_size),
            ("video_sink_size", self.video_sink_size),
        ):
            if (size - 1) % self.video_chunk_size:
                raise ValueError(f"{name} must be 1 + n * video_chunk_size for audio alignment")


class EchoWMKVWindow:
    """One bounded ``sink + recent FIFO`` KV window (per layer, per attention).

    K/V are stored *un-roped*; every forward re-applies the fixed window RoPE
    template, so the sliding window always occupies the template's positions.
    Updates are transactional over ``[start, start + tokens)``: an earlier
    noisier version of the same range is dropped while committed history is
    retained. Shapes are ``(B, capacity, local_heads, head_dim)`` so each rank
    holds its own (TP x SP) head shard of the shared token window.
    """

    def __init__(
        self,
        batch_size: int,
        capacity: int,
        num_heads: int,
        head_dim: int,
        local_attn_size: int,
        sink_tokens: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        if capacity <= 0:
            raise ValueError("KV cache capacity must be positive")
        self.k = torch.zeros(batch_size, capacity, num_heads, head_dim, device=device, dtype=dtype)
        self.v = torch.zeros_like(self.k)
        self.positions = torch.full((capacity,), -1, device=device, dtype=torch.long)
        self.length = 0
        self.local_attn_size = local_attn_size
        self.sink_tokens = sink_tokens

    def update(self, start: int, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Insert the global token range ``[start, start + S)`` and return the active window."""
        if torch.is_grad_enabled():
            raise RuntimeError("causal KV caches are inference-only")
        if k.shape != v.shape or k.ndim != 4:
            raise ValueError("KV tensors must have matching (B, S, H, D) shapes")
        length = int(self.length)
        old_positions = self.positions[:length]
        old_k = self.k[:, :length]
        old_v = self.v[:, :length]
        end = start + k.shape[1]

        keep_old = old_positions < start
        positions = torch.cat([old_positions[keep_old], torch.arange(start, end, device=k.device)], dim=0)
        merged_k = torch.cat([old_k[:, keep_old], k], dim=1)
        merged_v = torch.cat([old_v[:, keep_old], v], dim=1)

        local = self.local_attn_size
        sink = self.sink_tokens
        if local >= 0 and positions.numel() > local:
            if not 0 <= sink < local:
                raise ValueError(f"expected 0 <= sink_tokens < local_attn_size, got {sink}/{local}")
            sink_mask = positions < sink
            recent_budget = local - int(sink_mask.sum())
            recent_start = max(sink, end - recent_budget)
            keep = sink_mask | (positions >= recent_start)
            positions = positions[keep]
            merged_k = merged_k[:, keep]
            merged_v = merged_v[:, keep]

        active = positions.numel()
        if active > self.k.shape[1]:
            raise ValueError(f"KV cache overflow: {active} active tokens exceed capacity {self.k.shape[1]}")
        self.k[:, :active].copy_(merged_k)
        self.v[:, :active].copy_(merged_v)
        self.positions[:active].copy_(positions)
        self.length = active
        return self.k[:, :active], self.v[:, :active]


class EchoWMTextKV:
    """Init-once cross-attention KV over the (fixed-length) text context."""

    def __init__(
        self,
        batch_size: int,
        seq_len: int,
        num_heads: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.k = torch.zeros(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
        self.v = torch.zeros_like(self.k)
        self.length = 0
        self.is_init = False

    def get(self, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.is_init:
            size = k.shape[1]
            if size > self.k.shape[1]:
                raise ValueError(f"text KV overflow: {size} tokens exceed capacity {self.k.shape[1]}")
            self.k[:, :size].copy_(k)
            self.v[:, :size].copy_(v)
            self.length = size
            self.is_init = True
        size = int(self.length)
        return self.k[:, :size], self.v[:, :size]


@dataclass
class EchoWMLayerCaches:
    """All per-layer caches used by one Echo-WM forward."""

    video_self: EchoWMKVWindow
    video_text: EchoWMTextKV
    audio_self: EchoWMKVWindow | None = None
    audio_text: EchoWMTextKV | None = None
    a2v: EchoWMKVWindow | None = None
    v2a: EchoWMKVWindow | None = None
    video_ucpe: EchoWMKVWindow | None = None
    # RoPE templates shared by every layer (set once per session). The cross
    # templates are temporal-axis-only (dim 2048, audio head geometry): the
    # video one templates a2v queries / v2a keys, the audio one the reverse.
    video_rope: tuple[torch.Tensor, torch.Tensor] | None = None
    audio_rope: tuple[torch.Tensor, torch.Tensor] | None = None
    video_cross_rope: tuple[torch.Tensor, torch.Tensor] | None = None
    audio_cross_rope: tuple[torch.Tensor, torch.Tensor] | None = None
    a2v_q_slices: dict[tuple[int, int], tuple[int, int]] = field(default_factory=dict)
    v2a_q_slices: dict[tuple[int, int], tuple[int, int]] = field(default_factory=dict)
    # Bounded-anchor-translation state for the UCPE branch.
    ucpe_full_viewmats: torch.Tensor | None = None
    ucpe_full_Ks: torch.Tensor | None = None
    ucpe_bounded: bool = False


def build_video_positions(
    num_frames: int,
    height: int,
    width: int,
    *,
    fps: float,
    spatial_scale: int = 32,
    temporal_scale: int = 8,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Per-token ``(time, height, width)`` bounds, ``(B, 3, F*H*W, 2)``.

    Coordinates live in pixel/second space with the causal first-frame fix
    (latent frame 0 spans one pixel frame instead of ``temporal_scale``), and
    token order follows the reference patchifier's ``(f, h, w)`` row-major
    layout. Values are float seconds / pixels; the caller casts as needed.
    """
    latent_h = height // spatial_scale
    latent_w = width // spatial_scale
    f = torch.arange(num_frames, device=device)
    h = torch.arange(latent_h, device=device)
    w = torch.arange(latent_w, device=device)
    grid = torch.stack(torch.meshgrid(f, h, w, indexing="ij"), dim=0)  # (3, F, H, W)
    starts = grid.flatten(1)  # (3, F*H*W) in latent coordinates
    ends = starts + 1
    coords = torch.stack((starts, ends), dim=-1).float()  # (3, T, 2)
    # Latent -> pixel/second space: scale each axis by the VAE factors, then
    # causal_fix (the first latent frame's temporal stride is 1, not 8 — the
    # reference shifts the whole temporal axis and clamps at 0), then convert
    # the temporal axis to seconds.
    scales = torch.tensor([temporal_scale, spatial_scale, spatial_scale], device=device).float().unsqueeze(1)
    coords = coords * scales.unsqueeze(-1)
    coords[0] = (coords[0] + 1 - temporal_scale).clamp(min=0)
    coords[0] = coords[0] / fps
    return coords.unsqueeze(0)  # (1, 3, T, 2)


def build_audio_positions(
    num_frames: int,
    *,
    sample_rate: int = 16000,
    hop_length: int = 160,
    latent_downsample: int = 4,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Per-frame audio timestamps ``(B, 1, T, 2)`` in seconds.

    Matches the reference causal audio patchifier: latent frame ``f`` covers
    mel frames ``[4f + 1 - 4, ...]`` clipped at zero, converted to seconds by
    ``hop_length / sample_rate``.
    """

    def frame_time(latent_index: torch.Tensor) -> torch.Tensor:
        mel = latent_index * latent_downsample
        mel = (mel + 1 - latent_downsample).clamp(min=0)
        return mel * hop_length / sample_rate

    latent_index = torch.arange(num_frames, device=device)
    start = frame_time(latent_index)
    end = frame_time(latent_index + 1)
    coords = torch.stack((start, end), dim=-1).float()  # (T, 2)
    return coords[None, None]  # (1, 1, T, 2)


def _freq_grid(theta: float, max_pos_count: int, inner_dim: int) -> torch.Tensor:
    """float64 NumPy frequency grid (reference ``frequencies_precision="float64"``)."""
    n_elem = 2 * max_pos_count
    count = inner_dim // n_elem
    pow_indices = np.power(
        theta,
        np.linspace(
            np.log(1) / np.log(theta),
            np.log(theta) / np.log(theta),
            count,
            dtype=np.float64,
        ),
    )
    return torch.tensor(pow_indices * math.pi / 2, dtype=torch.float32)


def make_split_rope(
    positions: torch.Tensor,
    *,
    dim: int,
    num_heads: int,
    theta: float = 10000.0,
    max_pos: list[int],
    out_dtype: torch.dtype = torch.bfloat16,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split-style RoPE ``(cos, sin)`` of shape ``(B, H, T, D/2)``.

    ``positions`` is ``(B, axes, T, 2)`` per-axis bounds; the middle of each
    bound is the effective index. The frequency grid is generated in float64
    (NumPy) and trigonometry evaluated in float32 before the final cast.
    """
    positions = positions.to(device)
    n_pos_dims = positions.shape[1]
    if n_pos_dims != len(max_pos):
        raise ValueError(f"position dims ({n_pos_dims}) must match max_pos {max_pos}")
    indices = _freq_grid(theta, n_pos_dims, dim).to(device)
    middle = (positions[..., 0] + positions[..., 1]) / 2.0
    fractional = torch.stack([middle[:, i] / max_pos[i] for i in range(n_pos_dims)], dim=-1)
    freqs = (indices * (fractional.unsqueeze(-1) * 2 - 1)).transpose(-1, -2).flatten(2)
    expected = dim // 2
    pad_size = expected - freqs.shape[-1]
    cos, sin = freqs.cos(), freqs.sin()
    if pad_size != 0:
        cos = torch.cat([torch.ones_like(cos[:, :, :pad_size]), cos], dim=-1)
        sin = torch.cat([torch.zeros_like(sin[:, :, :pad_size]), sin], dim=-1)
    b = cos.shape[0]
    t = cos.shape[1]
    cos = cos.reshape(b, t, num_heads, -1).swapaxes(1, 2)
    sin = sin.reshape(b, t, num_heads, -1).swapaxes(1, 2)
    return cos.to(out_dtype), sin.to(out_dtype)


def make_cross_rope_template(
    positions: torch.Tensor,
    *,
    dim: int,
    num_heads: int,
    theta: float = 10000.0,
    max_pos: int,
    out_dtype: torch.dtype = torch.bfloat16,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cross-modal (a2v/v2a) RoPE template from the temporal axis only."""
    temporal = positions[:, 0:1, :, :]
    return make_split_rope(
        temporal,
        dim=dim,
        num_heads=num_heads,
        theta=theta,
        max_pos=[max_pos],
        out_dtype=out_dtype,
        device=device,
    )


def compute_cross_slices(
    video_frames: int,
    patches_per_frame: int,
    cache: EchoWMCacheConfig,
) -> tuple[dict[tuple[int, int], tuple[int, int]], dict[tuple[int, int], tuple[int, int]]]:
    """Query RoPE slot maps for a2v (keyed by audio block) and v2a (by video block).

    Each block's queries are placed at the tail of the (capped) window of the
    query-side template, matching the reference ``configure_bounded_caches``.
    """
    video_blocks = causal_video_blocks(video_frames, cache.video_chunk_size)
    audio_blocks = causal_audio_blocks(video_frames, cache.video_chunk_size)
    audio_to_video: dict[tuple[int, int], tuple[int, int]] = {}
    video_to_audio: dict[tuple[int, int], tuple[int, int]] = {}
    for (video_start, video_end), (audio_start, audio_end) in zip(video_blocks, audio_blocks, strict=True):
        video_query_end = min(video_end, cache.video_local_attn_size) * patches_per_frame
        audio_to_video[(audio_start, audio_end)] = (
            video_query_end - (video_end - video_start) * patches_per_frame,
            video_query_end,
        )
        audio_query_end = min(audio_end, cache.audio_local_attn_size)
        video_to_audio[(video_start * patches_per_frame, video_end * patches_per_frame)] = (
            audio_query_end - (audio_end - audio_start),
            audio_query_end,
        )
    return audio_to_video, video_to_audio
