# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Configuration for AR-Diffusion engine-level KV cache management."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

KV_GATHER_ENV = "VLLM_OMNI_AR_DIFFUSION_KV_GATHER"


@dataclass(frozen=True)
class QVGConfig:
    method: str = "qvg"
    bits: int = 4
    num_stages: int = 1
    group_size: int = 64
    num_centroids: int = 256
    kmeans_max_iters: int = 100
    keep_recent_chunks: int = 1
    seed: int = 0

    def __post_init__(self) -> None:
        if self.method != "qvg":
            raise ValueError(f"Unsupported AR-Diffusion history compression method: {self.method!r}")
        for name in self.__dataclass_fields__:
            if name != "method":
                value = getattr(self, name)
                if isinstance(value, bool) or not isinstance(value, int):
                    raise ValueError(f"QVG {name} must be an integer, got {value!r}")
        if self.bits not in (2, 4):
            raise ValueError("QVG bits must be 2 or 4")
        if not 1 <= self.num_stages <= 4:
            raise ValueError("QVG num_stages must be between 1 and 4")
        if self.group_size not in (16, 32, 64):
            raise ValueError("QVG group_size must be 16, 32, or 64")
        if not 1 <= self.num_centroids <= 256:
            raise ValueError("QVG num_centroids must be between 1 and 256")
        if self.kmeans_max_iters <= 0 or self.keep_recent_chunks < 0 or self.seed < 0:
            raise ValueError("QVG requires positive kmeans_max_iters and non-negative keep_recent_chunks and seed")


def contiguous_kv_gather_enabled() -> bool:
    """Whether the experimental contiguous-K/V gather attention path is on (``VLLM_OMNI_AR_DIFFUSION_KV_GATHER=1``).

    The one place the switch is read: the KV manager consults it when it budgets and allocates the history
    staging buffers, and the attention dispatch when it picks the path, so the two cannot disagree. It is the
    only consumer of ``ARDiffusionKVConfig.reuse_history_staging``.
    """
    return os.environ.get(KV_GATHER_ENV, "0") == "1"


@dataclass
class ARDiffusionKVConfig:
    """Settings for the AR-Diffusion paged KV cache.

    Disabled by default: when ``enable`` is False the AR-Diffusion engine behaves exactly
    like the base ``DiffusionEngine`` (no pool, no paged KV).
    """

    enable: bool = False
    # Persistent KV tokens materialized per paged cache block.
    chunk_size: int = 0
    # Resident window in chunks. ``None`` means full attention (no eviction).
    window_chunks: int | None = None
    # Protected leading chunks (attention sink); never evicted.
    sink_chunks: int = 0
    # Boundary reset vs. sliding replacement.
    reset_at_boundary: bool = False
    # Fraction of free device memory used to admit additional resident
    # sessions. One session is admitted whenever it fits actual free memory.
    gpu_memory_fraction: float = 0.1
    # When CUDA graph / torch.compile is on (not enforce_eager), pre-capture the
    # DiT graphs for every window-fill shape at load time via a synthetic rollout,
    # so the serving run is fast from the first chunk. No effect when eager.
    warmup_cudagraph: bool = True
    # Keep one contiguous K/V staging buffer per layer and refresh only the tokens that changed.
    # Within one AR block every forward attends the same history and differs only in the current
    # chunk, so re-gathering the whole visible window per forward re-copies bytes that did not move.
    # Off by default: it costs two max-sequence buffers per layer, per rank. See
    # ARDiffusionPagedForwardContext.history_staging.
    reuse_history_staging: bool = False
    # Also capture the post-window-boundary (reset-cycle) forward during warm-up.
    warmup_capture_reset: bool = False
    compression: QVGConfig | dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if isinstance(self.compression, dict):
            self.compression = QVGConfig(**self.compression)
        if self.compression is not None:
            if not isinstance(self.compression, QVGConfig):
                raise TypeError("AR-Diffusion compression must be a QVGConfig, a dict, or None")
            if self.reuse_history_staging:
                raise ValueError("QVG history compression requires reuse_history_staging=False")
            if self.reset_at_boundary:
                raise ValueError("QVG history compression currently requires reset_at_boundary=False")

    @property
    def sliding_window(self) -> int | None:
        """Window size in tokens, or ``None`` for full attention."""
        if self.window_chunks is None:
            return None
        return self.window_chunks * self.chunk_size
