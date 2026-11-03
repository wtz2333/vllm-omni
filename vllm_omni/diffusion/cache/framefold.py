# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn.functional as F

Reduce = Callable[[torch.Tensor], torch.Tensor]
Rotate = Callable[[torch.Tensor, int], torch.Tensor]
RoPE = Callable[..., tuple[torch.Tensor, torch.Tensor]]


@dataclass(frozen=True)
class FrameFoldConfig:
    budget_frames: int
    sink_frames: int
    recent_frames: int = 3
    pooling_factor: int = 4

    def __post_init__(self) -> None:
        for name in ("budget_frames", "sink_frames", "recent_frames", "pooling_factor"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < (0 if name == "sink_frames" else 1):
                raise ValueError(f"FrameFold {name} has an invalid value: {value!r}")

    def validate_block(self, frames: int) -> None:
        if frames <= 0 or self.budget_frames < self.sink_frames + self.recent_frames + frames + 1:
            raise ValueError(
                "FrameFold budget must fit sink, recent history, the current block and a history representative."
            )


def fit_value_merge(
    query: torch.Tensor,
    original_key: torch.Tensor,
    original_value: torch.Tensor,
    merged_key: torch.Tensor,
    left_value: torch.Tensor,
    right_value: torch.Tensor,
    reduce: Reduce,
) -> tuple[torch.Tensor, torch.Tensor]:
    # 候选使用完整 heads 的统计；分布式调用方归约拟合统计和失真。
    scale = query.shape[-1] ** -0.5
    original_weights = torch.einsum("qhd,ckhd->chqk", query.float(), original_key.float()).mul(scale).softmax(-1)
    response = torch.einsum("chqk,ckhd->cqhd", original_weights, original_value.float())
    weights = torch.einsum("qhd,ckhd->chqk", query.float(), merged_key.float()).mul(scale).softmax(-1)
    base = torch.einsum("chqk,ckhd->cqhd", weights, right_value.float())
    delta = torch.einsum("chqk,ckhd->cqhd", weights, left_value.float() - right_value.float())
    stats = reduce(torch.stack((((response - base) * delta).sum((1, 2, 3)), delta.square().sum((1, 2, 3))), -1))
    numerator, denominator = stats.unbind(-1)
    alpha = torch.where(denominator > 0, numerator / denominator.clamp_min(torch.finfo(stats.dtype).tiny), 0).clamp(
        0, 1
    )
    fitted = base + alpha[:, None, None, None] * delta
    distortion = reduce((response - fitted).norm(dim=-1).sum((1, 2)))
    if not torch.isfinite(alpha).all() or not torch.isfinite(distortion).all():
        raise RuntimeError("FrameFold fitting produced non-finite coefficients or distortions.")
    return alpha, distortion


class FrameFoldLayerCache:
    def __init__(
        self,
        config: FrameFoldConfig,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        height: int,
        width: int,
        rope: RoPE,
        slots: torch.Tensor | None = None,
    ) -> None:
        if height <= 0 or width <= 0 or key.ndim != 3 or key.shape != value.shape:
            raise ValueError("FrameFold requires positive spatial geometry and matching [tokens, heads, dim] K/V.")
        if not key.is_floating_point() or key.dtype != value.dtype or key.device != value.device:
            raise ValueError("FrameFold K/V must share a floating-point dtype and device.")
        capacity = config.budget_frames * height * width
        if (slots is None and key.shape[0] != capacity) or (slots is not None and slots.numel() != capacity):
            raise ValueError("FrameFold storage capacity does not match its frame budget.")
        self.config = config
        self.key, self.value, self.slots = key, value, slots
        self.height, self.width, self.rope = height, width, rope
        self.spans: list[int] = []
        self.absolute_end = 0
        self.query_history: torch.Tensor | None = None
        self.merge_count = 0
        self._pending_start: int | None = None
        self._pending_frames = 0
        self._plan: list[tuple[int, float, float]] = []

    @property
    def tokens_per_frame(self) -> int:
        return self.height * self.width

    def _pool(self, value: torch.Tensor) -> torch.Tensor:
        frames, _, heads, dim = value.shape
        pooled_height = math.ceil(self.height / self.config.pooling_factor)
        pooled_width = math.ceil(self.width / self.config.pooling_factor)
        grid = value.reshape(frames, self.height, self.width, heads * dim).permute(0, 3, 1, 2)
        pooled = F.adaptive_avg_pool2d(grid.float(), (pooled_height, pooled_width))
        return pooled.permute(0, 2, 3, 1).reshape(frames, -1, heads, dim)

    @staticmethod
    def _merge(value: torch.Tensor, index: int, coefficient: float) -> torch.Tensor:
        merged = (coefficient * value[index].float() + (1 - coefficient) * value[index + 1].float()).to(value.dtype)
        return torch.cat((value[:index], merged.unsqueeze(0), value[index + 2 :]), 0)

    def _select(
        self, key: torch.Tensor, value: torch.Tensor, spans: list[int], rotate: Rotate, reduce: Reduce
    ) -> tuple[int, float, float]:
        indices = list(range(self.config.sink_frames, len(spans) - self.config.recent_frames - 1))
        if not indices or self.query_history is None:
            raise RuntimeError("FrameFold has no eligible history pair or completed query probes.")
        pooled_key = self._pool(rotate(key, 0))
        pooled_value = self._pool(value)
        left = torch.tensor(indices, device=key.device)
        right = left + 1
        beta = [spans[index] / (spans[index] + spans[index + 1]) for index in indices]
        merged_key = torch.cat(
            [
                self._pool(
                    rotate(
                        (weight * key[index].float() + (1 - weight) * key[index + 1].float())
                        .to(key.dtype)
                        .unsqueeze(0),
                        index,
                    )
                )
                for index, weight in zip(indices, beta)
            ]
        )
        alpha, distortion = fit_value_merge(
            self.query_history.mean(0),
            torch.cat((pooled_key[left], pooled_key[right]), 1),
            torch.cat((pooled_value[left], pooled_value[right]), 1),
            merged_key,
            pooled_value[left],
            pooled_value[right],
            reduce,
        )
        selected = int(distortion.argmin().item())
        return indices[selected], beta[selected], float(alpha[selected].item())

    def prepare(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        current_start: int,
        rotate: Rotate,
        reduce: Reduce,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[int]]:
        if query.ndim != 3 or query.shape != key.shape or key.shape != value.shape:
            raise ValueError("FrameFold requires matching [tokens, heads, dim] Q/K/V.")
        if key.shape[1:] != self.key.shape[1:] or key.dtype != self.key.dtype or value.dtype != self.value.dtype:
            raise ValueError("FrameFold Q/K/V geometry and dtype must match the reserved storage.")
        if key.shape[0] % self.tokens_per_frame or current_start != self.absolute_end:
            raise ValueError("FrameFold blocks must be frame-aligned and contiguous in absolute token positions.")
        frames = key.shape[0] // self.tokens_per_frame
        self.config.validate_block(frames)
        if self._pending_start is not None and (current_start != self._pending_start or frames != self._pending_frames):
            raise RuntimeError("FrameFold cannot replace an uncommitted block.")
        count = len(self.spans) * self.tokens_per_frame
        history_key = self.key[:count] if self.slots is None else self.key.index_select(0, self.slots[:count])
        history_value = self.value[:count] if self.slots is None else self.value.index_select(0, self.slots[:count])
        history_key = history_key.unflatten(0, (len(self.spans), self.tokens_per_frame))
        history_value = history_value.unflatten(0, (len(self.spans), self.tokens_per_frame))
        spans = self.spans.copy()
        plan = self._plan.copy() if self._pending_start is not None else []
        for index, beta, alpha in plan:
            history_key = self._merge(history_key, index, beta)
            history_value = self._merge(history_value, index, alpha)
            spans[index : index + 2] = [spans[index] + spans[index + 1]]
        while len(spans) + frames > self.config.budget_frames:
            index, beta, alpha = self._select(history_key, history_value, spans, rotate, reduce)
            history_key = self._merge(history_key, index, beta)
            history_value = self._merge(history_value, index, alpha)
            spans[index : index + 2] = [spans[index] + spans[index + 1]]
            plan.append((index, beta, alpha))
        self._pending_start, self._pending_frames, self._plan = current_start, frames, plan
        query = rotate(query.unflatten(0, (frames, self.tokens_per_frame)), len(spans)).flatten(0, 1)
        next_key = torch.cat((history_key.flatten(0, 1), key), 0)
        next_value = torch.cat((history_value.flatten(0, 1), value), 0)
        encoded_key = rotate(next_key.unflatten(0, (len(spans) + frames, self.tokens_per_frame)), 0).flatten(0, 1)
        return query, encoded_key, next_key, next_value, spans + [1] * frames

    def commit(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, spans: list[int]) -> None:
        if self._pending_start is None:
            raise RuntimeError("FrameFold commit requires a prepared block.")
        if key.shape != value.shape or key.shape[0] != len(spans) * self.tokens_per_frame:
            raise ValueError("FrameFold commit must match the prepared spatial geometry.")
        probes = self._pool(query.unflatten(0, (self._pending_frames, self.tokens_per_frame)))
        if self.query_history is not None:
            probes = torch.cat((self.query_history, probes), 0)
        # 复制截断后的 Q，避免保留已经移出近期窗口的底层存储。
        probes = probes[-self.config.recent_frames :].clone()
        if self.slots is None:
            self.key[: key.shape[0]].copy_(key)
            self.value[: value.shape[0]].copy_(value)
        else:
            self.key.index_copy_(0, self.slots[: key.shape[0]], key)
            self.value.index_copy_(0, self.slots[: value.shape[0]], value)
        self.spans = spans
        self.absolute_end += self._pending_frames * self.tokens_per_frame
        self.query_history = probes
        self.merge_count += len(self._plan)
        self._pending_start, self._plan = None, []
