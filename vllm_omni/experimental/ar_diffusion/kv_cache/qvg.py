# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import torch

from vllm_omni.experimental.ar_diffusion.kv_cache.config import QVGConfig
from vllm_omni.experimental.ar_diffusion.kv_cache.paged import visible_window_blocks
from vllm_omni.experimental.ar_diffusion.kv_cache.paged_attention import _to_device_async
from vllm_omni.platforms import current_omni_platform


@dataclass(frozen=True)
class QVGTensor:
    centroids: list[torch.Tensor]
    assignments: list[torch.Tensor]
    residual: torch.Tensor
    scales: torch.Tensor

    @property
    def nbytes(self) -> int:
        return sum(
            t.numel() * t.element_size() for t in (*self.centroids, *self.assignments, self.residual, self.scales)
        )


class QVGCodec:
    def __init__(self, config: QVGConfig) -> None:
        from quant_videogen.kmeans.kmeans_euclid import _euclid_iter
        from quant_videogen.real.accumulate import nstage_accum
        from quant_videogen.real.quant_pack import quant_pack
        from quant_videogen.real.smooth import minus_centroid

        self.config = config
        self._iterate = _euclid_iter
        self._kmeans = self._cluster
        self._subtract = minus_centroid
        self._pack = quant_pack
        self._unpack = nstage_accum

    def _centroids_converged(self, previous: torch.Tensor, current: torch.Tensor, sizes: torch.Tensor) -> bool:
        weights = sizes.float().unsqueeze(-1)
        movement = ((current.float() - previous.float()).square() * weights).sum(dim=(1, 2))
        magnitude = (previous.float().square() * weights).sum(dim=(1, 2)).clamp_min(1e-12)
        return bool(torch.all(movement <= magnitude * self.config.kmeans_tolerance**2))

    def _cluster(
        self, tensor: torch.Tensor, n_clusters: int, max_iters: int, init_centroids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        if max_iters <= 0 or init_centroids.shape[1] != n_clusters:
            raise ValueError("QVG clustering requires positive iterations and matching centroid count")
        config = self.config
        centroids = init_centroids
        squared_norm = tensor.square().sum(dim=-1)
        for index in range(max_iters):
            updated, ids, sizes = self._iterate(tensor, squared_norm, centroids)
            iteration = index + 1
            check = (
                config.kmeans_tolerance > 0
                and iteration >= config.kmeans_min_iters
                and iteration < max_iters
                and iteration % config.kmeans_check_interval == 0
            )
            converged = check and self._centroids_converged(centroids, updated, sizes)
            centroids = updated
            if converged:
                break
        return ids, centroids, sizes, iteration

    def encode(self, tensor: torch.Tensor) -> QVGTensor:
        tokens, heads, head_size = tensor.shape
        config = self.config
        if head_size % config.group_size or tokens < config.num_centroids:
            raise ValueError("QVG requires head_size divisible by group_size and at least num_centroids tokens")
        residual = tensor.transpose(0, 1).contiguous()
        generator = torch.Generator(device=tensor.device).manual_seed(config.seed)
        centroids_list, assignments = [], []
        for _ in range(config.num_stages):
            # 独立生成初始质心，避免修改视频采样使用的全局 RNG。
            indices = torch.randint(tokens, (heads, config.num_centroids), device=tensor.device, generator=generator)
            initial = residual.gather(1, indices.unsqueeze(-1).expand(-1, -1, head_size))
            ids, centroids, _, _ = self._kmeans(
                residual,
                n_clusters=config.num_centroids,
                max_iters=config.kmeans_max_iters,
                init_centroids=initial,
            )
            ids = ids.to(torch.uint8).reshape(1, heads, tokens)
            centroids = centroids.reshape(1, heads, config.num_centroids, head_size)
            centroids_list.append(centroids)
            assignments.append(ids)
            residual = self._subtract(residual.unsqueeze(0), ids, centroids).squeeze(0)
        packed, scales, _ = self._pack(
            residual.unsqueeze(0),
            block_size=config.group_size,
            num_bits=config.bits,
            scale_precision=torch.float8_e4m3fn,
            pack_output_int8=True,
            asymmetric=False,
        )
        return QVGTensor(centroids_list, assignments, packed, scales)

    def decode(self, tensor: QVGTensor, dtype: torch.dtype) -> torch.Tensor:
        return self._unpack(
            centroids_list=tensor.centroids,
            cluster_ids_list=tensor.assignments,
            residual_quant=tensor.residual,
            scales=tensor.scales,
            block_size=self.config.group_size,
            num_bits=self.config.bits,
            PACK_INPUT_INT8=True,
            CLUSTER_ID_INT8=True,
            output_dtype=dtype,
        )[0].transpose(0, 1)


KVPair = tuple[torch.Tensor, torch.Tensor] | tuple[QVGTensor, QVGTensor]


@dataclass(frozen=True)
class _Chunk:
    start: int
    length: int
    layers: tuple[KVPair, ...]

    @property
    def end(self) -> int:
        return self.start + self.length


@dataclass
class QVGForwardContext:
    history: QVGHistory
    seq_len: int
    commit_current: bool
    block_table: torch.Tensor
    query_start_loc: torch.Tensor
    seq_lens: torch.Tensor
    max_seq_len: int
    layers: dict[int, tuple[torch.Tensor, torch.Tensor]] = field(default_factory=dict)
    visited: set[int] = field(default_factory=set)


@dataclass(frozen=True)
class QVGLayerContext:
    layer_idx: int
    forward_ctx: QVGForwardContext

    def read(self, current_start: int, key: torch.Tensor, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.forward_ctx.history.materialize(self, current_start, key, value)

    def token_ranges(self) -> tuple[tuple[int, int], ...]:
        history = self.forward_ctx.history
        if history._active is not self.forward_ctx:
            raise RuntimeError("QVG position metadata belongs to a retired forward")
        return (
            *history._ranges(history._end, self.forward_ctx.seq_len),
            (history._end, history._end + self.forward_ctx.seq_len),
        )


class QVGHistory:
    def __init__(
        self,
        codec: QVGCodec,
        *,
        num_layers: int,
        tokens_per_frame: int,
        window_tokens: int,
        sink_tokens: int,
        workspace: tuple[torch.Tensor, torch.Tensor],
        block_size: int = 16,
        agree: Callable[[bool], bool] = bool,
    ) -> None:
        self.codec = codec
        self.num_layers = num_layers
        self.tokens_per_frame = tokens_per_frame
        self.window_tokens = window_tokens
        self.sink_tokens = sink_tokens
        self.workspace = workspace
        self.block_size = block_size
        self._attention_blocks = (
            visible_window_blocks(
                chunk_size=tokens_per_frame, block_size=block_size, sink_tokens=sink_tokens, window_tokens=window_tokens
            )
            + 1
        )
        self.agree = agree
        self._chunks: tuple[_Chunk, ...] = ()
        self._end = 0
        self._active: QVGForwardContext | None = None
        self._closed = False

    @property
    def nbytes(self) -> int:
        return sum(
            tensor.nbytes if isinstance(tensor, QVGTensor) else tensor.numel() * tensor.element_size()
            for chunk in self._chunks
            for pair in chunk.layers
            for tensor in pair
        )

    def _ranges(self, end: int, current_tokens: int = 0) -> tuple[tuple[int, int], ...]:
        prefix_end = min(end, self.sink_tokens)
        tail_start = max(prefix_end, end + current_tokens - self.window_tokens)
        return tuple((start, stop) for start, stop in ((0, prefix_end), (tail_start, end)) if start < stop)

    def prepare(self, seq_len: int, commit_current: bool) -> list[QVGLayerContext]:
        if self._closed:
            raise RuntimeError("QVG history is closed")
        if self._active is not None and self._active.commit_current:
            raise RuntimeError("QVG clean history must be committed before preparing another forward")
        if seq_len <= 0 or seq_len % self.tokens_per_frame:
            raise ValueError("QVG current tokens must contain a positive whole number of latent frames")
        if seq_len > self.window_tokens:
            raise ValueError("QVG recent window must hold the entire current chunk")
        visible_tokens = sum(stop - start for start, stop in self._ranges(self._end, seq_len)) + seq_len
        pages = -(-visible_tokens // self.block_size)
        device = self.workspace[0].device
        block_table = _to_device_async(
            torch.tensor([[*range(pages), *([0] * (self._attention_blocks - pages))]], dtype=torch.int32), device
        )
        query_start_loc = _to_device_async(torch.tensor([0, seq_len], dtype=torch.int32), device)
        seq_lens = _to_device_async(torch.tensor([visible_tokens], dtype=torch.int32), device)
        self._active = QVGForwardContext(
            self,
            seq_len,
            commit_current,
            block_table,
            query_start_loc,
            seq_lens,
            self._attention_blocks * self.block_size,
        )
        return [QVGLayerContext(index, self._active) for index in range(self.num_layers)]

    def materialize(
        self,
        context: QVGLayerContext,
        current_start: int,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        forward = context.forward_ctx
        if self._closed or forward is not self._active:
            raise RuntimeError("QVG attention received a retired forward context")
        if current_start != self._end:
            raise ValueError(f"QVG expected absolute token offset {self._end}, got {current_start}")
        if not 0 <= context.layer_idx < self.num_layers or context.layer_idx in forward.visited:
            raise RuntimeError("QVG requires exactly one attention read per layer and forward")
        k_buffer, v_buffer = self.workspace
        expected = (1, forward.seq_len, *k_buffer.shape[2:])
        for tensor in (key, value):
            if tuple(tensor.shape) != expected or tensor.dtype != k_buffer.dtype or tensor.device != k_buffer.device:
                raise ValueError(f"QVG current K/V must have shape {expected}, dtype/device matching its workspace")
        offset = 0
        for start, stop in self._ranges(self._end, forward.seq_len):
            copied = 0
            for chunk in self._chunks:
                left, right = max(start, chunk.start), min(stop, chunk.end)
                if left >= right:
                    continue
                pair = chunk.layers[context.layer_idx]
                for buffer, data in zip((k_buffer, v_buffer), pair):
                    decoded = self.codec.decode(data, buffer.dtype) if isinstance(data, QVGTensor) else data
                    buffer[0, offset : offset + right - left].copy_(decoded[left - chunk.start : right - chunk.start])
                offset += right - left
                copied += right - left
            if copied != stop - start:
                raise RuntimeError("QVG history has missing committed rows")
        end = offset + forward.seq_len
        if end > k_buffer.shape[1]:
            raise RuntimeError("QVG attention exceeds its admitted workspace capacity")
        k_buffer[:, offset:end].copy_(key)
        v_buffer[:, offset:end].copy_(value)
        if forward.commit_current:
            # 当前 clean chunk 必须独立保存，不能引用下一层会覆盖的工作区或 QKV 输出。
            forward.layers[context.layer_idx] = (key[0].clone(), value[0].clone())
        forward.visited.add(context.layer_idx)
        return k_buffer[:, :end], v_buffer[:, :end]

    def commit(self) -> int:
        forward = self._active
        if forward is None or not forward.commit_current:
            return 0
        if len(forward.layers) != self.num_layers:
            raise RuntimeError("QVG cannot commit a chunk with missing layers")
        new_end = self._end + forward.seq_len
        current = _Chunk(self._end, forward.seq_len, tuple(forward.layers[index] for index in range(self.num_layers)))
        retained = self._ranges(new_end)
        chunks = tuple(
            chunk
            for chunk in (*self._chunks, current)
            if any(chunk.start < stop and chunk.end > start for start, stop in retained)
        )
        replacements = []
        error: Exception | None = None
        try:
            for chunk in chunks:
                protected = chunk.start < self.sink_tokens
                recent = chunk.end > new_end - self.codec.config.keep_recent_chunks * forward.seq_len
                if protected or recent or isinstance(chunk.layers[0][0], QVGTensor):
                    replacements.append(chunk)
                    continue
                layers = []
                for key, value in chunk.layers:
                    if not isinstance(key, torch.Tensor) or not isinstance(value, torch.Tensor):
                        raise TypeError("QVG chunk contains inconsistent K/V representations")
                    layers.append((self.codec.encode(key), self.codec.encode(value)))
                replacements.append(_Chunk(chunk.start, chunk.length, tuple(layers)))
            if self.workspace[0].device.type != "cpu":
                current_omni_platform.synchronize()
        except Exception as exc:
            # 先让所有 SP rank 得到提交失败的判定，再抛出本地错误。
            error = exc
        if not self.agree(error is None):
            forward.layers.clear()
            self._active = None
            raise RuntimeError("QVG history compression failed on a worker; previous history was retained") from error
        if error is not None:
            forward.layers.clear()
            self._active = None
            raise error
        self._chunks = tuple(replacements)
        self._end = new_end
        forward.layers.clear()
        self._active = None
        return forward.seq_len

    def close(self) -> None:
        self._chunks = ()
        if self._active is not None:
            self._active.layers.clear()
        self._active = None
        self._closed = True
