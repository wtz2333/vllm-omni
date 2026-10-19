# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""EdenLive's Wan 5B denoiser with runner-owned, unrotated self-attention KV.

Reference: VAST-AI-Research/EdenLive, commit 8f5d16e03910c1dd72632c96ccf9d44dc27670b8.
Wan's layers and checkpoint names are shared with Diffusers; EdenLive changes
the input channels, chunk attention and the position of cached keys.
"""

from dataclasses import dataclass
from types import SimpleNamespace

import torch
from diffusers.models.normalization import FP32LayerNorm
from diffusers.models.transformers.transformer_wan import (
    WanAttention,
    WanRotaryPosEmbed,
    WanTimeTextImageEmbedding,
    WanTransformerBlock,
)
from torch import nn

from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.experimental.ar_diffusion.kv_cache.state import ARDiffusionKVState

CHUNK_FRAMES = 3
WINDOW_FRAMES = 18
SINK_FRAMES = 6
KV_BRANCH = "main"


@dataclass(frozen=True)
class EdenLiveAttentionContext:
    state: ARDiffusionKVState
    write_slots: torch.Tensor
    read_slots: torch.Tensor
    query_start: int


def prepare_attention_context(
    state: ARDiffusionKVState,
    *,
    start_frame: int,
    frames: int,
    tokens_per_frame: int,
    commit: bool,
    device: torch.device,
) -> EdenLiveAttentionContext:
    """Select exact sink/recent rows, including when a frame straddles pages."""
    adapter = state.adapter(KV_BRANCH)
    if adapter.absolute_num_computed_tokens != start_frame * tokens_per_frame:
        raise ValueError("EdenLive latent position does not match the bound AR session")
    contexts = state.get_kv_caches(KV_BRANCH, seq_len=frames * tokens_per_frame, commit_current=commit)
    forward = contexts[0].forward_ctx
    forward.ensure_video_slots(device)
    query_start = min(start_frame, WINDOW_FRAMES - CHUNK_FRAMES)
    sink = min(start_frame, SINK_FRAMES) * tokens_per_frame
    recent = query_start * tokens_per_frame - sink
    end = adapter.num_computed_tokens
    positions = torch.cat((torch.arange(sink, device="cpu"), torch.arange(end - recent, end, device="cpu")))
    block_size = state.kv_cache.block_size
    indices, inverse = (positions // block_size).unique(sorted=True, return_inverse=True)
    block_ids = state.kv_cache.block_ids_at(adapter, indices.tolist())
    if state.kv_cache.null_block_id in block_ids:
        raise RuntimeError("EdenLive's fixed attention window contains an evicted KV page")
    physical = torch.tensor(block_ids, dtype=torch.long, device="cpu")[inverse]
    history_slots = (physical * block_size + positions % block_size).to(device)
    write_slots = forward.current_video_slot_mapping
    if write_slots is None:
        raise RuntimeError("AR-Diffusion did not allocate current KV slots")
    return EdenLiveAttentionContext(state, write_slots, torch.cat((history_slots, write_slots)), query_start)


def apply_rope(value: torch.Tensor, rotary: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    # EdenLive casts the loaded DiT, including RoPE buffers, to the runtime dtype.
    cosine, sine = rotary
    even, odd = value.unflatten(-1, (-1, 2)).unbind(-1)
    cosine, sine = cosine[..., 0::2], sine[..., 1::2]
    return torch.stack((even * cosine - odd * sine, even * sine + odd * cosine), dim=-1).flatten(-2).to(value.dtype)


def attend(
    layer: WanAttention,
    attention: Attention,
    hidden: torch.Tensor,
    encoder: torch.Tensor | None = None,
    *,
    context: EdenLiveAttentionContext | None = None,
    layer_index: int = 0,
    rotary: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    source = hidden if encoder is None else encoder
    query = layer.norm_q(layer.to_q(hidden)).unflatten(-1, (layer.heads, -1))
    key = layer.norm_k(layer.to_k(source)).unflatten(-1, (layer.heads, -1))
    value = layer.to_v(source).unflatten(-1, (layer.heads, -1))
    if context is not None:
        key_pool = context.state.kv_cache.key_cache(layer_index).flatten(0, 1)
        value_pool = context.state.kv_cache.value_cache(layer_index).flatten(0, 1)
        # Probes write scratch; only the separate clean forward gets managed pages.
        key_pool[context.write_slots] = key[0].to(key_pool.dtype)
        value_pool[context.write_slots] = value[0].to(value_pool.dtype)
        key = key_pool[context.read_slots].unsqueeze(0)
        value = value_pool[context.read_slots].unsqueeze(0)
    if rotary is not None:
        query = apply_rope(query, tuple(freq[:, -query.shape[1] :] for freq in rotary))
        key = apply_rope(key, rotary)
    result = attention(query, key, value).flatten(2, 3).to(query.dtype)
    return layer.to_out[1](layer.to_out[0](result))


class EdenLiveBlock(WanTransformerBlock):
    def __init__(self, dim: int, ffn_dim: int, num_heads: int, eps: float, cross_attn_norm: bool) -> None:
        super().__init__(dim, ffn_dim, num_heads, cross_attn_norm=cross_attn_norm, eps=eps)
        scale = (dim // num_heads) ** -0.5
        self.self_attention = Attention(
            num_heads, dim // num_heads, causal=False, softmax_scale=scale, role="self", qkv_layout="BSND"
        )
        self.cross_attention = Attention(
            num_heads,
            dim // num_heads,
            causal=False,
            softmax_scale=scale,
            role="cross",
            qkv_layout="BSND",
            computation="cross",
            cross_parallel_layout="local_q_replicated_kv",
        )

    def forward(self, hidden, encoder, temb, rotary, context, layer_index):
        # Preserve the reference's FP32 modulation, normalization and residual adds.
        shift, scale, gate, ff_shift, ff_scale, ff_gate = (self.scale_shift_table.unsqueeze(0) + temb.float()).unbind(2)
        normed = (self.norm1(hidden.float()) * (1 + scale) + shift).to(hidden.dtype)
        out = attend(self.attn1, self.self_attention, normed, context=context, layer_index=layer_index, rotary=rotary)
        hidden = (hidden.float() + out * gate).to(hidden.dtype)
        normed = self.norm2(hidden.float()).to(hidden.dtype)
        hidden = hidden + attend(self.attn2, self.cross_attention, normed, encoder)
        normed = (self.norm3(hidden.float()) * (1 + ff_scale) + ff_shift).to(hidden.dtype)
        return (hidden.float() + self.ffn(normed).float() * ff_gate).to(hidden.dtype)


class EdenLiveTransformer(nn.Module):
    """Single-device 5B architecture; control channels are not latent channels."""

    def __init__(self, config: dict, *, control_dim: int = 1024) -> None:
        super().__init__()
        self.config = SimpleNamespace(**{key: value for key, value in config.items() if not key.startswith("_")})
        cfg = self.config
        if tuple(cfg.patch_size) != (1, 2, 2) or cfg.in_channels != cfg.out_channels:
            raise ValueError("EdenLive 5B requires TI2V latents and patch_size=(1, 2, 2)")
        if getattr(cfg, "image_dim", None) is not None or getattr(cfg, "added_kv_proj_dim", None) is not None:
            raise ValueError("EdenLive 5B does not use CLIP image conditioning")
        self.control_dim = control_dim
        dim = cfg.num_attention_heads * cfg.attention_head_dim
        self.patch_embedding = nn.Conv3d(cfg.in_channels + control_dim, dim, cfg.patch_size, stride=cfg.patch_size)
        self.rope = WanRotaryPosEmbed(cfg.attention_head_dim, tuple(cfg.patch_size), cfg.rope_max_seq_len)
        self.condition_embedder = WanTimeTextImageEmbedding(dim, cfg.freq_dim, dim * 6, cfg.text_dim)
        self.blocks = nn.ModuleList(
            EdenLiveBlock(dim, cfg.ffn_dim, cfg.num_attention_heads, cfg.eps, cfg.cross_attn_norm)
            for _ in range(cfg.num_layers)
        )
        self.norm_out = FP32LayerNorm(dim, cfg.eps, elementwise_affine=False)
        self.scale_shift_table = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)
        self.proj_out = nn.Linear(dim, cfg.out_channels * 4)

    @property
    def dtype(self) -> torch.dtype:
        return self.patch_embedding.weight.dtype

    def forward(
        self,
        latents: torch.Tensor,
        control: torch.Tensor,
        timestep: torch.Tensor,
        prompt_embeds: torch.Tensor,
        *,
        state: ARDiffusionKVState,
        start_frame: int,
        commit: bool = False,
    ) -> torch.Tensor:
        batch, channels, frames, height, width = latents.shape
        if batch != 1 or channels != self.config.out_channels or not 1 <= frames <= CHUNK_FRAMES:
            raise ValueError("EdenLive expects one sample with 1-3 latent frames per chunk")
        if control.shape != (batch, self.control_dim, frames, height, width):
            raise ValueError("EdenLive proxy features must match the current latent chunk")
        tokens_per_frame = (height // 2) * (width // 2)
        context = prepare_attention_context(
            state,
            start_frame=start_frame,
            frames=frames,
            tokens_per_frame=tokens_per_frame,
            commit=commit,
            device=latents.device,
        )
        # expand() supplies geometry only; no window-sized latent allocation.
        geometry = latents[:, :1, :1].expand(1, 1, context.query_start + frames, height, width)
        rotary = self.rope(geometry)
        hidden = self.patch_embedding(torch.cat((latents, control), dim=1).to(self.dtype)).flatten(2).transpose(1, 2)
        timestep_seq_len = None
        if timestep.ndim == 2:
            timestep = timestep.expand(batch, frames).repeat_interleave(tokens_per_frame, dim=1)
            timestep_seq_len = frames * tokens_per_frame
        temb, projection, encoder, _ = self.condition_embedder(
            timestep.flatten(), prompt_embeds, timestep_seq_len=timestep_seq_len
        )
        projection = projection.unflatten(-1, (6, -1))
        if timestep_seq_len is None:
            # Preserve the reference's scalar projection GEMM before broadcasting.
            temb, projection = temb.unsqueeze(1), projection.unsqueeze(1)
        for index, block in enumerate(self.blocks):
            hidden = block(hidden, encoder, projection, rotary, context, index)
        shift, scale = (self.scale_shift_table.unsqueeze(0) + temb.unsqueeze(2)).unbind(2)
        hidden = (self.norm_out(hidden.float()) * (1 + scale) + shift).to(hidden.dtype)
        hidden = self.proj_out(hidden).reshape(batch, frames, height // 2, width // 2, 1, 2, 2, -1)
        output = hidden.permute(0, 7, 1, 4, 2, 5, 3, 6).flatten(6, 7).flatten(4, 5).flatten(2, 3)
        if commit:
            state.commit_paged_context(KV_BRANCH)
        return output
