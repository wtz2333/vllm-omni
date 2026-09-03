# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Echo-WM audio-video world-model transformer.

Port of the Echo-WM reference ``LTXModel`` (an LTX-2.3 audio-video DiT with a
pure-UCPE camera branch on every block) onto vLLM-Omni primitives:

* tensor parallelism via ``QKVParallelLinear``/``ColumnParallelLinear``/
  ``RowParallelLinear`` with global-statistics QK RMS norms;
* Ulysses sequence parallelism expressed *inside* the windowed attentions:
  projections run on the rank-local token shard, a head/sequence all-to-all
  produces the shared full-token window with per-rank head shards, and the
  bounded sink+FIFO KV caches store that post-all-to-all form. The declarative
  ``_sp_plan`` hook framework assumes a single token stream, which the joint
  audio/video rollout (two streams plus per-attention caches) does not match;
* the causal bounded-cache semantics of the reference rollout: transactional
  in-place block updates, un-roped cached K/V re-roped against a fixed window
  template every forward, and bounded-anchor-translation UCPE.

Numerics are kept line-for-line against the reference implementation; the CPU
tests pin the correspondence.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from vllm.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import ColumnParallelLinear, RowParallelLinear
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from vllm_omni.diffusion.distributed.comm import SeqAllToAll4D
from vllm_omni.diffusion.distributed.parallel_state import (
    get_sp_group,
    get_ulysses_parallel_rank,
    get_ulysses_parallel_world_size,
)
from vllm_omni.diffusion.models.ltx2.ltx2_transformer import (
    LTX2AdaLayerNormSingle,
    LTX2Attention,
    LTX2AudioVideoAttnProcessor,
    LTX2FeedForward,
    apply_split_rotary_emb,
)

from .causal_cache import (
    EchoWMCacheConfig,
    EchoWMKVWindow,
    EchoWMLayerCaches,
    EchoWMTextKV,
)
from .ucpe import PropeDotProductAttention, active_sink_fifo_indices, prepare_apply_fns, rebase_viewmat_translation

logger = init_logger(__name__)


def rms_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Affine-free RMS norm over the last dimension (reference ``rms_norm``)."""
    return torch.nn.functional.rms_norm(x, (x.shape[-1],), weight=None, eps=eps)


def _linear_out(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Call a parallel linear and unwrap the (output, bias) tuple it may return."""
    out = module(x)
    return out[0] if isinstance(out, tuple) else out


# ---------------------------------------------------------------------------
# Ulysses collectives for the windowed audio/video attentions.
# ---------------------------------------------------------------------------


def _contiguous_split_lengths(total: int, world: int) -> list[int]:
    """``tensor.chunk`` semantics: earlier ranks take the remainder."""
    base, rem = divmod(total, world)
    return [base + (1 if i < rem else 0) for i in range(world)]


@dataclass(frozen=True)
class _UlyssesState:
    world_size: int
    rank: int
    group: Any
    tp_size: int
    tp_rank: int

    @property
    def active(self) -> bool:
        return self.world_size > 1


def _ulysses() -> _UlyssesState:
    """Resolve the Ulysses group; degrades to a no-op when uninitialized.

    Outside the diffusion engine (unit tests, single-process construction) the
    SP coordinator is not initialized: sequence parallelism is then inactive by
    definition. Under the engine the groups exist before model load, so the
    forward path always observes the configured world size.
    """
    try:
        sp = get_sp_group()
        return _UlyssesState(
            world_size=get_ulysses_parallel_world_size(),
            rank=get_ulysses_parallel_rank(),
            group=sp.ulysses_group,
            tp_size=get_tensor_model_parallel_world_size(),
            tp_rank=get_tensor_model_parallel_rank(),
        )
    except Exception:
        return _UlyssesState(
            world_size=1,
            rank=0,
            group=None,
            tp_size=get_tensor_model_parallel_world_size(),
            tp_rank=get_tensor_model_parallel_rank(),
        )


def _rank_head_slice(uly: _UlyssesState, total_heads: int) -> tuple[int, int]:
    """Global head range this rank owns inside an attention with ``total_heads`` heads."""
    tp_local = total_heads // uly.tp_size
    heads = tp_local // uly.world_size
    start = uly.tp_rank * tp_local + uly.rank * heads
    return start, heads


def _pad_local_tokens(x: torch.Tensor, uly: _UlyssesState, global_len: int) -> torch.Tensor:
    """Pad the rank-local token shard so every rank holds the same chunk length."""
    if not uly.active:
        return x
    lens = _contiguous_split_lengths(global_len, uly.world_size)
    padded = max(lens)
    if x.shape[1] < padded:
        x = F.pad(x, (0, 0, 0, 0, 0, padded - x.shape[1]))
    elif x.shape[1] > padded:
        raise ValueError(f"local shard {x.shape[1]} exceeds the padded chunk length {padded}")
    return x


def _strip_global_tokens(x: torch.Tensor, uly: _UlyssesState, global_len: int) -> torch.Tensor:
    """Remove the per-shard padding columns introduced by ``_pad_local_tokens``."""
    if not uly.active:
        return x
    lens = _contiguous_split_lengths(global_len, uly.world_size)
    padded = max(lens)
    if x.shape[1] != padded * uly.world_size:
        raise ValueError(f"expected {padded * uly.world_size} padded global tokens, got {x.shape[1]}")
    chunks = []
    offset = 0
    for length in lens:
        chunks.append(x[:, offset : offset + length])
        offset += padded
    return torch.cat(chunks, dim=1)


def _a2a_to_window(x: torch.Tensor, uly: _UlyssesState, global_len: int) -> torch.Tensor:
    """(B, S_local, H_tp, D) -> (B, S_full, H_uly, D): enter the shared window form."""
    if not uly.active:
        return x
    padded = _pad_local_tokens(x, uly, global_len)
    out = SeqAllToAll4D.apply(uly.group, padded.contiguous(), 2, 1, False)
    return _strip_global_tokens(out, uly, global_len)


def _a2a_to_local(x: torch.Tensor, uly: _UlyssesState, global_len: int) -> torch.Tensor:
    """Inverse of :func:`_a2a_to_window`: (B, S_full, H_uly, D) -> (B, S_local, H_tp, D)."""
    if not uly.active:
        return x
    padded_out = _pad_local_tokens(x, uly, global_len)
    local = SeqAllToAll4D.apply(uly.group, padded_out.contiguous(), 1, 2, False)
    lens = _contiguous_split_lengths(global_len, uly.world_size)
    return local[:, : lens[uly.rank]]


def _slice_rope_heads(
    rope: tuple[torch.Tensor, torch.Tensor] | None,
    uly: _UlyssesState,
    total_heads: int,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Select this rank's (TP x SP) head rows from a full-head RoPE template."""
    if rope is None:
        return None
    cos, sin = rope
    if cos.ndim != 4 or cos.shape[1] != total_heads:
        raise ValueError(f"rope template must be (B, {total_heads}, T, D/2), got {tuple(cos.shape)}")
    start, heads = _rank_head_slice(uly, total_heads)
    return cos[:, start : start + heads], sin[:, start : start + heads]


def _slice_rope_rows(
    rope: tuple[torch.Tensor, torch.Tensor],
    start: int,
    end: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos, sin = rope
    return cos[:, :, start:end, :], sin[:, :, start:end, :]


# ---------------------------------------------------------------------------
# Causal attention processor with windowed KV caches and manual Ulysses.
# ---------------------------------------------------------------------------


class EchoWMCausalAttnProcessor(LTX2AudioVideoAttnProcessor):
    """Attention math for Echo-WM's windowed causal rollout.

    Extends the LTX-2 processor with three Echo-WM behaviours:

    * ``kv_cache``: transactional window update of un-roped K/V, then RoPE
      applied against the fixed window template (query at the window tail for
      self-attention; via caller-provided slot maps for a2v/v2a).
    * ``crossattn_cache``: init-once text K/V reuse.
    * Ulysses: an all-to-all converts the rank-local token shard into the
      shared full-token window with per-rank head shards around the cache and
      RoPE; the output is converted back before gating/``to_out``.
    """

    def __call__(  # noqa: PLR0912, PLR0913
        self,
        attn: LTX2Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        query_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        key_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        perturbation_mask: torch.Tensor | None = None,
        kv_cache: EchoWMKVWindow | None = None,
        kv_cache_start: int = 0,
        crossattn_cache: EchoWMTextKV | None = None,
        q_rope: tuple[torch.Tensor, torch.Tensor] | None = None,
        k_rope: tuple[torch.Tensor, torch.Tensor] | None = None,
        q_rope_slice: tuple[int, int] | None = None,
        global_query_len: int | None = None,
    ) -> torch.Tensor:
        if attention_mask is not None:
            raise NotImplementedError(
                "Echo-WM connectors replace text padding with learnable registers, so the "
                "context mask is always fully attendable; pass attention_mask=None"
            )
        if query_rotary_emb is not None or key_rotary_emb is not None:
            raise ValueError("pass rope via the q_rope/k_rope window templates")
        is_self_attention = encoder_hidden_states is None
        uly = _ulysses()

        gate_logits = None
        if attn.to_gate_logits is not None:
            gate_logits = attn.to_gate_logits(hidden_states)
            if isinstance(gate_logits, tuple):
                gate_logits = gate_logits[0]

        if is_self_attention:
            encoder_hidden_states = hidden_states

        query, key, value = self._project_qkv(
            attn=attn,
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            is_self_attention=is_self_attention,
        )
        query = attn.norm_q(query).to(dtype=value.dtype)
        key = attn.norm_k(key).to(dtype=value.dtype)

        total_heads = attn.total_num_heads
        heads, head_dim = attn.query_num_heads, attn.head_dim
        query = query.unflatten(2, (heads, head_dim))
        key = key.unflatten(2, (heads, head_dim))
        value = value.unflatten(2, (heads, head_dim))

        windowed = kv_cache is not None
        if windowed and uly.active:
            if global_query_len is None:
                raise ValueError("global_query_len is required for windowed attention under sequence parallelism")
            query = _a2a_to_window(query, uly, global_query_len)
            key = _a2a_to_window(key, uly, global_query_len)
            value = _a2a_to_window(value, uly, global_query_len)

        if windowed:
            key, value = kv_cache.update(kv_cache_start, key, value)
            active = key.shape[1]
            q_len = query.shape[1]
            q_slice = q_rope_slice if q_rope_slice is not None else (active - q_len, active)
            k_slice = (0, active)
        elif crossattn_cache is not None:
            key, value = crossattn_cache.get(key, value)
            q_slice = k_slice = None
        else:
            q_slice = k_slice = None

        local_q_rope = _slice_rope_heads(q_rope, uly, total_heads)
        if local_q_rope is not None:
            if k_rope is None:
                raise ValueError("k_rope template is required when q_rope is given")
            if attn.rope_type != "split":
                raise NotImplementedError(f"Echo-WM requires split rope, got {attn.rope_type}")
            local_k_rope = _slice_rope_heads(k_rope, uly, total_heads)
            # apply_split_rotary_emb takes (B, S, H*D) and reshapes internally.
            flat_query = query.flatten(2, 3)
            flat_key = key.flatten(2, 3)
            if windowed:
                q_rows = _slice_rope_rows(local_q_rope, *q_slice)
                k_rows = _slice_rope_rows(local_k_rope, *k_slice)
                flat_query = apply_split_rotary_emb(flat_query, q_rows, head_dim=head_dim)
                flat_key = apply_split_rotary_emb(flat_key, k_rows, head_dim=head_dim)
            else:
                flat_query = apply_split_rotary_emb(flat_query, local_q_rope, head_dim=head_dim)
                flat_key = apply_split_rotary_emb(flat_key, local_k_rope, head_dim=head_dim)
            query = flat_query.unflatten(2, (heads, head_dim))
            key = flat_key.unflatten(2, (heads, head_dim))

        out = attn.attn(query, key, value, None)
        if windowed and uly.active:
            out = _a2a_to_local(out, uly, global_query_len)

        out = out.flatten(2, 3).to(query.dtype)
        if perturbation_mask is not None:
            out = out * perturbation_mask + value.flatten(2, 3) * (1 - perturbation_mask)

        if gate_logits is not None:
            out = out.unflatten(2, (heads, head_dim))
            gates = 2.0 * torch.sigmoid(gate_logits)
            out = out * gates.unsqueeze(-1)
            out = out.flatten(2, 3)

        hidden_out = attn.to_out[0](out)
        if isinstance(hidden_out, tuple):
            hidden_out = hidden_out[0]
        return attn.to_out[1](hidden_out)


# ---------------------------------------------------------------------------
# Pure-UCPE camera branch
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EchoWMUCPEConfig:
    """Configuration of the pure-UCPE camera branch (one per model)."""

    enabled: bool = True
    attn_dim: int = 1024
    num_heads: int = 8
    patches_x: int = 40
    patches_y: int = 22
    image_width: int = 1280
    image_height: int = 704
    freq_base: float = 100.0


class EchoWMUCPEBranch(nn.Module):
    """Pure-UCPE branch added to the video self-attention residual."""

    def __init__(
        self,
        video_dim: int,
        config: EchoWMUCPEConfig,
        *,
        quant_config: Any = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.num_heads = config.num_heads
        self.head_dim = config.attn_dim // config.num_heads
        if config.attn_dim % config.num_heads != 0 or self.head_dim % 4 != 0:
            raise ValueError("UCPE attention dimension must be divisible by heads and by 4")
        self.ucpe_q_proj = ColumnParallelLinear(
            video_dim,
            config.attn_dim,
            bias=False,
            gather_output=False,
            return_bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.ucpe_q_proj" if prefix else "ucpe_q_proj",
        )
        self.ucpe_k_proj = ColumnParallelLinear(
            video_dim,
            config.attn_dim,
            bias=False,
            gather_output=False,
            return_bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.ucpe_k_proj" if prefix else "ucpe_k_proj",
        )
        self.ucpe_v_proj = ColumnParallelLinear(
            video_dim,
            config.attn_dim,
            bias=False,
            gather_output=False,
            return_bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.ucpe_v_proj" if prefix else "ucpe_v_proj",
        )
        self.ucpe_out_proj = RowParallelLinear(
            config.attn_dim,
            video_dim,
            bias=True,
            input_is_parallel=True,
            return_bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.ucpe_out_proj" if prefix else "ucpe_out_proj",
        )
        self.ucpe_prope = PropeDotProductAttention(
            head_dim=self.head_dim,
            patches_x=config.patches_x,
            patches_y=config.patches_y,
            image_width=config.image_width,
            image_height=config.image_height,
            freq_base=config.freq_base,
        )

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        # ColumnParallelLinear shards the flat feature rows, which are
        # head-major, so the local slice is a contiguous run of local heads.
        # Output is (B, S, H, D) — the cache layout; PRoPE transposes.
        b, s, _ = x.shape
        return x.view(b, s, -1, self.head_dim)

    def forward(
        self,
        norm_vx: torch.Tensor,
        viewmats: torch.Tensor,
        Ks: torch.Tensor,  # noqa: N803
        *,
        kv_cache: EchoWMKVWindow | None,
        kv_cache_start: int,
        patches_per_frame: int,
        full_viewmats: torch.Tensor | None = None,
        full_Ks: torch.Tensor | None = None,  # noqa: N803
        global_query_len: int | None = None,
    ) -> torch.Tensor:
        uly = _ulysses()
        q = self._split_heads(_linear_out(self.ucpe_q_proj, norm_vx))
        k = self._split_heads(_linear_out(self.ucpe_k_proj, norm_vx))
        v = self._split_heads(_linear_out(self.ucpe_v_proj, norm_vx))
        if uly.active:
            if global_query_len is None:
                raise ValueError("global_query_len is required for UCPE under sequence parallelism")
            q = _a2a_to_window(q, uly, global_query_len)
            k = _a2a_to_window(k, uly, global_query_len)
            v = _a2a_to_window(v, uly, global_query_len)

        if full_viewmats is None or full_Ks is None or kv_cache is None:
            raise ValueError("the causal UCPE branch requires bounded caches and full camera arrays")
        # Bounded anchor translation: cache raw (pre-PRoPE) K/V, then apply
        # PRoPE to the active sink+FIFO window rebased at a common anchor.
        k, v = kv_cache.update(kv_cache_start, k, v)
        # PRoPE operates on (B, heads, seqlen, head_dim).
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        ppf = patches_per_frame
        current_start = kv_cache_start // ppf
        current_end = current_start + q.shape[2] // ppf
        indices, anchor_index = active_sink_fifo_indices(
            current_end,
            kv_cache.local_attn_size // ppf,
            kv_cache.sink_tokens // ppf,
            full_viewmats.device,
        )
        anchor = full_viewmats[:, anchor_index : anchor_index + 1]
        q_viewmats = rebase_viewmat_translation(full_viewmats[:, current_start:current_end], anchor)
        k_viewmats = rebase_viewmat_translation(full_viewmats.index_select(1, indices), anchor)
        kwargs = dict(
            head_dim=self.ucpe_prope.head_dim,
            patches_x=self.ucpe_prope.patches_x,
            patches_y=self.ucpe_prope.patches_y,
            image_width=self.ucpe_prope.image_width,
            image_height=self.ucpe_prope.image_height,
            coeffs_x=None
            if self.ucpe_prope.coeffs_x_0 is None
            else (
                self.ucpe_prope.coeffs_x_0,
                self.ucpe_prope.coeffs_x_1,
            ),
            coeffs_y=None
            if self.ucpe_prope.coeffs_y_0 is None
            else (
                self.ucpe_prope.coeffs_y_0,
                self.ucpe_prope.coeffs_y_1,
            ),
        )
        apply_q, _, apply_out = prepare_apply_fns(
            viewmats=q_viewmats, Ks=full_Ks[:, current_start:current_end].float(), **kwargs
        )
        _, apply_kv, _ = prepare_apply_fns(viewmats=k_viewmats, Ks=full_Ks.index_select(1, indices).float(), **kwargs)
        q = self.ucpe_prope.transform(apply_q, q)
        k = self.ucpe_prope.transform(apply_kv, k)
        v = self.ucpe_prope.transform(apply_kv, v)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        out = self.ucpe_prope.transform(apply_out, out).transpose(1, 2)  # back to (B, S, H, D)

        if uly.active:
            out = _a2a_to_local(out, uly, global_query_len)
        out = out.to(dtype=self.ucpe_out_proj.weight.dtype)
        b, s, _, _ = out.shape
        return _linear_out(self.ucpe_out_proj, out.reshape(b, s, -1))


# ---------------------------------------------------------------------------
# Transformer block
# ---------------------------------------------------------------------------


class EchoWMTransformerBlock(nn.Module):
    """One audio-video block: video self(+ucpe), video text, audio self, audio
    text, a2v, v2a, then the two feed-forwards — in the reference order.

    The reference interleaves video-text and audio-self in the opposite order;
    the two operate on disjoint tensors, so the result is identical.
    """

    def __init__(  # noqa: PLR0913
        self,
        idx: int,
        num_layers: int,
        video_dim: int,
        video_heads: int,
        video_head_dim: int,
        cross_attention_dim: int,
        audio_dim: int,
        audio_heads: int,
        audio_head_dim: int,
        audio_cross_attention_dim: int,
        norm_eps: float,
        apply_gated_attention: bool,
        cross_attention_adaln: bool,
        ucpe: EchoWMUCPEConfig | None,
        quant_config: Any = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.idx = idx
        self.num_layers = num_layers
        self.norm_eps = norm_eps
        self.cross_attention_adaln = cross_attention_adaln
        shared = dict(
            norm_eps=norm_eps,
            rope_type="split",
            apply_gated_attention=apply_gated_attention,
            quant_config=quant_config,
        )
        # Echo-WM's QK norm is RMS over the full (heads x head_dim) feature, so
        # at TP > 1 it needs the global-statistics norm: that is the
        # "rms_norm_across_heads" implementation in the shared LTX-2 attention.
        qk_norm = "rms_norm_across_heads"
        self.attn1 = LTX2Attention(
            query_dim=video_dim,
            heads=video_heads,
            kv_heads=video_heads,
            dim_head=video_head_dim,
            cross_attention_dim=None,
            qk_norm=qk_norm,
            prefix=f"{prefix}.attn1" if prefix else "attn1",
            **shared,
        )
        self.attn2 = LTX2Attention(
            query_dim=video_dim,
            heads=video_heads,
            kv_heads=video_heads,
            dim_head=video_head_dim,
            cross_attention_dim=cross_attention_dim,
            qk_norm=qk_norm,
            prefix=f"{prefix}.attn2" if prefix else "attn2",
            **shared,
        )
        self.ff = LTX2FeedForward(video_dim, quant_config=quant_config, prefix=f"{prefix}.ff" if prefix else "ff")
        sst = 6 + (3 if cross_attention_adaln else 0)
        self.scale_shift_table = nn.Parameter(torch.empty(sst, video_dim))

        self.audio_attn1 = LTX2Attention(
            query_dim=audio_dim,
            heads=audio_heads,
            kv_heads=audio_heads,
            dim_head=audio_head_dim,
            cross_attention_dim=None,
            qk_norm=qk_norm,
            prefix=f"{prefix}.audio_attn1" if prefix else "audio_attn1",
            **shared,
        )
        self.audio_attn2 = LTX2Attention(
            query_dim=audio_dim,
            heads=audio_heads,
            kv_heads=audio_heads,
            dim_head=audio_head_dim,
            cross_attention_dim=audio_cross_attention_dim,
            qk_norm=qk_norm,
            prefix=f"{prefix}.audio_attn2" if prefix else "audio_attn2",
            **shared,
        )
        self.audio_ff = LTX2FeedForward(
            audio_dim, quant_config=quant_config, prefix=f"{prefix}.audio_ff" if prefix else "audio_ff"
        )
        self.audio_scale_shift_table = nn.Parameter(torch.empty(sst, audio_dim))

        # a2v: Q video, K/V audio; v2a mirrored. Both use the audio head
        # geometry (heads x head_dim = 2048).
        self.audio_to_video_attn = LTX2Attention(
            query_dim=video_dim,
            heads=audio_heads,
            kv_heads=audio_heads,
            dim_head=audio_head_dim,
            cross_attention_dim=audio_dim,
            qk_norm=qk_norm,
            prefix=f"{prefix}.audio_to_video_attn" if prefix else "audio_to_video_attn",
            **shared,
        )
        self.video_to_audio_attn = LTX2Attention(
            query_dim=audio_dim,
            heads=audio_heads,
            kv_heads=audio_heads,
            dim_head=audio_head_dim,
            cross_attention_dim=video_dim,
            qk_norm=qk_norm,
            prefix=f"{prefix}.video_to_audio_attn" if prefix else "video_to_audio_attn",
            **shared,
        )
        self.scale_shift_table_a2v_ca_video = nn.Parameter(torch.empty(5, video_dim))
        self.scale_shift_table_a2v_ca_audio = nn.Parameter(torch.empty(5, audio_dim))

        if cross_attention_adaln:
            self.prompt_scale_shift_table = nn.Parameter(torch.empty(2, video_dim))
            self.audio_prompt_scale_shift_table = nn.Parameter(torch.empty(2, audio_dim))

        # Every attention runs through the causal processor: it carries the
        # windowed-cache and rope-template arguments the base processor would
        # silently drop.
        for attention in (
            self.attn1,
            self.attn2,
            self.audio_attn1,
            self.audio_attn2,
            self.audio_to_video_attn,
            self.video_to_audio_attn,
        ):
            attention.set_processor(EchoWMCausalAttnProcessor())

        self.ucpe = (
            EchoWMUCPEBranch(
                video_dim,
                ucpe,
                quant_config=quant_config,
                prefix=f"{prefix}" if prefix else "",
            )
            if ucpe is not None
            else None
        )

    @staticmethod
    def _ada(scale_shift_table: torch.Tensor, timestep: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Reference ``get_ada_values``: table rows + per-token AdaLN rows."""
        num = scale_shift_table.shape[0]
        values = (
            scale_shift_table[None, None].to(device=timestep.device, dtype=timestep.dtype)
            + timestep.reshape(timestep.shape[0], timestep.shape[1], num, -1)
        ).unbind(dim=2)
        return values

    def _text_cross_attention(  # noqa: PLR0913
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        attn: LTX2Attention,
        scale_shift_table: torch.Tensor,
        prompt_scale_shift_table: torch.Tensor | None,
        timestep: torch.Tensor,
        prompt_timestep: torch.Tensor | None,
        crossattn_cache: EchoWMTextKV | None,
    ) -> torch.Tensor:
        if self.cross_attention_adaln:
            shift_q, scale_q, gate = self._ada(scale_shift_table, timestep)[6:9]
            if prompt_timestep is None or prompt_scale_shift_table is None:
                raise ValueError("cross-attention AdaLN requires prompt_timestep and prompt_scale_shift_table")
            batch = x.shape[0]
            shift_kv, scale_kv = (
                prompt_scale_shift_table[None, None].to(device=x.device, dtype=x.dtype)
                + prompt_timestep.reshape(batch, prompt_timestep.shape[1], 2, -1)
            ).unbind(dim=2)
            attn_input = rms_norm(x, self.norm_eps) * (1 + scale_q) + shift_q
            encoder_hidden_states = context * (1 + scale_kv) + shift_kv
            return (
                attn(
                    attn_input,
                    encoder_hidden_states=encoder_hidden_states,
                    crossattn_cache=crossattn_cache,
                )
                * gate
            )
        return attn(rms_norm(x, self.norm_eps), encoder_hidden_states=context, crossattn_cache=crossattn_cache)

    def forward(  # noqa: PLR0912, PLR0913, PLR0915
        self,
        vx: torch.Tensor,
        ax: torch.Tensor | None,
        video_timestep: torch.Tensor,
        audio_timestep: torch.Tensor | None,
        video_context: torch.Tensor,
        audio_context: torch.Tensor | None,
        caches: EchoWMLayerCaches,
        *,
        video_token_start: int = 0,
        audio_token_start: int = 0,
        prompt_timestep: torch.Tensor | None = None,
        audio_prompt_timestep: torch.Tensor | None = None,
        cross_scale_shift_timestep: torch.Tensor | None = None,
        cross_gate_timestep: torch.Tensor | None = None,
        audio_cross_scale_shift_timestep: torch.Tensor | None = None,
        audio_cross_gate_timestep: torch.Tensor | None = None,
        ucpe_viewmats: torch.Tensor | None = None,
        ucpe_Ks: torch.Tensor | None = None,  # noqa: N803
        patches_per_frame: int = 1,
        video_global_len: int | None = None,
        audio_global_len: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if ax is not None and (audio_context is None or audio_timestep is None):
            raise ValueError("audio forwards require audio context and timesteps")
        if (
            self.cross_attention_adaln
            and ax is not None
            and (
                cross_scale_shift_timestep is None
                or cross_gate_timestep is None
                or audio_cross_scale_shift_timestep is None
                or audio_cross_gate_timestep is None
            )
        ):
            raise ValueError("cross-attention AdaLN requires the AV-CA timesteps")

        # 1. video self-attention (+ optional UCPE residual)
        vshift, vscale, vgate = self._ada(self.scale_shift_table, video_timestep)[0:3]
        norm_vx = rms_norm(vx, self.norm_eps) * (1 + vscale) + vshift
        attn_out = self.attn1(
            norm_vx,
            encoder_hidden_states=None,
            kv_cache=caches.video_self,
            kv_cache_start=video_token_start,
            q_rope=caches.video_rope,
            k_rope=caches.video_rope,
            global_query_len=video_global_len,
        )
        if self.ucpe is not None and ucpe_viewmats is not None and ucpe_Ks is not None:
            attn_out = attn_out + self.ucpe(
                norm_vx,
                ucpe_viewmats,
                ucpe_Ks,
                kv_cache=caches.video_ucpe,
                kv_cache_start=video_token_start,
                patches_per_frame=patches_per_frame,
                full_viewmats=caches.ucpe_full_viewmats,
                full_Ks=caches.ucpe_full_Ks,
                global_query_len=video_global_len,
            )
        vx = vx + attn_out * vgate
        del vshift, vscale, vgate, norm_vx, attn_out

        # 2. video text cross-attention (AdaLN-modulated)
        vx = vx + self._text_cross_attention(
            vx,
            video_context,
            self.attn2,
            self.scale_shift_table,
            getattr(self, "prompt_scale_shift_table", None),
            video_timestep,
            prompt_timestep,
            caches.video_text,
        )

        if ax is not None:
            # 3. audio self-attention
            ashift, ascale, agate = self._ada(self.audio_scale_shift_table, audio_timestep)[0:3]
            norm_ax = rms_norm(ax, self.norm_eps) * (1 + ascale) + ashift
            ax = (
                ax
                + self.audio_attn1(
                    norm_ax,
                    encoder_hidden_states=None,
                    kv_cache=caches.audio_self,
                    kv_cache_start=audio_token_start,
                    q_rope=caches.audio_rope,
                    k_rope=caches.audio_rope,
                    global_query_len=audio_global_len,
                )
                * agate
            )
            del ashift, ascale, agate, norm_ax

            # 4. audio text cross-attention
            ax = ax + self._text_cross_attention(
                ax,
                audio_context,
                self.audio_attn2,
                self.audio_scale_shift_table,
                getattr(self, "audio_prompt_scale_shift_table", None),
                audio_timestep,
                audio_prompt_timestep,
                caches.audio_text,
            )

            # 5. audio-video cross-attention
            vx_norm3 = rms_norm(vx, self.norm_eps)
            ax_norm3 = rms_norm(ax, self.norm_eps)

            # a2v: Q video, K/V audio. Video Q uses table rows 0:2, audio K/V
            # rows 0:2, gate row 4.
            video_ca = self._ada(self.scale_shift_table_a2v_ca_video[:4, :], cross_scale_shift_timestep)
            scale_ca_video_a2v, shift_ca_video_a2v = (t.squeeze(2) for t in video_ca[0:2])
            (gate_out_a2v,) = (
                t.squeeze(2) for t in self._ada(self.scale_shift_table_a2v_ca_video[4:, :], cross_gate_timestep)
            )
            audio_ca = self._ada(self.scale_shift_table_a2v_ca_audio[:4, :], audio_cross_scale_shift_timestep)
            scale_ca_audio_a2v, shift_ca_audio_a2v = (t.squeeze(2) for t in audio_ca[0:2])
            vx_scaled = vx_norm3 * (1 + scale_ca_video_a2v) + shift_ca_video_a2v
            ax_scaled = ax_norm3 * (1 + scale_ca_audio_a2v) + shift_ca_audio_a2v
            del scale_ca_video_a2v, shift_ca_video_a2v, scale_ca_audio_a2v, shift_ca_audio_a2v
            q_slice = caches.a2v_q_slices.get((audio_token_start, audio_token_start + ax.shape[1]))
            if q_slice is None:
                raise ValueError(f"missing a2v query RoPE slice at audio start {audio_token_start}")
            vx = (
                vx
                + self.audio_to_video_attn(
                    vx_scaled,
                    encoder_hidden_states=ax_scaled,
                    kv_cache=caches.a2v,
                    kv_cache_start=audio_token_start,
                    q_rope=caches.video_cross_rope,
                    k_rope=caches.audio_cross_rope,
                    q_rope_slice=q_slice,
                    global_query_len=video_global_len,
                )
                * gate_out_a2v
            )
            del gate_out_a2v, vx_scaled, ax_scaled

            # v2a: Q audio (audio table rows 2:4), K/V video (video rows 2:4).
            audio_ca_v2a = self._ada(self.scale_shift_table_a2v_ca_audio[:4, :], audio_cross_scale_shift_timestep)
            scale_ca_audio_v2a, shift_ca_audio_v2a = (t.squeeze(2) for t in audio_ca_v2a[2:4])
            video_ca_v2a = self._ada(self.scale_shift_table_a2v_ca_video[:4, :], cross_scale_shift_timestep)
            scale_ca_video_v2a, shift_ca_video_v2a = (t.squeeze(2) for t in video_ca_v2a[2:4])
            (gate_out_v2a,) = (
                t.squeeze(2) for t in self._ada(self.scale_shift_table_a2v_ca_audio[4:, :], audio_cross_gate_timestep)
            )
            ax_scaled = ax_norm3 * (1 + scale_ca_audio_v2a) + shift_ca_audio_v2a
            vx_scaled = vx_norm3 * (1 + scale_ca_video_v2a) + shift_ca_video_v2a
            del scale_ca_audio_v2a, shift_ca_audio_v2a, scale_ca_video_v2a, shift_ca_video_v2a
            q_slice = caches.v2a_q_slices.get((video_token_start, video_token_start + vx_norm3.shape[1]))
            if q_slice is None:
                raise ValueError(f"missing v2a query RoPE slice at video start {video_token_start}")
            ax = (
                ax
                + self.video_to_audio_attn(
                    ax_scaled,
                    encoder_hidden_states=vx_scaled,
                    kv_cache=caches.v2a,
                    kv_cache_start=video_token_start,
                    q_rope=caches.audio_cross_rope,
                    k_rope=caches.video_cross_rope,
                    q_rope_slice=q_slice,
                    global_query_len=audio_global_len,
                )
                * gate_out_v2a
            )
            del gate_out_v2a, ax_scaled, vx_scaled, vx_norm3, ax_norm3

        # 6. feed-forwards
        vshift_mlp, vscale_mlp, vgate_mlp = self._ada(self.scale_shift_table, video_timestep)[3:6]
        vx_scaled = rms_norm(vx, self.norm_eps) * (1 + vscale_mlp) + vshift_mlp
        vx = vx + self.ff(vx_scaled) * vgate_mlp

        if ax is not None:
            ashift_mlp, ascale_mlp, agate_mlp = self._ada(self.audio_scale_shift_table, audio_timestep)[3:6]
            ax_scaled = rms_norm(ax, self.norm_eps) * (1 + ascale_mlp) + ashift_mlp
            ax = ax + self.audio_ff(ax_scaled) * agate_mlp

        return vx, ax


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


_ECHOWM_CHECKPOINT_CONTRACT: dict[str, Any] = {
    "num_layers": 48,
    "num_attention_heads": 32,
    "attention_head_dim": 128,
    "in_channels": 128,
    "out_channels": 128,
    "audio_num_attention_heads": 32,
    "audio_attention_head_dim": 64,
    "audio_in_channels": 128,
    "audio_out_channels": 128,
    "cross_attention_dim": 4096,
    "audio_cross_attention_dim": 2048,
    "timestep_scale_multiplier": 1000,
    "positional_embedding_theta": 10000.0,
    "apply_gated_attention": True,
    "cross_attention_adaln": True,
    "rope_type": "split",
}

_NAME_PREFIXES = ("model.diffusion_model.", "transformer.")


class EchoWMTransformer3DModel(nn.Module):
    """Echo-WM Flash audio-video DiT (velocity model)."""

    _repeated_blocks = ["EchoWMTransformerBlock"]
    _layerwise_offload_blocks_attrs = ["transformer_blocks"]
    packed_modules_mapping = {"to_qkv": ["to_q", "to_k", "to_v"]}
    stacked_params_mapping = [
        (".to_qkv.", ".to_q.", "q"),
        (".to_qkv.", ".to_k.", "k"),
        (".to_qkv.", ".to_v.", "v"),
    ]
    # The checkpoint keeps the UCPE projections directly on the block; the port
    # nests them under the ``ucpe`` branch module.
    _param_renames = {
        ".ucpe_q_proj.": ".ucpe.ucpe_q_proj.",
        ".ucpe_k_proj.": ".ucpe.ucpe_k_proj.",
        ".ucpe_v_proj.": ".ucpe.ucpe_v_proj.",
        ".ucpe_out_proj.": ".ucpe.ucpe_out_proj.",
        ".q_norm.": ".norm_q.",
        ".k_norm.": ".norm_k.",
    }

    def __init__(  # noqa: PLR0913
        self,
        *,
        num_layers: int = 48,
        num_attention_heads: int = 32,
        attention_head_dim: int = 128,
        in_channels: int = 128,
        out_channels: int = 128,
        audio_num_attention_heads: int = 32,
        audio_attention_head_dim: int = 64,
        audio_in_channels: int = 128,
        audio_out_channels: int = 128,
        cross_attention_dim: int = 4096,
        audio_cross_attention_dim: int = 2048,
        norm_eps: float = 1e-6,
        positional_embedding_theta: float = 10000.0,
        positional_embedding_max_pos: tuple[int, ...] = (20, 2048, 2048),
        audio_positional_embedding_max_pos: tuple[int, ...] = (20,),
        timestep_scale_multiplier: int = 1000,
        av_ca_timestep_scale_multiplier: float = 1000.0,
        rope_type: str = "split",
        apply_gated_attention: bool = True,
        cross_attention_adaln: bool = True,
        ucpe: EchoWMUCPEConfig | None = None,
        quant_config: Any = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        tp_size = get_tensor_model_parallel_world_size()
        if num_attention_heads % tp_size or audio_num_attention_heads % tp_size:
            raise ValueError(
                f"attention heads ({num_attention_heads}/{audio_num_attention_heads}) must be divisible by "
                f"tensor parallel size {tp_size}"
            )
        sp_size = _ulysses().world_size
        if (num_attention_heads // tp_size) % sp_size or (audio_num_attention_heads // tp_size) % sp_size:
            raise ValueError(f"TP-local heads must be divisible by sequence parallel size {sp_size}")
        if ucpe is not None and ucpe.num_heads % (tp_size * sp_size):
            raise ValueError(f"UCPE heads ({ucpe.num_heads}) must be divisible by tp*sp = {tp_size * sp_size}")

        self.config = SimpleNamespace(
            num_layers=num_layers,
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            in_channels=in_channels,
            out_channels=out_channels,
            audio_num_attention_heads=audio_num_attention_heads,
            audio_attention_head_dim=audio_attention_head_dim,
            audio_in_channels=audio_in_channels,
            audio_out_channels=audio_out_channels,
            cross_attention_dim=cross_attention_dim,
            audio_cross_attention_dim=audio_cross_attention_dim,
            norm_eps=norm_eps,
            positional_embedding_theta=positional_embedding_theta,
            positional_embedding_max_pos=tuple(positional_embedding_max_pos),
            audio_positional_embedding_max_pos=tuple(audio_positional_embedding_max_pos),
            timestep_scale_multiplier=timestep_scale_multiplier,
            av_ca_timestep_scale_multiplier=av_ca_timestep_scale_multiplier,
            apply_gated_attention=apply_gated_attention,
            cross_attention_adaln=cross_attention_adaln,
            ucpe=ucpe,
        )
        self.inner_dim = num_attention_heads * attention_head_dim
        self.audio_inner_dim = audio_num_attention_heads * audio_attention_head_dim
        self.timestep_scale_multiplier = timestep_scale_multiplier

        self.patchify_proj = nn.Linear(in_channels, self.inner_dim, bias=True)
        self.adaln_single = LTX2AdaLayerNormSingle(self.inner_dim, num_mod_params=9)
        self.prompt_adaln_single = (
            LTX2AdaLayerNormSingle(self.inner_dim, num_mod_params=2) if cross_attention_adaln else None
        )
        self.scale_shift_table = nn.Parameter(torch.empty(2, self.inner_dim))
        self.norm_out = nn.LayerNorm(self.inner_dim, elementwise_affine=False, eps=norm_eps)
        self.proj_out = nn.Linear(self.inner_dim, out_channels)

        self.audio_patchify_proj = nn.Linear(audio_in_channels, self.audio_inner_dim, bias=True)
        self.audio_adaln_single = LTX2AdaLayerNormSingle(self.audio_inner_dim, num_mod_params=9)
        self.audio_prompt_adaln_single = (
            LTX2AdaLayerNormSingle(self.audio_inner_dim, num_mod_params=2) if cross_attention_adaln else None
        )
        self.audio_scale_shift_table = nn.Parameter(torch.empty(2, self.audio_inner_dim))
        self.audio_norm_out = nn.LayerNorm(self.audio_inner_dim, elementwise_affine=False, eps=norm_eps)
        self.audio_proj_out = nn.Linear(self.audio_inner_dim, audio_out_channels)

        # AV cross-attention AdaLN embedders (always at cross sigma = 1).
        self.av_ca_video_scale_shift_adaln_single = LTX2AdaLayerNormSingle(self.inner_dim, 4)
        self.av_ca_audio_scale_shift_adaln_single = LTX2AdaLayerNormSingle(self.audio_inner_dim, 4)
        self.av_ca_a2v_gate_adaln_single = LTX2AdaLayerNormSingle(self.inner_dim, 1)
        self.av_ca_v2a_gate_adaln_single = LTX2AdaLayerNormSingle(self.audio_inner_dim, 1)

        self.transformer_blocks = nn.ModuleList(
            [
                EchoWMTransformerBlock(
                    idx=idx,
                    num_layers=num_layers,
                    video_dim=self.inner_dim,
                    video_heads=num_attention_heads,
                    video_head_dim=attention_head_dim,
                    cross_attention_dim=cross_attention_dim,
                    audio_dim=self.audio_inner_dim,
                    audio_heads=audio_num_attention_heads,
                    audio_head_dim=audio_attention_head_dim,
                    audio_cross_attention_dim=audio_cross_attention_dim,
                    norm_eps=norm_eps,
                    apply_gated_attention=apply_gated_attention,
                    cross_attention_adaln=cross_attention_adaln,
                    ucpe=ucpe,
                    quant_config=quant_config,
                    prefix=f"{prefix}transformer_blocks.{idx}" if prefix else f"transformer_blocks.{idx}",
                )
                for idx in range(num_layers)
            ]
        )
        self.ucpe_config = ucpe

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
        *,
        ucpe: EchoWMUCPEConfig | None = None,
        quant_config: Any = None,
        prefix: str = "",
    ) -> EchoWMTransformer3DModel:
        if config.get("_class_name") not in (None, "AVTransformer3DModel", "EchoWMTransformer3DModel"):
            raise ValueError(f"unexpected Echo-WM transformer class {config.get('_class_name')!r}")
        kwargs = dict(
            num_layers=config.get("num_layers", 48),
            num_attention_heads=config.get("num_attention_heads", 32),
            attention_head_dim=config.get("attention_head_dim", 128),
            in_channels=config.get("in_channels", 128),
            out_channels=config.get("out_channels", 128),
            audio_num_attention_heads=config.get("audio_num_attention_heads", 32),
            audio_attention_head_dim=config.get("audio_attention_head_dim", 64),
            audio_in_channels=config.get("audio_in_channels", 128),
            audio_out_channels=config.get("audio_out_channels", 128),
            cross_attention_dim=config.get("cross_attention_dim", 4096),
            audio_cross_attention_dim=config.get("audio_cross_attention_dim", 2048),
            norm_eps=config.get("norm_eps", 1e-6),
            positional_embedding_theta=config.get("positional_embedding_theta", 10000.0),
            positional_embedding_max_pos=tuple(config.get("positional_embedding_max_pos", (20, 2048, 2048))),
            audio_positional_embedding_max_pos=tuple(config.get("audio_positional_embedding_max_pos", (20,))),
            timestep_scale_multiplier=config.get("timestep_scale_multiplier", 1000),
            av_ca_timestep_scale_multiplier=config.get("av_ca_timestep_scale_multiplier", 1000.0),
            rope_type=config.get("rope_type", "split"),
            apply_gated_attention=config.get("apply_gated_attention", True),
            cross_attention_adaln=config.get("cross_attention_adaln", True),
        )
        # The checkpoint contract pins the official Echo-WM Flash geometry;
        # deviations must be rejected before weights load. Tests that need a
        # tiny model construct the class directly.
        for key, expected in _ECHOWM_CHECKPOINT_CONTRACT.items():
            if kwargs[key] != expected:
                raise ValueError(f"Echo-WM Flash checkpoint contract: {key} must be {expected!r}, got {kwargs[key]!r}")
        kwargs.update(ucpe=ucpe, quant_config=quant_config, prefix=prefix)
        return cls(**kwargs)

    # -- caches ---------------------------------------------------------------

    def local_head_counts(self) -> tuple[int, int]:
        """(video, audio) heads per rank in the post-all-to-all window form."""
        tp_size = get_tensor_model_parallel_world_size()
        sp_size = _ulysses().world_size
        return (
            self.config.num_attention_heads // (tp_size * sp_size),
            self.config.audio_num_attention_heads // (tp_size * sp_size),
        )

    def allocate_caches(
        self,
        *,
        batch_size: int,
        patches_per_frame: int,
        text_seq_len: int,
        cache_config: EchoWMCacheConfig,
        device: torch.device,
        dtype: torch.dtype,
    ) -> list[EchoWMLayerCaches]:
        """Allocate one :class:`EchoWMLayerCaches` per layer, TP/SP sharded."""
        video_heads, audio_heads = self.local_head_counts()
        video_window = cache_config.video_local_attn_size * patches_per_frame
        video_sink = cache_config.video_sink_size * patches_per_frame
        audio_window = cache_config.audio_local_attn_size
        audio_sink = cache_config.audio_sink_size

        def window(num_heads: int, head_dim: int, capacity: int, local: int, sink: int) -> EchoWMKVWindow:
            return EchoWMKVWindow(batch_size, capacity, num_heads, head_dim, local, sink, device, dtype)

        caches: list[EchoWMLayerCaches] = []
        for _ in range(self.config.num_layers):
            layer = EchoWMLayerCaches(
                video_self=window(video_heads, self.config.attention_head_dim, video_window, video_window, video_sink),
                video_text=EchoWMTextKV(
                    batch_size, text_seq_len, video_heads, self.config.attention_head_dim, device, dtype
                ),
                audio_self=window(
                    audio_heads, self.config.audio_attention_head_dim, audio_window, audio_window, audio_sink
                ),
                audio_text=EchoWMTextKV(
                    batch_size, text_seq_len, audio_heads, self.config.audio_attention_head_dim, device, dtype
                ),
                a2v=window(audio_heads, self.config.audio_attention_head_dim, audio_window, audio_window, audio_sink),
                v2a=window(audio_heads, self.config.audio_attention_head_dim, video_window, video_window, video_sink),
            )
            if self.ucpe_config is not None:
                layer.video_ucpe = window(
                    self.ucpe_config.num_heads // (get_tensor_model_parallel_world_size() * _ulysses().world_size),
                    self.ucpe_config.attn_dim // self.ucpe_config.num_heads,
                    video_window,
                    video_window,
                    video_sink,
                )
            caches.append(layer)
        return caches

    # -- forward ----------------------------------------------------------------

    def _prepare_timesteps(
        self, sigma: float, adaln: LTX2AdaLayerNormSingle, tokens: int, dtype: torch.dtype, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scaled = torch.full((tokens,), sigma * self.timestep_scale_multiplier, device=device, dtype=torch.float32)
        timestep, embedded = adaln(scaled, hidden_dtype=dtype)
        return timestep.view(1, tokens, -1), embedded.view(1, tokens, -1)

    def forward(  # noqa: PLR0912, PLR0913
        self,
        *,
        video_tokens: torch.Tensor,
        audio_tokens: torch.Tensor | None,
        video_sigma: float,
        audio_sigma: float | None = None,
        video_context: torch.Tensor | None = None,
        audio_context: torch.Tensor | None = None,
        caches: list[EchoWMLayerCaches] | None = None,
        video_token_start: int = 0,
        audio_token_start: int = 0,
        ucpe_viewmats: torch.Tensor | None = None,
        ucpe_Ks: torch.Tensor | None = None,  # noqa: N803
        patches_per_frame: int = 1,
        video_global_len: int | None = None,
        audio_global_len: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """One causal forward: returns velocity predictions in token space.

        ``video_tokens``/``audio_tokens`` are the *current block's* patchified
        latents; history comes from ``caches``. Sigmas are per-modality floats
        (the image-sink commit passes ``video_sigma=0.0``).
        """
        if caches is None:
            raise ValueError("Echo-WM inference requires causal caches")
        dtype = self.patchify_proj.weight.dtype
        device = self.patchify_proj.weight.device

        vx = self.patchify_proj(video_tokens.to(dtype))
        video_timestep, embedded_video = self._prepare_timesteps(
            video_sigma, self.adaln_single, vx.shape[1], dtype, device
        )
        ax = None
        audio_timestep = None
        embedded_audio = None
        if audio_tokens is not None:
            if audio_context is None:
                raise ValueError("audio forwards require audio_context")
            ax = self.audio_patchify_proj(audio_tokens.to(dtype))
            sigma = audio_sigma if audio_sigma is not None else video_sigma
            audio_timestep, embedded_audio = self._prepare_timesteps(
                sigma, self.audio_adaln_single, ax.shape[1], dtype, device
            )
        if video_context is None:
            raise ValueError("video forwards require video_context")

        # Prompt AdaLN: the rollout keeps sigma = 1 for the text branch; each
        # modality embeds it with its own prompt AdaLN.
        prompt_timestep = None
        audio_prompt_timestep = None
        if self.prompt_adaln_single is not None:
            prompt_timestep = self._prepare_timesteps(1.0, self.prompt_adaln_single, 1, dtype, device)[0]
            audio_prompt_timestep = self._prepare_timesteps(1.0, self.audio_prompt_adaln_single, 1, dtype, device)[0]
        # AV cross-attention AdaLN at the constant cross sigma = 1.
        cross_step = torch.full((1,), float(self.timestep_scale_multiplier), device=device, dtype=torch.float32)
        av_factor = self.config.av_ca_timestep_scale_multiplier / self.timestep_scale_multiplier
        cross_scale_shift = self.av_ca_video_scale_shift_adaln_single(cross_step, hidden_dtype=dtype)[0].view(1, 1, -1)
        cross_gate = self.av_ca_a2v_gate_adaln_single(cross_step * av_factor, hidden_dtype=dtype)[0].view(1, 1, -1)
        audio_cross_scale_shift = self.av_ca_audio_scale_shift_adaln_single(cross_step, hidden_dtype=dtype)[0].view(
            1, 1, -1
        )
        audio_cross_gate = self.av_ca_v2a_gate_adaln_single(cross_step * av_factor, hidden_dtype=dtype)[0].view(
            1, 1, -1
        )

        if video_global_len is None:
            video_global_len = vx.shape[1]
        if audio_global_len is None and ax is not None:
            audio_global_len = ax.shape[1]

        for layer_caches, block in zip(caches, self.transformer_blocks, strict=True):
            vx, ax = block(
                vx,
                ax,
                video_timestep,
                audio_timestep,
                video_context,
                audio_context,
                layer_caches,
                video_token_start=video_token_start,
                audio_token_start=audio_token_start,
                prompt_timestep=prompt_timestep,
                audio_prompt_timestep=audio_prompt_timestep,
                cross_scale_shift_timestep=cross_scale_shift,
                cross_gate_timestep=cross_gate,
                audio_cross_scale_shift_timestep=audio_cross_scale_shift,
                audio_cross_gate_timestep=audio_cross_gate,
                ucpe_viewmats=ucpe_viewmats,
                ucpe_Ks=ucpe_Ks,
                patches_per_frame=patches_per_frame,
                video_global_len=video_global_len,
                audio_global_len=audio_global_len,
            )

        vx = self._process_output(self.scale_shift_table, self.norm_out, self.proj_out, vx, embedded_video)
        if ax is not None:
            ax = self._process_output(
                self.audio_scale_shift_table, self.audio_norm_out, self.audio_proj_out, ax, embedded_audio
            )
        return vx, ax

    @staticmethod
    def _process_output(
        scale_shift_table: torch.Tensor,
        norm_out: nn.Module,
        proj_out: nn.Linear,
        x: torch.Tensor,
        embedded_timestep: torch.Tensor,
    ) -> torch.Tensor:
        scale_shift_values = (
            scale_shift_table[None, None].to(device=x.device, dtype=x.dtype) + embedded_timestep[:, :, None]
        )
        shift, scale = scale_shift_values[:, :, 0], scale_shift_values[:, :, 1]
        x = norm_out(x)
        x = x * (1 + scale) + shift
        return proj_out(x)

    # -- weights ---------------------------------------------------------------

    def load_weights(self, weights: Any) -> set[str]:
        params_dict = dict(self.named_parameters())
        loaded: set[str] = set()

        def normalize(name: str) -> str:
            for prefix in _NAME_PREFIXES:
                if name.startswith(prefix):
                    return normalize(name[len(prefix) :])
            return name

        for name, weight in weights:
            original = name
            name = normalize(name)
            target_name = name
            for ckpt_pattern, model_pattern in self._param_renames.items():
                if ckpt_pattern in target_name:
                    target_name = target_name.replace(ckpt_pattern, model_pattern)
            shard_id: str | None = None
            param = params_dict.get(target_name)
            if param is None:
                # Direct miss: the self-attention projections are fused, so a
                # separate to_q/to_k/to_v checkpoint tensor maps onto the fused
                # QKV parameter with its stacked shard loader.
                for fused_pattern, ckpt_pattern, sid in self.stacked_params_mapping:
                    if ckpt_pattern in target_name:
                        fused_name = target_name.replace(ckpt_pattern, fused_pattern)
                        if fused_name in params_dict:
                            target_name = fused_name
                            shard_id = sid
                            param = params_dict[fused_name]
                            break
            if param is None:
                raise KeyError(f"unknown Echo-WM transformer weight: {original!r} (looked up as {target_name!r})")
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            if shard_id is not None:
                weight_loader(param, weight, shard_id)
            else:
                weight_loader(param, weight)
            loaded.add(original)
            loaded.add(target_name)
        return loaded
