# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Echo-WM text conditioning stack: Gemma feature extraction + connectors.

Port of the reference ``EmbeddingsProcessor``: the 49 stacked Gemma hidden
layers are per-token RMS-normalized, rescaled, and projected per modality
(``text_embedding_projection``), then each modality runs through an 8-layer
1-D connector whose padded positions are replaced by learnable registers
(``{video,audio}_embeddings_connector``). The connectors use the same
LTX-2 primitives as the DiT, so TP sharding comes for free.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from vllm_omni.diffusion.models.echo_wm.transformer import EchoWMCausalAttnProcessor, rms_norm
from vllm_omni.diffusion.models.ltx2.ltx2_transformer import LTX2Attention, LTX2FeedForward

__all__ = ["EchoWMConnectorBlock", "EchoWMEmbeddingsConnector", "EchoWMTextStack"]


class EchoWMConnectorBlock(nn.Module):
    """One 1-D connector block: RMS-norm self-attention and feed-forward."""

    def __init__(
        self,
        dim: int,
        heads: int,
        head_dim: int,
        *,
        norm_eps: float = 1e-6,
        apply_gated_attention: bool = True,
        quant_config=None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.attn1 = LTX2Attention(
            query_dim=dim,
            heads=heads,
            kv_heads=heads,
            dim_head=head_dim,
            cross_attention_dim=None,
            qk_norm="rms_norm_across_heads",
            norm_eps=norm_eps,
            rope_type="split",
            apply_gated_attention=apply_gated_attention,
            quant_config=quant_config,
            prefix=f"{prefix}.attn1" if prefix else "attn1",
        )
        self.attn1.set_processor(EchoWMCausalAttnProcessor())
        self.ff = LTX2FeedForward(dim, quant_config=quant_config, prefix=f"{prefix}.ff" if prefix else "ff")

    def forward(self, hidden_states: torch.Tensor, rope: tuple[torch.Tensor, torch.Tensor] | None) -> torch.Tensor:
        normed = rms_norm(hidden_states)
        attn_out = self.attn1(
            normed,
            encoder_hidden_states=None,
            q_rope=rope,
            k_rope=rope,
        )
        hidden_states = attn_out + hidden_states
        return self.ff(rms_norm(hidden_states)) + hidden_states


class EchoWMEmbeddingsConnector(nn.Module):
    """Learnable-register connector over one modality's projected features."""

    def __init__(
        self,
        *,
        num_layers: int,
        heads: int,
        head_dim: int,
        num_registers: int = 128,
        rope_theta: float = 10000.0,
        rope_max_pos: int = 4096,
        apply_gated_attention: bool = True,
        quant_config=None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.num_registers = num_registers
        self.rope_theta = rope_theta
        self.rope_max_pos = rope_max_pos
        self.heads = heads
        inner_dim = heads * head_dim
        self.transformer_1d_blocks = nn.ModuleList(
            [
                EchoWMConnectorBlock(
                    inner_dim,
                    heads,
                    head_dim,
                    apply_gated_attention=apply_gated_attention,
                    quant_config=quant_config,
                    prefix=f"{prefix}.transformer_1d_blocks.{idx}" if prefix else f"transformer_1d_blocks.{idx}",
                )
                for idx in range(num_layers)
            ]
        )
        if num_registers:
            self.learnable_registers = nn.Parameter(
                torch.rand(num_registers, inner_dim, dtype=torch.bfloat16) * 2.0 - 1.0
            )

    @property
    def inner_dim(self) -> int:
        return self.transformer_1d_blocks[0].attn1.inner_dim

    def _replace_padded_with_registers(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Move real tokens to the front and fill the tail with tiled registers.

        ``attention_mask`` is the binary (B, T) Gemma mask; the reference does
        the same rearrangement through flipped additive masks.
        """
        seq_len = hidden_states.shape[1]
        if seq_len % self.num_registers:
            raise ValueError(f"connector sequence length {seq_len} must be divisible by {self.num_registers} registers")
        repeats = seq_len // self.num_registers
        registers = torch.tile(self.learnable_registers.to(hidden_states.dtype), (repeats, 1))
        keep = attention_mask.bool()
        counts = keep.sum(dim=-1)
        if (counts != counts[0]).any():
            raise NotImplementedError("connector register replacement requires equal prompt lengths per batch")
        n = int(counts[0])
        b, _, dim = hidden_states.shape
        out = registers.unsqueeze(0).expand(b, -1, -1).clone()
        out[:, :n] = hidden_states[keep].view(b, n, dim)
        return out

    def forward(
        self,
        features: torch.Tensor,
        attention_mask: torch.Tensor,
        rope: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        hidden_states = self._replace_padded_with_registers(features, attention_mask)
        for block in self.transformer_1d_blocks:
            hidden_states = block(hidden_states, rope)
        return rms_norm(hidden_states)


class EchoWMTextStack(nn.Module):
    """Per-token RMS feature extraction, per-modality projection, connectors."""

    def __init__(
        self,
        *,
        gemma_hidden_size: int = 3840,
        gemma_num_layers: int = 48,
        video_dim: int = 4096,
        audio_dim: int = 2048,
        connector_num_layers: int = 8,
        video_heads: int = 32,
        video_head_dim: int = 128,
        audio_heads: int = 32,
        audio_head_dim: int = 64,
        num_registers: int = 128,
        rope_theta: float = 10000.0,
        rope_max_pos: int = 4096,
        apply_gated_attention: bool = True,
        quant_config=None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        flat_dim = gemma_hidden_size * (gemma_num_layers + 1)
        self.embedding_dim = gemma_hidden_size
        self.video_dim = video_dim
        self.audio_dim = audio_dim
        self.video_aggregate_embed = nn.Linear(flat_dim, video_dim, bias=True)
        self.audio_aggregate_embed = nn.Linear(flat_dim, audio_dim, bias=True)
        self.video_connector = EchoWMEmbeddingsConnector(
            num_layers=connector_num_layers,
            heads=video_heads,
            head_dim=video_head_dim,
            num_registers=num_registers,
            rope_theta=rope_theta,
            rope_max_pos=rope_max_pos,
            apply_gated_attention=apply_gated_attention,
            quant_config=quant_config,
            prefix=f"{prefix}video_connector" if prefix else "video_connector",
        )
        self.audio_connector = EchoWMEmbeddingsConnector(
            num_layers=connector_num_layers,
            heads=audio_heads,
            head_dim=audio_head_dim,
            num_registers=num_registers,
            rope_theta=rope_theta,
            rope_max_pos=rope_max_pos,
            apply_gated_attention=apply_gated_attention,
            quant_config=quant_config,
            prefix=f"{prefix}audio_connector" if prefix else "audio_connector",
        )
        self.rope_theta = rope_theta
        self.rope_max_pos = rope_max_pos
        self._rope_cache: dict[tuple[int, torch.device], tuple[torch.Tensor, torch.Tensor]] = {}

    def _connector_rope(
        self, seq_len: int, num_heads: int, head_dim: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Split RoPE over plain token indices (reference connector layout)."""
        key = (seq_len, num_heads, head_dim, device, dtype)
        cached = self._rope_cache.get(key)
        if cached is not None:
            return cached
        from vllm_omni.diffusion.models.echo_wm.causal_cache import _freq_grid

        inner_dim = num_heads * head_dim
        # One position axis: the frequency grid carries inner_dim // 2 angles.
        indices = _freq_grid(self.rope_theta, 1, inner_dim).to(device)
        grid = torch.arange(seq_len, device=device, dtype=torch.float32)
        # (1, T, 1): a single-axis fractional grid matching the multi-axis
        # layout used by make_split_rope, so the shared formula applies.
        fractional = (grid / self.rope_max_pos)[None, :, None]
        freqs = (indices * (fractional.unsqueeze(-1) * 2 - 1)).transpose(-1, -2).flatten(2)
        expected = inner_dim // 2
        pad = expected - freqs.shape[-1]
        cos, sin = freqs.cos(), freqs.sin()
        if pad != 0:
            cos = torch.cat([torch.ones_like(cos[:, :, :pad]), cos], dim=-1)
            sin = torch.cat([torch.zeros_like(sin[:, :, :pad]), sin], dim=-1)
        b, t = cos.shape[0], cos.shape[1]
        cos = cos.reshape(b, t, num_heads, -1).swapaxes(1, 2).to(dtype)
        sin = sin.reshape(b, t, num_heads, -1).swapaxes(1, 2).to(dtype)
        self._rope_cache[key] = (cos, sin)
        return cos, sin

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Map stacked Gemma hidden states ``(B, T, D, L)`` to per-modality contexts."""
        if hidden_states.ndim != 4:
            raise ValueError(f"expected stacked hidden states (B, T, D, L), got {tuple(hidden_states.shape)}")
        b, t, d, num_layers = hidden_states.shape
        # Per-token RMS norm over the hidden axis, per layer; zero the padding.
        variance = torch.mean(hidden_states**2, dim=2, keepdim=True)
        normed = hidden_states * torch.rsqrt(variance + 1e-6)
        normed = normed.reshape(b, t, d * num_layers)
        keep = attention_mask.bool().unsqueeze(-1)
        normed = torch.where(keep, normed, torch.zeros_like(normed)).to(hidden_states.dtype)

        video = self.video_aggregate_embed(normed * math.sqrt(self.video_dim / self.embedding_dim))
        audio = self.audio_aggregate_embed(normed * math.sqrt(self.audio_dim / self.embedding_dim))

        device, dtype = video.device, video.dtype
        video_attn = self.video_connector.transformer_1d_blocks[0].attn1
        audio_attn = self.audio_connector.transformer_1d_blocks[0].attn1
        video_rope = self._connector_rope(t, video_attn.total_num_heads, video_attn.head_dim, device, dtype)
        audio_rope = self._connector_rope(t, audio_attn.total_num_heads, audio_attn.head_dim, device, dtype)
        video_encoding = self.video_connector(video, attention_mask, video_rope)
        audio_encoding = self.audio_connector(audio, attention_mask, audio_rope)
        return video_encoding, audio_encoding

    def load_weights(self, weights) -> set[str]:
        """Load from the reference single-file checkpoint naming."""
        params = dict(self.named_parameters())
        loaded: set[str] = set()

        def normalize(name: str) -> str:
            mapping = {
                "text_embedding_projection.video_aggregate_embed": "video_aggregate_embed",
                "text_embedding_projection.audio_aggregate_embed": "audio_aggregate_embed",
                "model.diffusion_model.video_embeddings_connector": "video_connector",
                "model.diffusion_model.audio_embeddings_connector": "audio_connector",
            }
            for old, new in mapping.items():
                if name.startswith(old):
                    target = new + name[len(old) :]
                    return target.replace(".q_norm.", ".norm_q.").replace(".k_norm.", ".norm_k.")
            return None

        fused_mapping = [
            (".to_qkv.", ".to_q.", "q"),
            (".to_qkv.", ".to_k.", "k"),
            (".to_qkv.", ".to_v.", "v"),
        ]
        for name, weight in weights:
            target = normalize(name)
            if target is None:
                continue
            shard_id: str | None = None
            param = params.get(target)
            if param is None:
                for fused_pattern, ckpt_pattern, sid in fused_mapping:
                    if ckpt_pattern in target:
                        fused_name = target.replace(ckpt_pattern, fused_pattern)
                        if fused_name in params:
                            target = fused_name
                            shard_id = sid
                            param = params[fused_name]
                            break
            if param is None:
                raise KeyError(f"unknown Echo-WM text-stack weight: {name!r} (looked up as {target!r})")
            loader = getattr(param, "weight_loader", default_weight_loader)
            if shard_id is not None:
                loader(param, weight, shard_id)
            else:
                loader(param, weight)
            loaded.add(name)
            loaded.add(target)
        return loaded
