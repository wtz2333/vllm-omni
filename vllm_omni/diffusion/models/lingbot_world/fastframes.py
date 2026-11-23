# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from safetensors.torch import load_file
from torch import nn

from vllm_omni.experimental.ar_diffusion.kv_cache.paged import pool_write_chunk

if TYPE_CHECKING:
    from vllm_omni.diffusion.models.lingbot_world.dmd_block import ARBlockContext
    from vllm_omni.diffusion.models.lingbot_world.transformer import CausalLingBotWorldTransformer3DModel


@dataclass
class FastFramesCapture:
    """The final denoise probe's inputs to the two FastFrames networks."""

    capture_pixels: bool
    kv: tuple[torch.Tensor, torch.Tensor] | None = None
    pixel_inputs: tuple[torch.Tensor, torch.Tensor] | None = None
    rope: tuple[torch.Tensor, torch.Tensor] | None = None


class LingBotFastFrames:
    """Apple's learned networks adapted to Omni's existing stepwise pipeline."""

    height = 464
    width = 832
    frames = 4
    rows = height // 16
    cols = width // 16
    layers = 30
    heads = 12
    head_dim = 128

    def __init__(self, *, pixel_decoder: str | None, kv_refiner: str | None, device: torch.device) -> None:
        self.pixel_head: nn.Module | None = None
        self.refiner: nn.Module | None = None
        if pixel_decoder is not None:
            from fastframes.models.pixel_decoder import PixelDecoder

            self.pixel_head = self._load(PixelDecoder, pixel_decoder, device)
        if kv_refiner is not None:
            from fastframes.models.kv_refinement import KVRefinementNetwork

            self.refiner = self._load(KVRefinementNetwork, kv_refiner, device)

    @staticmethod
    def _load(network: type[nn.Module], weights: str, device: torch.device) -> nn.Module:
        state = load_file(Path(weights).expanduser())
        with torch.device("meta"):
            if "stem_in.weight" in state:
                model = network(d_a=state["stem_in.weight"].shape[0])
            else:
                model = network()
        model.load_state_dict(state, strict=True, assign=True)
        return model.to(device=device, dtype=torch.bfloat16).eval().requires_grad_(False)

    def capture(self, *, first_chunk: bool, pixels: bool, device: torch.device) -> FastFramesCapture | None:
        capture_pixels = pixels and self.pixel_head is not None
        if self.refiner is not None and not first_chunk:
            shape = (self.layers, self.frames * self.rows * self.cols, self.heads, self.head_dim)
            keys = torch.empty(shape, device=device, dtype=torch.bfloat16)
            values = torch.empty_like(keys)
            return FastFramesCapture(capture_pixels=capture_pixels, kv=(keys, values))
        return FastFramesCapture(capture_pixels=True) if capture_pixels else None

    def state_bytes_per_session(self) -> int:
        # Final-probe K/V survives across scheduler calls. Pixel decode holds
        # the features and two latent frames of input per causal convolution.
        hidden = self.heads * self.head_dim
        frame_bytes = self.rows * self.cols * hidden * torch.bfloat16.itemsize
        kv_bytes = 2 * self.layers * self.frames * frame_bytes if self.refiner is not None else 0
        if self.pixel_head is None:
            return kv_bytes
        decoder_blocks = len(self.pixel_head.conv_low) + len(self.pixel_head.conv_high)
        return kv_bytes + (self.frames + 2 * decoder_blocks) * frame_bytes

    def new_decoder(self) -> nn.Module:
        from apps.lingbot.decoder import StreamingDecoder

        return StreamingDecoder(self.pixel_head)

    def refine_and_commit(
        self,
        capture: FastFramesCapture,
        *,
        latents: torch.Tensor,
        condition: torch.Tensor,
        timestep: float,
        transformer: CausalLingBotWorldTransformer3DModel,
        ar: ARBlockContext,
    ) -> None:
        if capture.kv is None or capture.rope is None or self.refiner is None:
            raise RuntimeError("FastFrames KV refinement requires the final denoise probe's canonical K/V and RoPE.")
        keys, values = capture.kv
        capture.kv = None
        clean = torch.cat((latents.to(condition.dtype), condition), dim=1)
        # Self-attention in the refinement stem is spatial, independently
        # per latent frame. It shares the DiT's clean-input patch projection.
        stem = transformer.patch_embedding(clean)[0].flatten(2).permute(1, 2, 0)
        step = self.refiner.embed_timestep(timestep, 1, stem.device, stem.dtype)
        phi = self.refiner.compute_phi(stem, step.expand(stem.shape[0], -1)).flatten(0, 1)
        refined_k, refined_v = self.refiner.repair_all(keys.flatten(2), values.flatten(2), phi, step)
        refined_k = transformer.blocks[0].self_attn.rotary_embedding(refined_k.reshape_as(keys), *capture.rope)
        refined_v = refined_v.reshape_as(values)
        tokens = keys.shape[1]
        layer_caches = ar.state.get_kv_caches(ar.branch, seq_len=tokens, commit_current=True)
        layer_caches[0].forward_ctx.prepare(device=latents.device, action_len=0, query_len=tokens)
        for layer, key, value in zip(layer_caches, refined_k, refined_v, strict=True):
            slots = layer.to_layer_inputs()
            pool_write_chunk(slots.key_pool, slots.value_pool, key.unsqueeze(0), value.unsqueeze(0), slots.video_slots)
        ar.state.commit_paged_context(ar.branch)
