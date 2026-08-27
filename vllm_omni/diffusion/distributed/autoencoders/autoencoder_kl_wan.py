# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
from diffusers.models.autoencoders import AutoencoderKLWan
from diffusers.models.autoencoders.autoencoder_kl_wan import unpatchify
from diffusers.models.autoencoders.vae import DecoderOutput
from diffusers.utils.accelerate_utils import apply_forward_hook
from vllm.logger import init_logger

from vllm_omni.diffusion.distributed.autoencoders import wan_spatial_shard
from vllm_omni.diffusion.distributed.autoencoders.distributed_vae_executor import (
    DistributedOperator,
    DistributedVaeMixin,
    GridSpec,
    TileTask,
)
from vllm_omni.platforms import current_omni_platform

logger = init_logger(__name__)


@dataclass
class WanVAEStreamingDecodeState:
    """Per-session causal-convolution state for incremental Wan VAE decode."""

    feat_cache: list[Any]
    decoded_latent_frames: int = 0
    used_cache_slots: int = 0
    batch_size: int | None = None
    latent_height: int | None = None
    latent_width: int | None = None
    dtype: torch.dtype | None = None
    device: torch.device | None = None

    def bind_input(self, z: torch.Tensor) -> None:
        if z.ndim != 5:
            raise ValueError(f"Wan streaming VAE decode expects a 5D latent tensor, got shape {tuple(z.shape)}.")
        if z.shape[0] != 1:
            raise ValueError(f"Wan streaming VAE decode supports batch size 1, got {z.shape[0]}.")
        if z.shape[2] <= 0:
            raise ValueError("Wan streaming VAE decode requires at least one latent frame.")
        signature = (int(z.shape[0]), int(z.shape[3]), int(z.shape[4]), z.dtype, z.device)
        current = (self.batch_size, self.latent_height, self.latent_width, self.dtype, self.device)
        if self.batch_size is None:
            self.batch_size, self.latent_height, self.latent_width, self.dtype, self.device = signature
        elif current != signature:
            raise ValueError(
                "Wan streaming VAE state cannot change batch, spatial shape, dtype, or device: "
                f"expected {current}, got {signature}."
            )

    def nbytes(self) -> int:
        """Return deduplicated storage bytes retained by the feature cache."""

        storages: set[tuple[str, int | None, int, int]] = set()
        total = 0
        for value in self.feat_cache:
            if not isinstance(value, torch.Tensor):
                continue
            storage = value.untyped_storage()
            key = (value.device.type, value.device.index, storage.data_ptr(), storage.nbytes())
            if key in storages:
                continue
            storages.add(key)
            total += storage.nbytes()
        return total

    def clear(self) -> None:
        """Drop every retained tensor and make the state reusable from frame zero."""

        self.feat_cache[:] = [None] * len(self.feat_cache)
        self.decoded_latent_frames = 0
        self.used_cache_slots = 0
        self.batch_size = None
        self.latent_height = None
        self.latent_width = None
        self.dtype = None
        self.device = None


class OmniAutoencoderKLWan(AutoencoderKLWan):
    def _execution_context(self):
        try:
            first_param = next(self.parameters())
        except StopIteration:
            return nullcontext()

        dtype = first_param.dtype
        if dtype not in (torch.float16, torch.bfloat16):
            return nullcontext()

        return current_omni_platform.create_autocast_context(
            device_type=first_param.device.type,
            dtype=dtype,
            enabled=True,
        )

    def encode(self, x: torch.Tensor, return_dict: bool = True):
        with self._execution_context():
            return super().encode(x, return_dict=return_dict)

    def decode(self, z: torch.Tensor, return_dict: bool = True):
        with self._execution_context():
            return super().decode(z, return_dict=return_dict)

    def create_streaming_decode_state(self) -> WanVAEStreamingDecodeState:
        """Allocate an empty decoder cache owned by one external session."""

        cached_counts = getattr(self, "_cached_conv_counts", None)
        if not isinstance(cached_counts, dict):
            raise RuntimeError("Wan VAE streaming decode requires cached decoder convolution counts.")
        decoder_slots = int(cached_counts.get("decoder", 0))
        if decoder_slots <= 0:
            raise RuntimeError("Wan VAE streaming decode requires at least one decoder cache slot.")
        return WanVAEStreamingDecodeState(feat_cache=[None] * decoder_slots)

    @apply_forward_hook
    def decode_streaming(
        self,
        z: torch.Tensor,
        state: WanVAEStreamingDecodeState,
        *,
        return_dict: bool = True,
    ) -> DecoderOutput | tuple[torch.Tensor]:
        """Decode new latent frames while preserving causal state across calls."""

        if not isinstance(state, WanVAEStreamingDecodeState):
            raise TypeError("Wan streaming VAE decode requires WanVAEStreamingDecodeState.")
        if self.use_tiling:
            raise NotImplementedError("Wan stateful streaming VAE decode does not support spatial tiling.")
        state.bind_input(z)
        decoded_frames: list[torch.Tensor] = []
        with self._execution_context():
            x = self.post_quant_conv(z)
            for frame_index in range(z.shape[2]):
                feat_idx = [0]
                decoded = self.decoder(
                    x[:, :, frame_index : frame_index + 1],
                    feat_cache=state.feat_cache,
                    feat_idx=feat_idx,
                    first_chunk=state.decoded_latent_frames == 0,
                )
                if feat_idx[0] > len(state.feat_cache):
                    raise RuntimeError(
                        "Wan decoder consumed more cache slots than allocated: "
                        f"used={feat_idx[0]}, allocated={len(state.feat_cache)}."
                    )
                state.used_cache_slots = max(state.used_cache_slots, feat_idx[0])
                state.decoded_latent_frames += 1
                decoded_frames.append(decoded)

        output = torch.cat(decoded_frames, dim=2)
        if self.config.patch_size is not None:
            output = unpatchify(output, patch_size=self.config.patch_size)
        output = torch.clamp(output, min=-1.0, max=1.0)
        if not return_dict:
            return (output,)
        return DecoderOutput(sample=output)


class DistributedAutoencoderKLWan(OmniAutoencoderKLWan, DistributedVaeMixin):
    @classmethod
    def from_pretrained(cls, *args: Any, **kwargs: Any):
        model = super().from_pretrained(*args, **kwargs)
        model.init_distributed()
        return model

    def tile_split(self, z: torch.Tensor) -> tuple[list[TileTask], GridSpec]:
        _, _, num_frames, height, width = z.shape
        sample_height = height * self.spatial_compression_ratio
        sample_width = width * self.spatial_compression_ratio

        tile_latent_min_height = self.tile_sample_min_height // self.spatial_compression_ratio
        tile_latent_min_width = self.tile_sample_min_width // self.spatial_compression_ratio
        tile_latent_stride_height = self.tile_sample_stride_height // self.spatial_compression_ratio
        tile_latent_stride_width = self.tile_sample_stride_width // self.spatial_compression_ratio
        tile_sample_stride_height = self.tile_sample_stride_height
        tile_sample_stride_width = self.tile_sample_stride_width
        if self.config.patch_size is not None:
            sample_height = sample_height // self.config.patch_size
            sample_width = sample_width // self.config.patch_size
            tile_sample_stride_height = tile_sample_stride_height // self.config.patch_size
            tile_sample_stride_width = tile_sample_stride_width // self.config.patch_size
            blend_height = self.tile_sample_min_height // self.config.patch_size - tile_sample_stride_height
            blend_width = self.tile_sample_min_width // self.config.patch_size - tile_sample_stride_width
        else:
            blend_height = self.tile_sample_min_height - tile_sample_stride_height
            blend_width = self.tile_sample_min_width - tile_sample_stride_width

        tiletask_list = []
        for i in range(0, height, tile_latent_stride_height):
            for j in range(0, width, tile_latent_stride_width):
                time_list = []
                for k in range(num_frames):
                    self._conv_idx = [0]
                    tile = z[:, :, k : k + 1, i : i + tile_latent_min_height, j : j + tile_latent_min_width]
                    time_list.append(tile)
                tiletask_list.append(
                    TileTask(
                        len(tiletask_list),
                        (i // tile_latent_stride_height, j // tile_latent_stride_width),
                        time_list,
                        workload=time_list[0].shape[3] * time_list[0].shape[4],
                    )
                )
        tile_spec = {
            "sample_height": sample_height,
            "sample_width": sample_width,
            "blend_height": blend_height,
            "blend_width": blend_width,
            "tile_sample_stride_height": tile_sample_stride_height,
            "tile_sample_stride_width": tile_sample_stride_width,
        }
        grid_spec = GridSpec(
            split_dims=(3, 4),
            grid_shape=(tiletask_list[-1].grid_coord[0] + 1, tiletask_list[-1].grid_coord[1] + 1),
            tile_spec=tile_spec,
            output_dtype=self.dtype,
        )
        return tiletask_list, grid_spec

    def tile_exec(self, task: TileTask) -> torch.Tensor:
        """Decode a single latent tile into RGB space."""
        self.clear_cache()
        time = []
        with self._execution_context():
            for k in range(len(task.tensor)):
                self._conv_idx = [0]
                tile = self.post_quant_conv(task.tensor[k])
                decoded = self.decoder(tile, feat_cache=self._feat_map, feat_idx=self._conv_idx, first_chunk=(k == 0))
                time.append(decoded)
        result = torch.cat(time, dim=2)
        return result

    def encode_tile_split(self, x: torch.Tensor) -> tuple[list[TileTask], GridSpec]:
        _, _, num_frames, height, width = x.shape
        encode_spatial_compression_ratio = self.spatial_compression_ratio
        # Scale tile parameters for patchified coordinate system
        tile_sample_min_height = self.tile_sample_min_height
        tile_sample_min_width = self.tile_sample_min_width
        tile_sample_stride_height = self.tile_sample_stride_height
        tile_sample_stride_width = self.tile_sample_stride_width
        if self.config.patch_size is not None:
            assert encode_spatial_compression_ratio % self.config.patch_size == 0
            encode_spatial_compression_ratio = self.spatial_compression_ratio // self.config.patch_size
            # When input is patchified, scale tile parameters accordingly
            tile_sample_min_height = tile_sample_min_height // self.config.patch_size
            tile_sample_min_width = tile_sample_min_width // self.config.patch_size
            tile_sample_stride_height = tile_sample_stride_height // self.config.patch_size
            tile_sample_stride_width = tile_sample_stride_width // self.config.patch_size

        latent_height = height // encode_spatial_compression_ratio
        latent_width = width // encode_spatial_compression_ratio

        tile_latent_min_height = tile_sample_min_height // encode_spatial_compression_ratio
        tile_latent_min_width = tile_sample_min_width // encode_spatial_compression_ratio
        tile_latent_stride_height = tile_sample_stride_height // encode_spatial_compression_ratio
        tile_latent_stride_width = tile_sample_stride_width // encode_spatial_compression_ratio

        blend_height = tile_latent_min_height - tile_latent_stride_height
        blend_width = tile_latent_min_width - tile_latent_stride_width

        tiletask_list = []
        temporal_compression = self.config.scale_factor_temporal
        for i in range(0, height, tile_sample_stride_height):
            for j in range(0, width, tile_sample_stride_width):
                time_list = []
                frame_range = 1 + (num_frames - 1) // temporal_compression
                for k in range(frame_range):
                    if k == 0:
                        tile = x[:, :, :1, i : i + tile_sample_min_height, j : j + tile_sample_min_width]
                    else:
                        tile = x[
                            :,
                            :,
                            1 + temporal_compression * (k - 1) : 1 + temporal_compression * k,
                            i : i + tile_sample_min_height,
                            j : j + tile_sample_min_width,
                        ]
                    time_list.append(tile)
                tiletask_list.append(
                    TileTask(
                        len(tiletask_list),
                        (i // tile_sample_stride_height, j // tile_sample_stride_width),
                        time_list,
                        workload=time_list[0].shape[3] * time_list[0].shape[4],
                    )
                )

        grid_spec = GridSpec(
            split_dims=(3, 4),
            grid_shape=(tiletask_list[-1].grid_coord[0] + 1, tiletask_list[-1].grid_coord[1] + 1),
            tile_spec={
                "latent_height": latent_height,
                "latent_width": latent_width,
                "blend_height": blend_height,
                "blend_width": blend_width,
                "tile_latent_stride_height": tile_latent_stride_height,
                "tile_latent_stride_width": tile_latent_stride_width,
            },
            output_dtype=self.dtype,
        )
        return tiletask_list, grid_spec

    def encode_tile_exec(self, task: TileTask) -> torch.Tensor:
        """Encode a single sample tile into latent space."""
        self.clear_cache()
        time = []
        for k, tile in enumerate(task.tensor):
            self._enc_conv_idx = [0]
            encoded = self.encoder(tile, feat_cache=self._enc_feat_map, feat_idx=self._enc_conv_idx)
            encoded = self.quant_conv(encoded)
            time.append(encoded)
        result = torch.cat(time, dim=2)
        self.clear_cache()
        return result

    def encode_tile_merge(
        self, coord_tensor_map: dict[tuple[int, ...], torch.Tensor], grid_spec: GridSpec
    ) -> torch.Tensor:
        """Merge encoded tiles into a full latent tensor."""
        grid_h, grid_w = grid_spec.grid_shape
        result_rows = []
        for i in range(grid_h):
            result_row = []
            for j in range(grid_w):
                tile = coord_tensor_map[(i, j)]
                if i > 0:
                    tile = self.blend_v(coord_tensor_map[(i - 1, j)], tile, grid_spec.tile_spec["blend_height"])
                if j > 0:
                    tile = self.blend_h(coord_tensor_map[(i, j - 1)], tile, grid_spec.tile_spec["blend_width"])
                result_row.append(
                    tile[
                        :,
                        :,
                        :,
                        : grid_spec.tile_spec["tile_latent_stride_height"],
                        : grid_spec.tile_spec["tile_latent_stride_width"],
                    ]
                )
            result_rows.append(torch.cat(result_row, dim=-1))

        enc = torch.cat(result_rows, dim=3)[
            :, :, :, : grid_spec.tile_spec["latent_height"], : grid_spec.tile_spec["latent_width"]
        ]
        return enc

    def tile_merge(self, coord_tensor_map: dict[tuple[int, ...], torch.Tensor], grid_spec: GridSpec) -> torch.Tensor:
        """Merge decoded tiles into a full image."""
        grid_h, grid_w = grid_spec.grid_shape
        result_rows = []
        self.clear_cache()
        for i in range(grid_h):
            result_row = []
            for j in range(grid_w):
                tile = coord_tensor_map[(i, j)]
                if i > 0:
                    tile = self.blend_v(coord_tensor_map[(i - 1, j)], tile, grid_spec.tile_spec["blend_height"])
                if j > 0:
                    tile = self.blend_h(coord_tensor_map[(i, j - 1)], tile, grid_spec.tile_spec["blend_width"])
                result_row.append(
                    tile[
                        :,
                        :,
                        :,
                        : grid_spec.tile_spec["tile_sample_stride_height"],
                        : grid_spec.tile_spec["tile_sample_stride_width"],
                    ]
                )
            result_rows.append(torch.cat(result_row, dim=-1))

        dec = torch.cat(result_rows, dim=3)[
            :, :, :, : grid_spec.tile_spec["sample_height"], : grid_spec.tile_spec["sample_width"]
        ]

        if self.config.patch_size is not None:
            dec = unpatchify(dec, patch_size=self.config.patch_size)

        dec = torch.clamp(dec, min=-1.0, max=1.0)
        return dec

    def _spatial_shard_decode_split_dim(self) -> str | None:
        mode = self.distributed_executor.parallel_mode
        if mode == "spatial_shard_width":
            return "width"
        if mode == "spatial_shard_height":
            return "height"
        return None

    def _spatial_shard_decode_enabled(self, z: torch.Tensor) -> bool:
        split_dim = self._spatial_shard_decode_split_dim()
        if split_dim is None:
            return False
        if z.ndim != 5:
            logger.warning("Wan VAE spatial sharded decode expects 5D latent input; falling back to tiled decode.")
            return False
        if not self.is_distributed_enabled():
            return False

        group = self.distributed_executor.group
        world_size = dist.get_world_size(group=group)
        requested_size = int(self.distributed_executor.parallel_size)
        if requested_size != world_size:
            logger.warning(
                "Wan VAE spatial sharded decode currently requires vae_patch_parallel_size "
                "to match the DIT group size; falling back to tiled decode. "
                "requested=%s dit_group=%s split_dim=%s",
                requested_size,
                world_size,
                split_dim,
            )
            return False
        return True

    def tiled_decode(self, z: torch.Tensor, return_dict: bool = True):
        if not self.is_distributed_enabled():
            return super().tiled_decode(z, return_dict=return_dict)

        if self._spatial_shard_decode_enabled(z):
            split_dim = self._spatial_shard_decode_split_dim()
            logger.debug("Decode running with Wan VAE spatial_shard_%s mode", split_dim)
            return wan_spatial_shard.spatial_shard_decode(
                self,
                z,
                group=self.distributed_executor.group,
                return_dict=return_dict,
                split_dim=split_dim,
            )

        logger.debug("Decode running with distributed executor")
        result = self.distributed_executor.execute(
            z,
            DistributedOperator(split=self.tile_split, exec=self.tile_exec, merge=self.tile_merge),
            broadcast_result=False,
        )
        if not return_dict:
            return (result,)

        return DecoderOutput(sample=result)

    def tiled_encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode using distributed VAE executor.

        Note: x is already patchified by parent's _encode() before calling this method.
        """
        if not self.is_distributed_enabled():
            return super().tiled_encode(x)

        logger.debug("Encode running with distributed executor")
        self.clear_cache()
        result = self.distributed_executor.execute(
            x,
            DistributedOperator(
                split=self.encode_tile_split,
                exec=self.encode_tile_exec,
                merge=self.encode_tile_merge,
            ),
            broadcast_result=True,
        )
        self.clear_cache()
        return result
