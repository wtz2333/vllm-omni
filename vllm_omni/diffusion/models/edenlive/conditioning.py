# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Released EdenLive 5B Proxy-DINO preprocessing and projection."""

from contextlib import nullcontext

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel


class ProxyDinoEncoder(nn.Module):
    def __init__(self, backbone: nn.Module, control_dim: int = 1024) -> None:
        super().__init__()
        self.backbone = backbone
        self.global_projection = nn.Linear(backbone.config.hidden_size, control_dim, dtype=torch.float32)
        self.patch_projection = nn.Linear(backbone.config.hidden_size, control_dim, dtype=torch.float32)

    def forward(self, frames: torch.Tensor, *, latent_size: tuple[int, int], dtype: torch.dtype) -> torch.Tensor:
        """Map selected [T,3,H,W] RGB frames in [0,1] onto the latent grid."""
        pixels = F.interpolate(
            frames.float().contiguous(), size=(480, 832), mode="bilinear", align_corners=False, antialias=True
        )
        mean = pixels.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
        std = pixels.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
        # The released 5B recipe autocasts both backbone and projection to BF16.
        with torch.autocast(device_type=pixels.device.type, dtype=dtype, enabled=dtype != torch.float32):
            # vLLM disables cuDNN SDPA globally; match the reference locally.
            attention = sdpa_kernel(SDPBackend.CUDNN_ATTENTION) if pixels.device.type == "cuda" else nullcontext()
            with attention:
                tokens = self.backbone(pixel_values=(pixels - mean) / std).last_hidden_state
            first = 1 + self.backbone.config.num_register_tokens
            grid_h, grid_w = 480 // self.backbone.config.patch_size, 832 // self.backbone.config.patch_size
            grid = tokens[:, first : first + grid_h * grid_w].reshape(frames.shape[0], grid_h, grid_w, -1)
            # The reference stacks per-frame grids before projection (a contiguous GEMM input).
            features = F.normalize(self.patch_projection(grid.contiguous()), p=2, dim=-1).float()
        features = features.permute(3, 0, 1, 2).unsqueeze(0).contiguous().to(dtype)
        return F.interpolate(features, size=(frames.shape[0], *latent_size), mode="trilinear", align_corners=False)


def projection_weights(weights: dict[str, torch.Tensor], model_keys: set[str] | None = None) -> dict[str, torch.Tensor]:
    """Normalize the released encoder checkpoint, including its frozen backbone."""
    converted = {}
    for name, value in weights.items():
        for prefix in ("model.module.", "model.", "module."):
            if name.startswith(prefix):
                name = name[len(prefix) :]
                break
        name = name.removeprefix("encoder.")
        if not name.startswith(("backbone.", "global_projection.", "patch_projection.")):
            raise ValueError(f"Unexpected EdenLive projection parameter: {name}")
        wrapped = "backbone.model." + name.removeprefix("backbone.")
        if name.startswith("backbone.") and model_keys is not None and wrapped in model_keys:
            name = wrapped
        if name in converted:
            raise ValueError(f"Duplicate EdenLive projection parameter: {name}")
        converted[name] = value
    expected = {f"{part}.{kind}" for part in ("global_projection", "patch_projection") for kind in ("weight", "bias")}
    if not expected.issubset(converted):
        raise ValueError(f"Incomplete EdenLive projection checkpoint: expected {sorted(expected)}")
    return converted
