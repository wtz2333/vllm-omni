# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Load the released component tree without copying or converting checkpoints on disk."""

import json
from pathlib import Path

import torch
from diffusers.loaders.single_file_utils import convert_wan_vae_to_diffusers

from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_wan import DistributedAutoencoderKLWan
from vllm_omni.diffusion.utils.wan_vae import fix_wan22_residual_vae_keys
from vllm_omni.transformers_utils.repo_utils import hf_api


def component_file(model: str, filename: str, revision: str | None = None) -> str:
    if Path(model).is_dir():
        path = Path(model) / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing EdenLive component: {path}")
        return str(path)
    return hf_api().hf_hub_download(repo_id=model, filename=filename, revision=revision)


def component_config(model: str, subfolder: str, revision: str | None = None) -> dict:
    with open(component_file(model, f"{subfolder}/config.json", revision)) as handle:
        return json.load(handle)


def load_vae(model: str, dtype: torch.dtype, revision: str | None = None) -> DistributedAutoencoderKLWan:
    config = component_config(model, "vae-5b", revision)
    if config.get("z_dim") != 48 or config.get("patch_size") != 2 or not config.get("is_residual"):
        raise ValueError("EdenLive 5B requires the 48-channel residual Wan2.2 VAE")
    # The released JSON describes RGB; Diffusers constructs the folded convolutions.
    config.update(in_channels=12, out_channels=12)
    path = component_file(model, "vae-5b/Wan2.2_VAE.pth", revision)
    source = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    converted = fix_wan22_residual_vae_keys(source, convert_wan_vae_to_diffusers(source))
    with torch.device("meta"):
        vae = DistributedAutoencoderKLWan.from_config(config)
    vae.load_state_dict(converted, strict=True, assign=True)
    vae = vae.to(dtype=dtype).eval()
    vae.init_distributed()
    return vae


def decoder_state_bytes(vae: DistributedAutoencoderKLWan, height: int, width: int) -> int:
    """Bound the two retained input frames per causal convolution, using actual geometry."""
    decoder = vae.decoder
    height //= vae.config.scale_factor_spatial
    width //= vae.config.scale_factor_spatial
    elements = 2 * decoder.conv_in.in_channels * height * width
    for block in decoder.mid_block.resnets:
        elements += 2 * (block.conv1.in_channels + block.conv2.in_channels) * height * width
    for stage in decoder.up_blocks:
        for block in stage.resnets:
            elements += 2 * (block.conv1.in_channels + block.conv2.in_channels) * height * width
        if stage.upsampler is not None:
            if stage.upsampler.mode == "upsample3d":
                elements += 2 * stage.upsampler.time_conv.in_channels * height * width
            height *= 2
            width *= 2
    elements += 2 * decoder.conv_out.in_channels * height * width
    return elements * torch.empty((), dtype=vae.dtype).element_size()
