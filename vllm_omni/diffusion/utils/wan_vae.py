# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Native Wan2.2 residual VAE checkpoint conversion shared by world models."""

import torch


def fix_wan22_residual_vae_keys(
    source: dict[str, torch.Tensor], converted: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    """Restore Wan2.2 residual block nesting lost by Diffusers' converter."""
    converted = {
        key: value
        for key, value in converted.items()
        if not key.startswith(("encoder.down_blocks.", "decoder.up_blocks.", "decoder.upsamples."))
    }
    tail_replacements = {
        "residual.0.": "norm1.",
        "residual.2.": "conv1.",
        "residual.3.": "norm2.",
        "residual.6.": "conv2.",
        "shortcut.": "conv_shortcut.",
    }

    for key, value in source.items():
        parts = key.split(".")
        if len(parts) < 6:
            continue
        if parts[:2] == ["encoder", "downsamples"] and parts[3] == "downsamples":
            block, layer, tail = parts[2], int(parts[4]), ".".join(parts[5:])
            component = f"resnets.{layer}" if layer < 2 else "downsampler"
            prefix = f"encoder.down_blocks.{block}.{component}."
        elif parts[:2] == ["decoder", "upsamples"] and parts[3] == "upsamples":
            block, layer, tail = parts[2], int(parts[4]), ".".join(parts[5:])
            component = f"resnets.{layer}" if layer < 3 else "upsampler"
            prefix = f"decoder.up_blocks.{block}.{component}."
        else:
            continue
        for old, new in tail_replacements.items():
            tail = tail.replace(old, new)
        converted[prefix + tail] = value

    return converted
