# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The released 5B four-step stochastic flow schedule (shift=5)."""

import torch
from diffusers.utils.torch_utils import randn_tensor

TIMESTEPS = (1000.0, 937.5, 833.3333333333, 625.0)


def step(latents: torch.Tensor, prediction: torch.Tensor, index: int, generator: torch.Generator) -> torch.Tensor:
    if not 0 <= index < len(TIMESTEPS):
        raise ValueError("EdenLive uses exactly four denoising steps")
    # The reference keeps the model timestep in FP32 but casts sigma to latent dtype.
    sigma = latents.new_tensor(TIMESTEPS[index] / 1000)
    clean = latents - sigma * prediction
    if index == len(TIMESTEPS) - 1:
        return clean
    sigma_next = latents.new_tensor(TIMESTEPS[index + 1] / 1000)
    noise = randn_tensor(clean.shape, generator=generator, device=clean.device, dtype=clean.dtype)
    return ((1 - sigma_next) * clean + sigma_next * noise).to(latents.dtype)
