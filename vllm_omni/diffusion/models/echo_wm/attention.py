# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""SDPA kernel selection matching the released Echo-WM inference path."""

from contextlib import nullcontext

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel


def upstream_sdpa_kernel(device: torch.device | str, *, text_cross: bool = False):
    """Preserve the reference's flash/self and efficient/text-cross kernels.

    The released text mask is all zeros, but makes XFormers choose its
    efficient kernel. Dropping that mask must not also change the kernel.
    Newer PyTorch can otherwise prefer cuDNN, whose BF16 rounding differs
    enough to change a long distilled rollout. CPU and unsupported fused
    shapes retain the ordinary math implementation.
    """
    if torch.device(device).type != "cuda":
        return nullcontext()
    kernels = [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]
    if not text_cross:
        kernels.insert(0, SDPBackend.FLASH_ATTENTION)
    return sdpa_kernel(kernels)
