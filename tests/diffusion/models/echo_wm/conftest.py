# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only process groups and kernels for Echo-WM contract tests."""

from contextlib import contextmanager
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch


@contextmanager
def cpu_distributed(init_method: str, *, rank: int = 0, world_size: int = 1, sp_size: int = 1):
    from vllm_omni.diffusion.distributed.parallel_state import (
        destroy_distributed_env,
        init_distributed_environment,
        initialize_model_parallel,
    )

    # Initialize Gloo first: the diffusion convenience initializer otherwise
    # sets the platform's default accelerator device, even with backend=gloo.
    torch.distributed.init_process_group(
        "gloo", init_method=init_method, rank=rank, world_size=world_size, timeout=timedelta(seconds=60)
    )
    try:
        init_distributed_environment(world_size=world_size, rank=rank, local_rank=rank, backend="gloo")
        initialize_model_parallel(sequence_parallel_size=sp_size, ulysses_degree=sp_size, backend="gloo")
        yield
    finally:
        destroy_distributed_env()


@pytest.fixture(autouse=True)
def _init_distributed(tmp_path):
    with cpu_distributed((tmp_path / "distributed-init").as_uri()):
        yield


@contextmanager
def cpu_kernels(*, sp_size: int = 1):
    from vllm.model_executor.layers.utils import default_unquantized_gemm

    from vllm_omni.diffusion.attention.backends.sdpa import SDPABackend
    from vllm_omni.diffusion.config import set_current_diffusion_config
    from vllm_omni.diffusion.data import AttentionConfig

    config = SimpleNamespace(
        diffusion_attention_config=AttentionConfig(default="TORCH_SDPA"),
        parallel_config=SimpleNamespace(ring_degree=1, sequence_parallel_size=sp_size),
    )
    with pytest.MonkeyPatch.context() as patch, set_current_diffusion_config(config):
        patch.setattr(
            "vllm.model_executor.layers.linear.dispatch_unquantized_gemm",
            lambda: default_unquantized_gemm,
        )
        # A CUDA build probes device capability before honoring TORCH_SDPA.
        # Select the real CPU-capable kernel directly; attention math is real.
        patch.setattr("vllm_omni.diffusion.attention.selector._cached_get_backend_cls", lambda *_a, **_kw: SDPABackend)
        yield


@pytest.fixture(autouse=True)
def _force_cpu_kernels():
    with cpu_kernels():
        yield
