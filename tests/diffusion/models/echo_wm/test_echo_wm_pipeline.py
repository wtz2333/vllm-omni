# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Echo-WM pipeline tests.

The centerpiece drives the reference ``causal_rollout`` (``ltx-causal``) and
the port's ``EchoWMCausalPipeline.forward`` on identical tiny weights and
inputs, pinning the full rollout semantics: noise draw order, the DMD
transition math, block iteration, clean commits, and output assembly.
Skipped when the reference repository is absent (CI); the request-parsing
contracts always run.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]

_ECHO_ROOT = Path("/data/wtz2333/WorldModel/JoyAI-Echo/echo_wm")

_TINY_TRANSFORMER_CONFIG = {
    "num_layers": 2,
    "num_attention_heads": 2,
    "attention_head_dim": 8,
    "in_channels": 16,
    "out_channels": 16,
    "audio_num_attention_heads": 2,
    "audio_attention_head_dim": 4,
    "audio_in_channels": 16,
    "audio_out_channels": 16,
    "cross_attention_dim": 16,
    "audio_cross_attention_dim": 8,
    "positional_embedding_max_pos": [20, 16, 16],
    "audio_positional_embedding_max_pos": [20],
    "rope_type": "split",
    "apply_gated_attention": True,
    "cross_attention_adaln": True,
}


@pytest.fixture(autouse=True)
def _init_distributed():
    from vllm.distributed.parallel_state import (
        cleanup_dist_env_and_memory,
        init_distributed_environment,
        initialize_model_parallel,
    )

    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29717")
    init_distributed_environment(
        world_size=1,
        rank=0,
        local_rank=0,
        distributed_init_method="env://",
    )
    initialize_model_parallel()
    yield
    cleanup_dist_env_and_memory()


@pytest.fixture(autouse=True)
def _force_default_gemm(monkeypatch):
    from vllm.model_executor.layers.utils import default_unquantized_gemm

    monkeypatch.setattr(
        "vllm.model_executor.layers.linear.dispatch_unquantized_gemm",
        lambda: default_unquantized_gemm,
    )


@pytest.fixture(autouse=True)
def _force_torch_sdpa():
    from vllm_omni.diffusion.config import set_current_diffusion_config
    from vllm_omni.diffusion.data import AttentionConfig

    od_config = SimpleNamespace(
        diffusion_attention_config=AttentionConfig(default="TORCH_SDPA"),
        parallel_config=SimpleNamespace(ring_degree=1),
    )
    with set_current_diffusion_config(od_config):
        yield


@pytest.fixture(autouse=True)
def _tiny_contract(monkeypatch):
    """Relax the checkpoint contract to the tiny geometry for CPU tests."""
    from vllm_omni.diffusion.models.echo_wm import transformer as transformer_module

    monkeypatch.setattr(
        transformer_module,
        "_ECHOWM_CHECKPOINT_CONTRACT",
        {
            "num_layers": 2,
            "num_attention_heads": 2,
            "attention_head_dim": 8,
            "in_channels": 16,
            "out_channels": 16,
            "audio_num_attention_heads": 2,
            "audio_attention_head_dim": 4,
            "audio_in_channels": 16,
            "audio_out_channels": 16,
            "cross_attention_dim": 16,
            "audio_cross_attention_dim": 8,
            "timestep_scale_multiplier": 1000,
            "positional_embedding_theta": 10000.0,
            "apply_gated_attention": True,
            "cross_attention_adaln": True,
            "rope_type": "split",
        },
    )


def _tiny_od_config(tmp_path) -> SimpleNamespace:
    """An od_config pointing at a tiny single-file checkpoint."""
    from safetensors.torch import save_file

    checkpoint = tmp_path / "echo-wm-tiny.safetensors"
    save_file(
        {"model.diffusion_model.dummy": torch.zeros(1)},
        str(checkpoint),
        metadata={"config": __import__("json").dumps({"transformer": _TINY_TRANSFORMER_CONFIG})},
    )
    return SimpleNamespace(
        model=str(checkpoint),
        model_config={"echo_wm_height": 64, "echo_wm_width": 128},
        parallel_config=SimpleNamespace(pipeline_parallel_size=1),
        quantization_config=None,
    )


def test_pipeline_parses_request_and_rejects_bad_shapes(tmp_path):
    from vllm_omni.diffusion.models.echo_wm.pipeline import EchoWMCausalPipeline

    pipeline = EchoWMCausalPipeline(od_config=_tiny_od_config(tmp_path))
    ppf = 8
    action = {
        "ucpe_viewmats": torch.eye(4).repeat(1, 4, 1, 1),
        "ucpe_Ks": torch.eye(3).repeat(1, 4, 1, 1),
    }
    embeds = (torch.randn(1, 6, 16), torch.randn(1, 6, 8))
    request = SimpleNamespace(
        num_reqs=1,
        prompts=[{"prompt": ""}],
        sampling_params=SimpleNamespace(
            num_outputs_per_prompt=1,
            num_frames=25,
            num_inference_steps=4,
            seed=42,
            extra_args={
                "echo_wm_image_latent": torch.randn(1, ppf, 16),
                "echo_wm_prompt_embeds": embeds,
                "echo_wm_action": action,
                "echo_wm_height": 64,
                "echo_wm_width": 128,
            },
        ),
    )
    inputs = pipeline._parse_request(request)
    assert inputs.num_frames == 25  # 1 + 3 * 8 pixel frames
    bad = SimpleNamespace(
        num_reqs=1,
        prompts=[""],
        sampling_params=SimpleNamespace(
            num_outputs_per_prompt=1,
            num_frames=25,
            num_inference_steps=8,
            seed=42,
            extra_args=request.sampling_params.extra_args,
        ),
    )
    with pytest.raises(ValueError, match="num_inference_steps=4"):
        pipeline._parse_request(bad)


def test_pipeline_forward_matches_reference_rollout(tmp_path):
    if not (_ECHO_ROOT / "ltx-causal" / "src").exists():
        pytest.skip("Echo-WM reference repository not available")
    for package in ("ltx-core/src", "ltx-causal/src"):
        sys.path.insert(0, str(_ECHO_ROOT / package))

    from ltx_causal.cache import CausalCacheConfig
    from ltx_causal.causal_wrapper import CausalModelWrapper
    from ltx_causal.rollout import causal_rollout
    from ltx_causal.scheduling import causal_audio_frames

    # Build the reference tiny model (identical to the transformer parity test).
    import tests.diffusion.models.echo_wm.test_echo_wm_transformer as transformer_tests
    from vllm_omni.diffusion.models.echo_wm import transformer as transformer_module
    from vllm_omni.diffusion.models.echo_wm.pipeline import EchoWMCausalPipeline

    ref = transformer_tests._reference_modules()
    torch.manual_seed(31)
    reference_model = transformer_tests._tiny_reference(ref).eval()

    od_config = _tiny_od_config(tmp_path)
    pipeline = EchoWMCausalPipeline(od_config=od_config)
    pipeline.dtype = torch.float32
    pipeline.device = torch.device("cpu")
    # Rebuild the transformer on CPU float32 and copy the reference weights.
    pipeline.transformer = transformer_module.EchoWMTransformer3DModel(
        **{
            key: value
            for key, value in _TINY_TRANSFORMER_CONFIG.items()
            if key
            not in (
                "positional_embedding_max_pos",
                "audio_positional_embedding_max_pos",
                "rope_type",
                "apply_gated_attention",
                "cross_attention_adaln",
            )
        },
        positional_embedding_max_pos=(20, 16, 16),
        audio_positional_embedding_max_pos=(20,),
        apply_gated_attention=True,
        cross_attention_adaln=True,
        ucpe=transformer_module.EchoWMUCPEConfig(
            attn_dim=16, num_heads=2, patches_x=4, patches_y=2, image_width=64, image_height=32
        ),
    ).eval()
    transformer_tests._copy_reference_weights(reference_model, pipeline.transformer)

    latent_frames = 7  # 1 + 2 * 3
    num_frames = (latent_frames - 1) * 8 + 1
    ppf = 8
    action = transformer_tests._se3_cameras(latent_frames)
    embeds = (torch.randn(1, 6, 16), torch.randn(1, 6, 8))
    image_latent = torch.randn(1, ppf, 16)
    seed = 123

    # Reference rollout.
    wrapper = CausalModelWrapper(
        reference_model,
        patches_per_frame=ppf,
        cache=CausalCacheConfig(video_local_attn_size=19, video_sink_size=7),
    )
    video_positions = transformer_tests.build_video_positions(latent_frames, height=64, width=128, fps=24.0)
    audio_positions = transformer_tests.build_audio_positions(causal_audio_frames(latent_frames))
    with torch.inference_mode():
        ref_video, ref_audio = causal_rollout(
            wrapper=wrapper,
            clean_video=torch.cat([image_latent, torch.zeros(1, (latent_frames - 1) * ppf, 16)], dim=1),
            clean_audio=torch.zeros(1, causal_audio_frames(latent_frames), 16),
            video_positions=video_positions,
            audio_positions=audio_positions,
            video_context=embeds[0],
            audio_context=embeds[1],
            context_mask=None,
            action_cond={"ucpe_viewmats": action[0], "ucpe_Ks": action[1]},
            seed=seed,
        )

    # Port rollout through the pipeline.
    request = SimpleNamespace(
        num_reqs=1,
        prompts=[{"prompt": ""}],
        sampling_params=SimpleNamespace(
            num_outputs_per_prompt=1,
            num_frames=num_frames,
            num_inference_steps=4,
            seed=seed,
            extra_args={
                "echo_wm_image_latent": image_latent,
                "echo_wm_prompt_embeds": embeds,
                "echo_wm_action": {"ucpe_viewmats": action[0], "ucpe_Ks": action[1]},
                "echo_wm_height": 64,
                "echo_wm_width": 128,
            },
        ),
    )
    with torch.inference_mode():
        output = pipeline.forward(request)
    latents = output.output["payload"]["latents"]
    torch.testing.assert_close(latents["video"], ref_video, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(latents["audio"], ref_audio, rtol=1e-4, atol=1e-4)
    assert latents["latent_frames"] == latent_frames
    assert latents["audio_frames"] == causal_audio_frames(latent_frames)
