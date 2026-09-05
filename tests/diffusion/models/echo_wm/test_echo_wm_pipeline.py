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

_ECHO_ROOT = Path(os.environ.get("ECHOWM_REFERENCE_ROOT", ""))

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
    "connector_num_layers": 1,
    "connector_num_attention_heads": 2,
    "connector_attention_head_dim": 8,
    "audio_connector_num_attention_heads": 2,
    "audio_connector_attention_head_dim": 4,
}


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
            if not key.startswith(("connector_", "audio_connector_"))
            and key
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


@pytest.mark.parametrize("missing_weight", [False, True])
def test_standard_loader_tracks_pipeline_parameter_names(tmp_path, missing_weight):
    from safetensors.torch import save_file
    from torch import nn

    from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
    from vllm_omni.diffusion.models.echo_wm.pipeline import EchoWMCausalPipeline
    from vllm_omni.diffusion.models.echo_wm.text_stack import EchoWMTextStack
    from vllm_omni.diffusion.models.echo_wm.transformer import EchoWMTransformer3DModel

    # Keep real checkpoint mapping and standard-loader coverage checks while
    # using two tiny modules so the regression does not need model weights.
    transformer = EchoWMTransformer3DModel.__new__(EchoWMTransformer3DModel)
    nn.Module.__init__(transformer)
    transformer.patchify_proj = nn.Linear(1, 1)
    text_stack = EchoWMTextStack.__new__(EchoWMTextStack)
    nn.Module.__init__(text_stack)
    text_stack.video_aggregate_embed = nn.Linear(1, 1)
    pipeline = EchoWMCausalPipeline.__new__(EchoWMCausalPipeline)
    nn.Module.__init__(pipeline)
    pipeline.transformer = transformer
    pipeline.text_stack = text_stack
    pipeline.weights_sources = []
    pipeline.model_path = str(tmp_path / "weights.safetensors")
    weights = {
        "model.diffusion_model.patchify_proj.weight": torch.full((1, 1), 2.0),
        "model.diffusion_model.patchify_proj.bias": torch.full((1,), 3.0),
        "text_embedding_projection.video_aggregate_embed.weight": torch.full((1, 1), 4.0),
        "text_embedding_projection.video_aggregate_embed.bias": torch.full((1,), 5.0),
    }
    if missing_weight:
        weights.pop("text_embedding_projection.video_aggregate_embed.bias")
    save_file(weights, pipeline.model_path)
    loader = DiffusersPipelineLoader.__new__(DiffusersPipelineLoader)
    loader.quant_config = None
    if missing_weight:
        with pytest.raises(ValueError, match="text_stack.video_aggregate_embed.bias"):
            loader.load_weights(pipeline)
    else:
        loader.load_weights(pipeline)
        assert transformer.patchify_proj.weight.item() == 2.0
        assert transformer.patchify_proj.bias.item() == 3.0
        assert text_stack.video_aggregate_embed.weight.item() == 4.0
        assert text_stack.video_aggregate_embed.bias.item() == 5.0


def _public_request(request_id="echo", seed=0):
    from vllm_omni.diffusion.request import OmniDiffusionRequest
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    return OmniDiffusionRequest(
        request_id=request_id,
        prompt="test prompt",
        sampling_params=OmniDiffusionSamplingParams(
            num_frames=49,
            num_inference_steps=4,
            output_type="latent",
            seed=seed,
            extra_args={
                "echo_wm_image_latent": torch.ones(1, 8, 16),
                "echo_wm_prompt_embeds": (torch.ones(1, 6, 16), torch.ones(1, 6, 8)),
                "echo_wm_action": {
                    "ucpe_viewmats": torch.eye(4).repeat(1, 7, 1, 1),
                    "ucpe_Ks": torch.eye(3).repeat(1, 7, 1, 1),
                },
            },
        ),
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_public_request_preserves_zero_seed_geometry_and_weight_dtype(tmp_path, dtype):
    from vllm_omni.diffusion.models.echo_wm.pipeline import EchoWMCausalPipeline
    from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch

    config = _tiny_od_config(tmp_path)
    config.dtype = dtype
    pipeline = EchoWMCausalPipeline(od_config=config).to(dtype=dtype)
    request = _public_request()
    parsed = pipeline._parse_request(DiffusionRequestBatch([request]))
    assert parsed.seed == 0
    assert (parsed.height, parsed.width) == (64, 128)
    assert parsed.image_latent_tokens.dtype == dtype
    request.sampling_params.height = 96
    with pytest.raises(ValueError, match="must match model_config"):
        pipeline._parse_request(DiffusionRequestBatch([request]))


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("decode_media", [False, True])
def test_real_scheduler_runner_completes_multiple_chunks_and_requests(tmp_path, monkeypatch, streaming, decode_media):
    from contextlib import nullcontext

    from vllm_omni.diffusion.models.echo_wm.pipeline import EchoWMCausalPipeline
    from vllm_omni.diffusion.sched import StepScheduler
    from vllm_omni.diffusion.worker import diffusion_model_runner as runner_module
    from vllm_omni.diffusion.worker.diffusion_model_runner import DiffusionModelRunner
    from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch

    config = _tiny_od_config(tmp_path)
    config.streaming_output = streaming
    config.dtype = torch.float32
    config.cache_backend = None
    config.model_class_name = "EchoWMCausalPipeline"
    config.parallel_config.use_hsdp = False
    pipeline = EchoWMCausalPipeline(od_config=config)

    # Only the expensive DiT is substituted. Real pipeline preparation, noise
    # transitions, chunk commits, StepScheduler and Runner batching all execute.
    def probe(session, inputs, video, audio, video_sigma, audio_sigma, video_start, audio_start):
        return (
            video - video_sigma * 0.125 if video is not None else None,
            audio - (audio_sigma if audio_sigma is not None else video_sigma) * 0.25 if audio is not None else None,
        )

    def decode(video, audio, *, height, width, generator):
        num_frames = (video.shape[1] // 8 - 1) * 8 + 1
        frames = torch.arange(num_frames, dtype=torch.uint8)[:, None, None, None]
        samples = torch.arange(audio.shape[1] * 10, dtype=torch.float32)[None]
        return {"video": frames.expand(-1, height, width, 3), "audio": samples, "sample_rate": 24000}

    pipeline._media = SimpleNamespace(decode=decode)
    monkeypatch.setattr(pipeline, "_forward_block", probe)
    monkeypatch.setattr(runner_module, "set_forward_context", lambda **kwargs: nullcontext())
    runner = DiffusionModelRunner.__new__(DiffusionModelRunner)
    runner.pipeline = pipeline
    runner.od_config = config
    runner.vllm_config = None
    runner.device = torch.device("cpu")
    runner.state_cache = {}
    runner.kv_transfer_manager = SimpleNamespace(receive_multi_kv_cache_distributed=lambda *a, **kw: None)
    scheduler = StepScheduler()
    scheduler.initialize(config)
    final_videos = []
    for request_index in range(2):
        request = _public_request(f"echo-{request_index}", seed=request_index)
        if decode_media:
            request.sampling_params.output_type = "pil"
        baseline = pipeline.forward(DiffusionRequestBatch([request])).output["payload"]
        expected = baseline if decode_media else baseline["latents"]
        scheduler.add_request(request)
        emitted = []
        for tick in range(12):
            scheduled = scheduler.schedule()
            result = runner._execute_stepwise(scheduled, validate_kv_metadata=True, record_output_peak_memory=False)
            item = result.get_request_output(request.request_id)
            assert item is not None
            assert item.result is None or item.result.error is None
            assert item.step_index == tick + 1
            assert item.finished == (tick == 11)
            if item.result is not None:
                emitted.append(item.result)
            scheduler.update_from_output(scheduled, result)
        assert not scheduler.has_requests()
        assert request.request_id not in runner.state_cache
        assert [item.chunk_index for item in emitted] == ([0, 1, 2] if streaming else [2])
        assert [item.finished for item in emitted] == ([False, False, True] if streaming else [True])
        for modality in ("video", "audio"):
            payloads = [item.output["payload"] for item in emitted]
            actual = torch.cat(
                [(payload if decode_media else payload["latents"])[modality] for payload in payloads],
                dim=(0 if modality == "video" else -1) if decode_media else 1,
            )
            torch.testing.assert_close(actual, expected[modality], rtol=0, atol=0)
        if decode_media:
            from vllm_omni.diffusion.output_formatter import (
                format_diffusion_outputs,
                normalize_diffusion_postprocess_output,
            )

            for chunk in emitted:
                formatted = format_diffusion_outputs(
                    request=request,
                    od_config=config,
                    diffusion_output=chunk,
                    output_data=chunk.output,
                    postprocess_output=normalize_diffusion_postprocess_output(chunk.output),
                )[0]
                assert formatted.images[0].shape[-3:] == (64, 128, 3)
                assert formatted.multimodal_output["audio_sample_rate"] == 24000
        final_videos.append(expected["video"])
    if not decode_media:
        assert not torch.equal(final_videos[0], final_videos[1])


def test_public_image_and_action_string_reach_preprocessing(tmp_path, monkeypatch):
    from vllm_omni.diffusion.models.echo_wm.pipeline import EchoWMCausalPipeline
    from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch

    pipeline = EchoWMCausalPipeline(od_config=_tiny_od_config(tmp_path))
    request = _public_request()
    request.prompt = {"prompt": "camera moves forward", "multi_modal_data": {"image": ["frame.png"]}}
    request.sampling_params.extra_args.pop("echo_wm_image_latent")
    request.sampling_params.extra_args["echo_wm_action"] = "w-49"

    def encode(image, *, height, width):
        assert image == "frame.png"
        assert (height, width) == (64, 128)
        return torch.ones(1, 8, 16)

    monkeypatch.setattr(pipeline, "_encode_image", encode)
    parsed = pipeline._parse_request(DiffusionRequestBatch([request]))
    assert parsed.ucpe_viewmats.shape == (1, 7, 4, 4)
    assert parsed.ucpe_viewmats[0, -1, 2, 3] > parsed.ucpe_viewmats[0, 0, 2, 3]


@pytest.mark.parametrize(
    "invalid,match",
    [
        ({"num_inference_steps": 0}, "num_inference_steps=4"),
        ({"echo_wm_timesteps": (1000, 750, 500)}, "exactly four"),
        ({"fps": float("nan")}, "fps must be positive"),
    ],
)
def test_request_rejects_invalid_step_schedule_and_fps(tmp_path, invalid, match):
    from vllm_omni.diffusion.models.echo_wm.pipeline import EchoWMCausalPipeline
    from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch

    pipeline = EchoWMCausalPipeline(od_config=_tiny_od_config(tmp_path))
    request = _public_request()
    for key, value in invalid.items():
        if key.startswith("echo_wm_"):
            request.sampling_params.extra_args[key] = value
        else:
            setattr(request.sampling_params, key, value)
    with pytest.raises(ValueError, match=match):
        pipeline._parse_request(DiffusionRequestBatch([request]))


@pytest.mark.parametrize("bundle", [False, True])
def test_prompt_encoder_resolves_gemma_bundle_or_regular_directory(tmp_path, monkeypatch, bundle):
    from torch import nn
    from transformers import AutoTokenizer, Gemma3ForConditionalGeneration

    from vllm_omni.diffusion.models.echo_wm.pipeline import EchoWMCausalPipeline

    if bundle:
        (tmp_path / "text_encoder").mkdir()
        (tmp_path / "tokenizer").mkdir()
    seen = {}

    class Tokenizer:
        pad_token = None
        eos_token = "<eos>"
        padding_side = "right"

        def __call__(self, prompt, **kwargs):
            assert prompt == "a scene"
            assert self.padding_side == "left"
            assert self.pad_token == "<eos>"
            return SimpleNamespace(input_ids=torch.tensor([[2, 3]]), attention_mask=torch.ones(1, 2))

    class Encoder(nn.Module):
        def model(self, **kwargs):
            assert not self.training
            assert kwargs["output_hidden_states"] is True
            return SimpleNamespace(hidden_states=(torch.ones(1, 2, 3), torch.full((1, 2, 3), 2.0)))

    class Connector(nn.Module):
        def forward(self, hidden, mask):
            assert hidden.shape == (1, 2, 3, 2)
            assert mask.shape == (1, 2)
            return hidden[..., 0], hidden[..., 1]

    def tokenizer_load(path, **kwargs):
        seen["tokenizer"] = path
        assert kwargs["local_files_only"] is True
        return Tokenizer()

    def model_load(path, **kwargs):
        seen["model"] = path
        assert kwargs["local_files_only"] is True
        assert kwargs["dtype"] == torch.bfloat16
        return Encoder()

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", tokenizer_load)
    monkeypatch.setattr(Gemma3ForConditionalGeneration, "from_pretrained", model_load)
    pipeline = EchoWMCausalPipeline.__new__(EchoWMCausalPipeline)
    nn.Module.__init__(pipeline)
    pipeline._gemma_path = str(tmp_path)
    pipeline._gemma = None
    pipeline.device = torch.device("cpu")
    pipeline.dtype = torch.bfloat16
    pipeline.text_stack = Connector()
    video, audio = pipeline._encode_prompt(" a scene ")
    assert seen == {
        "model": str(tmp_path / "text_encoder" if bundle else tmp_path),
        "tokenizer": str(tmp_path / "tokenizer" if bundle else tmp_path),
    }
    torch.testing.assert_close(video, torch.ones(1, 2, 3))
    torch.testing.assert_close(audio, torch.full((1, 2, 3), 2.0))


def test_hf_gemma_loader_converts_legacy_bundle_weight_names(tmp_path):
    from safetensors.torch import save_file
    from transformers import Gemma3Config, Gemma3ForConditionalGeneration

    config = Gemma3Config(
        text_config={
            "vocab_size": 32,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 8,
            "max_position_embeddings": 32,
            "sliding_window": 16,
        },
        vision_config={
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "image_size": 4,
            "patch_size": 2,
        },
        mm_tokens_per_image=4,
    )
    original = Gemma3ForConditionalGeneration(config).eval()
    config.save_pretrained(tmp_path)
    legacy = {}
    for name, value in original.state_dict().items():
        if name == "lm_head.weight":
            continue  # The released checkpoint ties this to embed_tokens.
        name = name.replace("model.language_model.", "language_model.model.")
        name = name.replace("model.vision_tower.", "vision_tower.")
        name = name.replace("model.multi_modal_projector.", "multi_modal_projector.")
        legacy[name] = value.clone()
    save_file(legacy, str(tmp_path / "model.safetensors"))
    restored, info = Gemma3ForConditionalGeneration.from_pretrained(
        tmp_path,
        dtype=torch.float32,
        local_files_only=True,
        output_loading_info=True,
    )
    assert not info["missing_keys"]
    assert not info["unexpected_keys"]
    for name, value in original.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], value, rtol=0, atol=0)
