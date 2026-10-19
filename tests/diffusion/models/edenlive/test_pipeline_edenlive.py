# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU integration through config discovery, the loader and AR session binding."""

import json

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKLWan
from safetensors.torch import save_file
from torch import nn
from transformers import BatchEncoding, DINOv3ViTConfig
from transformers.modeling_outputs import BaseModelOutput
from vllm.config.load import LoadConfig

from tests.diffusion.models.edenlive.test_edenlive import TINY_CONFIG, make_state
from tests.helpers.stage_config import get_deploy_config_path
from vllm_omni.config.resolver import resolve_omni_config
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.models.edenlive import pipeline_edenlive as module
from vllm_omni.diffusion.models.edenlive.weights import decoder_state_bytes, load_vae
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.utils import StepRequestState
from vllm_omni.experimental.ar_diffusion.streaming_decode import WanStreamingDecoder
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]
ENGINE = "vllm_omni.experimental.ar_diffusion.engine.ARDiffusionEngine"


def tiny_vae_config():
    return dict(
        base_dim=4,
        decoder_base_dim=4,
        z_dim=48,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_downsample=[False, True, True],
        in_channels=12,
        out_channels=12,
        patch_size=2,
        is_residual=True,
        scale_factor_spatial=16,
        scale_factor_temporal=4,
        latents_mean=[0.0] * 48,
        latents_std=[1.0] * 48,
    )


class TinyTextEncoder(nn.Module):
    def forward(self, ids, mask):
        return BaseModelOutput(last_hidden_state=torch.ones(ids.shape[0], ids.shape[1], 16))


class TinyBackbone(nn.Module):
    config = DINOv3ViTConfig(hidden_size=4, num_attention_heads=1, patch_size=16, num_register_tokens=4)

    def forward(self, pixel_values):
        patches = F.avg_pool2d(pixel_values, 16).flatten(2).transpose(1, 2)
        patches = F.pad(patches, (0, 1))
        registers = torch.zeros(patches.shape[0], 5, 4, dtype=patches.dtype)
        return BaseModelOutput(last_hidden_state=torch.cat((registers, patches), dim=1))


@pytest.fixture
def pipeline_components(cpu_config, tmp_path, monkeypatch):
    torch.manual_seed(19)
    config = {**TINY_CONFIG, "in_channels": 48, "out_channels": 48}
    (tmp_path / "dit-5b").mkdir()
    (tmp_path / "dit-5b/config.json").write_text(json.dumps(config))
    (tmp_path / "vision-encoder").mkdir()
    projection = module.ProxyDinoEncoder(TinyBackbone())
    save_file(
        {name: value for name, value in projection.state_dict().items() if "projection" in name},
        tmp_path / "vision-encoder/projection.safetensors",
    )
    vae = AutoencoderKLWan(**tiny_vae_config()).eval()

    def tokenizer(*args, **kwargs):
        return BatchEncoding(
            {
                "input_ids": torch.zeros(1, 512, dtype=torch.long),
                "attention_mask": torch.ones(1, 512, dtype=torch.long),
            }
        )

    monkeypatch.setattr(module, "get_local_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(module.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: tokenizer)
    monkeypatch.setattr(module.UMT5EncoderModel, "from_pretrained", lambda *args, **kwargs: TinyTextEncoder())
    monkeypatch.setattr(module.AutoModel, "from_pretrained", lambda *args, **kwargs: TinyBackbone())
    monkeypatch.setattr(module, "load_vae", lambda *args: vae)
    cpu_config.engine_backend = ENGINE
    cpu_config.model_config = {"ar_diffusion_height": 32, "ar_diffusion_width": 32}
    return cpu_config


def request(frames=85, output_type="latent"):
    return OmniDiffusionRequest(
        request_id="rollout",
        prompt={
            "prompt": "a rendered world",
            "multi_modal_data": {
                "image": np.zeros((32, 32, 3), dtype=np.uint8),
                "video": np.stack([np.full((32, 32, 3), i, dtype=np.uint8) for i in range(frames)]),
            },
        },
        sampling_params=OmniDiffusionSamplingParams(
            height=32,
            width=32,
            num_frames=frames,
            num_inference_steps=4,
            guidance_scale=1.0,
            output_type=output_type,
            seed=42,
        ),
    )


@pytest.mark.parametrize("streaming", [False, True])
def test_model_index_selects_ar_engine_and_real_deploy(tmp_path, streaming):
    (tmp_path / "model_index.json").write_text(json.dumps({"_class_name": "EdenLivePipeline"}))
    result = resolve_omni_config(
        str(tmp_path),
        trust_remote_code=False,
        deploy_config_path=get_deploy_config_path("edenlive_5b_streaming.yaml") if streaming else None,
        cli_overrides={},
        stage_overrides=None,
        strategy_config_path=None,
    )
    assert result.pipeline_config.model_type == "edenlive"
    stage = result.stage_configs[0]
    assert stage.diffusion_config.engine_backend == ENGINE
    assert stage.diffusion_config.model_class_name == "EdenLivePipeline"
    assert bool(stage.diffusion_config.streaming_output) is streaming
    assert bool(stage.diffusion_config.step_execution) is streaming


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_standard_loader_reads_widened_checkpoint(pipeline_components, tmp_path, dtype):
    pipeline_components.dtype = dtype
    reference = module.EdenLiveTransformer({**TINY_CONFIG, "in_channels": 48, "out_channels": 48})
    save_file(reference.state_dict(), tmp_path / "dit-5b/diffusion_pytorch_model.safetensors")
    loader = DiffusersPipelineLoader(LoadConfig(), pipeline_components)
    pipeline = loader.load_model(load_device="cpu", device=torch.device("cpu"))
    assert isinstance(pipeline, module.EdenLivePipeline)
    assert pipeline.transformer.patch_embedding.in_channels == 1072
    for key, value in reference.state_dict().items():
        torch.testing.assert_close(pipeline.transformer.state_dict()[key], value.to(dtype), rtol=0, atol=0)
    assert all(parameter.dtype == dtype for parameter in pipeline.transformer.parameters())
    assert all(buffer.dtype == dtype for buffer in pipeline.transformer.rope.buffers())


def test_stepwise_and_request_rollouts_match_after_window_and_short_tail(pipeline_components, mocker):
    pipeline = module.EdenLivePipeline(od_config=pipeline_components)
    ar = make_state(tokens_per_frame=1, name="rollout")
    try:
        with pipeline.bind_ar_diffusion_state("rollout", ar):
            expected = pipeline(request()).output["payload"]["latents"]
            fresh = request()
            state = StepRequestState(request_id=fresh.request_id, prompt=fresh.prompt, sampling=fresh.sampling_params)
            pipeline.prepare_encode(state)
            spy = mocker.spy(pipeline.transformer, "forward")
            outputs = []
            while not state.request_denoise_completed:
                for _ in range(4):
                    pipeline.step_scheduler(state, pipeline.denoise_step(None, states=[state]))
                    if state.chunk_index == 0:
                        torch.testing.assert_close(state.latents[:, :, :1], state.extra["condition"], rtol=0, atol=0)
                output = pipeline.post_decode(state)
                outputs.append(output.output["payload"]["latents"])
            actual = torch.cat(outputs, dim=2)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            assert actual.shape == (1, 48, 22, 2, 2)
            assert len(outputs) == 8 and outputs[-1].shape[2] == 1
            assert spy.call_count == 8 * 5  # Four probes plus a separate clean KV commit.
            assert ar.adapter("main").absolute_num_computed_tokens == 22
            for call in spy.call_args_list[4::5]:
                assert call.kwargs["commit"] is True
                assert torch.count_nonzero(call.args[2]) == 0
        assert pipeline._ar_state is None
    finally:
        ar.close()


def test_streaming_vae_matches_full_decode_and_budget():
    torch.manual_seed(23)
    vae = AutoencoderKLWan(**tiny_vae_config()).eval()
    decoder = WanStreamingDecoder(vae)
    state = decoder.new_decode_state("vae")
    latents = torch.randn(1, 48, 7, 2, 2)
    with torch.no_grad():
        expected = vae.decode(latents).sample
        chunks = [decoder.decode_chunk(chunk, state) for chunk in latents.split(3, dim=2)]
    assert [chunk.shape[2] for chunk in chunks] == [9, 12, 4]
    torch.testing.assert_close(torch.cat(chunks, dim=2), expected, rtol=1e-5, atol=1e-5)
    actual_bytes = sum(item.numel() * item.element_size() for item in state.feat_map if isinstance(item, torch.Tensor))
    assert actual_bytes <= decoder_state_bytes(vae, 32, 32)
    decoder.release(state)
    assert not any(isinstance(item, torch.Tensor) for item in state.feat_map)


def test_native_wan22_vae_checkpoint_loads_strictly(tmp_path, monkeypatch):
    from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_wan import DistributedAutoencoderKLWan
    from vllm_omni.diffusion.models.lance.wan_vae import WanVAE_

    native = WanVAE_(dim=4, dec_dim=4, z_dim=48, temperal_downsample=(False, True, True)).eval()
    folder = tmp_path / "vae-5b"
    folder.mkdir()
    config = {**tiny_vae_config(), "in_channels": 3, "out_channels": 3}
    (folder / "config.json").write_text(json.dumps(config))
    torch.save(native.state_dict(), folder / "Wan2.2_VAE.pth")
    monkeypatch.setattr(DistributedAutoencoderKLWan, "init_distributed", lambda self: None)
    loaded = load_vae(str(tmp_path), torch.float32)
    assert loaded.encoder.conv_in.in_channels == 12
    assert loaded.decoder.conv_out.out_channels == 12
    assert sum(p.numel() for p in loaded.parameters()) == sum(p.numel() for p in native.parameters())


def test_failed_forward_releases_kv_and_decoder_state(pipeline_components, monkeypatch):
    pipeline = module.EdenLivePipeline(od_config=pipeline_components)
    ar = make_state(tokens_per_frame=1, name="rollout")
    original = pipeline.decoder.decode_chunk
    calls = 0

    def fail_second_chunk(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("decode failed")
        return original(*args, **kwargs)

    monkeypatch.setattr(pipeline.decoder, "decode_chunk", fail_second_chunk)
    try:
        with pipeline.bind_ar_diffusion_state("rollout", ar):
            with pytest.raises(RuntimeError, match="decode failed"):
                pipeline(request(21, output_type="np"))
            assert not pipeline._decode_states
            assert ar.adapter("main").absolute_num_computed_tokens == 0
        assert pipeline._ar_state is None
    finally:
        ar.close()


@pytest.fixture
def real_runner(pipeline_components, monkeypatch):
    from vllm.config import VllmConfig

    from vllm_omni.diffusion.worker import diffusion_model_runner
    from vllm_omni.experimental.ar_diffusion import runner as ar_runner

    for platform in (diffusion_model_runner.current_omni_platform, ar_runner.current_omni_platform):
        monkeypatch.setattr(platform, "is_available", lambda: False)
        monkeypatch.setattr(platform, "synchronize", lambda: None)
        monkeypatch.setattr(platform, "max_memory_reserved", lambda: 0)
        monkeypatch.setattr(platform, "max_memory_allocated", lambda: 0)
    pipeline_components.step_execution = True
    pipeline_components.streaming_output = True
    runner = ar_runner.ARDiffusionModelRunner(VllmConfig(), pipeline_components, torch.device("cpu"))
    runner.pipeline = module.EdenLivePipeline(od_config=pipeline_components)
    runner._preallocate_kv_cache(available_bytes=1 << 26)
    yield runner
    runner.release_step_requests(list(runner.state_cache))


def scheduled(step_id, *, new=False, finished=False):
    from vllm_omni.diffusion.sched.interface import CachedRequestData, DiffusionSchedulerOutput, NewRequestData

    return DiffusionSchedulerOutput(
        step_id=step_id,
        scheduled_new_reqs=[NewRequestData("rollout", request(21, output_type="np"))] if new else [],
        scheduled_cached_reqs=CachedRequestData([] if new or finished else ["rollout"]),
        finished_req_ids={"rollout"} if finished else set(),
        num_running_reqs=0 if finished else 1,
        num_waiting_reqs=0,
    )


def test_real_ar_runner_streams_two_chunks_and_releases_session(real_runner):
    chunks = []
    for index in range(8):
        output = real_runner.execute_stepwise(scheduled(index, new=index == 0)).get_request_output("rollout")
        if output.result is not None:
            assert not output.result.error
            chunks.append(output.result.output["payload"]["video"])
        if output.finished:
            break
    assert output.finished
    assert [chunk.shape for chunk in chunks] == [(9, 32, 32, 3), (12, 32, 32, 3)]
    assert not real_runner._sessions
    assert not real_runner.pipeline._decode_states
    real_runner.execute_stepwise(scheduled(9, finished=True))
    assert not real_runner.state_cache


def test_real_ar_runner_cancel_discards_partial_rollout(real_runner):
    for index in range(4):
        output = real_runner.execute_stepwise(scheduled(index, new=index == 0)).get_request_output("rollout")
        assert not output.finished
        if output.result is not None:
            break
    assert real_runner.pipeline._decode_states
    real_runner.release_step_requests(["rollout"])
    assert not real_runner._sessions
    assert not real_runner.pipeline._decode_states
    assert not real_runner.state_cache
    assert real_runner.pipeline._ar_state is None
