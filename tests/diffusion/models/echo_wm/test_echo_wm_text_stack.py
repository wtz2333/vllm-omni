# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Echo-WM text-stack parity tests against the reference ``EmbeddingsProcessor``.

Reference parity skips when the upstream checkout is absent. The CPU Gloo
SP regression runs without that optional dependency.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch

from vllm_omni.diffusion.models.echo_wm.text_stack import EchoWMTextStack

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]

_ECHO_ROOT = Path(os.environ.get("ECHOWM_REFERENCE_ROOT", ""))


def _reference():
    if not (_ECHO_ROOT / "ltx-core" / "src").exists():
        pytest.skip("Echo-WM reference repository not available")
    sys.path.insert(0, str(_ECHO_ROOT / "ltx-core" / "src"))
    from ltx_core.text_encoders.gemma.encoders.encoder_configurator import EmbeddingsProcessorConfigurator

    config = {
        "transformer": {
            "num_attention_heads": 2,
            "attention_head_dim": 8,
            "audio_num_attention_heads": 2,
            "audio_attention_head_dim": 4,
            "connector_num_attention_heads": 2,
            "connector_attention_head_dim": 8,
            "audio_connector_num_attention_heads": 2,
            "audio_connector_attention_head_dim": 4,
            "connector_num_layers": 2,
            "connector_positional_embedding_max_pos": [64],
            "connector_num_learnable_registers": 8,
            "connector_apply_gated_attention": True,
            "caption_proj_before_connector": True,
            "caption_projection_first_linear": False,
            "caption_proj_input_norm": False,
            "caption_projection_second_linear": False,
            "rope_type": "split",
            "frequencies_precision": "float64",
        }
    }
    torch.manual_seed(5)
    processor = EmbeddingsProcessorConfigurator.from_config(config)
    with torch.no_grad():
        for param in processor.parameters():
            if param.dtype.is_floating_point:
                param.normal_(0.0, 0.02)
    return processor


def _mine() -> EchoWMTextStack:
    torch.manual_seed(9)
    stack = EchoWMTextStack(
        gemma_hidden_size=3840,
        gemma_num_layers=48,
        video_dim=16,
        audio_dim=8,
        connector_num_layers=2,
        video_heads=2,
        video_head_dim=8,
        audio_heads=2,
        audio_head_dim=4,
        num_registers=128,
        rope_max_pos=64,
    )
    return stack


def _copy_weights(reference, stack: EchoWMTextStack) -> None:
    ref_state = dict(reference.named_parameters())
    with torch.no_grad():
        for name, param in stack.named_parameters():
            if ".to_qkv." in name:
                prefix, suffix = name.split(".to_qkv.", 1)
                param.copy_(
                    torch.cat(
                        [
                            ref_state[f"{prefix}.to_q.{suffix}"],
                            ref_state[f"{prefix}.to_k.{suffix}"],
                            ref_state[f"{prefix}.to_v.{suffix}"],
                        ],
                        dim=0,
                    )
                )
                continue
            ref_name = name.replace(".norm_q.", ".q_norm.").replace(".norm_k.", ".k_norm.")
            if name.endswith("aggregate_embed.weight") or name.endswith("aggregate_embed.bias"):
                ref_name = "feature_extractor." + name
            source = ref_state[ref_name]
            assert source.shape == param.shape, (ref_name, source.shape, param.shape)
            param.copy_(source)


def test_text_stack_matches_reference_embeddings_processor():
    reference = _reference()
    stack = _mine().eval()
    _copy_weights(reference, stack)

    torch.manual_seed(21)
    batch, seq, hidden, layers = 1, 128, 3840, 49
    hidden_states = tuple(torch.randn(batch, seq, hidden) for _ in range(layers))
    lengths = [100]
    attention_mask = torch.zeros(batch, seq, dtype=torch.int64)
    attention_mask[:, seq - lengths[0] :] = 1  # left padding, like the Gemma tokenizer

    with torch.inference_mode():
        reference_out = reference.process_hidden_states(hidden_states, attention_mask, padding_side="left")
        video, audio = stack(torch.stack(hidden_states, dim=-1), attention_mask)

    torch.testing.assert_close(video, reference_out.video_encoding, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(audio, reference_out.audio_encoding, rtol=1e-4, atol=1e-4)
    # After register replacement every position is attendable, so the emitted
    # mask is all-ones and the DiT runs its text cross-attention unmasked.
    assert reference_out.attention_mask.bool().all()


def test_text_stack_load_weights_maps_checkpoint_names():
    stack = _mine()
    checkpoint = {}
    for name, param in stack.named_parameters():
        if ".to_qkv." in name:
            prefix, suffix = name.split(".to_qkv.", 1)
            shard = param.shape[0] // 3
            if prefix.startswith("video_connector"):
                ckpt_prefix = "model.diffusion_model.video_embeddings_connector." + prefix[len("video_connector.") :]
            elif prefix.startswith("audio_connector"):
                ckpt_prefix = "model.diffusion_model.audio_embeddings_connector." + prefix[len("audio_connector.") :]
            else:
                ckpt_prefix = prefix
            for projection in ("q", "k", "v"):
                checkpoint[f"{ckpt_prefix}.to_{projection}.{suffix}"] = torch.randn(shard, *param.shape[1:])
            continue
        ref_name = name.replace(".norm_q.", ".q_norm.").replace(".norm_k.", ".k_norm.")
        if name.startswith("video_connector"):
            ckpt_name = "model.diffusion_model.video_embeddings_connector." + ref_name[len("video_connector.") :]
        elif name.startswith("audio_connector"):
            ckpt_name = "model.diffusion_model.audio_embeddings_connector." + ref_name[len("audio_connector.") :]
        else:
            ckpt_name = "text_embedding_projection." + ref_name
        checkpoint[ckpt_name] = torch.randn_like(param)
    covered = stack.load_weights(list(checkpoint.items()))
    assert set(checkpoint) <= covered
    # Non-text checkpoint names are skipped so the pipeline can dispatch by
    # prefix; the strict loader check lives one level up.
    assert "not_a_text_weight" not in stack.load_weights([("not_a_text_weight", torch.zeros(1))])


def _tiny_parallel_text_stack():
    stack = (
        EchoWMTextStack(
            gemma_hidden_size=8,
            gemma_num_layers=2,
            video_dim=16,
            audio_dim=8,
            connector_num_layers=2,
            video_heads=2,
            video_head_dim=8,
            audio_heads=2,
            audio_head_dim=4,
            num_registers=8,
            rope_max_pos=64,
        )
        .float()
        .eval()
    )

    # Parallel linears allocate empty storage for checkpoint loading.
    # Initialize all weights before creating the SP=1 oracle.
    with torch.no_grad():
        for parameter in stack.parameters():
            parameter.normal_(0.0, 0.02)
    return stack


def _text_stack_sp_worker(rank, init_method, fixture_path):
    from tests.diffusion.models.echo_wm.conftest import cpu_distributed, cpu_kernels

    torch.set_num_threads(1)
    with cpu_distributed(init_method, rank=rank, world_size=2, sp_size=2), cpu_kernels(sp_size=2):
        fixture = torch.load(fixture_path, weights_only=True)
        stack = _tiny_parallel_text_stack()
        stack.load_state_dict(fixture["weights"])
        with torch.inference_mode():
            video, audio = stack(fixture["hidden_states"], fixture["attention_mask"])
        # Connectors are fully replicated across SP ranks, so GEMM shapes and
        # all tensor values must stay identical to the single-rank baseline.
        torch.testing.assert_close(video, fixture["video"], rtol=0, atol=0)
        torch.testing.assert_close(audio, fixture["audio"], rtol=0, atol=0)
        assert not torch.cuda.is_initialized()


@pytest.mark.parallel
def test_text_stack_sp2_gloo_matches_single_rank(tmp_path):
    """Exercise Gemma feature projection and both connectors with active SP."""
    torch.manual_seed(17)
    stack = _tiny_parallel_text_stack()
    hidden_states = torch.randn(1, 8, 8, 3)
    attention_mask = torch.tensor([[0, 0, 0, 1, 1, 1, 1, 1]])
    with torch.inference_mode():
        video, audio = stack(hidden_states, attention_mask)
    fixture_path = tmp_path / "text-stack-baseline.pt"
    torch.save(
        {
            "weights": stack.state_dict(),
            "hidden_states": hidden_states,
            "attention_mask": attention_mask,
            "video": video,
            "audio": audio,
        },
        fixture_path,
    )
    torch.multiprocessing.spawn(
        _text_stack_sp_worker,
        args=((tmp_path / "text-sp-init").as_uri(), fixture_path),
        nprocs=2,
        join=True,
    )
