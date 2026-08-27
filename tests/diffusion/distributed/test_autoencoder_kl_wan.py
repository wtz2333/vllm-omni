from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.distributed.autoencoders import autoencoder_kl_wan as wan_vae_module
from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_wan import OmniAutoencoderKLWan

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _DummyOmniAutoencoderKLWan(OmniAutoencoderKLWan):
    def __init__(self, *, dtype: torch.dtype):
        torch.nn.Module.__init__(self)
        self.register_parameter("dummy_weight", torch.nn.Parameter(torch.ones(1, dtype=dtype)))


class _StreamingDecoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.first_chunk_calls: list[bool] = []

    def forward(
        self,
        x: torch.Tensor,
        *,
        feat_cache: list,
        feat_idx: list[int],
        first_chunk: bool,
    ) -> torch.Tensor:
        self.first_chunk_calls.append(first_chunk)
        feat_cache[0] = x.detach().clone()
        feat_idx[0] += 1
        output_frames = 1 if first_chunk else 4
        return x[:, :3, :1].expand(-1, -1, output_frames, -1, -1).clone()


class _StreamingDummyOmniAutoencoderKLWan(OmniAutoencoderKLWan):
    def __init__(self) -> None:
        torch.nn.Module.__init__(self)
        self.register_parameter("dummy_weight", torch.nn.Parameter(torch.ones(1)))
        self.post_quant_conv = torch.nn.Identity()
        self.decoder = _StreamingDecoder()
        self._cached_conv_counts = {"decoder": 1}
        self.use_tiling = False

    @property
    def config(self) -> SimpleNamespace:
        return SimpleNamespace(patch_size=None)


def test_wan_vae_execution_context_handles_fp32():
    model = _DummyOmniAutoencoderKLWan(dtype=torch.float32)
    with model._execution_context():
        output = model.dummy_weight + 1
    assert output.dtype == torch.float32


def test_wan_vae_execution_context_handles_bf16():
    model = _DummyOmniAutoencoderKLWan(dtype=torch.bfloat16)
    with model._execution_context():
        output = model.dummy_weight + 1
    assert output.dtype == torch.bfloat16


def test_wan_vae_execution_context_uses_platform_autocast(mocker):
    sentinel = object()
    platform = mocker.Mock()
    platform.create_autocast_context.return_value = sentinel
    mocker.patch.object(wan_vae_module, "current_omni_platform", platform)

    model = _DummyOmniAutoencoderKLWan(dtype=torch.bfloat16)

    assert model._execution_context() is sentinel
    platform.create_autocast_context.assert_called_once_with(
        device_type=model.dummy_weight.device.type,
        dtype=torch.bfloat16,
        enabled=True,
    )


def test_wan_streaming_decode_preserves_causal_state_across_calls() -> None:
    model = _StreamingDummyOmniAutoencoderKLWan()
    state = model.create_streaming_decode_state()

    first = model.decode_streaming(torch.ones(1, 16, 3, 2, 2), state, return_dict=False)[0]
    second = model.decode_streaming(torch.ones(1, 16, 3, 2, 2), state, return_dict=False)[0]

    assert first.shape == (1, 3, 9, 2, 2)
    assert second.shape == (1, 3, 12, 2, 2)
    assert model.decoder.first_chunk_calls == [True, False, False, False, False, False]
    assert state.decoded_latent_frames == 6
    assert state.used_cache_slots == 1
    assert state.nbytes() == 16 * 2 * 2 * torch.tensor([], dtype=torch.float32).element_size()


def test_wan_streaming_decode_state_rejects_shape_change_and_clears_storage() -> None:
    model = _StreamingDummyOmniAutoencoderKLWan()
    state = model.create_streaming_decode_state()
    model.decode_streaming(torch.ones(1, 16, 1, 2, 2), state)

    with pytest.raises(ValueError, match="cannot change batch, spatial shape, dtype, or device"):
        model.decode_streaming(torch.ones(1, 16, 1, 2, 3), state)

    state.clear()
    assert state.nbytes() == 0
    assert state.decoded_latent_frames == 0
    assert state.batch_size is None
