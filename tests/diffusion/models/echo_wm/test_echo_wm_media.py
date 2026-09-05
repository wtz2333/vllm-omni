# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Input conditioning contracts independent of model weights."""

import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from vllm_omni.diffusion.models.echo_wm.actions import build_action_condition, parse_action_string
from vllm_omni.diffusion.models.echo_wm.media import preprocess_image

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def test_camera_static_and_translation_alignment():
    static = build_action_condition("none-25", num_frames=25, width=128, height=64)
    torch.testing.assert_close(static["ucpe_viewmats"], torch.eye(4).repeat(1, 4, 1, 1))
    moving = build_action_condition("w-24", num_frames=25, width=128, height=64)
    expected = torch.arange(4).float() * (8 * 0.05 / 30.0)
    torch.testing.assert_close(moving["ucpe_viewmats"][0, :, 2, 3], expected)
    assert moving["ucpe_Ks"].shape == (1, 4, 3, 3)
    assert moving["ucpe_Ks"].dtype == torch.float32
    assert moving["ucpe_Ks"][0, 0, 0, 2] == 64


@pytest.mark.parametrize("action", ["", "w-0", "w--1", "x-2", "w"])
def test_camera_rejects_invalid_actions(action):
    with pytest.raises(ValueError):
        parse_action_string(action)


def test_camera_matches_reference_actions(monkeypatch):
    root = Path(os.environ.get("ECHOWM_REFERENCE_ROOT", ""))
    reference_path = root / "helpers/action_camera.py"
    if not reference_path.exists():
        pytest.skip("Set ECHOWM_REFERENCE_ROOT for optional upstream camera parity")
    spec = importlib.util.spec_from_file_location("echo_wm_reference_camera", reference_path)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    params = dict(
        num_frames=169,
        image_width=1280,
        image_height=704,
        translation_speed=0.05,
        rotation_speed_deg=0.4,
        pitch_limit_deg=40.0,
        fov_deg=70.0,
        fps=24,
    )
    original = reference.build_action_pt_from_string("k-32,i-32,sw-32,j-40,none-32", **params)
    expected = original["c2ws_raw"][None]
    expected = torch.linalg.inv(expected[:, :1]) @ expected
    expected[..., :3, 3] /= 30
    result = build_action_condition("k-32,i-32,sw-32,j-40,none-32", num_frames=169, width=1280, height=704)
    torch.testing.assert_close(result["ucpe_viewmats"], expected[:, ::8], rtol=0, atol=0)
    torch.testing.assert_close(result["ucpe_Ks"][0, 0], original["K_pix"], rtol=0, atol=0)


def test_image_preprocessing_formats_match(tmp_path):
    pixels = np.random.default_rng(42).integers(0, 256, (63, 95, 3), dtype=np.uint8)
    image = Image.fromarray(pixels)
    path = tmp_path / "conditioning.png"
    image.save(path)
    expected = preprocess_image(path, height=64, width=128)
    assert expected.shape == (1, 3, 1, 64, 128)
    assert torch.isfinite(expected).all()
    assert expected.min() >= -1 and expected.max() <= 1
    torch.testing.assert_close(preprocess_image(image, height=64, width=128), expected, rtol=0, atol=0)
    torch.testing.assert_close(
        preprocess_image(torch.from_numpy(pixels), height=64, width=128), expected, rtol=0, atol=0
    )
    with pytest.raises(ValueError, match="uint8"):
        preprocess_image(torch.zeros(16, 16, 3), height=64, width=128)


def test_image_encoder_preserves_reference_patchifier_layout(monkeypatch):
    from vllm_omni.diffusion.models.echo_wm.media import EchoWMMediaAdapter

    latents = torch.arange(128 * 8, dtype=torch.float32).reshape(1, 128, 1, 2, 4)
    adapter = EchoWMMediaAdapter("unused.safetensors", device=torch.device("cpu"), dtype=torch.float32)
    monkeypatch.setattr(adapter, "_load_component", lambda _: lambda image: latents)
    image = Image.fromarray(np.zeros((64, 128, 3), dtype=np.uint8))
    output = adapter.encode_image(image, height=64, width=128)
    assert output.shape == (1, 8, 128)
    assert output.stride(1) == 1
    torch.testing.assert_close(output[0, 3], latents[0, :, 0, 0, 3], rtol=0, atol=0)


def test_decoder_uses_reference_video_memory_layout(monkeypatch):
    import sys
    from types import ModuleType, SimpleNamespace

    from vllm_omni.diffusion.models.echo_wm.media import EchoWMMediaAdapter

    for name in ("ltx_core", "ltx_core.model"):
        if name not in sys.modules:
            module = ModuleType(name)
            module.__path__ = []
            monkeypatch.setitem(sys.modules, name, module)
    video_module = ModuleType("ltx_core.model.video_vae")
    audio_module = ModuleType("ltx_core.model.audio_vae")
    observed = {}

    def decode_video(latent, decoder, **kwargs):
        observed["video"] = latent
        yield torch.zeros(25, 64, 128, 3, dtype=torch.uint8)

    def decode_audio(latent, decoder, vocoder):
        observed["audio"] = latent
        return SimpleNamespace(waveform=torch.zeros(2, 48000), sampling_rate=48000)

    video_module.decode_video = decode_video
    video_module.TilingConfig = SimpleNamespace(default=lambda: None)
    audio_module.decode_audio = decode_audio
    monkeypatch.setitem(sys.modules, "ltx_core.model.video_vae", video_module)
    monkeypatch.setitem(sys.modules, "ltx_core.model.audio_vae", audio_module)
    adapter = EchoWMMediaAdapter("unused.safetensors", device=torch.device("cpu"))
    monkeypatch.setattr(adapter, "_load_component", lambda _: object())
    video = torch.arange(4 * 8 * 128, dtype=torch.float32).reshape(1, 32, 128)
    audio = torch.zeros(1, 27, 128)
    adapter.decode(video, audio, height=64, width=128, generator=torch.Generator())
    assert observed["video"].shape == (1, 128, 4, 2, 4)
    assert observed["video"].is_contiguous()
    torch.testing.assert_close(observed["video"][0, :, 2, 1, 3], video[0, 23].bfloat16(), rtol=0, atol=0)
    assert observed["audio"].shape == (1, 8, 27, 16)
