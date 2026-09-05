# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Echo-WM image conditioning and audiovisual decoding.

Uses the released Echo-WM ``ltx_core`` VAE implementations as an optional
runtime dependency. The DiT and text connector run in vLLM-Omni. Components
are loaded only for encoding/decoding and released before the next phase.
"""

from __future__ import annotations

import json
import math
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from safetensors import safe_open


def preprocess_image(
    image: Image.Image | torch.Tensor | str | Path, *, height: int, width: int, device: torch.device | str = "cpu"
) -> torch.Tensor:
    """Match the reference CRF-33 conditioning, resize, and center crop."""
    import av

    if isinstance(image, (str, Path)):
        with Image.open(image) as source:
            pixels = np.array(source.convert("RGB"))
    elif isinstance(image, Image.Image):
        pixels = np.array(image.convert("RGB"))
    elif isinstance(image, torch.Tensor):
        if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != torch.uint8:
            raise ValueError("Echo-WM image tensors must be uint8 RGB with shape (H, W, 3)")
        pixels = image.detach().cpu().numpy()
    else:
        raise TypeError("Echo-WM image must be a PIL image, RGB uint8 tensor, or image path")
    source_h, source_w = pixels.shape[:2]
    if min(source_h, source_w) < 2:
        raise ValueError("Echo-WM input image must be at least 2 pixels on each axis")
    # This codec round trip is part of the released conditioning recipe.
    with BytesIO() as buffer:
        with av.open(buffer, "w", format="mp4") as container:
            stream = container.add_stream("libx264", rate=1, options={"crf": "33", "preset": "veryfast"})
            stream.height = source_h // 2 * 2
            stream.width = source_w // 2 * 2
            frame = av.VideoFrame.from_ndarray(pixels[: stream.height, : stream.width], format="rgb24")
            frame = frame.reformat(format="yuv420p")
            container.mux(stream.encode(frame))
            container.mux(stream.encode())
        encoded = buffer.getvalue()
    with av.open(BytesIO(encoded)) as container:
        pixels = next(container.decode(video=0)).to_ndarray(format="rgb24")
    tensor = torch.tensor(pixels, dtype=torch.float32, device=device).permute(2, 0, 1).unsqueeze(0)
    source_h, source_w = tensor.shape[-2:]
    scale = max(height / source_h, width / source_w)
    resized_h, resized_w = math.ceil(source_h * scale), math.ceil(source_w * scale)
    tensor = torch.nn.functional.interpolate(tensor, size=(resized_h, resized_w), mode="bilinear", align_corners=False)
    top, left = (resized_h - height) // 2, (resized_w - width) // 2
    tensor = tensor[:, :, top : top + height, left : left + width]
    return tensor.unsqueeze(2) / 127.5 - 1.0


class EchoWMMediaAdapter:
    """Load only the VAE component needed by the current media phase."""

    def __init__(self, checkpoint_path: str, *, device: torch.device, dtype: torch.dtype = torch.bfloat16) -> None:
        self.checkpoint_path = checkpoint_path
        self.device = torch.device(device)
        self.dtype = dtype

    def _load_component(self, component: str) -> torch.nn.Module:
        try:
            from ltx_core.model.audio_vae import AudioDecoderConfigurator, VocoderConfigurator
            from ltx_core.model.video_vae import VideoDecoderConfigurator, VideoEncoderConfigurator
        except ImportError as exc:
            raise ImportError(
                "Echo-WM image/audio/video processing requires the released JoyAI-Echo ltx_core package. "
                "Add JoyAI-Echo/echo_wm/ltx-core/src to PYTHONPATH; see examples/offline_inference/echo_wm/README.md."
            ) from exc
        configurators = {
            "video_encoder": VideoEncoderConfigurator,
            "video_decoder": VideoDecoderConfigurator,
            "audio_decoder": AudioDecoderConfigurator,
            "vocoder": VocoderConfigurator,
        }
        prefixes = {
            "video_encoder": "vae.encoder.",
            "video_decoder": "vae.decoder.",
            "audio_decoder": "audio_vae.decoder.",
            "vocoder": "vocoder.",
        }
        prefix = prefixes[component]
        statistics = (
            "audio_vae.per_channel_statistics." if component == "audio_decoder" else "vae.per_channel_statistics."
        )
        with safe_open(self.checkpoint_path, framework="pt", device="cpu") as checkpoint:
            config = json.loads(checkpoint.metadata()["config"])
            with torch.device("meta"):
                model = configurators[component].from_config(config)
            weights = {}
            for name in checkpoint.keys():
                if name.startswith(prefix):
                    target = name.removeprefix(prefix)
                elif component != "vocoder" and name.startswith(statistics):
                    target = "per_channel_statistics." + name.removeprefix(statistics)
                else:
                    continue
                tensor = checkpoint.get_tensor(name)
                weights[target] = tensor.to(dtype=self.dtype) if tensor.is_floating_point() else tensor
            model.load_state_dict(weights, strict=True, assign=True)
        return model.to(self.device).eval()

    @torch.inference_mode()
    def encode_image(self, image: Image.Image | torch.Tensor | str | Path, *, height: int, width: int) -> torch.Tensor:
        pixels = preprocess_image(image, height=height, width=width, device=self.device).to(dtype=self.dtype)
        encoder = self._load_component("video_encoder")
        latents = encoder(pixels)
        # Upstream VideoLatentPatchifier(1): b c f h w -> b (f h w) c.
        return latents.permute(0, 2, 3, 4, 1).flatten(1, 3)

    @torch.inference_mode()
    def decode(
        self,
        video_latents: torch.Tensor,
        audio_latents: torch.Tensor,
        *,
        height: int,
        width: int,
        generator: torch.Generator,
    ) -> dict[str, Any]:
        try:
            from ltx_core.model.audio_vae import decode_audio
            from ltx_core.model.video_vae import TilingConfig, decode_video
        except ImportError as exc:
            raise ImportError("Echo-WM decoding requires JoyAI-Echo/echo_wm/ltx-core/src on PYTHONPATH") from exc
        batch, _, channels = video_latents.shape
        latent_h, latent_w = height // 32, width // 32
        # The native VideoLatentTools decoder input is contiguous B,C,F,H,W.
        # Keeping a channels-last view selects different BF16 cuDNN kernels.
        video = video_latents.reshape(batch, -1, latent_h, latent_w, channels).permute(0, 4, 1, 2, 3).contiguous()
        audio = audio_latents.reshape(batch, -1, 8, 16).permute(0, 2, 1, 3)
        # The reference decodes audio before iterating the lazily decoded video.
        audio_decoder = self._load_component("audio_decoder")
        vocoder = self._load_component("vocoder")
        decoded_audio = decode_audio(audio.to(device=self.device, dtype=self.dtype), audio_decoder, vocoder)
        waveform = decoded_audio.waveform.detach().float().cpu()
        sample_rate = int(decoded_audio.sampling_rate)
        del audio_decoder, vocoder, decoded_audio
        video_decoder = self._load_component("video_decoder")
        frames = [
            chunk.cpu()
            for chunk in decode_video(
                video.to(device=self.device, dtype=self.dtype),
                video_decoder,
                tiling_config=TilingConfig.default(),
                generator=generator,
            )
        ]
        return {"video": torch.cat(frames, dim=0), "audio": waveform, "sample_rate": sample_rate}
