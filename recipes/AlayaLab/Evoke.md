# Alaya-EVOKE for causal video generation

## Summary

- Vendor: Alaya Lab
- Models: `AlayaLab/Evoke`, `AlayaLab/Evoke-Turbo`
- Task: causal text-to-video and image-to-video diffusion rollout
- Recipe: three CFG-free denoising steps for the released post-distilled and Turbo checkpoints

Evoke is a causal world model. It runs a short diffusion denoise for each
video chunk and carries latent history into the next chunk. The denoiser uses
the Helios-family parameter layout, while the pipeline owns Evoke's nested
checkpoint layout and rollout defaults.

## Text-to-video

Run the shared text-to-video entrypoint:

```bash
cd examples/offline_inference/text_to_video
python text_to_video.py \
  --model AlayaLab/Evoke \
  --prompt "A camera moves through a quiet neon city at night." \
  --height 384 --width 640 --num-frames 73 \
  --num-inference-steps 3 --guidance-scale 1.0 \
  --output evoke_t2v.mp4
```

Turbo uses the same command with `--model AlayaLab/Evoke-Turbo`. The shared
base VAE, tokenizer, text encoder and scheduler are resolved from
`AlayaLab/Evoke`; Turbo contributes the merged transformer weights.

## Limitations

Camera-conditioned v2v and long-horizon geometry-bank evaluation also require
the separate ViGeo/depth assets from the upstream Evoke repository. The native
pipeline accepts sink-frame and camera metadata, but this recipe only covers
the standard T2V/I2V serving path.

References: [Evoke model card](https://huggingface.co/AlayaLab/Evoke),
[Evoke-Turbo model card](https://huggingface.co/AlayaLab/Evoke-Turbo),
[official implementation](https://github.com/AlayaLab/Evoke).
