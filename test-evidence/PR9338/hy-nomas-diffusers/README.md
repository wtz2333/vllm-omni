# Hy-Nomas-Diffusers evidence for PR #9338

Implementation: `257856d3afce694803f83fd60674f292b7a34583`.
The runtime remains dependent on the pinned original Hy-Nomas components. All candidate runs use the published Diffusers-format checkpoint and its bundled text encoder/tokenizer.

`metrics.json` contains the numerical accuracy and performance results; `manifest.json` records revisions and workload. Comparison clips use reference on the left and vLLM-Omni on the right. The six scenarios cover text, prompt segments, and camera control, each in resident and CPU-offload modes. Numerical errors and SSIM are computed from all decoded float pixels before uint8 conversion and MP4 encoding. GIFs are reduced previews; comparison MP4s preserve the full 49-frame clips.

`streams/` additionally contains viewing copies of the previously accepted SP2 three-turn session (147 frames), live prompt/camera interaction (477 frames), cancellation (23 frames), and subsequent fresh request (477 frames). These copies are scaled to 960 pixels wide and recompressed for viewing; frame counts are checked. They are not used for the numerical metrics.

Performance uses one prompt, 704×1280, 121 frames, eight steps, BF16, seed 42, CFG 1, flow shift 5, batch/concurrency one. Each configuration has two exact-shape warmups and five measured requests. The same physical two-H200 pair runs SP1, SP2, SP1+offload, SP2+offload, then SP1 again. SP1 always uses the first device. The second SP1 run checks drift. Raw warmup and measured samples are in `performance/`.

An initial SP2 memory-collection attempt stopped before generation because control RPCs return only the primary rank's value. The measurement extension now gathers each rank's counters through the existing CPU process group outside timed calls. The initial failure did not produce measured SP2 requests. After obtaining a new available GPU pair, the entire matrix, including the initial SP1 baseline, was rerun on that pair. Earlier exploratory SP1 samples are excluded from these tables; model code was unchanged.

E2E begins immediately before `Omni.generate` and ends after its complete CPU tensor result arrives. It includes text encoding, denoising, KV work, VAE decoding, output conversion and transport. It excludes loading, memory RPCs, hashing, video encoding and disk writes. Each rank's PyTorch allocated/reserved peaks are sampled outside timed calls. Process-tree host RSS/PSS are sampled once per second across the full run, including loading. Generation frame rate is 121 divided by mean E2E, not a saturation benchmark. Nearest-rank p95 with five samples is the maximum.

## Reproduce

Check out implementation commit `257856d3afce694803f83fd60674f292b7a34583` in vLLM-Omni. Install the pinned reference dependency and prepare the candidate and reference checkpoints as documented in the model recipe. Keep this evidence directory outside the vLLM-Omni source checkout. Run from the clean implementation checkout with the evidence scripts directory on `PYTHONPATH`:

```bash
export HY_NOMAS_MODEL=/path/to/Hy-Nomas-Diffusers
export HY_NOMAS_REFERENCE_MODEL=/path/to/Hy-Nomas
export PYTHONPATH=/path/to/evidence/scripts:$PYTHONPATH
CUDA_VISIBLE_DEVICES=0 python /path/to/evidence/scripts/benchmark.py --sp 1 --name sp1
CUDA_VISIBLE_DEVICES=0,1 python /path/to/evidence/scripts/benchmark.py --sp 2 --name sp2
CUDA_VISIBLE_DEVICES=0 python /path/to/evidence/scripts/benchmark.py --sp 1 --offload --name sp1_offload
CUDA_VISIBLE_DEVICES=0,1 python /path/to/evidence/scripts/benchmark.py --sp 2 --offload --name sp2_offload
CUDA_VISIBLE_DEVICES=0 python /path/to/evidence/scripts/benchmark.py --sp 1 --name sp1_repeat
```

Use new result directories when repeating. The small worker extension only resets and reads memory counters; it does not replace inference or insert synchronization inside timed calls. The accuracy capture plugin calls the original implementations unchanged and records their outputs:

```bash
export HY_ACCURACY_OUTPUT=/path/to/new-accuracy-output
CUDA_VISIBLE_DEVICES=0 python -m pytest tests/e2e/accuracy/test_hy_nomas.py -p capture_accuracy -k test_reference_decoded_video -m 'local_model and cuda' --run-level core_model -q -s
```

Accuracy capture additionally requires `scikit-image`, `imageio`, ffmpeg, and a font supported by ffmpeg drawtext. The original tests still enforce exact equality. GPU CI and saturated/concurrent throughput were not run. Codex assisted with measurement, capture and reporting.
