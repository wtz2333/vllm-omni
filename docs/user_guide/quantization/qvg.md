# QVG history KV compression

[Quant VideoGen](https://arxiv.org/html/2602.02958v5) compresses committed
autoregressive video history using centroid smoothing and packed INT2/INT4
residuals. This experimental integration supports LingBot World v2 through
`ARDiffusionEngine`, with eager CUDA execution and one active sequence.
GPU quality, memory, and latency qualification is still required.

## Install the core package

Keep the PyTorch, Triton, Transformers, and CUDA versions required by your
vLLM installation. Install the pinned QVG core without its experiment extras
or its older PyTorch pin:

```bash
python -m pip install 'termcolor>=2.3' 'loguru>=0.7'
python -m pip install --no-deps \
  'quant-videogen @ git+https://github.com/svg-project/Quant-VideoGen.git@0601468f2dbba6a17ac7086faec6d41527cad188'
```

The core is imported when QVG storage is initialized. Missing dependencies
fail initialization; other deployments do not require this package.

## Configure LingBot World v2

Use the QVG deployment profile with the existing realtime video API:

```bash
vllm serve robbyant/lingbot-world-v2-14b-causal-fast-diffusers --omni \
  --deploy-config vllm_omni/deploy/lingbot_world_v2_qvg.yaml
```

The profile enables `model_config.ar_diffusion_kv_config.compression`:

```yaml
ar_diffusion_kv_config:
  compression:
    method: qvg
    bits: 4
    num_stages: 1
    group_size: 64
    num_centroids: 256
    kmeans_max_iters: 100
    kmeans_min_iters: 4
    kmeans_check_interval: 4
    kmeans_tolerance: 0.001
    keep_recent_chunks: 1
    seed: 0
```

| Field | Default | Meaning |
| --- | --- | --- |
| `bits` | `4` | Packed residual precision, `2` or `4`. |
| `num_stages` | `1` | Residual smoothing stages, from `1` to `4`. |
| `group_size` | `64` | Residual scale group, `16`, `32`, or `64`; must divide the head size. |
| `num_centroids` | `256` | Centroids per head and stage, from `1` to `256`; must fit one generated chunk. |
| `kmeans_max_iters` | `100` | Positive iteration bound. |
| `kmeans_min_iters` | `4` | Earliest iteration eligible for convergence, subject to the hard maximum. |
| `kmeans_check_interval` | `4` | Check convergence once per interval rather than synchronizing every iteration. |
| `kmeans_tolerance` | `0.001` | Per-head, token-weighted relative RMS centroid movement; `0` runs the fixed iteration budget. |
| `keep_recent_chunks` | `1` | Completed generated chunks retained in floating point. |
| `seed` | `0` | Independent k-means initialization seed. |

Chunks overlapping the attention sink remain in floating point. Reads select
the exact sink and recent-window rows, even when those rows cover only part
of a compressed chunk. Eviction releases whole compressed chunks once no
visible row references them; retained data is never recompressed.

Clustering uses the pinned QVG Lloyd update kernels. The hard iteration bound
always applies; convergence is checked only after the minimum iteration count
and at the configured interval. Each head must satisfy the movement threshold.
Validate the tolerance on the target workload before changing it.

## Combine with weight quantization

History compression has its own configuration. Existing `quantization_config`
and `--diffusion-quantization-config` continue to select Linear/weight
quantization. For example, add the following to the server command to select
INT8 transformer weights and activations as well as QVG history:

```bash
--diffusion-quantization-config '{"transformer":{"method":"int8"}}'
```

Each weight method must already support the model and hardware. Validate
the combined output independently; CPU routing tests do not establish GPU
quality or performance. SVDQuant retains its existing SM103 restriction.

## Storage and lifecycle

The runner owns the compressed history for each session and logical KV
branch. No floating-point history pool or per-layer full-history staging is
allocated in QVG mode. A shared single-layer workspace reconstructs visible
K/V for ordinary floating-point attention. Pending clean K/V remains private
until every layer and sequence-parallel worker completes compression.
Reset, cancellation, failed execution, close, and LRU eviction release the
same session-owned state as the ordinary AR-Diffusion path.

This version conservatively retains the ordinary floating-point admission
budget and adds reservations for partial chunks, the read workspace, and
compression temporaries. Smaller packed storage does not automatically admit
more sessions. Measure actual allocated/reserved memory together with kernel
temporaries; the paper's KV compression ratio is not a total-device-memory
reduction guarantee.

LingBot stores pre-RoPE keys and restores the absolute spatial and temporal
positions after reconstruction. Native paged, direct, and QVG reads use the
same attention entry point. Partial pages are trimmed to the exact sink
and recent-frame window. On CUDA, `VLLM_OMNI_AR_DIFFUSION_KV_GATHER=1`
selects contiguous varlen attention for both native and QVG reads. Reconstructed
QVG inputs enter the kernel directly without another gather. Use the same
setting in the reference run: paged and contiguous kernels can differ in
floating-point accumulation. Ulysses head ownership is preserved.
It rejects graph/compiled execution, noisy-cell
pipelines, first-block KV sharing, last-step KV reuse, boundary resets,
full-history staging reuse, and explicit runtime attention quantization.
It requires the normal clean-x0 commit. Cross-attention stays in its native
representation. Parallel execution still needs its own GPU qualification.

CPU coverage exercises the real history manager, codec adapter, and LingBot
attention routing with controlled substitutes for QVG's external GPU
primitives. It does not execute the CUDA/Triton compression kernels.
