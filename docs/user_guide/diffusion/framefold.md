# FrameFold for LingBot-World 2.0

LingBot-World 2.0 causal-fast supports an optional FrameFold history KV policy.
FIFO remains the default. FrameFold merges adjacent historical latent frames,
preserving complete spatial grids while fitting the merged values against
recent historical attention responses. It uses the existing checkpoint.

## Configuration

Set the policy on the diffusion stage of a LingBot deployment:

```yaml
enforce_eager: true
model_config:
  lingbot_kv_policy: framefold
  lingbot_reuse_last_step_kv: false
  ar_diffusion_height: 480
  ar_diffusion_width: 832
```

For the complete stepwise topology, use
`vllm_omni/deploy/lingbot_world_v2_stepwise.yaml` and add
`lingbot_kv_policy: framefold` to its stage's `model_config`. Keep its existing
AR-Diffusion engine and streaming settings. The same model configuration works
for complete requests.

Use `lingbot_kv_policy: fifo`, or omit the setting, for FIFO. Unknown policies
are rejected. FrameFold requires eager execution and clean-x0 KV commits;
combining it with `lingbot_reuse_last_step_kv: true` is rejected.
It also rejects `lingbot_share_first_block_kv: true`, whose snapshots contain
post-RoPE keys without the required historical queries, and
`lingbot_npu_leased_post_attention: true`.

The checkpoint's attention window supplies the total budget, including the
current block. Its sink frames remain protected. FrameFold also protects the
three most recent completed latent frames and uses their pooled queries to
evaluate merges, with spatial pooling factor four. The budget must fit the
sink, recent history, current block and at least one historical representative.
For the 18-frame window and nine sink frames, a three-frame current block
leaves three slots for historical representatives.

## Execution and memory

Complete requests use request-local buffers. AR-Diffusion sessions reserve a
fixed set of pages from the runtime's KV pool and release them on reset,
close, cancellation or failure. These pages store pre-RoPE keys and values;
attention materializes keys at the compacted cache positions. Absolute frame
positions for camera controls and output continue to advance independently.

Denoising probes reuse the merge plan without changing committed K/V or the
saved historical queries. Clean-x0 forwards publish the updated history and
capture queries for the next block. Queries come from the clean forward; the
paper does not specify the denoising forward used by its unpublished code.

The memory reservation includes pooled queries for every layer and a
conservative bound for fitting and attention workspace. Saving more temporal
history within the same entry budget does not establish a reduction in peak
device memory. TP and Ulysses aggregate fitting statistics and candidate
distortions across all heads before selecting the same merge on each rank.

## Validation status

CPU coverage checks the fitting formula, repeated merges, pre-RoPE key
construction, real RoPE attention with contiguous and pool-backed storage,
clean commits, storage isolation and release. The CPU tests do not establish
GPU kernel compatibility, distributed runtime correctness, generated-video
quality or performance. CUDA Graph execution is not supported by this policy.

The intended benefit is long-video consistency at a fixed KV capacity. The
FrameFold paper reports additional compute at matched capacity, so latency
and peak memory must be measured separately on the intended workload.

Reference: [FrameFold paper, version 1](https://arxiv.org/html/2610.14568v1).
