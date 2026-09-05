# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pure-UCPE camera branch for Echo-WM.

The PRoPE (projection rotary position embedding) math is ported from the
Echo-WM reference implementation, which in turn derives from

    "Cameras as Relative Positional Encoding" https://arxiv.org/pdf/2507.10496

(MIT licensed). The transforms map attention features through per-camera
projection matrices built from camera extrinsics (``viewmats``) and
intrinsics (``Ks``), plus 2-D split-style RoPE over the latent patch grid.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial

import torch
import torch.nn.functional as F

__all__ = [
    "PropeDotProductAttention",
    "prepare_apply_fns",
    "active_sink_fifo_indices",
    "rebase_viewmat_translation",
]


def active_sink_fifo_indices(
    current_end: int, local_size: int, sink_size: int, device: torch.device
) -> tuple[torch.Tensor, int]:
    """Indices represented by a bounded ``sink + recent FIFO`` cache."""
    if local_size <= 0 or sink_size < 0 or sink_size >= local_size:
        raise ValueError(f"invalid sink/FIFO layout: local={local_size}, sink={sink_size}")
    if current_end <= local_size:
        return torch.arange(current_end, device=device), 0
    recent_start = max(sink_size, current_end - (local_size - sink_size))
    return (
        torch.cat(
            (
                torch.arange(sink_size, device=device),
                torch.arange(recent_start, current_end, device=device),
            )
        ),
        recent_start,
    )


def rebase_viewmat_translation(viewmats: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
    """Apply one common right-side translation, preserving relative cameras."""
    with torch.autocast(device_type=viewmats.device.type, enabled=False):
        matrices = viewmats.float()
        anchor = anchor.float()
        shift = -(anchor[..., :3, :3].transpose(-1, -2) @ anchor[..., :3, 3:4])
        result = matrices.clone()
        result[..., :3, 3:4] += result[..., :3, :3] @ shift
    return result


class PropeDotProductAttention(torch.nn.Module):
    """PRoPE attention with precomputed RoPE coefficients.

    The coefficients and geometry are fixed at construction; the per-forward
    camera-dependent transforms are prepared by
    :meth:`precompute_and_cache_apply_fns` and consumed by the three
    ``apply_to_*`` helpers, mirroring the reference implementation.
    """

    coeffs_x_0: torch.Tensor
    coeffs_x_1: torch.Tensor
    coeffs_y_0: torch.Tensor
    coeffs_y_1: torch.Tensor

    def __init__(
        self,
        head_dim: int,
        patches_x: int,
        patches_y: int,
        image_width: int,
        image_height: int,
        freq_base: float = 100.0,
        freq_scale: float = 1.0,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.patches_x = patches_x
        self.patches_y = patches_y
        self.image_width = image_width
        self.image_height = image_height
        self.freq_base = freq_base
        self.freq_scale = freq_scale
        coeffs_x = _rope_precompute_coeffs(
            torch.tile(torch.arange(patches_x, device="cpu"), (patches_y,)),
            freq_base=freq_base,
            freq_scale=freq_scale,
            feat_dim=head_dim // 4,
        )
        coeffs_y = _rope_precompute_coeffs(
            torch.repeat_interleave(torch.arange(patches_y, device="cpu"), patches_x),
            freq_base=freq_base,
            freq_scale=freq_scale,
            feat_dim=head_dim // 4,
        )
        # ModelLedger builds the action branch outside its meta-device scope:
        # these geometry constants are computed on CPU in FP32, then moved.
        self._coefficients_cpu = dict(
            coeffs_x_0=coeffs_x[0],
            coeffs_x_1=coeffs_x[1],
            coeffs_y_0=coeffs_y[0],
            coeffs_y_1=coeffs_y[1],
        )
        for name, value in self._coefficients_cpu.items():
            self.register_buffer(name, value, persistent=False)

    def _apply(self, fn, recurse=True):
        result = super()._apply(fn, recurse=recurse)
        # Restore from the CPU constants, including after dtype conversion or
        # meta/to_empty round trips. Widening rounded BF16 tables is not enough.
        for name, value in self._coefficients_cpu.items():
            setattr(self, name, value.to(device=getattr(self, name).device, dtype=torch.float32))
        for name in ("apply_fn_q", "apply_fn_kv", "apply_fn_o"):
            if hasattr(self, name):
                setattr(self, name, None)
        return result

    def ensure_coefficients(self, device: torch.device) -> None:
        for name, value in self._coefficients_cpu.items():
            current = getattr(self, name)
            if current.device != device or current.dtype != torch.float32:
                setattr(self, name, value.to(device=device, dtype=torch.float32))

    def apply_fns_ready(self) -> bool:
        return getattr(self, "apply_fn_q", None) is not None

    def precompute_and_cache_apply_fns(
        self,
        viewmats: torch.Tensor,
        Ks: torch.Tensor | None,  # noqa: N803  # noqa: N803
        coeffs_x: tuple[torch.Tensor, torch.Tensor] | None = None,
        coeffs_y: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> None:
        self.ensure_coefficients(viewmats.device)
        batch, cameras, _, _ = viewmats.shape
        if viewmats.shape != (batch, cameras, 4, 4):
            raise ValueError(f"expected viewmats (B, C, 4, 4), got {tuple(viewmats.shape)}")
        if Ks is not None and Ks.shape != (batch, cameras, 3, 3):
            raise ValueError(f"expected Ks (B, C, 3, 3), got {tuple(Ks.shape)}")
        cached_x = None if self.coeffs_x_0 is None else (self.coeffs_x_0, self.coeffs_x_1)
        cached_y = None if self.coeffs_y_0 is None else (self.coeffs_y_0, self.coeffs_y_1)
        self.apply_fn_q, self.apply_fn_kv, self.apply_fn_o = prepare_apply_fns(
            head_dim=self.head_dim,
            viewmats=viewmats,
            Ks=Ks,
            patches_x=self.patches_x,
            patches_y=self.patches_y,
            image_width=self.image_width,
            image_height=self.image_height,
            coeffs_x=cached_x if coeffs_x is None else coeffs_x,
            coeffs_y=cached_y if coeffs_y is None else coeffs_y,
        )

    def apply_to_q(self, q: torch.Tensor) -> torch.Tensor:
        return self.apply_fn_q(q)

    def apply_to_kv(self, kv: torch.Tensor) -> torch.Tensor:
        return self.apply_fn_kv(kv)

    def apply_to_o(self, o: torch.Tensor) -> torch.Tensor:
        return self.apply_fn_o(o)

    def transform(self, apply_fn: Callable[[torch.Tensor], torch.Tensor], value: torch.Tensor) -> torch.Tensor:
        """Apply one PRoPE transform in float32, preserving dtype."""
        dtype = value.dtype
        with torch.autocast(device_type=value.device.type, enabled=False):
            return apply_fn(value.float()).to(dtype)


def prepare_apply_fns(
    head_dim: int,
    viewmats: torch.Tensor,
    Ks: torch.Tensor | None,  # noqa: N803
    patches_x: int,
    patches_y: int,
    image_width: int,
    image_height: int,
    coeffs_x: tuple[torch.Tensor, torch.Tensor] | None = None,
    coeffs_y: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[
    Callable[[torch.Tensor], torch.Tensor],
    Callable[[torch.Tensor], torch.Tensor],
    Callable[[torch.Tensor], torch.Tensor],
]:
    """Prepare transforms for PRoPE-style positional encoding."""
    device = viewmats.device
    batch, cameras, _, _ = viewmats.shape
    dtype = viewmats.dtype

    if Ks is not None:
        # Normalize camera intrinsics. K is image<-camera, viewmats is
        # camera<-world, so P = lift(K) @ viewmats is image<-world.
        Ks_norm = torch.zeros_like(Ks)
        Ks_norm[..., 0, 0] = Ks[..., 0, 0] / image_width
        Ks_norm[..., 1, 1] = Ks[..., 1, 1] / image_height
        Ks_norm[..., 0, 2] = Ks[..., 0, 2] / image_width - 0.5
        Ks_norm[..., 1, 2] = Ks[..., 1, 2] / image_height - 0.5
        Ks_norm[..., 2, 2] = 1.0
        del Ks

        P = torch.einsum("...ij,...jk->...ik", _lift_k(Ks_norm), viewmats)
        P_T = P.transpose(-1, -2)
        P_inv = torch.einsum("...ij,...jk->...ik", _invert_se3(viewmats), _lift_k(_invert_k(Ks_norm)))
    else:
        P = viewmats
        P_T = P.transpose(-1, -2)
        P_inv = _invert_se3(viewmats)

    if P.shape != P_inv.shape or P.shape[-2:] != (4, 4):
        raise ValueError(f"expected projection matrices (B, C, 4, 4), got {tuple(P.shape)}")

    if coeffs_x is None:
        coeffs_x = _rope_precompute_coeffs(
            torch.tile(torch.arange(patches_x, device=device), (patches_y * cameras,)),
            freq_base=100.0,
            freq_scale=1.0,
            feat_dim=head_dim // 4,
            dtype=dtype,
        )
    if coeffs_y is None:
        coeffs_y = _rope_precompute_coeffs(
            torch.tile(torch.repeat_interleave(torch.arange(patches_y, device=device), patches_x), (cameras,)),
            freq_base=100.0,
            freq_scale=1.0,
            feat_dim=head_dim // 4,
            dtype=dtype,
        )

    if head_dim % 4 != 0:
        raise ValueError(f"head_dim must be divisible by 4, got {head_dim}")
    transforms_q = [
        (partial(_apply_tiled_projmat, matrix=P_T), head_dim // 2),
        (partial(_rope_apply_coeffs, coeffs=coeffs_x), head_dim // 4),
        (partial(_rope_apply_coeffs, coeffs=coeffs_y), head_dim // 4),
    ]
    transforms_kv = [
        (partial(_apply_tiled_projmat, matrix=P_inv), head_dim // 2),
        (partial(_rope_apply_coeffs, coeffs=coeffs_x), head_dim // 4),
        (partial(_rope_apply_coeffs, coeffs=coeffs_y), head_dim // 4),
    ]
    transforms_o = [
        (partial(_apply_tiled_projmat, matrix=P), head_dim // 2),
        (partial(_rope_apply_coeffs, coeffs=coeffs_x, inverse=True), head_dim // 4),
        (partial(_rope_apply_coeffs, coeffs=coeffs_y, inverse=True), head_dim // 4),
    ]

    apply_fn_q = partial(_apply_block_diagonal, func_size_pairs=transforms_q)
    apply_fn_kv = partial(_apply_block_diagonal, func_size_pairs=transforms_kv)
    apply_fn_o = partial(_apply_block_diagonal, func_size_pairs=transforms_o)
    return apply_fn_q, apply_fn_kv, apply_fn_o


def _apply_tiled_projmat(
    feats: torch.Tensor,
    matrix: torch.Tensor,
) -> torch.Tensor:
    """Apply a projection matrix to features, per camera or per ray."""
    batch, num_heads, seqlen, feat_dim = feats.shape
    matrix = matrix.to(device=feats.device, dtype=feats.dtype)
    D = matrix.shape[-1]
    if feat_dim % D != 0:
        raise ValueError(f"feat_dim={feat_dim} must be divisible by D={D}")

    if matrix.shape[1] == seqlen:
        feats_ = feats.view(batch, num_heads, seqlen, feat_dim // D, D)
        out = torch.einsum("btij,bntpj->bntpi", matrix, feats_)
        return out.reshape(feats.shape)

    cameras = matrix.shape[1]
    if seqlen <= cameras or seqlen % cameras != 0:
        raise ValueError(f"seqlen {seqlen} must be a multiple of cameras {cameras}")
    return torch.einsum(
        "bcij,bncpkj->bncpki",
        matrix,
        feats.reshape((batch, num_heads, cameras, -1, feat_dim // D, D)),
    ).reshape(feats.shape)


def _rope_precompute_coeffs(
    positions: torch.Tensor,
    freq_base: float,
    freq_scale: float,
    feat_dim: int,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute RoPE coefficients (split-style ordering)."""
    if positions.ndim != 1:
        raise ValueError(f"positions must be 1-D, got {positions.ndim}")
    if feat_dim % 2 != 0:
        raise ValueError(f"feat_dim must be even, got {feat_dim}")
    num_freqs = feat_dim // 2
    freqs = freq_scale * (
        freq_base
        ** (-torch.arange(num_freqs, device=positions.device, dtype=torch.float32)[None, None, None, :] / num_freqs)
    )
    angles = positions.to(torch.float32)[None, None, :, None] * freqs
    return torch.cos(angles).to(dtype), torch.sin(angles).to(dtype)


def _rope_apply_coeffs(
    feats: torch.Tensor,
    coeffs: tuple[torch.Tensor, torch.Tensor],
    inverse: bool = False,
) -> torch.Tensor:
    """Apply RoPE coefficients to features ('split' ordering)."""
    cos, sin = coeffs
    if cos.shape[2] != feats.shape[2]:
        n_repeats = feats.shape[2] // cos.shape[2]
        cos = cos.repeat(1, 1, n_repeats, 1)
        sin = sin.repeat(1, 1, n_repeats, 1)
    if cos.shape[-1] != feats.shape[-1] // 2:
        raise ValueError("coefficients do not match the feature dimension")
    x_in = feats[..., : feats.shape[-1] // 2]
    y_in = feats[..., feats.shape[-1] // 2 :]
    return torch.cat(
        (
            [cos * x_in + sin * y_in, -sin * x_in + cos * y_in]
            if not inverse
            else [cos * x_in - sin * y_in, sin * x_in + cos * y_in]
        ),
        dim=-1,
    )


def _apply_block_diagonal(
    feats: torch.Tensor,
    func_size_pairs: list[tuple[Callable[[torch.Tensor], torch.Tensor], int]],
) -> torch.Tensor:
    """Apply a block-diagonal function to an input array."""
    funcs, block_sizes = zip(*func_size_pairs, strict=True)
    if feats.shape[-1] != sum(block_sizes):
        raise ValueError(f"feature dim {feats.shape[-1]} does not match block sizes {block_sizes}")
    x_blocks = torch.split(feats, block_sizes, dim=-1)
    return torch.cat([f(x_block) for f, x_block in zip(funcs, x_blocks, strict=True)], dim=-1)


def _invert_se3(transforms: torch.Tensor) -> torch.Tensor:
    """Invert a 4x4 SE(3) matrix."""
    if transforms.shape[-2:] != (4, 4):
        raise ValueError("SE3 transforms must be 4x4")
    rinv = transforms[..., :3, :3].transpose(-1, -2)
    out = torch.zeros_like(transforms)
    out[..., :3, :3] = rinv
    out[..., :3, 3] = -torch.einsum("...ij,...j->...i", rinv, transforms[..., :3, 3])
    out[..., 3, 3] = 1.0
    return out


def _lift_k(ks: torch.Tensor) -> torch.Tensor:
    """Lift 3x3 matrices to homogeneous 4x4 matrices."""
    if ks.shape[-2:] != (3, 3):
        raise ValueError("intrinsics must be 3x3")
    out = torch.zeros(ks.shape[:-2] + (4, 4), device=ks.device, dtype=ks.dtype)
    out[..., :3, :3] = ks
    out[..., 3, 3] = 1.0
    return out


def _invert_k(ks: torch.Tensor) -> torch.Tensor:
    """Invert 3x3 intrinsics matrices (no skew)."""
    if ks.shape[-2:] != (3, 3):
        raise ValueError("intrinsics must be 3x3")
    out = torch.zeros_like(ks)
    out[..., 0, 0] = 1.0 / ks[..., 0, 0]
    out[..., 1, 1] = 1.0 / ks[..., 1, 1]
    out[..., 0, 2] = -ks[..., 0, 2] / ks[..., 0, 0]
    out[..., 1, 2] = -ks[..., 1, 2] / ks[..., 1, 1]
    out[..., 2, 2] = 1.0
    return out


def ucpe_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    viewmats: torch.Tensor,
    Ks: torch.Tensor | None,  # noqa: N803
    patches_x: int,
    patches_y: int,
    image_width: int,
    image_height: int,
) -> torch.Tensor:
    """One-shot PRoPE self-attention (recomputes transforms per call)."""
    batch, num_heads, seqlen, head_dim = q.shape
    cameras = viewmats.shape[1]
    if seqlen != cameras * patches_x * patches_y:
        raise ValueError(f"seqlen {seqlen} != cameras {cameras} x patches {patches_x}x{patches_y}")
    apply_fn_q, apply_fn_kv, apply_fn_o = prepare_apply_fns(
        head_dim=head_dim,
        viewmats=viewmats,
        Ks=Ks,
        patches_x=patches_x,
        patches_y=patches_y,
        image_width=image_width,
        image_height=image_height,
    )
    out = F.scaled_dot_product_attention(
        query=apply_fn_q(q),
        key=apply_fn_kv(k),
        value=apply_fn_kv(v),
    )
    return apply_fn_o(out)
