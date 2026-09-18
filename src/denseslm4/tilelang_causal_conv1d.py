"""Small TileLang prototype of Dao-AILab causal-conv1d forward.

Supported for now:
- x:      (batch, dim, seqlen), CUDA contiguous
- weight: (dim, width), CUDA contiguous, width usually 2/3/4
- bias:   optional (dim,)
- activation: None, "silu", or "swish"
- forward only, no seq_idx / initial_states / final_states yet

Definition matches:
    F.conv1d(x, weight.unsqueeze(1), bias, padding=width - 1, groups=dim)[..., :seqlen]
"""

from __future__ import annotations

import torch
import tilelang
import tilelang.language as T


@tilelang.jit(out_idx=[-1])
def _causal_conv1d_fwd_no_bias_kernel(
    B: int,
    D: int,
    L: int,
    K: int,
    dtype: str = "float16",
    apply_silu: bool = False,
    block_size: int = 256,
):
    if K not in (2, 3, 4):
        raise NotImplementedError("prototype supports width 2, 3, or 4")

    @T.prim_func
    def kernel(
        A: T.Tensor((B, D, L), dtype),
        W: T.Tensor((D, K), dtype),
        O: T.Tensor((B, D, L), dtype),
    ):
        total = B * D * L
        with T.Kernel(T.ceildiv(total, block_size), threads=block_size) as bx:
            for tx in T.Parallel(block_size):
                idx = bx * block_size + tx
                if idx < total:
                    t = idx % L
                    d = (idx // L) % D
                    b = idx // (D * L)
                    zero = T.cast(0.0, T.float32)
                    if K == 2:
                        acc = (T.cast(A[b, d, t - 1], T.float32) * T.cast(W[d, 0], T.float32) if t >= 1 else zero) + T.cast(A[b, d, t], T.float32) * T.cast(W[d, 1], T.float32)
                    elif K == 3:
                        acc = (T.cast(A[b, d, t - 2], T.float32) * T.cast(W[d, 0], T.float32) if t >= 2 else zero) + (T.cast(A[b, d, t - 1], T.float32) * T.cast(W[d, 1], T.float32) if t >= 1 else zero) + T.cast(A[b, d, t], T.float32) * T.cast(W[d, 2], T.float32)
                    else:
                        acc = (T.cast(A[b, d, t - 3], T.float32) * T.cast(W[d, 0], T.float32) if t >= 3 else zero) + (T.cast(A[b, d, t - 2], T.float32) * T.cast(W[d, 1], T.float32) if t >= 2 else zero) + (T.cast(A[b, d, t - 1], T.float32) * T.cast(W[d, 2], T.float32) if t >= 1 else zero) + T.cast(A[b, d, t], T.float32) * T.cast(W[d, 3], T.float32)
                    if apply_silu:
                        O[b, d, t] = T.cast(acc / (1.0 + T.exp(-acc)), dtype)
                    else:
                        O[b, d, t] = T.cast(acc, dtype)

    return kernel


@tilelang.jit(out_idx=[-1])
def _causal_conv1d_fwd_bias_kernel(
    B: int,
    D: int,
    L: int,
    K: int,
    dtype: str = "float16",
    apply_silu: bool = False,
    block_size: int = 256,
):
    if K not in (2, 3, 4):
        raise NotImplementedError("prototype supports width 2, 3, or 4")

    @T.prim_func
    def kernel(
        A: T.Tensor((B, D, L), dtype),
        W: T.Tensor((D, K), dtype),
        bias: T.Tensor((D,), dtype),
        O: T.Tensor((B, D, L), dtype),
    ):
        total = B * D * L
        with T.Kernel(T.ceildiv(total, block_size), threads=block_size) as bx:
            for tx in T.Parallel(block_size):
                idx = bx * block_size + tx
                if idx < total:
                    t = idx % L
                    d = (idx // L) % D
                    b = idx // (D * L)
                    zero = T.cast(0.0, T.float32)
                    if K == 2:
                        acc = T.cast(bias[d], T.float32) + (T.cast(A[b, d, t - 1], T.float32) * T.cast(W[d, 0], T.float32) if t >= 1 else zero) + T.cast(A[b, d, t], T.float32) * T.cast(W[d, 1], T.float32)
                    elif K == 3:
                        acc = T.cast(bias[d], T.float32) + (T.cast(A[b, d, t - 2], T.float32) * T.cast(W[d, 0], T.float32) if t >= 2 else zero) + (T.cast(A[b, d, t - 1], T.float32) * T.cast(W[d, 1], T.float32) if t >= 1 else zero) + T.cast(A[b, d, t], T.float32) * T.cast(W[d, 2], T.float32)
                    else:
                        acc = T.cast(bias[d], T.float32) + (T.cast(A[b, d, t - 3], T.float32) * T.cast(W[d, 0], T.float32) if t >= 3 else zero) + (T.cast(A[b, d, t - 2], T.float32) * T.cast(W[d, 1], T.float32) if t >= 2 else zero) + (T.cast(A[b, d, t - 1], T.float32) * T.cast(W[d, 2], T.float32) if t >= 1 else zero) + T.cast(A[b, d, t], T.float32) * T.cast(W[d, 3], T.float32)
                    if apply_silu:
                        O[b, d, t] = T.cast(acc / (1.0 + T.exp(-acc)), dtype)
                    else:
                        O[b, d, t] = T.cast(acc, dtype)

    return kernel


@tilelang.jit(out_idx=[-1])
def _causal_conv1d_fwd_no_bias_vec_kernel(
    B: int,
    D: int,
    L: int,
    K: int,
    dtype: str = "float16",
    apply_silu: bool = False,
    block_size: int = 256,
    values_per_thread: int = 4,
):
    if K not in (2, 3, 4):
        raise NotImplementedError("prototype supports width 2, 3, or 4")

    @T.prim_func
    def kernel(
        A: T.Tensor((B, D, L), dtype),
        W: T.Tensor((D, K), dtype),
        O: T.Tensor((B, D, L), dtype),
    ):
        tiles_l = T.ceildiv(L, values_per_thread)
        total = B * D * tiles_l
        with T.Kernel(T.ceildiv(total, block_size), threads=block_size) as bx:
            for tx in T.Parallel(block_size):
                tile_idx = bx * block_size + tx
                if tile_idx < total:
                    l_tile = tile_idx % tiles_l
                    d = (tile_idx // tiles_l) % D
                    b = tile_idx // (D * tiles_l)
                    base_t = l_tile * values_per_thread
                    zero = T.cast(0.0, T.float32)
                    for vi in T.serial(values_per_thread):
                        t = base_t + vi
                        if t < L:
                            if K == 2:
                                acc = (T.cast(A[b, d, t - 1], T.float32) * T.cast(W[d, 0], T.float32) if t >= 1 else zero) + T.cast(A[b, d, t], T.float32) * T.cast(W[d, 1], T.float32)
                            elif K == 3:
                                acc = (T.cast(A[b, d, t - 2], T.float32) * T.cast(W[d, 0], T.float32) if t >= 2 else zero) + (T.cast(A[b, d, t - 1], T.float32) * T.cast(W[d, 1], T.float32) if t >= 1 else zero) + T.cast(A[b, d, t], T.float32) * T.cast(W[d, 2], T.float32)
                            else:
                                acc = (T.cast(A[b, d, t - 3], T.float32) * T.cast(W[d, 0], T.float32) if t >= 3 else zero) + (T.cast(A[b, d, t - 2], T.float32) * T.cast(W[d, 1], T.float32) if t >= 2 else zero) + (T.cast(A[b, d, t - 1], T.float32) * T.cast(W[d, 2], T.float32) if t >= 1 else zero) + T.cast(A[b, d, t], T.float32) * T.cast(W[d, 3], T.float32)
                            if apply_silu:
                                O[b, d, t] = T.cast(acc / (1.0 + T.exp(-acc)), dtype)
                            else:
                                O[b, d, t] = T.cast(acc, dtype)

    return kernel


@tilelang.jit(out_idx=[-1])
def _causal_conv1d_fwd_bias_vec_kernel(
    B: int,
    D: int,
    L: int,
    K: int,
    dtype: str = "float16",
    apply_silu: bool = False,
    block_size: int = 256,
    values_per_thread: int = 4,
):
    if K not in (2, 3, 4):
        raise NotImplementedError("prototype supports width 2, 3, or 4")

    @T.prim_func
    def kernel(
        A: T.Tensor((B, D, L), dtype),
        W: T.Tensor((D, K), dtype),
        bias: T.Tensor((D,), dtype),
        O: T.Tensor((B, D, L), dtype),
    ):
        tiles_l = T.ceildiv(L, values_per_thread)
        total = B * D * tiles_l
        with T.Kernel(T.ceildiv(total, block_size), threads=block_size) as bx:
            for tx in T.Parallel(block_size):
                tile_idx = bx * block_size + tx
                if tile_idx < total:
                    l_tile = tile_idx % tiles_l
                    d = (tile_idx // tiles_l) % D
                    b = tile_idx // (D * tiles_l)
                    base_t = l_tile * values_per_thread
                    zero = T.cast(0.0, T.float32)
                    bias_f = T.cast(bias[d], T.float32)
                    for vi in T.serial(values_per_thread):
                        t = base_t + vi
                        if t < L:
                            if K == 2:
                                acc = bias_f + (T.cast(A[b, d, t - 1], T.float32) * T.cast(W[d, 0], T.float32) if t >= 1 else zero) + T.cast(A[b, d, t], T.float32) * T.cast(W[d, 1], T.float32)
                            elif K == 3:
                                acc = bias_f + (T.cast(A[b, d, t - 2], T.float32) * T.cast(W[d, 0], T.float32) if t >= 2 else zero) + (T.cast(A[b, d, t - 1], T.float32) * T.cast(W[d, 1], T.float32) if t >= 1 else zero) + T.cast(A[b, d, t], T.float32) * T.cast(W[d, 2], T.float32)
                            else:
                                acc = bias_f + (T.cast(A[b, d, t - 3], T.float32) * T.cast(W[d, 0], T.float32) if t >= 3 else zero) + (T.cast(A[b, d, t - 2], T.float32) * T.cast(W[d, 1], T.float32) if t >= 2 else zero) + (T.cast(A[b, d, t - 1], T.float32) * T.cast(W[d, 2], T.float32) if t >= 1 else zero) + T.cast(A[b, d, t], T.float32) * T.cast(W[d, 3], T.float32)
                            if apply_silu:
                                O[b, d, t] = T.cast(acc / (1.0 + T.exp(-acc)), dtype)
                            else:
                                O[b, d, t] = T.cast(acc, dtype)

    return kernel


@tilelang.jit(out_idx=[-1])
def _causal_conv1d_fwd_no_bias_channel_tile_kernel(
    B: int,
    D: int,
    L: int,
    K: int,
    dtype: str = "float16",
    apply_silu: bool = False,
    block_size: int = 256,
    values_per_thread: int = 4,
):
    if K not in (2, 3, 4):
        raise NotImplementedError("prototype supports width 2, 3, or 4")

    @T.prim_func
    def kernel(
        A: T.Tensor((B, D, L), dtype),
        W: T.Tensor((D, K), dtype),
        O: T.Tensor((B, D, L), dtype),
    ):
        block_l = block_size * values_per_thread
        with T.Kernel(T.ceildiv(L, block_l), B * D, threads=block_size) as (bx, by):
            d = by % D
            b = by // D
            zero = T.cast(0.0, T.float32)
            w0 = T.cast(W[d, 0], T.float32)
            w1 = T.cast(W[d, 1], T.float32)
            w2 = T.cast(W[d, 2], T.float32) if K >= 3 else zero
            w3 = T.cast(W[d, 3], T.float32) if K >= 4 else zero
            for tx, vi in T.Parallel(block_size, values_per_thread):
                t = bx * block_l + vi * block_size + tx
                if t < L:
                    if K == 2:
                        acc = (T.cast(A[b, d, t - 1], T.float32) * w0 if t >= 1 else zero) + T.cast(A[b, d, t], T.float32) * w1
                    elif K == 3:
                        acc = (T.cast(A[b, d, t - 2], T.float32) * w0 if t >= 2 else zero) + (T.cast(A[b, d, t - 1], T.float32) * w1 if t >= 1 else zero) + T.cast(A[b, d, t], T.float32) * w2
                    else:
                        acc = (T.cast(A[b, d, t - 3], T.float32) * w0 if t >= 3 else zero) + (T.cast(A[b, d, t - 2], T.float32) * w1 if t >= 2 else zero) + (T.cast(A[b, d, t - 1], T.float32) * w2 if t >= 1 else zero) + T.cast(A[b, d, t], T.float32) * w3
                    if apply_silu:
                        O[b, d, t] = T.cast(acc / (1.0 + T.exp(-acc)), dtype)
                    else:
                        O[b, d, t] = T.cast(acc, dtype)

    return kernel


@tilelang.jit(out_idx=[-1])
def _causal_conv1d_fwd_bias_channel_tile_kernel(
    B: int,
    D: int,
    L: int,
    K: int,
    dtype: str = "float16",
    apply_silu: bool = False,
    block_size: int = 256,
    values_per_thread: int = 4,
):
    if K not in (2, 3, 4):
        raise NotImplementedError("prototype supports width 2, 3, or 4")

    @T.prim_func
    def kernel(
        A: T.Tensor((B, D, L), dtype),
        W: T.Tensor((D, K), dtype),
        bias: T.Tensor((D,), dtype),
        O: T.Tensor((B, D, L), dtype),
    ):
        block_l = block_size * values_per_thread
        with T.Kernel(T.ceildiv(L, block_l), B * D, threads=block_size) as (bx, by):
            d = by % D
            b = by // D
            zero = T.cast(0.0, T.float32)
            bias_f = T.cast(bias[d], T.float32)
            w0 = T.cast(W[d, 0], T.float32)
            w1 = T.cast(W[d, 1], T.float32)
            w2 = T.cast(W[d, 2], T.float32) if K >= 3 else zero
            w3 = T.cast(W[d, 3], T.float32) if K >= 4 else zero
            for tx, vi in T.Parallel(block_size, values_per_thread):
                t = bx * block_l + vi * block_size + tx
                if t < L:
                    if K == 2:
                        acc = bias_f + (T.cast(A[b, d, t - 1], T.float32) * w0 if t >= 1 else zero) + T.cast(A[b, d, t], T.float32) * w1
                    elif K == 3:
                        acc = bias_f + (T.cast(A[b, d, t - 2], T.float32) * w0 if t >= 2 else zero) + (T.cast(A[b, d, t - 1], T.float32) * w1 if t >= 1 else zero) + T.cast(A[b, d, t], T.float32) * w2
                    else:
                        acc = bias_f + (T.cast(A[b, d, t - 3], T.float32) * w0 if t >= 3 else zero) + (T.cast(A[b, d, t - 2], T.float32) * w1 if t >= 2 else zero) + (T.cast(A[b, d, t - 1], T.float32) * w2 if t >= 1 else zero) + T.cast(A[b, d, t], T.float32) * w3
                    if apply_silu:
                        O[b, d, t] = T.cast(acc / (1.0 + T.exp(-acc)), dtype)
                    else:
                        O[b, d, t] = T.cast(acc, dtype)

    return kernel


def causal_conv1d_tilelang(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: str | None = None,
    *,
    vectorized: bool | None = None,
    values_per_thread: int = 4,
    channel_tiled: bool | None = None,
) -> torch.Tensor:
    """TileLang forward prototype compatible with causal_conv1d_fn's common path."""
    if activation not in (None, "silu", "swish"):
        raise NotImplementedError('activation must be None, "silu", or "swish"')
    if x.ndim != 3:
        raise ValueError("x must have shape (batch, dim, seqlen)")
    if weight.ndim != 2:
        raise ValueError("weight must have shape (dim, width)")
    if x.shape[1] != weight.shape[0]:
        raise ValueError(f"dim mismatch: x.shape[1]={x.shape[1]} vs weight.shape[0]={weight.shape[0]}")
    if x.dtype != weight.dtype or (bias is not None and bias.dtype != x.dtype):
        raise ValueError("x, weight, and bias must have the same dtype")
    dtype_map = {
        torch.float16: "float16",
        torch.bfloat16: "bfloat16",
        torch.float32: "float32",
    }
    if x.dtype not in dtype_map:
        raise NotImplementedError("prototype currently supports float16, bfloat16, and float32")
    if not x.is_cuda or not weight.is_cuda or (bias is not None and not bias.is_cuda):
        raise ValueError("all tensors must be CUDA tensors")

    x = x.contiguous()
    weight = weight.contiguous()
    apply_silu = activation in ("silu", "swish")
    batch, dim, seqlen = x.shape
    width = weight.shape[1]
    tl_dtype = dtype_map[x.dtype]
    if vectorized is None:
        vectorized = batch * dim * seqlen >= 1_000_000
    if channel_tiled is None:
        # CUDA grid.y is limited to 65535 on many launch paths. The channel-tile
        # kernel maps grid.y to B * D, so fall back to the flattened vectorized
        # kernel for very large batch/channel products.
        channel_tiled = batch * dim * seqlen >= 1_000_000 and batch * dim <= 65_535
    if bias is None:
        if channel_tiled:
            kernel = _causal_conv1d_fwd_no_bias_channel_tile_kernel(
                batch, dim, seqlen, width, tl_dtype, apply_silu, values_per_thread=values_per_thread
            )
            return kernel(x, weight)
        if vectorized:
            kernel = _causal_conv1d_fwd_no_bias_vec_kernel(
                batch, dim, seqlen, width, tl_dtype, apply_silu, values_per_thread=values_per_thread
            )
            return kernel(x, weight)
        kernel = _causal_conv1d_fwd_no_bias_kernel(batch, dim, seqlen, width, tl_dtype, apply_silu)
        return kernel(x, weight)
    if channel_tiled:
        kernel = _causal_conv1d_fwd_bias_channel_tile_kernel(
            batch, dim, seqlen, width, tl_dtype, apply_silu, values_per_thread=values_per_thread
        )
        return kernel(x, weight, bias.contiguous())
    if vectorized:
        kernel = _causal_conv1d_fwd_bias_vec_kernel(
            batch, dim, seqlen, width, tl_dtype, apply_silu, values_per_thread=values_per_thread
        )
        return kernel(x, weight, bias.contiguous())
    kernel = _causal_conv1d_fwd_bias_kernel(batch, dim, seqlen, width, tl_dtype, apply_silu)
    return kernel(x, weight, bias.contiguous())


def causal_conv1d_ref(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None, activation: str | None = None):
    out = torch.nn.functional.conv1d(
        x,
        weight.unsqueeze(1),
        bias,
        padding=weight.shape[1] - 1,
        groups=x.shape[1],
    )[..., : x.shape[-1]]
    if activation in ("silu", "swish"):
        out = torch.nn.functional.silu(out)
    return out


if __name__ == "__main__":
    from causal_conv1d import causal_conv1d_fn

    torch.manual_seed(0)
    tests = [
        (2, 128, 512, 2, False, None),
        (2, 128, 512, 3, True, None),
        (2, 128, 512, 4, True, "silu"),
    ]
    for B, D, L, K, has_bias, activation in tests:
        x = torch.randn(B, D, L, device="cuda", dtype=torch.float16)
        w = torch.randn(D, K, device="cuda", dtype=torch.float16)
        bias = torch.randn(D, device="cuda", dtype=torch.float16) if has_bias else None

        out = causal_conv1d_tilelang(x, w, bias, activation=activation)
        ref = causal_conv1d_fn(x, w, bias, activation=activation)
        torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2)
        print(f"ok B={B} D={D} L={L} K={K} bias={has_bias} activation={activation}")
