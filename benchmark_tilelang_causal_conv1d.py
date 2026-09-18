from __future__ import annotations

import argparse
import statistics
from dataclasses import dataclass

import torch
from causal_conv1d import causal_conv1d_fn

from denseslm4.tilelang_causal_conv1d import causal_conv1d_tilelang


DTYPES = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}


@dataclass(frozen=True)
class Case:
    batch: int
    dim: int
    seqlen: int
    width: int
    dtype_name: str
    bias: bool = True
    activation: str | None = "silu"


def cuda_time_ms(fn, warmup: int, repeat: int) -> tuple[float, float, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(repeat):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return statistics.mean(times), statistics.median(times), min(times)


def run_case(case: Case, warmup: int, repeat: int, check: bool) -> dict[str, float | str | int]:
    dtype = DTYPES[case.dtype_name]
    x = torch.randn(case.batch, case.dim, case.seqlen, device="cuda", dtype=dtype)
    w = torch.randn(case.dim, case.width, device="cuda", dtype=dtype)
    bias = torch.randn(case.dim, device="cuda", dtype=dtype) if case.bias else None

    # Compile TileLang and warm official extension lazily before timing.
    y_tl = causal_conv1d_tilelang(x, w, bias, activation=case.activation)
    y_official = causal_conv1d_fn(x, w, bias, activation=case.activation)
    torch.cuda.synchronize()

    if check:
        atol = 2e-2 if dtype != torch.float32 else 1e-4
        rtol = 2e-2 if dtype != torch.float32 else 1e-4
        torch.testing.assert_close(y_tl, y_official, rtol=rtol, atol=atol)

    tl_mean, tl_median, tl_min = cuda_time_ms(
        lambda: causal_conv1d_tilelang(x, w, bias, activation=case.activation), warmup, repeat
    )
    off_mean, off_median, off_min = cuda_time_ms(
        lambda: causal_conv1d_fn(x, w, bias, activation=case.activation), warmup, repeat
    )

    return {
        "B": case.batch,
        "D": case.dim,
        "L": case.seqlen,
        "K": case.width,
        "dtype": case.dtype_name,
        "bias": str(case.bias),
        "act": str(case.activation),
        "tilelang_ms": tl_median,
        "official_ms": off_median,
        "speedup_official_over_tilelang": tl_median / off_median,
        "tilelang_min_ms": tl_min,
        "official_min_ms": off_min,
    }


def default_cases(dtype_names: list[str]) -> list[Case]:
    shapes = [
        (1, 128, 128),
        (1, 128, 512),
        (4, 256, 512),
        (4, 1024, 1024),
        (8, 2048, 2048),
    ]
    cases = []
    for dtype_name in dtype_names:
        for batch, dim, seqlen in shapes:
            for width in (2, 3, 4):
                cases.append(Case(batch, dim, seqlen, width, dtype_name, bias=True, activation="silu"))
    return cases


def print_table(rows: list[dict[str, float | str | int]]) -> None:
    headers = ["B", "D", "L", "K", "dtype", "tilelang_ms", "official_ms", "official/tilelang speedup"]
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join(["---"] * len(headers)) + "|")
    for r in rows:
        print(
            "| "
            + " | ".join(
                [
                    str(r["B"]),
                    str(r["D"]),
                    str(r["L"]),
                    str(r["K"]),
                    str(r["dtype"]),
                    f"{float(r['tilelang_ms']):.4f}",
                    f"{float(r['official_ms']):.4f}",
                    f"{float(r['speedup_official_over_tilelang']):.2f}x",
                ]
            )
            + " |"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtypes", nargs="+", default=["fp16", "bf16", "fp32"], choices=DTYPES.keys())
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--no-check", action="store_true")
    args = parser.parse_args()

    rows = []
    for case in default_cases(args.dtypes):
        try:
            row = run_case(case, args.warmup, args.repeat, check=not args.no_check)
        except Exception as exc:
            print(f"FAILED {case}: {type(exc).__name__}: {exc}")
            continue
        rows.append(row)
        print(
            f"done B={case.batch} D={case.dim} L={case.seqlen} K={case.width} {case.dtype_name}: "
            f"tilelang={row['tilelang_ms']:.4f} ms official={row['official_ms']:.4f} ms "
            f"speedup={row['speedup_official_over_tilelang']:.2f}x"
        )

    print()
    print_table(rows)


if __name__ == "__main__":
    main()
