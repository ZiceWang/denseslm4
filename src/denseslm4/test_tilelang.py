import torch
import tilelang
import tilelang.language as T


@tilelang.jit
def gemm(
    A,
    B,
    block_M: int = 64,
    block_N: int = 64,
    block_K: int = 32,
):
    # Eager JIT 写法：动态维度用 T.const 声明，不要写 shape="M K"。
    M, N, K = T.const("M, N, K")

    # Tensor 标注写在函数体里。注意这里是 [[M, K], dtype]，不是 shape="..."。
    A: T.Tensor[[M, K], T.float16]
    B: T.Tensor[[K, N], T.float16]

    C = T.empty((M, N), T.float32)

    # blockIdx.x 覆盖 N 方向，blockIdx.y 覆盖 M 方向。
    with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
        A_shared = T.alloc_shared((block_M, block_K), A.dtype)
        B_shared = T.alloc_shared((block_K, block_N), B.dtype)
        C_local = T.alloc_fragment((block_M, block_N), T.float32)

        T.clear(C_local)
        for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
            T.copy(A[by * block_M, ko * block_K], A_shared)
            T.copy(B[ko * block_K, bx * block_N], B_shared)
            T.gemm(A_shared, B_shared, C_local)

        T.copy(C_local, C[by * block_M, bx * block_N])

    return C


if __name__ == "__main__":
    M, N, K = 512, 512, 512
    A = torch.randn(M, K, device="cuda", dtype=torch.float16)
    B = torch.randn(K, N, device="cuda", dtype=torch.float16)

    C = gemm(A, B)
    ref = A.float() @ B.float()

    torch.testing.assert_close(C, ref, rtol=1e-2, atol=1e-1)
    print("✅ TileLang GEMM 运行成功！", C.shape, C.dtype)