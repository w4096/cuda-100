#include <cutlass/gemm/device/gemm.h>

extern "C" void gemm_cutlass(int M, int N, int K, float alpha, const half* A, const half* B, float beta, float* C) {
    using Gemm = cutlass::gemm::device::Gemm<
        half, cutlass::layout::RowMajor,
        half, cutlass::layout::RowMajor,
        float, cutlass::layout::RowMajor
    >;

    Gemm gemm_op;
    typename Gemm::Arguments args(
        {M, N, K},
        {A, K},
        {B, N},
        {C, N},
        {C, N},
        {alpha, beta}
    );
    gemm_op(args);
}
