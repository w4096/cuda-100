#include <mma.h>

using namespace nvcuda;

template<int WMMA_M, int WMMA_N, int WMMA_K>
__global__ void wmma_kernel(int M, int N, int K, float alpha, const half* A, const half* B, float beta, float* C){
    int lda = K;
    int ldb = N;
    int ldc = N;

    // tensor core fragment
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>              acc_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>              c_frag;

    // init of C fragment
    wmma::fill_fragment(acc_frag, 0.0f);

    int warp_x = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int warp_y = (blockIdx.y * blockDim.y + threadIdx.y);

    for(int i = 0; i < K; i += WMMA_K){
        int a_x = i;
        int a_y = warp_y * WMMA_M;

        int b_x = warp_x * WMMA_N;
        int b_y = i;

        if (a_x < K && a_y < M && b_x < N && b_y < K) {
            const half* A_ptr = &A[a_y * lda + a_x];
            const half* B_ptr = &B[b_y * ldb + b_x];

            // Load the inputs
            wmma::load_matrix_sync(a_frag, A_ptr, lda);
            wmma::load_matrix_sync(b_frag, B_ptr, ldb);

            // Perform the matrix multiplication
            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
    }

    int c_x = warp_x * WMMA_N;
    int c_y = warp_y * WMMA_M;

    if (c_x < N && c_y < M) {
        float* c_ptr = &C[c_y * ldc + c_x];
        if (alpha != 1.0f || beta != 0.0f) {
            wmma::load_matrix_sync(c_frag, c_ptr, ldc, wmma::mem_row_major);

            // scale by alpha and beta
            for(int i = 0; i < c_frag.num_elements; i++){
                c_frag.x[i] = alpha * acc_frag.x[i] + beta * c_frag.x[i];
            }

            // write memory back to C
            wmma::store_matrix_sync(c_ptr, c_frag, ldc, wmma::mem_row_major);
        } else {
            // write memory back to C
            wmma::store_matrix_sync(c_ptr, acc_frag, ldc, wmma::mem_row_major);
        }
    }
}

extern "C" void gemm_wmma_16x16x16(int M, int N, int K, float alpha, const half* A, const half* B, float beta, float* C) {
    constexpr int WMMA_M = 16;
    constexpr int WMMA_N = 16;
    constexpr int WMMA_K = 16;

    dim3 blockDim(128, 4);


    int rows_per_block = WMMA_M * blockDim.y;
    int cols_per_block = WMMA_N * (blockDim.x / 32);

    dim3 gridDim;
    gridDim.x = (N + cols_per_block -1) / cols_per_block;
    gridDim.y = (M + rows_per_block -1) / rows_per_block;

    wmma_kernel<WMMA_M, WMMA_N, WMMA_K><<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
    cudaDeviceSynchronize();
}
