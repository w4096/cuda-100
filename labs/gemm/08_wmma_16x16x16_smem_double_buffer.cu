#include <mma.h>

using namespace nvcuda;


/*
  WAPRS_GRID_SIZE = 4

  +------------------+--------------------+---------------------+---------------------+
 | warp0            | warp1              | warp2               | warp3               |
 +------------------+--------------------+---------------------+---------------------+
 | warp4            | warp5              | warp6               | warp7               |
 +------------------+--------------------+---------------------+---------------------+
 | warp8            | warp9              | warp10              | warp11              |
 +------------------+--------------------+---------------------+---------------------+
 | warp12           | warp13             | warp14              | warp15              |
 +------------------+--------------------+---------------------+---------------------+
*/

template<int WMMA_TILE_SIZE, int WAPRS_GRID_SIZE>
__global__ void wmma_smem_double_buffer_kernel(int M, int N, int K, float alpha, const half* A, const half* B, float beta, float* C){
    __shared__ half As[2][WAPRS_GRID_SIZE][WAPRS_GRID_SIZE][WMMA_TILE_SIZE][WMMA_TILE_SIZE];
    __shared__ half Bs[2][WAPRS_GRID_SIZE][WAPRS_GRID_SIZE][WMMA_TILE_SIZE][WMMA_TILE_SIZE];

    // tensor core fragment
    wmma::fragment<wmma::matrix_a, WMMA_TILE_SIZE, WMMA_TILE_SIZE, WMMA_TILE_SIZE, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_TILE_SIZE, WMMA_TILE_SIZE, WMMA_TILE_SIZE, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_TILE_SIZE, WMMA_TILE_SIZE, WMMA_TILE_SIZE, float>              acc_frag;
    wmma::fragment<wmma::accumulator, WMMA_TILE_SIZE, WMMA_TILE_SIZE, WMMA_TILE_SIZE, float>              c_frag;

    // init of C fragment
    wmma::fill_fragment(acc_frag, 0.0f);

    int warp_x = threadIdx.x / warpSize;
    int warp_y = threadIdx.y;
    int lane = threadIdx.x % warpSize;

    constexpr int BLOCK_M = WMMA_TILE_SIZE * WAPRS_GRID_SIZE;
    constexpr int BLOCK_N = BLOCK_M;
    constexpr int BLOCK_K = BLOCK_M;

    int lda = K;
    int ldb = N;
    int ldc = N;

    int smem_idx = 0;

    auto load_global_mem_to_smem = [&](int k, int smem_idx){
        // load A and B matrix to shared memory
        for (int i = 0; i < WMMA_TILE_SIZE * WMMA_TILE_SIZE; i += warpSize) {
            int idx = lane + i;
            int a_row = idx / WMMA_TILE_SIZE;
            int a_col = idx % WMMA_TILE_SIZE;
            
            int global_a_row = blockIdx.y * BLOCK_M + warp_y * WMMA_TILE_SIZE + a_row;
            int global_a_col = k + warp_x * WMMA_TILE_SIZE + a_col;

            if (global_a_row < M && global_a_col < K) {
                As[smem_idx][warp_y][warp_x][a_row][a_col] = A[global_a_row * lda + global_a_col];
            } else {
                As[smem_idx][warp_y][warp_x][a_row][a_col] = half(0.0f);
            }


            int b_row = idx / WMMA_TILE_SIZE;
            int b_col = idx % WMMA_TILE_SIZE;

            int global_b_row = k + warp_y * WMMA_TILE_SIZE + b_row;
            int global_b_col = blockIdx.x * BLOCK_N + warp_x * WMMA_TILE_SIZE + b_col;

            if (global_b_row < K && global_b_col < N) {
                Bs[smem_idx][warp_y][warp_x][b_row][b_col] = B[global_b_row * ldb + global_b_col];
            } else {
                Bs[smem_idx][warp_y][warp_x][b_row][b_col] = half(0.0f);
            }
        }
    };

    load_global_mem_to_smem(0, smem_idx);
    __syncthreads();

    for(int k = 0; k < K; k += BLOCK_K){

        // load A and B matrix to shared memory
        int next_k = k + BLOCK_K;
        int next_smem_idx = (smem_idx + 1) % 2;
        if(next_k < K){
            load_global_mem_to_smem(next_k, next_smem_idx);
        }
        
        for (int i = 0; i < WAPRS_GRID_SIZE; i++) {
            const half* A_ptr = &As[smem_idx][warp_y][i][0][0];
            const half* B_ptr = &Bs[smem_idx][i][warp_x][0][0];

            // Load the inputs
            wmma::load_matrix_sync(a_frag, A_ptr, WMMA_TILE_SIZE);
            wmma::load_matrix_sync(b_frag, B_ptr, WMMA_TILE_SIZE);

            // Perform the matrix multiplication
            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }

        __syncthreads();
        smem_idx = (smem_idx + 1) % 2;
    }

    int c_x = blockIdx.x * BLOCK_N + warp_x * WMMA_TILE_SIZE;
    int c_y = blockIdx.y * BLOCK_M + warp_y * WMMA_TILE_SIZE;
    
    if (c_x < N && c_y < M) {
        float* c_ptr = &C[c_y * ldc + c_x];
    
        wmma::load_matrix_sync(c_frag, c_ptr, ldc, wmma::mem_row_major);

        // scale by alpha and beta
        for(int i = 0; i < c_frag.num_elements; i++){
            c_frag.x[i] = alpha * acc_frag.x[i] + beta * c_frag.x[i];
        }

        // write memory back to C
        wmma::store_matrix_sync(c_ptr, c_frag, ldc, wmma::mem_row_major);
    }
}

extern "C" void gemm_wmma_16x16x16_smem_double_buffer(int M, int N, int K, float alpha, const half* A, const half* B, float beta, float* C) {
    constexpr int WMMA_TILE_SIZE = 16;
    constexpr int WAPRS_GRID_SIZE = 4; // number of warps per block in x dimension

    dim3 blockDim(WAPRS_GRID_SIZE * 32, WAPRS_GRID_SIZE);

    int BLOCK_M = WMMA_TILE_SIZE * WAPRS_GRID_SIZE; // block size in M dimension
    int BLOCK_N = WMMA_TILE_SIZE * WAPRS_GRID_SIZE; // block size in N dimension

    dim3 gridDim;
    gridDim.x = (N + BLOCK_N -1) / BLOCK_N;
    gridDim.y = (M + BLOCK_M -1) / BLOCK_M;
    wmma_smem_double_buffer_kernel<WMMA_TILE_SIZE, WAPRS_GRID_SIZE><<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
    cudaDeviceSynchronize();
}
