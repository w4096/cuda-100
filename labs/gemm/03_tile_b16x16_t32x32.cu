#include <cuda_fp16.h>

template<int TILE_DIM, int BLOCK_DIM_Y, int BLOCK_DIM_X>
__global__   void gemm_block_tiling_v2(const int M, const int N, const int K, const float alpha, const half* A,
                                     const half* B, const float beta, float* C) {
    __shared__ half As[TILE_DIM][TILE_DIM+1];
    __shared__ half Bs[TILE_DIM][TILE_DIM];

    // thread id in thread block
    int tid = blockDim.x * threadIdx.y + threadIdx.x;

    // the left upper corner of tile in matrix C
    int cx = blockIdx.x * TILE_DIM;
    int cy = blockIdx.y * TILE_DIM;

    constexpr int BLOCK_SIZE = BLOCK_DIM_Y * BLOCK_DIM_X;
    constexpr int BATCH_X = TILE_DIM / BLOCK_DIM_X;
    constexpr int BATCH_Y = TILE_DIM / BLOCK_DIM_Y;

    float accum[BATCH_Y][BATCH_X] = {0.0};

    for (int k = 0; k < K; k += TILE_DIM) {
        /* load global memory into shared memory, here I use a loop with
         * step size of BLOCK_SIZE to handle more elements in a thread. */
        for (int i = 0; i < TILE_DIM * TILE_DIM; i += BLOCK_SIZE) {
            int y = (i+tid) / TILE_DIM; // index in As and Bs
            int x = (i+tid) % TILE_DIM;
            
            int ax = k + x; // index in matrix A
            int ay = cy + y;
            int bx = cx + x; // index in matrix B
            int by = k + y;

            As[y][x] = (ax < K && ay < M) ? A[ay * K + ax] : half(0.0f);
            Bs[y][x] = (bx < N && by < K) ? B[by * N + bx] : half(0.0f);
        }
        __syncthreads();
        for (int i = 0; i < BATCH_Y; i++) {
            for (int j = 0; j < BATCH_X; j++) {
                int y = threadIdx.y + i * BLOCK_DIM_Y;
                int x = threadIdx.x + j * BLOCK_DIM_X;
                for (int ki = 0; ki < TILE_DIM; ki++) {
                    accum[i][j] += __half2float(As[y][ki]) * __half2float(Bs[ki][x]);
                }
            }
        }
        __syncthreads();
    }
    for (int i = 0; i < BATCH_Y; i++) {
        for (int j = 0; j < BATCH_X; j++) {
            int y = threadIdx.y + i * BLOCK_DIM_Y;
            int x = threadIdx.x + j * BLOCK_DIM_X;
            y = cy + y; // index in matrix C
            x = cx + x;

            if (y < M && x < N) {
                C[y * N + x] = alpha * accum[i][j] + beta * C[y * N + x];
            }
        }
    }
}

inline unsigned int cdiv(unsigned int a, unsigned int b) {
    return (a + b - 1) / b;
}

// Tile size: 32x32, Block size: 16x16, each thread computes 2x2 elements
extern "C" void gemm_tile_B16x16_T32x32(const int M, const int N, const int K, const float alpha,
                                            const half* A, const half* B, const float beta, float* C) {
    constexpr int BLOCK_DIM_X = 16;
    constexpr int BLOCK_DIM_Y = 16;
    constexpr int TILE_DIM = BLOCK_DIM_X * 2;
    dim3 threads = {BLOCK_DIM_X, BLOCK_DIM_Y};
    dim3 blocks = {cdiv(N, TILE_DIM), cdiv(M, TILE_DIM)};

    gemm_block_tiling_v2<TILE_DIM, BLOCK_DIM_Y, BLOCK_DIM_X><<<blocks, threads>>>(M, N, K, alpha, A, B, beta, C);
}
