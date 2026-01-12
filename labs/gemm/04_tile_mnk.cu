#include <cuda_fp16.h>

template<int Bm, int Bn, int Bk, int BLOCK_DIM_Y, int BLOCK_DIM_X>
__global__ void gemm_block_tiling_v3(const int M, const int N, const int K, const float alpha, const half* A,
                                     const half* B, const float beta, float* C) {
    __shared__ half As[Bm][Bk];
    __shared__ half Bs[Bk][Bn];

    // thread id in thread block
    int tid = blockDim.x * threadIdx.y + threadIdx.x;

    // the left upper corner of tile C in matrix C
    const unsigned int cx = blockIdx.x * Bn;
    const unsigned int cy = blockIdx.y * Bm;

    constexpr int BLOCK_SIZE = BLOCK_DIM_Y * BLOCK_DIM_X;
    constexpr int BATCH_X = Bn / BLOCK_DIM_X;
    constexpr int BATCH_Y = Bm / BLOCK_DIM_Y;

    float accum[BATCH_Y][BATCH_X] = {0.0};

    for (int k = 0; k < K; k += Bk) {
        for (int i = 0; i < Bm * Bk; i += BLOCK_SIZE) {
            const int idx = i + tid;
            int y = idx / Bk; // index in As
            int x = idx % Bk;

            int ax = k + x; // index in matrix A
            int ay = cy + y;
            As[y][x] = (ax < K && ay < M) ? A[ay * K + ax] : half(0.0f);
        }

        for (int i = 0; i < Bk * Bn; i += BLOCK_SIZE) {
            const int idx = i + tid;
            int y = idx / Bn; // index in Bs
            int x = idx % Bn;

            int bx = cx + x; // index in matrix B
            int by = k + y;
            Bs[y][x] = (bx < N && by < K) ? B[by * N + bx] : half(0.0f);
        }

        __syncthreads();
        for (int i = 0; i < BATCH_Y; i++) {
            for (int j = 0; j < BATCH_X; j++) {
                int y = threadIdx.y + i * BLOCK_DIM_Y;
                int x = threadIdx.x + j * BLOCK_DIM_X;
                for (int ki = 0; ki < Bk; ki++) {
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

            if (x < N && y < M) {
                C[y * N + x] = alpha * accum[i][j] + beta * C[y * N + x];
            }
        }
    }
}


extern "C" void gemm_tile_m64n64k16(const int M, const int N, const int K, const float alpha, const half* A,
                              const half* B, const float beta, float* C) {
    constexpr int BLOCK_DIM_X = 16;
    constexpr int BLOCK_DIM_Y = 16;
    constexpr int Bm = BLOCK_DIM_X * 4;
    constexpr int Bn = BLOCK_DIM_X * 4;
    constexpr int Bk = 16;

    dim3 blockDim(BLOCK_DIM_X, BLOCK_DIM_Y);
    dim3 gridDim((N + Bn - 1) / Bn, (M + Bm - 1) / Bm);

    gemm_block_tiling_v3<Bm, Bn, Bk, BLOCK_DIM_Y, BLOCK_DIM_X><<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}
