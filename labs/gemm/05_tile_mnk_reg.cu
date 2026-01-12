#include <cuda_fp16.h>


template<int Bm = 128, int Bn = 128, int Bk = 8, int BLOCK_DIM_Y=16, int BLOCK_DIM_X=16>
__global__ void gemm_block_tiling_v4(const int M, const int N, const int K, const float alpha,
                                     const half* A, const half* B, const float beta, float* C) {
    // shared memory used to store the tiles of A and B
    __shared__ half As[Bm][Bk];
    __shared__ half Bs[Bk][Bn];

    const unsigned int tid = threadIdx.x + threadIdx.y * blockDim.x;

    // the left upper corner of tile C in matrix C
    const unsigned int cx = blockIdx.x * Bn;
    const unsigned int cy = blockIdx.y * Bm;

    constexpr unsigned int BLOCK_SIZE = BLOCK_DIM_X * BLOCK_DIM_Y;

    const unsigned int tx = tid % BLOCK_DIM_X;
    const unsigned int ty = tid / BLOCK_DIM_X;

    /* tile C has size Bm x Bn, block has size  BLOCK_HEIGHT x BLOCK_WIDTH
     * each thread has to save tw x th element of matrix C */
    constexpr int tw = Bn / BLOCK_DIM_X;
    constexpr int th = Bm / BLOCK_DIM_Y;
    float t[th][tw] = {0.0};

    // save a row of matrix A
    half a[th];
    // save a column of matrix B
    half b[tw];

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

        for (int i = 0; i < Bk; i++) {
            // read elements into registers
            for (int y = 0, idx = 0; y < Bm; y += BLOCK_DIM_Y, idx++) {
                a[idx] = As[y + ty][i];
            }
            for (int x = 0, idx = 0; x < Bn; x += BLOCK_DIM_X, idx++) {
                b[idx] = Bs[i][x + tx];
            }

            for (int m = 0; m < th; m++) {
                for (int n = 0; n < tw; n++) {
                    t[m][n] += __half2float(a[m]) * __half2float(b[n]);
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int m = 0; m < th; m++) {
        #pragma unroll
        for (int n = 0; n < tw; n++) {
            int x = n * BLOCK_DIM_X + tx;
            int y = m * BLOCK_DIM_Y + ty;
            y = cy + y; // index in matrix C
            x = cx + x;
            if (x < N && y < M) {
                C[y * N + x] = alpha * t[m][n] + beta * C[y * N + x];
            }
        }
    }
}


extern "C" void gemm_tile_m64n64k16_reg(int M, int N, int K, float alpha, const half* A, const half* B, float beta, float* C) {
    constexpr int BLOCK_DIM_X = 16;
    constexpr int BLOCK_DIM_Y = 16;
    constexpr int Bm = 64;
    constexpr int Bn = 64;
    constexpr int Bk = 16;

    dim3 blockDim(BLOCK_DIM_X, BLOCK_DIM_Y);
    dim3 gridDim((N + Bn - 1) / Bn, (M + Bm - 1) / Bm);

    gemm_block_tiling_v4<Bm, Bn, Bk, BLOCK_DIM_Y, BLOCK_DIM_X><<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}
