#include <cuda_fp16.h>
#include <cuda_runtime.h>

__global__ void gemm_naive(const int M, int N, int K, float alpha, const half* A, const half* B, float beta, half* C) {
    const unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float t = 0.0;
        for (int i = 0; i < K; i++) {
            t += __half2float(A[row * K + i]) * __half2float(B[i * N + col]);
        }
        C[row * N + col] = alpha * t + beta * __half2float(C[row * N + col]);
    }
}

template<int Bm = 128, int Bn = 128, int Bk = 8, int BLOCK_SIZE = 256, int BLOCK_WIDTH = 16>
__global__ void gemm_block_tiling_v5(int M, int N, int K, float alpha,
                                     const half* A, const half* B, float beta, half* C) {
    // shared memory used to store the tiles of A and B
    __shared__ half As[Bm][Bk];
    __shared__ half Bs[Bk][Bn];

    // thread blocks are 1D
    const unsigned int tid = threadIdx.x;

    // the left upper corner of tile C in matrix C
    const unsigned int cx = blockIdx.x * Bn;
    const unsigned int cy = blockIdx.y * Bm;

    const unsigned int tx = tid % BLOCK_WIDTH;
    const unsigned int ty = tid / BLOCK_WIDTH;

    constexpr int BLOCK_HEIGHT = BLOCK_SIZE / BLOCK_WIDTH;

    /* tile C has size Bm x Bn, block has size  BLOCK_HEIGHT x BLOCK_WIDTH
     * each thread has to save tw x th element of matrix C */
    constexpr int tw = Bn / BLOCK_WIDTH;
    constexpr int th = Bm / BLOCK_HEIGHT;
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
            int Ay = cy + y;
            int Ax = k + x;
            if (Ay < M && Ax < K) {
                As[y][x] = A[Ay * K + Ax];
            }
        }

        for (int i = 0; i < Bk * Bn; i += BLOCK_SIZE) {
            const int idx = i + tid;
            int y = idx / Bn; // index in Bs
            int x = idx % Bn;
            int By = k + y;
            int Bx = cx + x;
            if (By < K && Bx < N) {
                Bs[y][x] = B[By * N + Bx];
            }
        }

        __syncthreads();

        for (int i = 0; i < Bk; i++) {
            // read elements into registers
            for (int y = 0, idx = 0; y < Bm; y += BLOCK_HEIGHT, idx++) {
                a[idx] = As[y + ty][i];
            }
            for (int x = 0, idx = 0; x < Bn; x += BLOCK_WIDTH, idx++) {
                b[idx] = Bs[i][x + tx];
            }

            for (int m = 0; m < th; m++) {
                for (int n = 0; n < tw; n++) {
                    t[m][n] += __half2float(a[m] * b[n]);
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int m = 0; m < th; m++) {
    #pragma unroll
        for (int n = 0; n < tw; n++) {
            int x = n * BLOCK_WIDTH + tx;
            int y = m * BLOCK_HEIGHT + ty;
            y = cy + y; // index in matrix C
            x = cx + x;
            if (y < M && x < N) {
                C[y * N + x] = alpha * t[m][n] + beta * __half2float(C[y * N + x]);
            }
        }
    }
}

// A, B, and C are device pointers
extern "C" void solve2(const half* A, const half* B, half* C, int M, int N, int K, float alpha, float beta) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (M + threadsPerBlock.y - 1) / threadsPerBlock.y);
    gemm_naive<<<blocksPerGrid, threadsPerBlock>>>(M, N, K, alpha, A, B, beta, C);
}

static unsigned int cdiv(unsigned int a, unsigned int b) {
    return (a + b - 1) / b;
}

extern "C" void solve(const half* A, const half* B, half* C, int M, int N, int K, float alpha, float beta) {
    constexpr int TILE_M = 128;
    constexpr int TILE_N = 128;

    dim3 threads{256};
    dim3 blocks{cdiv(N, TILE_N), cdiv(M, TILE_M)};

    gemm_block_tiling_v5<<<blocks, threads>>>(M, N, K, alpha, A, B, beta, C);
}
