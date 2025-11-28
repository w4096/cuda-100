#include <common/utils.h>

/*
 * C = alpha * A * B + beta * C
 *
 * A: M x K
 * B: K x N
 * C: M x N
 *
 *        K                                       N
 *    +------+               N              +-------------+
 *    |      |        +-------------+       |             |
 *  M |      |    *   |             |   =   |             |  M
 *    |      |        +-------------+       |             |
 *    |      |                              |             |
 *    +------+                              +-------------+
 *
 *       A                   B                     C
 */

__global__ void gemm_naive(const int M, const int N, const int K, const float alpha, const float* A, const float* B,
                           const float beta, float* C) {
    const unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float t = 0.0;
        for (int i = 0; i < K; i++) {
            t += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = alpha * t + beta * C[row * N + col];
    }
}

template<unsigned int TILE = 16, unsigned int BLOCK_SIZE = 256>
__global__ void gemm_block_tiling_v1(const int M, const int N, const int K, const float alpha, const float* A,
                                     const float* B, const float beta, float* C) {
    static_assert(TILE * TILE == BLOCK_SIZE);
    const unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    __syncthreads();

    float t = 0.0;
    for (int i = 0; i < K; i += TILE) {
        As[threadIdx.y][threadIdx.x] = A[row * K + i + threadIdx.x];
        Bs[threadIdx.y][threadIdx.x] = B[(i + threadIdx.y) * N + col];
        __syncthreads();
#pragma unroll
        for (int k = 0; k < TILE; k++) {
            t += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }
    C[row * N + col] = alpha * t + beta * C[row * N + col];
}

template<unsigned int TILE = 32, unsigned int BLOCK_SIZE = 256>
__global__ void gemm_block_tiling_v2(const int M, const int N, const int K, const float alpha, const float* A,
                                     const float* B, const float beta, float* C) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    // thread blocks are 1D
    int tid = threadIdx.x;

    // the left upper corner of tile C in matrix C
    int cx = blockIdx.x * TILE;
    int cy = blockIdx.y * TILE;

    // every thread calculates `ts` elements of C
    constexpr int ts = TILE * TILE / BLOCK_SIZE;
    float t[ts] = {0.0};

    for (int k = 0; k < K; k += TILE) {
        /* load global memory into shared memory, here I use a loop with
         * step size of BLOCK_SIZE to handle more elements in a thread. */
        for (int i = tid; i < TILE * TILE; i += BLOCK_SIZE) {
            int y = i / TILE; // index in As and Bs
            int x = i % TILE;
            As[y][x] = A[(cy + y) * K + k + x];
            Bs[y][x] = B[(k + y) * N + cx + x];
        }
        __syncthreads();

        // do dot product through K dimension
        for (int i = 0; i < TILE; i++) {
            for (int m = tid, ti = 0; m < TILE * TILE; m += BLOCK_SIZE, ti++) {
                int y = m / TILE;
                int x = m % TILE;
                t[ti] += As[y][i] * Bs[i][x];
            }
        }
        __syncthreads();
    }

    for (int m = tid, ti = 0; m < TILE * TILE; m += BLOCK_SIZE, ti++) {
        int y = m / TILE; // index in tile of C
        int x = m % TILE;

        y = cy + y; // index in matrix C
        x = cx + x;
        C[y * N + x] = alpha * t[ti] + beta * C[y * N + x];
    }
}

template<int Bm = 128, int Bn = 128, int Bk = 8, int BLOCK_SIZE = 256>
__global__ void gemm_block_tiling_v3(const int M, const int N, const int K, const float alpha, const float* A,
                                     const float* B, const float beta, float* C) {
    // shared memory used to store the tiles of A and B
    __shared__ float As[Bm][Bk];
    __shared__ float Bs[Bk][Bn];

    // thread blocks are 1D
    const unsigned int tid = threadIdx.x;

    // the left upper corner of tile C in matrix C
    const unsigned int cx = blockIdx.x * Bn;
    const unsigned int cy = blockIdx.y * Bm;

    // tile C has size Bm x Bn, each thread has to process ts elements
    constexpr int ts = Bm * Bn / BLOCK_SIZE;
    float t[ts] = {0.0};

    for (int k = 0; k < K; k += Bk) {
#pragma unroll
        for (int i = 0; i < Bm * Bk; i += BLOCK_SIZE) {
            const int idx = i + tid;
            int y = idx / Bk; // index in As
            int x = idx % Bk;
            As[y][x] = A[(cy + y) * K + k + x];
        }

#pragma unroll
        for (int i = 0; i < Bk * Bn; i += BLOCK_SIZE) {
            const int idx = i + tid;
            int y = idx / Bn; // index in Bs
            int x = idx % Bn;
            Bs[y][x] = B[(k + y) * N + cx + x];
        }

        __syncthreads();

#pragma unroll
        for (int i = 0; i < Bk; i++) {
#pragma unroll
            for (int m = 0, ti = 0; m < Bm * Bn; m += BLOCK_SIZE, ti++) {
                const int idx = m + tid;
                int y = idx / Bn; // index in tile C
                int x = idx % Bn;
                t[ti] += As[y][i] * Bs[i][x];
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int m = 0, ti = 0; m < Bm * Bn; m += BLOCK_SIZE, ti++) {
        const int idx = m + tid;
        int y = idx / Bn; // index in tile C
        int x = idx % Bn;

        y = cy + y; // index in matrix C
        x = cx + x;
        C[y * N + x] = alpha * t[ti] + beta * C[y * N + x];
    }
}

template<int Bm = 128, int Bn = 128, int Bk = 8, int BLOCK_SIZE = 256, int BLOCK_WIDTH = 16>
__global__ void gemm_block_tiling_v4(const int M, const int N, const int K, const float alpha, const float* A,
                                     const float* B, const float beta, float* C) {
    // shared memory used to store the tiles of A and B
    __shared__ float As[Bm][Bk];
    __shared__ float Bs[Bk][Bn];

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
    float a[th];
    // save a column of matrix B
    float b[tw];

    for (int k = 0; k < K; k += Bk) {
#pragma unroll
        for (int i = 0; i < Bm * Bk; i += BLOCK_SIZE) {
            const int idx = i + tid;
            int y = idx / Bk; // index in As
            int x = idx % Bk;
            As[y][x] = A[(cy + y) * K + k + x];
        }

#pragma unroll
        for (int i = 0; i < Bk * Bn; i += BLOCK_SIZE) {
            const int idx = i + tid;
            int y = idx / Bn; // index in Bs
            int x = idx % Bn;
            Bs[y][x] = B[(k + y) * N + cx + x];
        }

        __syncthreads();

#pragma unroll
        for (int i = 0; i < Bk; i++) {
// read elements into registers
#pragma unroll
            for (int y = 0, idx = 0; y < Bm; y += BLOCK_HEIGHT, idx++) {
                a[idx] = As[y + ty][i];
            }
#pragma unroll
            for (int x = 0, idx = 0; x < Bn; x += BLOCK_WIDTH, idx++) {
                b[idx] = Bs[i][x + tx];
            }

#pragma unroll
            for (int m = 0; m < th; m++) {
#pragma unroll
                for (int n = 0; n < tw; n++) {
                    t[m][n] += a[m] * b[n];
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
            C[y * N + x] = alpha * t[m][n] + beta * C[y * N + x];
        }
    }
}

#include <cublas_v2.h>

void gemm_cublas(int M, int N, int K, float alpha, const float* A, const float* B, float beta, float* C) {
    cublasHandle_t handle;
    cublasCreate_v2(&handle);
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, N, A, K, &beta, C, N);
}

void gemm(int kernel, int M, int N, int K, float alpha, const float* A, const float* B, float beta, float* C) {
    dim3 threads, blocks;
    const int Bm = 128;
    const int Bn = 128;

    switch (kernel) {
    case 0:
        threads = dim3(16, 16);
        blocks = dim3(cdiv(N, threads.x), cdiv(M, threads.y));
        gemm_naive<<<blocks, threads>>>(M, N, K, alpha, A, B, beta, C);
        break;
    case 1: {
        constexpr int TILE_SIZE = 16;
        threads = {16, 16};
        blocks = {cdiv(N, TILE_SIZE), cdiv(M, TILE_SIZE)};
        ASSERT(TILE_SIZE == threads.x && TILE_SIZE == threads.y);
        gemm_block_tiling_v1<TILE_SIZE><<<blocks, threads>>>(M, N, K, alpha, A, B, beta, C);
        break;
    }
    case 2: {
        constexpr int TILE_SIZE = 32;
        threads = 256;
        blocks = {cdiv(N, TILE_SIZE), cdiv(M, TILE_SIZE)};
        ASSERT(threads.y == 1);
        gemm_block_tiling_v2<TILE_SIZE, 256><<<blocks, threads>>>(M, N, K, alpha, A, B, beta, C);
        break;
    }
    case 3: {
        threads = 256;
        blocks = {cdiv(N, Bn), cdiv(M, Bm)};
        gemm_block_tiling_v3<<<blocks, threads>>>(M, N, K, alpha, A, B, beta, C);
        break;
    }
    case 4: {
        threads = 256;
        blocks = {cdiv(N, Bn), cdiv(M, Bm)};
        gemm_block_tiling_v4<<<blocks, threads>>>(M, N, K, alpha, A, B, beta, C);
        break;
    }
    case 100:
        gemm_cublas(M, N, K, alpha, A, B, beta, C);
        break;
    }
}