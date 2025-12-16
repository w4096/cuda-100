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

template<int TILE_DIM = 16>
__global__ void gemm_block_tiling_v1(const int M, const int N, const int K, const float alpha, const float* A,
                                     const float* B, const float beta, float* C) {
    const unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float As[TILE_DIM][TILE_DIM];
    __shared__ float Bs[TILE_DIM][TILE_DIM];

    __syncthreads();

    float accum = 0.0;
    for (int i = 0; i < K; i += TILE_DIM) {
        As[threadIdx.y][threadIdx.x] = A[row * K + i + threadIdx.x];
        Bs[threadIdx.y][threadIdx.x] = B[(i + threadIdx.y) * N + col];
        __syncthreads();

        for (int k = 0; k < TILE_DIM; k++) {
            accum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }
    C[row * N + col] = alpha * accum + beta * C[row * N + col];
}

template<int TILE_DIM, int BLOCK_DIM_Y, int BLOCK_DIM_X>
__global__   void gemm_block_tiling_v2(const int M, const int N, const int K, const float alpha, const float* A,
                                     const float* B, const float beta, float* C) {
    __shared__ float As[TILE_DIM][TILE_DIM+1];
    __shared__ float Bs[TILE_DIM][TILE_DIM];

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
            As[y][x] = A[(cy + y) * K + k + x];
            Bs[y][x] = B[(k + y) * N + cx + x];
        }
        __syncthreads();
        for (int i = 0; i < BATCH_Y; i++) {
            for (int j = 0; j < BATCH_X; j++) {
                int y = threadIdx.y + i * BLOCK_DIM_Y;
                int x = threadIdx.x + j * BLOCK_DIM_X;
                for (int ki = 0; ki < TILE_DIM; ki++) {
                    accum[i][j] += As[y][ki] * Bs[ki][x];
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
            C[y * N + x] = alpha * accum[i][j] + beta * C[y * N + x];
        }
    }
}

template<int Bm, int Bn, int Bk, int BLOCK_DIM_Y, int BLOCK_DIM_X>
__global__ void gemm_block_tiling_v3(const int M, const int N, const int K, const float alpha, const float* A,
                                     const float* B, const float beta, float* C) {
    __shared__ float As[Bm][Bk];
    __shared__ float Bs[Bk][Bn];

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
            As[y][x] = A[(cy + y) * K + k + x];
        }

        for (int i = 0; i < Bk * Bn; i += BLOCK_SIZE) {
            const int idx = i + tid;
            int y = idx / Bn; // index in Bs
            int x = idx % Bn;
            Bs[y][x] = B[(k + y) * N + cx + x];
        }

        __syncthreads();
        for (int i = 0; i < BATCH_Y; i++) {
            for (int j = 0; j < BATCH_X; j++) {
                int y = threadIdx.y + i * BLOCK_DIM_Y;
                int x = threadIdx.x + j * BLOCK_DIM_X;
                for (int ki = 0; ki < Bk; ki++) {
                    accum[i][j] += As[y][ki] * Bs[ki][x];
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
            C[y * N + x] = alpha * accum[i][j] + beta * C[y * N + x];
        }
    }
}

template<int Bm, int Bn, int Bk, int BLOCK_DIM_Y, int BLOCK_DIM_X>
__global__ void gemm_block_tiling_v4(const int M, const int N, const int K, const float alpha, const float* A,
                                     const float* B, const float beta, float* C) {
    // shared memory used to store the tiles of A and B
    __shared__ float As[Bm][Bk];
    __shared__ float Bs[Bk][Bn];

    // thread id in thread block
    int tid = blockDim.x * threadIdx.y + threadIdx.x;

    // the left upper corner of tile C in matrix C
    const unsigned int cx = blockIdx.x * Bn;
    const unsigned int cy = blockIdx.y * Bm;

    constexpr int BLOCK_SIZE = BLOCK_DIM_Y * BLOCK_DIM_X;
    constexpr int BATCH_X = Bn / BLOCK_DIM_X;
    constexpr int BATCH_Y = Bm / BLOCK_DIM_Y;

    float accum[BATCH_Y][BATCH_X] = {0.0};

    float a[BATCH_Y];
    float b[BATCH_X];

    for (int k = 0; k < K; k += Bk) {
        for (int i = 0; i < Bm * Bk; i += BLOCK_SIZE) {
            const int idx = i + tid;
            int y = idx / Bk; // index in As
            int x = idx % Bk;
            As[y][x] = A[(cy + y) * K + k + x];
        }
        for (int i = 0; i < Bk * Bn; i += BLOCK_SIZE) {
            const int idx = i + tid;
            int y = idx / Bn; // index in Bs
            int x = idx % Bn;
            Bs[y][x] = B[(k + y) * N + cx + x];
        }

        __syncthreads();


        for (int p = 0; p < Bk; p++) {
            for (int i = 0; i < BATCH_Y; i++) {
                int y = threadIdx.y + i * BLOCK_DIM_Y;
                a[i] = As[y][p];
            }
            for (int i = 0; i < BATCH_X; i++) {
                int x = threadIdx.x + i * BLOCK_DIM_X;
                b[i] = Bs[p][x];
            }
            for (int i = 0; i < BATCH_Y; i++) {
                for (int j = 0; j < BATCH_X; j++) {
                    accum[i][j] += a[i] * b[j];
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
            C[y * N + x] = alpha * accum[i][j] + beta * C[y * N + x];
        }
    }
}


template<int Bm = 128, int Bn = 128, int Bk = 8, int BLOCK_SIZE = 256, int BLOCK_WIDTH = 16>
__global__ void gemm_block_tiling_v5(const int M, const int N, const int K, const float alpha,
                                     const float* A, const float* B, const float beta, float* C) {
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
        for (int i = 0; i < Bm * Bk; i += BLOCK_SIZE) {
            const int idx = i + tid;
            int y = idx / Bk; // index in As
            int x = idx % Bk;
            As[y][x] = A[(cy + y) * K + k + x];
        }

        for (int i = 0; i < Bk * Bn; i += BLOCK_SIZE) {
            const int idx = i + tid;
            int y = idx / Bn; // index in Bs
            int x = idx % Bn;
            Bs[y][x] = B[(k + y) * N + cx + x];
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
    switch (kernel) {
    case 0: {
        dim3 threads = {16, 16};
        dim3 blocks = {cdiv(N, threads.x), cdiv(M, threads.y)};
        gemm_naive<<<blocks, threads>>>(M, N, K, alpha, A, B, beta, C);
        break;
    }
    case 1: {
        constexpr int TILE_DIM = 16;
        dim3 threads = {16, 16};
        dim3 blocks = {cdiv(N, TILE_DIM), cdiv(M, TILE_DIM)};
        ASSERT(TILE_DIM == threads.x && TILE_DIM == threads.y);
        gemm_block_tiling_v1<TILE_DIM><<<blocks, threads>>>(M, N, K, alpha, A, B, beta, C);
        break;
    }
    case 2: {
        constexpr int BLOCK_DIM_X = 16;
        constexpr int BLOCK_DIM_Y = 16;
        constexpr int TILE_DIM = BLOCK_DIM_X * 2;
        dim3 threads = {16, 16};
        dim3 blocks = {cdiv(N, TILE_DIM), cdiv(M, TILE_DIM)};
        gemm_block_tiling_v2<TILE_DIM, BLOCK_DIM_Y, BLOCK_DIM_X><<<blocks, threads>>>(M, N, K, alpha, A, B, beta, C);
        break;
    }
    case 3: {
        constexpr int BLOCK_DIM_X = 16;
        constexpr int BLOCK_DIM_Y = 16;
        constexpr int Bm = BLOCK_DIM_X * 4;
        constexpr int Bn = BLOCK_DIM_X * 4;
        constexpr int Bk = 16;
        dim3 threads = {16, 16};
        dim3 blocks = {cdiv(N, Bn), cdiv(M, Bm)};
        gemm_block_tiling_v3<Bm, Bn, Bk, BLOCK_DIM_Y, BLOCK_DIM_X>
            <<<blocks, threads>>>(M, N, K, alpha, A, B, beta, C);
        break;
    }
    case 4: {
        constexpr int BLOCK_DIM_X = 16;
        constexpr int BLOCK_DIM_Y = 16;
        constexpr int Bm = BLOCK_DIM_Y * 8;
        constexpr int Bn = BLOCK_DIM_X * 8;
        constexpr int Bk = 16;
        dim3 threads = {BLOCK_DIM_X, BLOCK_DIM_Y};
        dim3 blocks = {cdiv(N, Bn), cdiv(M, Bm)};
        gemm_block_tiling_v4<Bm, Bn, Bk, BLOCK_DIM_Y, BLOCK_DIM_X>
            <<<blocks, threads>>>(M, N, K, alpha, A, B, beta, C);
        break;
    }
    case 6: {
        constexpr int TILE_M = 128;
        constexpr int TILE_N = 128;

        dim3 threads{256};
        dim3 blocks{M / static_cast<unsigned int>(TILE_M), N / static_cast<unsigned int>(TILE_N)};
        gemm_block_tiling_v5<<<blocks, threads>>>(M, N, K, 1.0, A, B, 1.0, C);
        break;
    }
    case 100:
        gemm_cublas(M, N, K, alpha, A, B, beta, C);
        break;
    }
}