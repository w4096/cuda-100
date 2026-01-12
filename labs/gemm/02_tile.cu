#include <cuda_fp16.h>

template<int TILE_DIM = 16>
__global__ void kernel2(const int M, const int N, const int K, const float alpha, const half* A,
                                const half* B, const float beta, float* C) {
    const unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ half As[TILE_DIM][TILE_DIM];
    __shared__ half Bs[TILE_DIM][TILE_DIM];

    __syncthreads();

    float accum = 0.0;
    for (int i = 0; i < K; i += TILE_DIM) {
        int ax = i + threadIdx.x;
        int ay = row;

        if (ay < M && ax < K) {
            As[threadIdx.y][threadIdx.x] = A[ay * K + ax];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0;
        }

        int bx = col;
        int by = i + threadIdx.y;
        if (by < K && bx < N) {
            Bs[threadIdx.y][threadIdx.x] = B[by * N + bx];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0;
        }
        __syncthreads();


        for (int k = 0; k < TILE_DIM; k++) {
            accum += __half2float(As[threadIdx.y][k]) * __half2float(Bs[k][threadIdx.x]);
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = alpha * accum + beta * C[row * N + col];
    }
}

extern "C"
void gemm_tile_16x16(int M, int N, int K, float alpha, const half* A, const half* B, float beta, float* C) {
    const dim3 blockDim(16, 16);
    const dim3 gridDim((N + blockDim.x - 1) / blockDim.x,
                       (M + blockDim.y - 1) / blockDim.y);

    kernel2<16><<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

extern "C"
void gemm_tile_32x32(int M, int N, int K, float alpha, const half* A, const half* B, float beta, float* C) {
    const dim3 blockDim(32, 32);
    const dim3 gridDim((N + blockDim.x - 1) / blockDim.x,
                       (M + blockDim.y - 1) / blockDim.y);

    kernel2<32><<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}
