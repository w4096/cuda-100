#include <cuda_fp16.h>

__global__ void kernel1(const int M, const int N, const int K, const float alpha, const half* A, const half* B,
                        const float beta, float* C) {
    const unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float t = 0.0;
        for (int i = 0; i < K; i++) {
            t += __half2float(A[row * K + i]) * __half2float(B[i * N + col]);
        }
        C[row * N + col] = alpha * t + beta * C[row * N + col];
    }
}

extern "C" void gemm_naive(int M, int N, int K, float alpha, const half* A, const half* B, float beta, float* C) {
    dim3 blockSize(16, 16);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (M + blockSize.y - 1) / blockSize.y);

    kernel1<<<gridSize, blockSize>>>(M, N, K, alpha, A, B, beta, C);
    cudaDeviceSynchronize();
}
