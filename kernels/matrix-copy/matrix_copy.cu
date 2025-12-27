#include <cuda_runtime.h>

__global__ void copy_matrix_kernel(const float* __restrict__ A, float* __restrict__ B, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N / 4) {
        reinterpret_cast<float4*>(B)[idx] = reinterpret_cast<const float4*>(A)[idx];
    }
    if (blockIdx.x == 0 && threadIdx.x < N % 4) {
        int offset = N / 4 * 4;
        B[offset + threadIdx.x] = A[offset + threadIdx.x];
    }
}

// A, B are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, float* B, int N) {
    int total = N * N;
    int threadsPerBlock = 256;
    int blocksPerGrid = (std::max(total / 4, 1) + threadsPerBlock - 1) / threadsPerBlock;
    copy_matrix_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, total);
    cudaDeviceSynchronize();
}