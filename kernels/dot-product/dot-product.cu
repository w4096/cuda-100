#include <cuda_runtime.h>

__global__ void dot_product_vectorize_warp_shuffle_kernel(const float* A, const float* B, float* result, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    float sum = 0;
    if (tid < N / 4) {
        float4 a = reinterpret_cast<const float4*>(A)[tid];
        float4 b = reinterpret_cast<const float4*>(B)[tid];
        sum = a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
    }
    if (tid < N % 4) {
        int idx = (N / 4) * 4 + tid;
        sum += A[idx] * B[idx];
    }

    sum += __shfl_down_sync(0xFFFFFFFF, sum, 16);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 8);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 4);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 2);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 1);

    if (threadIdx.x % 32 == 0) {
        atomicAdd(result, sum);
    }
}

__global__ void dot_product_naive_kernel(const float* A, const float* B, float* result, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < N) {
        float sum = A[tid] * B[tid];
        atomicAdd(result, sum);
    }
}

static int cdiv(int a, int b) {
    return (a + b - 1) / b;
}

extern "C" {

void solve_naive(const float* A, const float* B, float* result, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    dot_product_naive_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, result, N);
    cudaDeviceSynchronize();
}


void solve_vectorize_warp_shuffle(const float* A, const float* B, float* result, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (cdiv(N, 4) + threadsPerBlock - 1) / threadsPerBlock;

    dot_product_vectorize_warp_shuffle_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, result, N);
    cudaDeviceSynchronize();
}

}
