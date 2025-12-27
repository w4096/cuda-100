#include <cuda_runtime.h>

__device__ __forceinline__ void relu_device(const float4* input, float4* output, int N) {
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    if (x < N) {
        float4 a = input[x];
        a.x = a.x < 0 ? 0 : a.x;
        a.y = a.y < 0 ? 0 : a.y;
        a.z = a.z < 0 ? 0 : a.z;
        a.w = a.w < 0 ? 0 : a.w;
        output[x] = a;
    }
}

__global__ void relu_kernel(const float* input, float* output, int N) {
    int x = threadIdx.x + blockDim.x * blockIdx.x;

    relu_device(reinterpret_cast<const float4*>(input), reinterpret_cast<float4*>(output), N / 4);

    if (x < N % 4) {
        float val = input[N - (N % 4) + x];
        output[N - (N % 4) + x] = val < 0 ? 0 : val;
    }
}

__global__ void relu_kernel_0(const float* input, float* output, int N) {
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    if (x < N) {
        output[x] = max(input[x], 0.0f);
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N / 4 + threadsPerBlock - 1) / threadsPerBlock;

    relu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize();
}

extern "C" void solve1(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    relu_kernel_0<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize();
}
