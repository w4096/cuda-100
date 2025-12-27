#include <cuda_runtime.h>

__global__ void silu_kernel(const float* input, float* output, int N) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    int N_vec = N / 4;
    if (idx < N_vec) {
        float4 val = reinterpret_cast<const float4*>(input)[idx];
        val.x = val.x / (1.0f + expf(-val.x));
        val.y = val.y / (1.0f + expf(-val.y));
        val.z = val.z / (1.0f + expf(-val.z));
        val.w = val.w / (1.0f + expf(-val.w));
        reinterpret_cast<float4*>(output)[idx] = val;
    }
    if (blockIdx.x == 0 && threadIdx.x < (N % 4)) {
        idx = N / 4 * 4 + threadIdx.x;
        float val = input[idx];
        output[idx] = val / (1.0f + expf(-val));
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (std::max(N / 4, 1) + threadsPerBlock - 1) / threadsPerBlock;

    silu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize();
}
