#include <cuda_runtime.h>

__device__ float warp_reduce_sum(float val) {
    unsigned int mask = __activemask();
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(mask, val, offset);
    }
    return val;
}

__device__ float block_reduce_sum(float &val) {
    __shared__ float smem[32]; // assuming max 1024 threads per block
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;

    val = warp_reduce_sum(val);
    if (lane == 0) {
        smem[wid] = val;
    }
    __syncthreads();

    if (wid == 0) {
        val = lane < (blockDim.x / warpSize) ? smem[lane] : 0.0f;
        val = warp_reduce_sum(val);
        if (lane == 0) {
            smem[0] = val;
        }
    }
    __syncthreads();
    return smem[0];
}

__global__ void rms_kernel(const float* input, float gamma, float beta, float* output, int N, float eps) {
    int tid = threadIdx.x;

    float squared_sum = 0.0f;
    int stride = blockDim.x * gridDim.x;
    for (int i = tid; i < N; i += stride) {
        squared_sum += input[i] * input[i];
    }

    squared_sum = block_reduce_sum(squared_sum);
    float rms = sqrtf(squared_sum / N + eps);

    for (int i = tid; i < N; i += stride) {
        output[i] = gamma * (input[i] / rms) + beta;
    }
}

extern "C" void solve(const float* input, float gamma, float beta, float* output, int N, float eps) {
    int threadsPerBlock = 512;
    rms_kernel<<<1, threadsPerBlock>>>(input, gamma, beta, output, N, eps);
    cudaDeviceSynchronize();
}
