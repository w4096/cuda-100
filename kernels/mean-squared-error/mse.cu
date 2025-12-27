#include <cuda_runtime.h>

__global__ void mean_squared_error_kernel(const float* predictions, const float* targets, float* mse, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    float sum = 0.0f;
    for (int i = 0; i < N; i += blockDim.x * gridDim.x) {
        int idx = tid + i;
        if (idx < N) {
            float diff = predictions[idx] - targets[idx];
            sum += diff * diff;
        }
    }

    // Reduce within warp
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    }

    // Write the result of each warp to shared memory
    __shared__ float smem[32]; // assuming max 1024 threads per block
    if (threadIdx.x % warpSize == 0) {
        smem[threadIdx.x / warpSize] = sum;
    }
    __syncthreads();

    // Reduce within block
    if (threadIdx.x < blockDim.x / warpSize) {
        sum = smem[threadIdx.x];
        unsigned mask = (1U << (blockDim.x / warpSize)) - 1;
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(mask, sum, offset);
        }
        if (threadIdx.x == 0) {
            atomicAdd(mse, sum / N);
        }
    }
}


// predictions, targets, mse are device pointers
extern "C" void solve(const float* predictions, const float* targets, float* mse, int N) {
    dim3 threadsPerBlock = 256;
    dim3 blocksPerGrid = (N + threadsPerBlock.x - 1) / threadsPerBlock.x;
    blocksPerGrid.x = max(blocksPerGrid.x / 4, 1);

    mean_squared_error_kernel<<<blocksPerGrid, threadsPerBlock>>>(predictions, targets, mse, N);
    cudaDeviceSynchronize();
}
