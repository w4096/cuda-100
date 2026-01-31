#include "utils/utils.h"
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <string>
#include <torch/torch.h>

__global__ void reduce_naive(const int* input, int n, int* output) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        atomicAdd(output, input[idx]);
    }
}

__global__ void reduce_with_shared_memory(const int* input, int n, int* output) {
    __shared__ int sdata[1];
    if (threadIdx.x == 0) {
        sdata[0] = 0;
    }
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        atomicAdd(&sdata[0], input[idx]);
    }

    __syncthreads();
    if (threadIdx.x == 0) {
        atomicAdd(output, sdata[0]);
    }
}

__global__ void reduce_0(const int* input, int n, int* output) {
    extern __shared__ int sdata[];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    // Load data into shared memory
    sdata[tid] = idx < n ? input[idx] : 0;
    __syncthreads();

    // Tree-based reduction in shared memory
    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2 * s) == 0) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write the result for this block to global memory
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

__global__ void reduce_1(const int* input, int n, int* output) {
    extern __shared__ int sdata[];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    // Load data into shared memory
    sdata[tid] = idx < n ? input[idx] : 0;
    __syncthreads();

    // Tree-based reduction in shared memory
    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        int index = 2 * s * tid;
        if (index < blockDim.x) {
            sdata[index] += sdata[index + s];
        }
        __syncthreads();
    }

    // Write the result for this block to global memory
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

__global__ void reduce_2(const int* input, int n, int* output) {
    extern __shared__ int sdata[];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    // Load data into shared memory
    sdata[tid] = idx < n ? input[idx] : 0;
    __syncthreads();

    // Tree-based reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write the result for this block to global memory
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

__global__ void reduce_3(const int* input, int n, int* output) {
    extern __shared__ int sdata[];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    // Load data into shared memory
    if (idx < n) {
        sdata[tid] = input[idx];
    }
    if (idx + blockDim.x * gridDim.x < n) {
        sdata[tid] += input[idx + blockDim.x * gridDim.x];
    }
    __syncthreads();

    // Tree-based reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write the result for this block to global memory
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

__global__ void reduce_4(const int* input, int n, int* output) {
    extern __shared__ int sdata[];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    // Load data into shared memory
    if (idx < n) {
        sdata[tid] = input[idx];
    }
    if (idx + blockDim.x * gridDim.x < n) {
        sdata[tid] += input[idx + blockDim.x * gridDim.x];
    }
    __syncthreads();

    // Tree-based reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s >= 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid < 32) {
        int sum = sdata[tid];
#pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }

        // Write the result for this block to global memory
        if (tid == 0) {
            output[blockIdx.x] = sum;
        }
    }
}

__global__ void reduce_4_1(const int* input, int n, int* output) {
    extern __shared__ int sdata[];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    // Load data into shared memory
    if (idx < n) {
        sdata[tid] = input[idx];
    }
    if (idx + blockDim.x * gridDim.x < n) {
        sdata[tid] += input[idx + blockDim.x * gridDim.x];
    }
    __syncthreads();

    // Tree-based reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s >= 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid < 32) {
        // use warp reduce intrinsic
        sdata[tid] = __reduce_add_sync(0xffffffff, sdata[tid]);

        __syncthreads();

        // Write the result for this block to global memory
        if (tid == 0) {
            output[blockIdx.x] = sdata[0];
        }
    }
}

template<int Chunks>
__global__ void reduce_5(const int* input, int n, int* output) {
    extern __shared__ int sdata[];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    // Load data into shared memory
    sdata[tid] = 0;
#pragma unroll
    for (int i = 0; i < Chunks && idx < n; i++) {
        sdata[tid] += input[idx];
        idx += blockDim.x * gridDim.x;
    }
    __syncthreads();

    // Tree-based reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s >= 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid < 32) {
        // use warp reduce intrinsic
        int sum = __reduce_add_sync(0xffffffff, sdata[tid]);

        // Write the result for this block to global memory
        if (tid == 0) {
            output[blockIdx.x] = sum;
        }
    }
}

template<int kernel>
int reduce(const int* nums, int n) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    if (kernel == 3 || kernel == 4) {
        blocks = (blocks + 1) / 2;
    } else if (kernel == 5) {
        blocks = (blocks + 3) / 4;
    }

    int* partial_sum;
    cudaMalloc(reinterpret_cast<void**>(&partial_sum), sizeof(int) * blocks);

    if (kernel == 0) {
        reduce_0<<<blocks, threads, threads * sizeof(int)>>>(nums, n, partial_sum);
    } else if (kernel == 1) {
        reduce_1<<<blocks, threads, threads * sizeof(int)>>>(nums, n, partial_sum);
    } else if (kernel == 2) {
        reduce_2<<<blocks, threads, threads * sizeof(int)>>>(nums, n, partial_sum);
    } else if (kernel == 3) {
        reduce_3<<<blocks, threads, threads * sizeof(int)>>>(nums, n, partial_sum);
    } else if (kernel == 4) {
        reduce_4<<<blocks, threads, threads * sizeof(int)>>>(nums, n, partial_sum);
    } else if (kernel == 5) {
        reduce_5<4><<<blocks, threads, threads * sizeof(int)>>>(nums, n, partial_sum);
    }
    cudaDeviceSynchronize();
    CHECK_CUDA_ERR(cudaGetLastError());

    if (blocks != 1) {
        int* input = partial_sum;
        cudaMalloc(&partial_sum, sizeof(int));
        int zero = 0;
        cudaMemcpy(partial_sum, &zero, sizeof(int), cudaMemcpyHostToDevice);
        n = blocks;
        blocks = (n + threads - 1) / threads;
        reduce_with_shared_memory<<<blocks, threads>>>(input, n, partial_sum);
        cudaDeviceSynchronize();
        CHECK_CUDA_ERR(cudaGetLastError());
        cudaFree(input);
    }

    int sum;
    cudaMemcpy(&sum, partial_sum, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(partial_sum);
    return sum;
}

int main() {
    int N = 1024 * 1024 * 400;

    int sum = 0;
    auto nums = torch::randint(2, N, torch::kInt).cuda();
    {
        Timer timer("torch::sum");

        sum = torch::sum(nums).item().toInt();
        std::cout << sum << std::endl;
    }

    {
        Timer timer("reduce_naive");

        int threads = 256;
        int blocks = (N + threads - 1) / threads;

        int h_sum = 0;
        int* d_sum;
        cudaMalloc(reinterpret_cast<void**>(&d_sum), sizeof(int));
        cudaMemcpy(d_sum, &h_sum, sizeof(int), cudaMemcpyHostToDevice);
        reduce_naive<<<blocks, threads>>>(nums.data_ptr<int>(), N, d_sum);
        CHECK_CUDA_ERR(cudaGetLastError());
        cudaDeviceSynchronize();

        // Check result
        cudaMemcpy(&h_sum, d_sum, sizeof(int), cudaMemcpyDeviceToHost);
        ASSERT(h_sum == sum);
    }

    {
        Timer timer("reduce_with_shared_memory");

        int threads = 256;
        int blocks = (N + threads - 1) / threads;

        int h_sum = 0;
        int* d_sum;
        cudaMalloc(reinterpret_cast<void**>(&d_sum), sizeof(int));
        cudaMemcpy(d_sum, &h_sum, sizeof(int), cudaMemcpyHostToDevice);
        reduce_with_shared_memory<<<blocks, threads>>>(nums.data_ptr<int>(), N, d_sum);
        CHECK_CUDA_ERR(cudaGetLastError());
        cudaDeviceSynchronize();

        // Check result
        cudaMemcpy(&h_sum, d_sum, sizeof(int), cudaMemcpyDeviceToHost);
        ASSERT(h_sum == sum);
    }

    {
        Timer timer("reduce_0");
        int s = reduce<0>(nums.data_ptr<int>(), nums.size(0));
        ASSERT(s == sum);
    }

    {
        Timer timer("reduce_1");
        int s = reduce<1>(nums.data_ptr<int>(), nums.size(0));
        ASSERT(s == sum);
    }

    {
        Timer timer("reduce_2");
        int s = reduce<2>(nums.data_ptr<int>(), nums.size(0));
        ASSERT(s == sum);
    }

    {
        Timer timer("reduce_3");
        int s = reduce<3>(nums.data_ptr<int>(), nums.size(0));
        ASSERT(s == sum);
    }

    {
        Timer timer("reduce_4");
        int s = reduce<4>(nums.data_ptr<int>(), nums.size(0));
        ASSERT(s == sum);
    }

    {
        Timer timer("reduce_5");
        int s = reduce<5>(nums.data_ptr<int>(), nums.size(0));
        ASSERT(s == sum);
    }

    return 0;
}
