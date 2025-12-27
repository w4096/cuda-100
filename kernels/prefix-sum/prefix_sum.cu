#include <cstdio>
#include <cuda_runtime.h>

template<int ELEMENTS_PER_THREAD>
__global__ void add_prefix_sum_for_blocks(float* output, const float* block_prefix_sums, int N) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x * ELEMENTS_PER_THREAD;
    if (blockIdx.x == 0)
        return; // First block has no offset

    float prefix_sum = block_prefix_sums[blockIdx.x - 1];

#pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        int index = tid + i * blockDim.x;
        if (index < N) {
            output[index] += prefix_sum;
        }
    }
}

template<int BLOCK_SIZE>
__global__ void kogge_stone_scan_double_buffer_kernel(const float* input, float* output, int N,
                                                      float* block_sums = nullptr) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    __shared__ float smem[BLOCK_SIZE * 2];

    smem[threadIdx.x] = idx < N ? input[idx] : 0.0f;
    __syncthreads();

    float* src = smem + BLOCK_SIZE;
    float* dst = smem;

    for (int stride = 1; stride < BLOCK_SIZE; stride *= 2) {
        float* temp = src;
        src = dst;
        dst = temp;

        if (threadIdx.x >= stride) {
            dst[threadIdx.x] = src[threadIdx.x] + src[threadIdx.x - stride];
        } else {
            dst[threadIdx.x] = src[threadIdx.x];
        }
        __syncthreads();
    }

    if (idx < N) {
        output[idx] = dst[threadIdx.x];
    }

    if (block_sums != nullptr && threadIdx.x == BLOCK_SIZE - 1) {
        block_sums[blockIdx.x] = dst[threadIdx.x];
    }
}

template<int BLOCK_SIZE>
__global__ void kogge_stone_scan_kernel(const float* input, float* output, int N, float* block_sums = nullptr) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    __shared__ float smem[BLOCK_SIZE];
    if (idx < N) {
        smem[threadIdx.x] = input[idx];
    } else {
        smem[threadIdx.x] = 0.0f;
    }
    __syncthreads();

    for (int stride = 1; stride < BLOCK_SIZE; stride *= 2) {
        float val = 0;
        if (threadIdx.x >= stride) {
            val = smem[threadIdx.x] + smem[threadIdx.x - stride];
        }
        __syncthreads();
        if (threadIdx.x >= stride) {
            smem[threadIdx.x] = val;
        }
        __syncthreads();
    }

    if (idx < N) {
        output[idx] = smem[threadIdx.x];
    }

    if (block_sums != nullptr && threadIdx.x == BLOCK_SIZE - 1) {
        block_sums[blockIdx.x] = smem[threadIdx.x];
    }
}

template<int BLOCK_SIZE>
__global__ void brent_kung_scan_kernel(const float* input, float* output, int N, float* block_sums = nullptr) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    __shared__ float smem[BLOCK_SIZE];

    smem[threadIdx.x] = tid < N ? input[tid] : 0.0f;
    __syncthreads();

    for (int stride = 1; stride < BLOCK_SIZE; stride *= 2) {
        int idx = (threadIdx.x + 1) * 2 * stride - 1;
        if (idx < BLOCK_SIZE) {
            smem[idx] += smem[idx - stride];
        }
        __syncthreads();
    }

    for (int stride = BLOCK_SIZE / 4; stride >= 1; stride /= 2) {
        int idx = (threadIdx.x + 1) * 2 * stride - 1;
        if (idx + stride < BLOCK_SIZE) {
            smem[idx + stride] += smem[idx];
        }
        __syncthreads();
    }

    if (tid < N) {
        output[tid] = smem[threadIdx.x];
    }

    if (block_sums != nullptr && threadIdx.x == BLOCK_SIZE - 1) {
        block_sums[blockIdx.x] = smem[threadIdx.x];
    }
}

template<int BLOCK_SIZE>
__global__ void brent_kung_scan_optimized_kernel(const float* input, float* output, int N,
                                                 float* block_sums = nullptr) {
    int i = threadIdx.x + blockDim.x * blockIdx.x * 2;
    constexpr int SECTION_SIZE = BLOCK_SIZE * 2;
    __shared__ float smem[SECTION_SIZE];

    if (i < N) {
        smem[threadIdx.x] = input[i];
    } else {
        smem[threadIdx.x] = 0.0f;
    }
    if (i + blockDim.x < N) {
        smem[threadIdx.x + blockDim.x] = input[i + blockDim.x];
    } else {
        smem[threadIdx.x + blockDim.x] = 0.0f;
    }

    __syncthreads();

    for (int stride = 1; stride <= BLOCK_SIZE; stride *= 2) {
        int idx = (threadIdx.x + 1) * 2 * stride - 1;
        if (idx < SECTION_SIZE) {
            smem[idx] += smem[idx - stride];
        }
        __syncthreads();
    }

    for (int stride = SECTION_SIZE / 4; stride > 0; stride /= 2) {
        int idx = (threadIdx.x + 1) * 2 * stride - 1;
        if (idx + stride < SECTION_SIZE) {
            smem[idx + stride] += smem[idx];
        }
        __syncthreads();
    }

    if (i < N) {
        output[i] = smem[threadIdx.x];
    }
    if (i + blockDim.x < N) {
        output[i + blockDim.x] = smem[threadIdx.x + blockDim.x];
    }

    if (block_sums != nullptr && threadIdx.x == 0) {
        block_sums[blockIdx.x] = smem[SECTION_SIZE - 1];
    }
}

template<int THREAD_BLOCK_SIZE, int ELEMENTS_PER_THREAD>
void scan_launch(const float* input, float* output, int N, const void* kernel) {
    dim3 threads = THREAD_BLOCK_SIZE;
    dim3 blocks = (std::max(1, N / ELEMENTS_PER_THREAD) + threads.x - 1) / threads.x;

    float* block_sums = nullptr;
    if (blocks.x > 1) {
        cudaMalloc(&block_sums, blocks.x * sizeof(float));
    }

    void* argv[] = {static_cast<void*>(&input), static_cast<void*>(&output), static_cast<void*>(&N),
                    static_cast<void*>(&block_sums)};
    cudaLaunchKernel(kernel, blocks, threads, argv, 0, nullptr);

    if (blocks.x > 1) {
        float* block_sums_prefix = nullptr;
        cudaMalloc(&block_sums_prefix, blocks.x * sizeof(float));
        scan_launch<THREAD_BLOCK_SIZE, ELEMENTS_PER_THREAD>(block_sums, block_sums_prefix, blocks.x, kernel);
        add_prefix_sum_for_blocks<ELEMENTS_PER_THREAD><<<blocks, threads>>>(output, block_sums_prefix, N);
    }
    cudaDeviceSynchronize();
}

extern "C" {

void kogge_stone_scan(const float* input, float* output, int N) {
    auto kernel = reinterpret_cast<const void*>(kogge_stone_scan_kernel<512>);
    scan_launch<512, 1>(input, output, N, kernel);
}

void kogge_stone_scan_double_buffer(const float* input, float* output, int N) {
    auto kernel = reinterpret_cast<const void*>(kogge_stone_scan_double_buffer_kernel<512>);
    scan_launch<512, 1>(input, output, N, kernel);
}

void brent_kung_scan(const float* input, float* output, int N) {
    auto kernel = reinterpret_cast<const void*>(brent_kung_scan_kernel<512>);
    scan_launch<512, 1>(input, output, N, kernel);
}

void brent_kung_scan_optimized(const float* input, float* output, int N) {
    auto kernel = reinterpret_cast<const void*>(brent_kung_scan_optimized_kernel<512>);
    scan_launch<512, 2>(input, output, N, kernel);
}

} // extern "C"