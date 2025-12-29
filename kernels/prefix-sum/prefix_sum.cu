#include <cstdio>
#include <cuda_runtime.h>

template<typename T, int ELEMENTS_PER_THREAD>
__global__ void add_prefix_sum_for_blocks(T* output, const T* block_prefix_sums, int N) {
    if (blockIdx.x == 0) {
        return; // First block has no offset
    }
    T prefix_sum = block_prefix_sums[blockIdx.x - 1];

    int tid = threadIdx.x + blockDim.x * blockIdx.x * ELEMENTS_PER_THREAD;

#pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        int index = tid + i * blockDim.x;
        if (index < N) {
            output[index] += prefix_sum;
        }
    }
}

template<typename T, int BLOCK_SIZE>
__global__ void kogge_stone_scan_double_buffer_kernel(const T* input, T* output, int N, T* block_sums = nullptr) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    __shared__ T smem[BLOCK_SIZE * 2];

    smem[threadIdx.x] = idx < N ? input[idx] : T(0);
    __syncthreads();

    T* src = smem + BLOCK_SIZE;
    T* dst = smem;

    for (int stride = 1; stride < BLOCK_SIZE; stride *= 2) {
        T* temp = src;
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

template<typename T, int BLOCK_SIZE>
__global__ void kogge_stone_scan_kernel(const T* input, T* output, int N, T* block_sums = nullptr) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    __shared__ T smem[BLOCK_SIZE];
    if (idx < N) {
        smem[threadIdx.x] = input[idx];
    } else {
        smem[threadIdx.x] = T(0);
    }
    __syncthreads();

    for (int stride = 1; stride < BLOCK_SIZE; stride *= 2) {
        T val = 0;
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
        block_sums[blockIdx.x] = smem[BLOCK_SIZE - 1];
    }
}

template<typename T, int BLOCK_SIZE>
__global__ void brent_kung_scan_kernel(const T* input, T* output, int N, T* block_sums = nullptr) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    __shared__ T smem[BLOCK_SIZE];

    smem[threadIdx.x] = tid < N ? input[tid] : T(0);
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

template<typename T, int BLOCK_SIZE>
__global__ void brent_kung_scan_optimized_kernel(const T* input, T* output, int N, T* block_sums = nullptr) {
    int i = threadIdx.x + blockDim.x * blockIdx.x * 2;
    constexpr int SECTION_SIZE = BLOCK_SIZE * 2;
    __shared__ T smem[SECTION_SIZE];

    if (i < N) {
        smem[threadIdx.x] = input[i];
    } else {
        smem[threadIdx.x] = T(0);
    }
    if (i + blockDim.x < N) {
        smem[threadIdx.x + blockDim.x] = input[i + blockDim.x];
    } else {
        smem[threadIdx.x + blockDim.x] = T(0);
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

static constexpr int KERNEL_KOGGE_STONE = 0;
static constexpr int KERNEL_KOGGE_STONE_DOUBLE_BUFFER = 1;
static constexpr int KERNEL_BRENT_KUNG = 2;
static constexpr int KERNEL_BRENT_KUNG_OPTIMIZED = 3;

template<typename T, int THREAD_BLOCK_SIZE, int ELEMENTS_PER_THREAD, int kernel>
void launch_scan_kernel(const T* input, T* output, int N) {
    dim3 threads = THREAD_BLOCK_SIZE;
    dim3 blocks = (std::max(1, N / ELEMENTS_PER_THREAD) + threads.x - 1) / threads.x;

    T* block_sums = nullptr;
    if (blocks.x > 1) {
        cudaMalloc(&block_sums, blocks.x * sizeof(T));
    }

    if constexpr (kernel == KERNEL_KOGGE_STONE) {
        kogge_stone_scan_kernel<T, THREAD_BLOCK_SIZE><<<blocks, threads>>>(input, output, N, block_sums);
    } else if constexpr (kernel == KERNEL_KOGGE_STONE_DOUBLE_BUFFER) {
        kogge_stone_scan_double_buffer_kernel<T, THREAD_BLOCK_SIZE><<<blocks, threads>>>(input, output, N, block_sums);
    } else if constexpr (kernel == KERNEL_BRENT_KUNG) {
        brent_kung_scan_kernel<T, THREAD_BLOCK_SIZE><<<blocks, threads>>>(input, output, N, block_sums);
    } else if constexpr (kernel == KERNEL_BRENT_KUNG_OPTIMIZED) {
        brent_kung_scan_optimized_kernel<T, THREAD_BLOCK_SIZE><<<blocks, threads>>>(input, output, N, block_sums);
    } else {
        static_assert(kernel >= 0 && kernel <= 3, "Invalid kernel type");
    }

    if (blocks.x > 1) {
        T* block_sums_prefix = nullptr;
        cudaMalloc(&block_sums_prefix, blocks.x * sizeof(T));
        launch_scan_kernel<T, THREAD_BLOCK_SIZE, ELEMENTS_PER_THREAD, kernel>(block_sums, block_sums_prefix, blocks.x);
        add_prefix_sum_for_blocks<T, ELEMENTS_PER_THREAD><<<blocks, threads>>>(output, block_sums_prefix, N);
        cudaFree(block_sums);
        cudaFree(block_sums_prefix);
    }
    cudaDeviceSynchronize();
}

extern "C" {

void kogge_stone_scan(const float* input, float* output, int N) {
    launch_scan_kernel<float, 512, 1, KERNEL_KOGGE_STONE>(input, output, N);
}

void kogge_stone_scan_double_buffer(const float* input, float* output, int N) {
    launch_scan_kernel<float, 512, 1, KERNEL_KOGGE_STONE_DOUBLE_BUFFER>(input, output, N);
}

void brent_kung_scan(const float* input, float* output, int N) {
    launch_scan_kernel<float, 512, 1, KERNEL_BRENT_KUNG>(input, output, N);
}

void brent_kung_scan_optimized(const float* input, float* output, int N) {
    launch_scan_kernel<float, 512, 2, KERNEL_BRENT_KUNG_OPTIMIZED>(input, output, N);
}

} // extern "C"