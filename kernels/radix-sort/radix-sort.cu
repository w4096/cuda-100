#include <cuda_runtime.h>

template<int ELEMENTS_PER_THREAD>
__global__ void radix_sort_1bit_count_kernel(const unsigned int* input, unsigned int* digits, int N,
                                             unsigned int shift) {
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        int idx = threadIdx.x + blockIdx.x * blockDim.x + i * blockDim.x * gridDim.x;
        if (idx < N) {
            unsigned int val = input[idx];
            unsigned int byte = (val >> shift) & 0x1;
            digits[idx] = byte;
        }
    }
}

#include <cstdio>

__global__ void radix_sort_1bit_reorder(const unsigned int* input, unsigned int* output, unsigned int* prefix_sum,
                                        int N, unsigned int shift) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        unsigned int val = input[idx];
        unsigned int bit = (val >> shift) & 0x1;

        int num_ones_before = idx > 0 ? prefix_sum[idx - 1] : 0;
        int num_ones_total = prefix_sum[N - 1];
        int dst = (bit == 0) ? idx - num_ones_before : N - num_ones_total + num_ones_before;
        output[dst] = val;
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
        block_sums[blockIdx.x] = smem[BLOCK_SIZE-1];
    }
}

template<typename T>
__global__ void add_prefix_sum_for_blocks(T* output, const T* block_prefix_sums, int N) {
    if (blockIdx.x == 0) {
        return; // First block has no offset
    }
    T prefix_sum = block_prefix_sums[blockIdx.x - 1];
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (idx < N) {
        output[idx] += prefix_sum;
    }
}


template<typename T, int THREAD_BLOCK_SIZE, int ELEMENTS_PER_THREAD = 1>
void kogge_stone_scan(const T* input, T* output, int N) {
    dim3 threads = THREAD_BLOCK_SIZE;
    dim3 blocks = (std::max(1, N / ELEMENTS_PER_THREAD) + threads.x - 1) / threads.x;

    T* block_sums = nullptr;
    if (blocks.x > 1) {
        cudaMalloc(&block_sums, blocks.x * sizeof(T));
    }

    kogge_stone_scan_kernel<T, THREAD_BLOCK_SIZE><<<blocks, threads>>>(input, output, N, block_sums);

    if (blocks.x > 1) {
        T* block_sums_prefix = nullptr;
        cudaMalloc(&block_sums_prefix, blocks.x * sizeof(T));
        kogge_stone_scan<T, THREAD_BLOCK_SIZE, ELEMENTS_PER_THREAD>(block_sums, block_sums_prefix, blocks.x);
        add_prefix_sum_for_blocks<T><<<blocks, threads>>>(output, block_sums_prefix, N);
        cudaFree(block_sums);
        cudaFree(block_sums_prefix);
    }

    cudaDeviceSynchronize();
}

// input, output are device pointers
extern "C" void solve(const unsigned int* input, unsigned int* output, int N) {
    unsigned int* temp;
    cudaMalloc(&temp, sizeof(unsigned int) * N);
    const unsigned int* data = input;
    unsigned int* out = temp;

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    unsigned int* digits;
    cudaMalloc(&digits, sizeof(unsigned int) * N);

    unsigned int* digits_prefix_sum;
    cudaMalloc(&digits_prefix_sum, sizeof(unsigned int) * N);

    for (int shift = 0; shift < 32; shift += 1) {
        radix_sort_1bit_count_kernel<1><<<blocksPerGrid, threadsPerBlock>>>(data, digits, N, shift);

        kogge_stone_scan<unsigned int, 256>(digits, digits_prefix_sum, N);

        radix_sort_1bit_reorder<<<blocksPerGrid, threadsPerBlock>>>(data, out, digits_prefix_sum, N, shift);

        // Swap input and output pointers
        data = out;
        out = (out == temp) ? output : temp;
    }

    if (data != output) {
        cudaMemcpy(output, data, sizeof(unsigned int) * N, cudaMemcpyDeviceToDevice);
    }
    cudaFree(temp);
    cudaFree(digits);
    cudaFree(digits_prefix_sum);
}

extern "C" void radix_sort_cpu(const unsigned int* input, unsigned int* output, int N) {
    unsigned int* temp = new unsigned int[N];
    unsigned int* data = const_cast<unsigned int*>(input);
    unsigned int* out = temp;

    for (int shift = 0; shift < 32; shift += 8) {
        int count[256] = {0};

        for (int i = 0; i < N; ++i) {
            int byte = (data[i] >> shift) & 0xFF;
            count[byte]++;
        }

        int accu[256];
        accu[0] = 0;
        for (int i = 1; i < 256; ++i) {
            accu[i] = accu[i - 1] + count[i - 1];
        }

        for (int i = 0; i < N; i++) {
            int byte = (data[i] >> shift) & 0xFF;
            out[accu[byte]++] = data[i];
        }

        data = out;
        out = (out == temp) ? output : temp;
    }

    if (data != output) {
        memcpy(output, data, N * sizeof(unsigned int));
    }

    delete[] temp;
}
