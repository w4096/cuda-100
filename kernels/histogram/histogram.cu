#include <cuda_runtime.h>

__global__ void histogram_naive_kernel(const int* __restrict__ input, int* histogram, int N, int num_bins) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < N) {
        int bin = input[idx];
        if (bin >= 0 && bin < num_bins) {
            atomicAdd(&histogram[bin], 1);
        }
    }
}

template <int ELEMENTS_PER_THREAD>
__global__ void histogram_shared_memory_kernel(const int* __restrict__ input, int* histogram, int N, int num_bins) {
    extern __shared__ int hist[];
    for (int i = threadIdx.x; i < num_bins; i += blockDim.x) {
        hist[i] = 0;
    }
    __syncthreads();

    int prev_bin_idx = -1;
    int accumulator = 0;
#pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; ++i) {
        int idx = (threadIdx.x + blockIdx.x * blockDim.x) + i * blockDim.x * gridDim.x;
        if (idx < N) {
            int bin = input[idx];
            if (bin == prev_bin_idx) {
                accumulator++;
            } else {
                if (accumulator > 0) {
                    atomicAdd(&hist[prev_bin_idx], accumulator);
                }
                accumulator = 1;
                prev_bin_idx = bin;
            }
        }
    }
    if (accumulator > 0) {
        atomicAdd(&hist[prev_bin_idx], accumulator);
    }

    __syncthreads();
    for (int i = threadIdx.x; i < num_bins; i += blockDim.x) {
        atomicAdd(&histogram[i], hist[i]);
    }
}


// input, histogram are device pointers
extern "C" void solve_naive(const int* input, int* histogram, int N, int num_bins) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    histogram_naive_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, histogram, N, num_bins);
    cudaDeviceSynchronize();
}


// input, histogram are device pointers
extern "C" void solve(const int* input, int* histogram, int N, int num_bins) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    constexpr int ELEMENTS_PER_THREAD = 4;
    blocksPerGrid = (blocksPerGrid + ELEMENTS_PER_THREAD - 1) / ELEMENTS_PER_THREAD;

    int smem = num_bins * sizeof(int);

    histogram_shared_memory_kernel<ELEMENTS_PER_THREAD><<<blocksPerGrid, threadsPerBlock, smem>>>(input, histogram, N, num_bins);
    cudaDeviceSynchronize();
}
