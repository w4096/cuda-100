#include <cuda_runtime.h>

template<typename T>
__device__ __forceinline__ T warp_reduce_sum(T val) {
    unsigned mask = __activemask();
    val += __shfl_xor_sync(mask, val, 16);
    val += __shfl_xor_sync(mask, val, 8);
    val += __shfl_xor_sync(mask, val, 4);
    val += __shfl_xor_sync(mask, val, 2);
    val += __shfl_xor_sync(mask, val, 1);
    return val;
}

template<typename T>
__device__ float block_reduce_sum(T* smem, T val) {
    auto warps = blockDim.x * blockDim.y / 32;
    val = warp_reduce_sum(val);
    if (threadIdx.x % 32 == 0) {
        smem[threadIdx.x / 32] = val;
    }
    __syncthreads();
    if (threadIdx.x < 32) {
        if (threadIdx.x < warps) {
            val = smem[threadIdx.x];
        } else {
            val = 0;
        }
        __syncwarp();
        val = warp_reduce_sum(val);
        smem[0] = val;
    }
    __syncthreads();
    return smem[0];
}

template<int ELEMENTS_PER_THREAD>
__global__ void count_equal_kernel(const int* input, int* output, int N, int K) {
    int x = threadIdx.x + blockDim.x * blockIdx.x;

    int equals = 0;
#pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        int idx = x * ELEMENTS_PER_THREAD + i;
        if (idx < N && input[idx] == K) {
            equals++;
        }
    }

    constexpr int MAX_WARPS_PER_BLOCK = 32;
    __shared__ int count[MAX_WARPS_PER_BLOCK];
    equals = block_reduce_sum<int>(count, equals);

    if (threadIdx.x == 0) {
        atomicAdd(output, equals);
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const int* input, int* output, int N, int K) {
    int threadsPerBlock = 256;
    constexpr int ELEMENTS_PER_THREAD = 16;
    int blocksPerGrid = (std::max(1, N / ELEMENTS_PER_THREAD) + threadsPerBlock - 1) / threadsPerBlock;

    count_equal_kernel<ELEMENTS_PER_THREAD><<<blocksPerGrid, threadsPerBlock>>>(input, output, N, K);
    cudaDeviceSynchronize();
}