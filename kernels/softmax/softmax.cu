#include <cuda/std/numeric>
#include <cuda_runtime.h>
#include <stdio.h>

template<typename T, typename ReduceOp>
__device__ __forceinline__ T warp_reduce(T val) {
    unsigned mask = __activemask();
    val = ReduceOp::apply(__shfl_xor_sync(mask, val, 16), val);
    val = ReduceOp::apply(__shfl_xor_sync(mask, val, 8), val);
    val = ReduceOp::apply(__shfl_xor_sync(mask, val, 4), val);
    val = ReduceOp::apply(__shfl_xor_sync(mask, val, 2), val);
    val = ReduceOp::apply(__shfl_xor_sync(mask, val, 1), val);
    return val;
}

template<typename T, typename ReduceOp>
__device__ __forceinline__ float block_reduce(T* smem, T val) {
    auto tid = threadIdx.y * blockDim.x + threadIdx.x;
    auto warps = blockDim.x * blockDim.y / 32;
    auto lane = tid % 32;
    auto warp_id = tid / 32;

    val = warp_reduce<T, ReduceOp>(val);
    if (lane == 0) {
        smem[warp_id] = val;
    }
    __syncthreads();
    if (tid < 32) {
        if (tid < warps) {
            val = smem[tid];
        } else {
            val = ReduceOp::default_value();
        }
        __syncwarp();
        val = warp_reduce<T, ReduceOp>(val);
        smem[0] = val;
    }
    __syncthreads();
    return smem[0];
}

template<typename T>
struct MaxOp {
    __device__ __forceinline__ static T apply(T a, T b) {
        return fmax(a, b);
    }
    __device__ __forceinline__ static T default_value() {
        return cuda::std::numeric_limits<T>::min();
    }
};

template<typename T>
struct SumOp {
    __device__ __forceinline__ static T apply(T a, T b) {
        return a + b;
    }
    __device__ __forceinline__ static T default_value() {
        return T(0);
    }
};

template<int ELEMENTS_PER_THREAD>
__global__ void softmax_kernel(const float* input, float* output, int N) {
    int x = threadIdx.x + blockDim.x * blockIdx.x;

    float max = MaxOp<float>::default_value();
    float expsum = 0.0f;
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        int idx = x + i * blockDim.x * gridDim.x;
        if (idx < N) {
            if (input[idx] > max) {
                expsum = expsum * expf(max - input[idx]);
                max = input[idx];
            }
            expsum += __expf(input[idx] - max);
        }
    }

    __shared__ float smem[32];
    float m = block_reduce<float, MaxOp<float>>(smem, max);
    if (m > max) {
        expsum = expsum * __expf(max - m);
        max = m;
    }

    expsum = block_reduce<float, SumOp<float>>(smem, expsum);

    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        int idx = x + i * blockDim.x * gridDim.x;
        if (idx < N) {
            output[idx] = __expf(input[idx] - max) / expsum;
        }
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    constexpr int ELEMENTS_PER_THREAD = 16;
    int blocksPerGrid = (std::max(1, N / ELEMENTS_PER_THREAD) + threadsPerBlock - 1) / threadsPerBlock;

    softmax_kernel<ELEMENTS_PER_THREAD><<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize();
}
