#include <cuda_runtime.h>

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

template<int ELEMENTS_PER_THREAD, typename T, typename ReduceOp>
__global__ void reduction_kernel(const T* input, T* output, int N) {
    T val = ReduceOp::default_value();
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        int idx = x + i * blockDim.x * gridDim.x;
        if (idx < N) {
            val = ReduceOp::apply(val, input[idx]);
        }
    }

    __shared__ T smem[32];
    val = block_reduce<T, ReduceOp>(smem, val);
    if (threadIdx.x == 0) {
        atomicAdd(output, val);
    }
}

struct SumOp {
    __device__ __forceinline__ static float apply(double a, float b) {
        return a + b;
    }
    __device__ __forceinline__ static float default_value() {
        return 0.0f;
    }
};

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    dim3 threads = 256;
    constexpr int ELEMENTS_PER_THREAD = 16;
    dim3 blocks = (std::max(N / ELEMENTS_PER_THREAD, 1) + threads.x - 1) / threads.x;

    reduction_kernel<ELEMENTS_PER_THREAD, float, SumOp><<<blocks, threads>>>(input, output, N);
    cudaDeviceSynchronize();
}
