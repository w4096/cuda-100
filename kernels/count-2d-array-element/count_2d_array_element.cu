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
__device__ __forceinline__ float block_reduce_sum(T* smem, T val) {
    auto tid = threadIdx.y * blockDim.x + threadIdx.x;
    auto warps = blockDim.x * blockDim.y / 32;
    auto lane = tid % 32;
    auto warp_id = tid / 32;

    val = warp_reduce_sum(val);
    if (lane == 0) {
        smem[warp_id] = val;
    }
    __syncthreads();
    if (tid < 32) {
        if (tid < warps) {
            val = smem[tid];
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

template<int ROWS_PER_THREAD, int COLS_PER_THREAD>
__global__ void count_2d_equal_kernel(const int* input, int* output, int N, int M, int K) {
    int row = threadIdx.y + blockDim.y * blockIdx.y;
    int col = threadIdx.x + blockDim.x * blockIdx.x;

    int equals = 0;
#pragma unroll
    for (int i = 0; i < ROWS_PER_THREAD; i++) {
#pragma unroll
        for (int j = 0; j < COLS_PER_THREAD; j++) {
            int r = row + i * blockDim.y * gridDim.y;
            int c = col + j * blockDim.x * gridDim.x;
            if (r < N && c < M) {
                if (input[r * M + c] == K) {
                    equals++;
                }
            }
        }
    }

    __shared__ int smem[32];
    equals = block_reduce_sum<int>(smem, equals);

    if (threadIdx.x == 0 && threadIdx.y == 0) {
        atomicAdd(output, equals);
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const int* input, int* output, int N, int M, int K) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((M + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    blocksPerGrid.x = std::max<int>(1, (blocksPerGrid.x + 3) / 4);
    blocksPerGrid.y = std::max<int>(1, (blocksPerGrid.y + 3) / 4);

    count_2d_equal_kernel<4, 4><<<blocksPerGrid, threadsPerBlock>>>(input, output, N, M, K);
    cudaDeviceSynchronize();
}