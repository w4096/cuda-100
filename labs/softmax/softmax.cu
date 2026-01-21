#include <cuda/std/numeric>

__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (size_t mask = 16; mask > 0; mask /= 2) {
        val = fmax(__shfl_xor_sync(0xffffffff, val, mask), val);
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
#pragma unroll
    for (size_t mask = 16; mask > 0; mask /= 2) {
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}

__device__  float shared_mem_reduce_max(float *smem, float val) {
    auto warps = blockDim.x * blockDim.y / 32;

    val = warp_reduce_max(val);

    if (threadIdx.x % 32 == 0) {
        smem[threadIdx.x / 32] = val;
    }
    __syncthreads();

    if (threadIdx.x < 32) {
        if (threadIdx.x < warps) {
            val = smem[threadIdx.x];
        } else {
            val = cuda::std::numeric_limits<float>::min();
        }
        __syncwarp();
        val = warp_reduce_max(val);
        smem[0] = val;
    }
    __syncthreads();
    return smem[0];
}

__device__  float shared_mem_reduce_sum(float *smem, float val) {
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


__global__ void softmax_warp_reduce(const float * __restrict__ in, float *out, int rows, int dim) {
    auto threads = gridDim.x * blockDim.x;
    auto warps =  threads / 32;
    auto global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    auto warp_idx = global_thread_idx / 32;
    auto lane = global_thread_idx % 32;

    for (unsigned int row = warp_idx; row < rows; row += warps) {
        const float *x = &in[row * dim];
        float *o = &out[row * dim];

        float max = cuda::std::numeric_limits<float>::min();
        for (auto col = lane; col < dim; col += 32) {
            max = fmax(x[col], max);
        }
        max = warp_reduce_max(max);

        float sum = 0;
        for (auto col = lane; col < dim; col += 32) {
            sum += __expf(x[col] - max);
        }
        sum = warp_reduce_sum(sum);

        for (auto col = lane; col < dim; col += 32) {
            o[col] = __expf(x[col] - max) / sum;
        }
    }
}

__global__ void online_softmax_warp_reduce(const float * __restrict__ in, float *out, int rows, int dim) {
    auto threads = gridDim.x * blockDim.x;
    auto warps =  threads / 32;
    auto global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    auto warp_idx = global_thread_idx / 32;
    auto lane = global_thread_idx % 32;

    for (unsigned int row = warp_idx; row < rows; row += warps) {
        const float *x = &in[row * dim];
        float *o = &out[row * dim];

        float sum = 0;
        float max = cuda::std::numeric_limits<float>::min();
        for (auto col = lane; col < dim; col += 32) {
            float m = fmax(x[col], max);
            if (col > lane && m > max) {
                sum = sum * __expf(max - m);
            }
            max = m;
            sum += __expf(x[col] - max);
        }
        float m = warp_reduce_max(max);
        if (m > max) {
            sum = sum * __expf(max - m);
            max = m;
        }
        sum = warp_reduce_sum(sum);

        for (auto col = lane; col < dim; col += 32) {
            o[col] = __expf(x[col] - max) / sum;
        }
    }
}

__global__ void softmax_block_reduce(const float * __restrict__ in, float *out, int rows, int dim) {
    __shared__ float smem[32];

    for (unsigned int row = blockIdx.x; row < rows; row += gridDim.x) {
        const float *x = &in[row * dim];
        float *o = &out[row * dim];

        float max = cuda::std::numeric_limits<float>::min();
        for (auto col = threadIdx.x; col < dim; col += blockDim.x) {
            max = fmax(x[col], max);
        }
        max = shared_mem_reduce_max(smem, max);

        float sum = 0;
        for (auto col = threadIdx.x; col < dim; col += blockDim.x) {
            sum += __expf(x[col] - max);
        }
        sum = shared_mem_reduce_sum(smem, sum);

        for (auto col = threadIdx.x; col < dim; col += blockDim.x) {
            o[col] = __expf(x[col] - max) / sum;
        }
    }
}

__global__ void online_softmax_block_reduce(const float * __restrict__ in, float *out, int rows, int dim) {
    __shared__ float smem[32];

    for (unsigned int row = blockIdx.x; row < rows; row += gridDim.x) {
        const float *x = &in[row * dim];
        float *o = &out[row * dim];

        float sum = 0;
        float max = cuda::std::numeric_limits<float>::min();
        for (auto col = threadIdx.x; col < dim; col += blockDim.x) {
            float m = fmax(x[col], max);
            if (col > threadIdx.x && m > max) {
                sum = sum * __expf(max - m);
            }
            max = m;
            sum += __expf(x[col] - max);
        }

        float m = shared_mem_reduce_max(smem, max);
        if (m > max) {
            sum = sum * __expf(max - m);
            max = m;
        }
        sum = shared_mem_reduce_sum(smem, sum);

        for (auto col = threadIdx.x; col < dim; col += blockDim.x) {
            o[col] = __expf(x[col] - max) / sum;
        }
    }
}

