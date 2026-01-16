#include "utils/utils.h"
#include <iostream>
#include <random>
#include <string>

template<typename T, typename OP>
__global__ void elementwise_v1(T* x, size_t N) {
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        OP()(&x[idx]);
    }
}

template<typename T, typename OP>
__global__ void elementwise_v2(T* x, size_t N) {
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    int grid_size = gridDim.x * blockDim.x;

    for (int i = 0; i < N; i += grid_size) {
        OP()(&x[i + idx]);
    }
}

template<typename T, typename OP, int ELEMENT_PER_THREAD>
__global__ void elementwise_v3(T* x, size_t N) {
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    idx *= ELEMENT_PER_THREAD;

#pragma unroll ELEMENT_PER_THREAD
    for (int i = 0; i < ELEMENT_PER_THREAD; i++) {
        if (idx + i < N) {
            OP()(&x[idx + i]);
        }
    }
}

template<typename T>
class Abs {
 public:
    __device__ __forceinline__ void operator()(T* x) const {
        *x = std::abs(*x);
    }
};

template<>
class Abs<float4> {
 public:
    __device__ __forceinline__ void operator()(float4* x) const {
        x->x = std::abs(x->x);
        x->y = std::abs(x->y);
        x->z = std::abs(x->z);
        x->w = std::abs(x->w);
    }
};

namespace elementwise {

template<typename T, typename OP>
__global__ void unary(T* a, size_t N) {
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        OP()(&a[idx]);
    }
}

template<typename T, typename OP>
__global__ void binary(T* a, T* b, size_t N) {
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        OP()(&a[idx], &b[idx]);
    }
}

template<typename T, typename OP>
__global__ void ternary(T* a, T* b, T* c, size_t N) {
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        OP()(&a[idx], &b[idx], &c[idx]);
    }
}

} // namespace elementwise

template<typename T>
class Copy {
 public:
    __device__ __forceinline__ void operator()(float* src, float* dist) const {
        *dist = *src;
    }
};

int main() {
    int N = 1024 * 1024 * 400;
    std::vector<float> x(N);

    std::uniform_real_distribution<float> dist(-10, 10);
    std::default_random_engine generator;
    for (int i = 0; i < N; ++i) {
        x[i] = dist(generator);
    }

    float* d_nums;
    cudaMalloc(&d_nums, sizeof(float) * N);
    cudaMemcpy(d_nums, x.data(), sizeof(float) * N, cudaMemcpyHostToDevice);

    {
        Timer timer("elementwise_v1<float, Abs<float>>");
        int threads = 256;
        int blocks = (N + threads - 1) / threads;

        elementwise_v1<float, Abs<float>><<<blocks, threads>>>(d_nums, N);
        cudaDeviceSynchronize();
        CHECK_CUDA_ERR(cudaGetLastError());
    }

    {
        Timer timer("elementwise_v1<float4, Abs<float4>>");
        int threads = 256;
        int blocks = (N + threads - 1) / threads;
        blocks = blocks / 4;

        auto f4 = reinterpret_cast<float4*>(d_nums);
        elementwise_v1<float4, Abs<float4>><<<blocks, threads>>>(f4, N / 4);
        cudaDeviceSynchronize();
        CHECK_CUDA_ERR(cudaGetLastError());
    }

    {
        Timer timer("elementwise_v2<float, Abs<float>>");
        int threads = 256;
        int blocks = (N + threads - 1) / threads;
        blocks = blocks / 4;

        elementwise_v2<float, Abs<float>><<<blocks, threads>>>(d_nums, N);
        cudaDeviceSynchronize();
        CHECK_CUDA_ERR(cudaGetLastError());
    }

    {
        Timer timer("elementwise_v2<float4, Abs<float4>>");
        int threads = 256;
        int blocks = (N + threads - 1) / threads;
        blocks = blocks / 4 / 4;

        auto f4 = reinterpret_cast<float4*>(d_nums);
        elementwise_v2<float4, Abs<float4>><<<blocks, threads>>>(f4, N / 4);
        cudaDeviceSynchronize();
        CHECK_CUDA_ERR(cudaGetLastError());
    }

    {
        Timer timer("elementwise_v3<float, Abs<float>, 4>");
        int threads = 256;
        int blocks = (N + threads - 1) / threads;
        blocks = blocks / 4;

        elementwise_v3<float, Abs<float>, 4><<<blocks, threads>>>(d_nums, N);
        cudaDeviceSynchronize();
        CHECK_CUDA_ERR(cudaGetLastError());
    }

    {
        Timer timer("elementwise_v3<float4, Abs<float4>, 4>");
        int threads = 256;
        int blocks = (N + threads - 1) / threads;
        blocks = blocks / 4 / 4;

        auto f4 = reinterpret_cast<float4*>(d_nums);
        elementwise_v3<float4, Abs<float4>, 4><<<blocks, threads>>>(f4, N / 4);
        cudaDeviceSynchronize();
        CHECK_CUDA_ERR(cudaGetLastError());
    }

    {
        Timer timer("elementwise::binary<float, Copy<float>>");
        float* d_out;
        cudaMalloc(&d_out, sizeof(float) * N);

        int threads = 256;
        int blocks = (N + threads - 1) / threads;
        elementwise::binary<float, Copy<float>><<<blocks, threads>>>(d_nums, d_out, N);
        cudaDeviceSynchronize();
        CHECK_CUDA_ERR(cudaGetLastError());
    }

    return 0;
}
