#include "common/utils.h"
#include <iostream>
#include <string>
#include <torch/torch.h>

__global__ void elementwise_abs(float* x, size_t N) {
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        x[idx] = std::abs(x[idx]);
    }
}

template<typename T, typename OP>
__global__ void elementwise(T* x, size_t N) {
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        OP()(&x[idx]);
    }
}

template<typename T>
class Abs {
 public:
    __device__ __forceinline__ void operator()(T* x) const {
        if (*x < 0) {
            *x = -*x;
        }
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

template<typename T>
class ReLU {
    __device__ __forceinline__ void operator()(T* x) const {
        *x = *x < 0 ? 0 : *x;
    }
};

int main() {
    int N = 1024 * 1024 * 200;

    auto x = torch::rand(N).cuda();
    {
        Timer timer("torch::abs");
        torch::abs_(x);
    }

    x = torch::rand(N).cuda();
    {
        Timer timer("elementwise_abs");
        int threads = 256;
        int blocks = (N + threads - 1) / threads;

        elementwise_abs<<<blocks, threads>>>(x.data_ptr<float>(), N);
        cudaDeviceSynchronize();
        CHECK_CUDA_ERR(cudaGetLastError());
    }

    x = torch::rand(N).cuda();
    {
        Timer timer("elementwise.abs");
        int threads = 256;
        int blocks = (N + threads - 1) / threads;

        elementwise<float, Abs<float>><<<blocks, threads>>>(x.data_ptr<float>(), N);
        cudaDeviceSynchronize();
        CHECK_CUDA_ERR(cudaGetLastError());
    }

    x = torch::rand(N).cuda();
    {
        Timer timer("elementwise.abs.float4");
        int threads = 256;
        int blocks = (N + threads - 1) / threads;
        blocks = blocks / 4;

        auto float4_x = reinterpret_cast<float4*>(x.data_ptr<float>());
        elementwise<float4, Abs<float4>><<<blocks, threads>>>(float4_x, N);
        cudaDeviceSynchronize();
        CHECK_CUDA_ERR(cudaGetLastError());
    }

    x = torch::rand(N).cuda();
    {
        Timer timer("torch::relu");
        torch::relu_(x);
    }

    x = torch::rand(N).cuda();
    {
        Timer timer("elementwise.relu");
        int threads = 256;
        int blocks = (N + threads - 1) / threads;

        elementwise<float, ReLU<float>><<<blocks, threads>>>(x.data_ptr<float>(), N);
        cudaDeviceSynchronize();
        CHECK_CUDA_ERR(cudaGetLastError());
    }

    return 0;
}
