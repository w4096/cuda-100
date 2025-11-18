#include "common/utils.h"
#include <iostream>
#include <string>
#include <torch/torch.h>


__global__ void elementwise_abs(float *x, size_t N) {
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    x[idx] = x[idx] > 0 ? x[idx] : -x[idx];
}


template<typename A, typename OP>
__global__ void elementwise(A *a, size_t N) {
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        OP()(&a[idx]);
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

template<typename T>
class ReLU {
    __device__ __forceinline__ void operator()(T *x) const {
        *x = *x < 0 ? 0 : *x;
    }
};


int main() {
    int N = 1024 * 1024 * 200;


    auto a = torch::rand(N).cuda();
    {
        Timer timer("torch::abs");
        torch::abs_(a);
    }


    a = torch::rand(N).cuda();
    {
        Timer timer("elementwise.abs");
        int threads = 256;
        int blocks = (N + threads - 1) / threads;

        elementwise<float, Abs<float>><<<blocks, threads>>>(a.data_ptr<float>(), N);
        cudaDeviceSynchronize();
        CHECK_CUDA_ERR(cudaGetLastError());
    }

    a = torch::rand(N).cuda();
    {
        Timer timer("elementwise_abs");
        int threads = 256;
        int blocks = (N + threads - 1) / threads;

        elementwise_abs<<<blocks, threads>>>(a.data_ptr<float>(), N);
        cudaDeviceSynchronize();
        CHECK_CUDA_ERR(cudaGetLastError());
    }

    a = torch::rand(N).cuda();
    {
        Timer timer("elementwise.abs.float4");
        int threads = 256;
        int blocks = (N + threads - 1) / threads;
        blocks = blocks / 4;

        auto x = reinterpret_cast<float4*>(a.data_ptr<float>());
        elementwise<float4, Abs<float4>><<<blocks, threads>>>(x, N);
        cudaDeviceSynchronize();
        CHECK_CUDA_ERR(cudaGetLastError());
    }

    a = torch::rand(N).cuda();
    {
        Timer timer("torch::relu");
        torch::relu_(a);
    }

    a = torch::rand(N).cuda();
    {
        Timer timer("elementwise.relu");
        int threads = 256;
        int blocks = (N + threads - 1) / threads;

        elementwise<float, ReLU<float>><<<blocks, threads>>>(a.data_ptr<float>(), N);
        cudaDeviceSynchronize();
        CHECK_CUDA_ERR(cudaGetLastError());
    }


    return 0;
}




