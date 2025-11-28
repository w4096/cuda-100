#include "common/utils.h"
#include "gemm.cuh"
#include <cstdio>
#include <cstdlib>
#include <random>

__global__ void diff_count(const float* a, const float* b, unsigned int N, const float delta, int* count) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < N) {
        if (std::abs(a[tid] - b[tid]) > delta) {
            atomicAdd(count, 1);
        }
    }
}

bool all_close(const float* a, const float* b, unsigned int N) {
    int* count;
    cudaMalloc(&count, sizeof(int));
    int zero = 0;
    cudaMemcpy(count, &zero, sizeof(int), cudaMemcpyHostToDevice);

    diff_count<<<cdiv(N, 256), 256>>>(a, b, N, 0.1, count);
    cudaDeviceSynchronize();
    cudaMemcpy(&zero, count, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(count);
    return zero == 0;
}

int main() {
    int M = 4096;
    int K = 4096;
    int N = 4096;

    auto* h_A = new float[M * K];
    auto* h_B = new float[K * N];
    auto* h_C = new float[M * N];

    std::normal_distribution<float> distribution(0, 1.0);
    std::default_random_engine generator;

    for (int i = 0; i < M * K; i++) {
        h_A[i] = distribution(generator);
    }
    for (int i = 0; i < K * N; i++) {
        h_B[i] = distribution(generator);
    }
    for (int i = 0; i < M * N; i++) {
        h_C[i] = distribution(generator);
    }

    float *d_A, *d_B, *d_C;

    cudaMalloc(reinterpret_cast<void**>(&d_A), M * K * sizeof(float));
    cudaMalloc(reinterpret_cast<void**>(&d_B), K * N * sizeof(float));
    cudaMalloc(reinterpret_cast<void**>(&d_C), M * N * sizeof(float));

    cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, M * N * sizeof(float), cudaMemcpyHostToDevice);

    {
        dim3 threads{32, 32};
        dim3 blocks{cdiv(N, threads.x), cdiv(M, threads.y)};
        {
            Timer t("0");
            gemm(0, M, N, K, 1.0, d_A, d_B, 1.0, d_C);
            cudaDeviceSynchronize();
        }
        CHECK_CUDA_ERR(cudaGetLastError());
    }

    {
        float* d_C1;
        cudaMalloc(reinterpret_cast<void**>(&d_C1), M * N * sizeof(float));
        cudaMemcpy(d_C1, h_C, M * N * sizeof(float), cudaMemcpyHostToDevice);
        {
            Timer t("1");
            gemm(1, M, N, K, 1.0, d_A, d_B, 1.0, d_C1);
            cudaDeviceSynchronize();
        }
        CHECK_CUDA_ERR(cudaGetLastError());
        ASSERT(all_close(d_C, d_C1, M * N));
        cudaFree(d_C1);
    }

    {
        float* d_C1;
        cudaMalloc(reinterpret_cast<void**>(&d_C1), M * N * sizeof(float));
        cudaMemcpy(d_C1, h_C, M * N * sizeof(float), cudaMemcpyHostToDevice);
        {
            Timer t("2");
            gemm(2, M, N, K, 1.0, d_A, d_B, 1.0, d_C1);
            cudaDeviceSynchronize();
        }
        CHECK_CUDA_ERR(cudaGetLastError());
        ASSERT(all_close(d_C, d_C1, M * N));
        cudaFree(d_C1);
    }

    {
        float* d_C1;
        cudaMalloc(reinterpret_cast<void**>(&d_C1), M * N * sizeof(float));
        cudaMemcpy(d_C1, h_C, M * N * sizeof(float), cudaMemcpyHostToDevice);
        {
            Timer t("3");
            gemm(3, M, N, K, 1.0, d_A, d_B, 1.0, d_C1);
            cudaDeviceSynchronize();
        }
        CHECK_CUDA_ERR(cudaGetLastError());
        ASSERT(all_close(d_C, d_C1, M * N));
        cudaFree(d_C1);
    }

    {
        float* d_C1;
        cudaMalloc(reinterpret_cast<void**>(&d_C1), M * N * sizeof(float));
        cudaMemcpy(d_C1, h_C, M * N * sizeof(float), cudaMemcpyHostToDevice);
        {
            Timer t("4");
            gemm(4, M, N, K, 1.0, d_A, d_B, 1.0, d_C1);
            cudaDeviceSynchronize();
        }
        CHECK_CUDA_ERR(cudaGetLastError());
        ASSERT(all_close(d_C, d_C1, M * N));
        cudaFree(d_C1);
    }

    {
        float* d_C1;
        cudaMalloc(reinterpret_cast<void**>(&d_C1), M * N * sizeof(float));
        cudaMemcpy(d_C1, h_C, M * N * sizeof(float), cudaMemcpyHostToDevice);
        {
            Timer t("100");
            gemm(100, M, N, K, 1.0, d_A, d_B, 1.0, d_C1);
            cudaDeviceSynchronize();
        }
        CHECK_CUDA_ERR(cudaGetLastError());
        ASSERT(all_close(d_C, d_C1, M * N));
        cudaFree(d_C1);
    }

    return 0;
}