#include "utils/utils.h"
#include <cassert>
#include <chrono>

__global__ void add(float* x, float* y, float* z, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        z[i] = x[i] + y[i];
    }
}

int main() {
    int N = 1024 * 1024 * 100;
    int SIZE = N * sizeof(float);

    auto* h_x = static_cast<float*>(malloc(SIZE));
    auto* h_y = static_cast<float*>(malloc(SIZE));
    auto* h_z = static_cast<float*>(malloc(SIZE));

    for (int i = 0; i < N; i++) {
        h_x[i] = 1.0f;
        h_y[i] = 2.0f;
    }

    {
        Timer timer("cpu");

        for (int i = 0; i < N; i++) {
            h_z[i] = h_x[i] + h_y[i];
        }
    }

    float *d_x, *d_y, *d_z;
    CHECK_CUDA_ERR(cudaMalloc(&d_x, SIZE));
    CHECK_CUDA_ERR(cudaMalloc(&d_y, SIZE));
    CHECK_CUDA_ERR(cudaMalloc(&d_z, SIZE));

    CHECK_CUDA_ERR(cudaMemcpy(d_x, h_x, SIZE, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(d_y, h_y, SIZE, cudaMemcpyHostToDevice));

    int threads = 128;
    int blocks = (N + threads - 1) / threads;
    {
        Timer timer("gpu");
        add<<<blocks, threads>>>(d_x, d_y, d_z, N);
        CHECK_CUDA_ERR(cudaGetLastError());
        cudaDeviceSynchronize();
    }

    CHECK_CUDA_ERR(cudaMemcpy(h_z, d_z, SIZE, cudaMemcpyDeviceToHost));

    for (int i = 0; i < N; i++) {
        ASSERT(h_z[i] == 3);
    }
    printf("All assertions passed! Result is correct.\n");

    free(h_x);
    free(h_y);
    free(h_z);
    CHECK_CUDA_ERR(cudaFree(d_x));
    CHECK_CUDA_ERR(cudaFree(d_y));
    CHECK_CUDA_ERR(cudaFree(d_z));

    CHECK_CUDA_ERR(cudaDeviceReset());

    return 0;
}
