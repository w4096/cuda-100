#include "common/utils.h"
#include <random>
#include "softmax.cuh"

void softmax(const float *in, float *out, int rows, int dim) {
    for (int row = 0; row < rows; ++row) {
        const float *x = &in[row * dim];
        float *o = &out[row * dim];

        float max = std::numeric_limits<float>::min();
        for (int i = 0; i < dim; i++) {
            max = std::max(max, x[i]);
        }

        float sum = 0;
        for (int i = 0; i < dim; i++) {
            sum += std::exp(x[i] - max);
        }

        for (int i = 0; i < dim; i++) {
            o[i] = std::exp(x[i] - max) / sum;
        }
    }
}

void check(const float *in, float *out, int n) {
    for (int i = 0; i < n; i++) {
        ASSERT(std::abs(in[i] - out[i]) < 1e-5);
    }
}

__global__ void clear(float *data) {
    data[threadIdx.x + blockIdx.x * blockDim.x] = 0;
}

int main() {
    constexpr int m = 4096;
    constexpr  int dim = 128;
    constexpr size_t size = m * dim * sizeof(float);

    float* h_in = (float*)malloc(size);
    float* h_out = (float*)malloc(size);

    std::default_random_engine generator;
    std::uniform_real_distribution<float> dist(-1000, 1000);

    for (int i = 0; i < m * dim; i++) {
        h_in[i] = dist(generator);
    }

    {
        Timer timer("cpu");
        softmax(h_in, h_out, m, dim);
    }

    float *d_in, *d_out;
    CHECK_CUDA_ERR(cudaMalloc(&d_in, size));
    CHECK_CUDA_ERR(cudaMalloc(&d_out, size));

    CHECK_CUDA_ERR(cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice));


    {
        Timer timer("warp");
        int threads = 256;
        int blocks = 4096;
        softmax_warp_reduce<<<blocks, threads>>>(d_in, d_out, m, dim);
        cudaDeviceSynchronize();
        CHECK_CUDA_ERR(cudaGetLastError());
    }

    float* h_out_1 = (float*)malloc(size);
    CHECK_CUDA_ERR(cudaMemcpy(h_out_1, d_out, size, cudaMemcpyDeviceToHost));
    check(h_out, h_out_1, dim * m);

    clear<<<m * dim / 256, 256>>>(d_out);
    cudaDeviceSynchronize();

    {
        Timer timer("block");
        int threads = 256;
        int blocks = 4096;
        online_softmax_block_reduce<<<blocks, threads>>>(d_in, d_out, m, dim);
        cudaDeviceSynchronize();
        CHECK_CUDA_ERR(cudaGetLastError());
    }

    CHECK_CUDA_ERR(cudaMemcpy(h_out_1, d_out, size, cudaMemcpyDeviceToHost));
    check(h_out, h_out_1, dim * m);

    return 0;
}
