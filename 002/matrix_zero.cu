#include <stdlib.h>
#include <assert.h>
#include <iostream>

__global__ void matrixZeroKernel(float *matrix, const int width, const int height) {
    const auto x = blockIdx.x * blockDim.x + threadIdx.x;
    const auto y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        matrix[y * width + x] = 0;
    }
}


int main(int argc, char **argv) {
    const int width = 1024;
    const int height = 512;
    auto *h_matrix  = new float[width * height];

    for (int i = 0; i < width * height; i++) {
        h_matrix[i] = 1;
    }

    float *d_matrix;
    cudaMalloc(reinterpret_cast<void **>(&d_matrix), width * height * sizeof(float));

    cudaMemcpy(d_matrix, h_matrix, width * height * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threads = {16, 16};
    dim3 blocks = {(width + threads.x - 1) / threads.x, (height + threads.y - 1) / threads.y};
    matrixZeroKernel<<<blocks, threads>>>(d_matrix, width, height);
    cudaDeviceSynchronize();

    cudaMemcpy(h_matrix, d_matrix, width * height * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < width * height; i++) {
        assert(h_matrix[i] == 0);
    }

    delete[] h_matrix;
    cudaFree(d_matrix);
}