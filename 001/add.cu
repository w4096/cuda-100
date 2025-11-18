#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <chrono>
#include <iostream>
#include <string>
#include "common/utils.h"

__global__ void add(float *x, float *y, float *z, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        z[i] = x[i] + y[i];
    }
}

int main() {
    int N = 1024 * 1024 * 100;
    int SIZE = N * sizeof(float);
    
    // 分配主机内存
    float *h_x = (float*)malloc(SIZE);
    float *h_y = (float*)malloc(SIZE);
    float *h_z = (float*)malloc(SIZE);

    // 初始化主机数据
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


    // 分配设备内存（添加错误检查）
    float *d_x, *d_y, *d_z;
    CHECK_CUDA_ERR(cudaMalloc(&d_x, SIZE));
    CHECK_CUDA_ERR(cudaMalloc(&d_y, SIZE));
    CHECK_CUDA_ERR(cudaMalloc(&d_z, SIZE));

    // 主机→设备数据传输（添加错误检查）
    CHECK_CUDA_ERR(cudaMemcpy(d_x, h_x, SIZE, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(d_y, h_y, SIZE, cudaMemcpyHostToDevice));

    // 计算 block 数：总元素数 / 每个block的线程数（向上取整）
    int threads = 128;
    int blocks = (N + threads - 1) / threads;
    {
        Timer timer("gpu");
        add<<<blocks, threads>>>(d_x, d_y, d_z, N);
        CHECK_CUDA_ERR(cudaGetLastError());
        cudaDeviceSynchronize();
    }


    // 设备→主机数据传输（添加错误检查）
    CHECK_CUDA_ERR(cudaMemcpy(h_z, d_z, SIZE, cudaMemcpyDeviceToHost));

    // 验证结果
    for (int i = 0; i < N; i++) {
        assert(h_z[i] == 3);       
    }
    printf("All assertions passed! Result is correct.\n");


    // 释放内存（避免泄漏）
    free(h_x);
    free(h_y);
    free(h_z);
    CHECK_CUDA_ERR(cudaFree(d_x));
    CHECK_CUDA_ERR(cudaFree(d_y));
    CHECK_CUDA_ERR(cudaFree(d_z));

    // 重置CUDA设备（可选，释放设备资源）
    CHECK_CUDA_ERR(cudaDeviceReset());

    return 0;
}
