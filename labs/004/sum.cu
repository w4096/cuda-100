#include "utils/utils.h"
#include <cassert>

__global__ void sum_kernel_naive(const int* nums, int N, int* result) {
    const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        atomicAdd(result, nums[idx]);
    }
}

__global__ void sum_kernel_shared_memory(const int* nums, int N, int* result) {
    __shared__ int sum;
    if (threadIdx.x == 0) {
        sum = 0;
    }
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        atomicAdd(&sum, nums[idx]);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        atomicAdd(result, sum);
    }
}

__global__ void sum_kernel_coalesced_access(const int* nums, int N, int stride, int* result) {
    __shared__ int sum;
    if (threadIdx.x == 0) {
        sum = 0;
    }
    __syncthreads();
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * stride;
    int local_sum = 0;
    for (int i = 0; i < stride; i++) {
        if (idx + i < N) {
            local_sum += nums[idx + i];
        }
    }
    atomicAdd(&sum, local_sum);

    __syncthreads();
    if (threadIdx.x == 0) {
        atomicAdd(result, sum);
    }
}

int main() {
    int N = 1024 * 1024 * 100;

    int* h_nums = (int*)malloc(N * sizeof(int));
    for (int i = 0; i < N; i++) {
        h_nums[i] = 1;
    }

    int* d_nums;
    cudaMalloc(&d_nums, N * sizeof(int));
    cudaMemcpy(d_nums, h_nums, N * sizeof(int), cudaMemcpyHostToDevice);
    {
        Timer t("naive");
        int* d_sum;
        cudaMalloc(&d_sum, sizeof(int));
        dim3 threads = {256};
        dim3 blocks = (N + threads.x - 1) / threads.x;
        sum_kernel_naive<<<blocks, threads>>>(d_nums, N, d_sum);
        cudaDeviceSynchronize();
        int sum = 0;
        cudaMemcpy(&sum, d_sum, sizeof(int), cudaMemcpyDeviceToHost);
        assert(sum == N);
    }
    {
        Timer t("shared_memory");
        int* d_sum;
        cudaMalloc(&d_sum, sizeof(int));
        dim3 threads = {256};
        dim3 blocks = (N + threads.x - 1) / threads.x;
        sum_kernel_shared_memory<<<blocks, threads>>>(d_nums, N, d_sum);
        cudaDeviceSynchronize();
        int sum = 0;
        cudaMemcpy(&sum, d_sum, sizeof(int), cudaMemcpyDeviceToHost);
        assert(sum == N);
    }
    {
        Timer t("coalesced_access");
        int* d_sum;
        cudaMalloc(&d_sum, sizeof(int));
        dim3 threads = {256};
        dim3 blocks = (N + threads.x - 1) / 8 / threads.x;
        sum_kernel_coalesced_access<<<blocks, threads>>>(d_nums, N, 8, d_sum);
        cudaDeviceSynchronize();
        int sum = 0;
        cudaMemcpy(&sum, d_sum, sizeof(int), cudaMemcpyDeviceToHost);
        assert(sum == N);
    }
}
