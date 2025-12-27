#include <cstdio>
#include <cuda_runtime.h>

__global__ void convolution_1d_kernel(const float* input, const float* kernel, float* output, int input_size,
                                      int kernel_size) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int output_size = input_size - kernel_size + 1;
    if (tid >= output_size) {
        return;
    }
    for (int i = 0; i < kernel_size; ++i) {
        output[tid] += input[tid + i] * kernel[i];
    }
}

__global__ void convolution_1d_kernel_smem(const float* input, const float* kernel, float* output, int input_size,
                                           int kernel_size) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    extern __shared__ float smem[];
    float* kernel_smem = smem;
    float* block_input_smem = smem + kernel_size;

    const int block_size = blockDim.x;
    int block_input_size = block_size + kernel_size - 1;
    if (block_input_size > input_size) {
        block_input_size = input_size;
    }

    // Load kernel into shared memory
    for (int i = threadIdx.x; i < kernel_size; i += block_size) {
        kernel_smem[i] = kernel[i];
    }

    // Load input into shared memory
    int bidx = blockIdx.x * blockDim.x;
    for (int i = threadIdx.x; i < block_input_size; i += block_size) {
        if (bidx + i < input_size) {
            block_input_smem[i] = input[bidx + i];
        }
    }

    __syncthreads();

    int output_size = input_size - kernel_size + 1;
    if (tid >= output_size) {
        return;
    }

    float sum = 0.0f;
    for (int i = 0; i < kernel_size; ++i) {
        sum += kernel_smem[i] * block_input_smem[threadIdx.x + i];
    }
    output[tid] = sum;
}

// input, kernel, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, const float* kernel, float* output, int input_size, int kernel_size) {
    int output_size = input_size - kernel_size + 1;
    int threadsPerBlock = 256;
    int blocksPerGrid = (output_size + threadsPerBlock - 1) / threadsPerBlock;

    // convolution_1d_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, kernel, output, input_size, kernel_size);
    int smem = sizeof(float) * (kernel_size + threadsPerBlock + kernel_size - 1);
    convolution_1d_kernel_smem<<<blocksPerGrid, threadsPerBlock, smem>>>(input, kernel, output, input_size,
                                                                         kernel_size);

    cudaDeviceSynchronize();
}
