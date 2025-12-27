#include <cuda_runtime.h>

__global__ void invert_kernel(unsigned char* image, int N) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    auto image4 = reinterpret_cast<uint4*>(image);
    auto image1 = reinterpret_cast<unsigned int*>(image);
    int vec_N = N >> 2;

    if (tid < vec_N) {
        uint4 pixels = image4[tid];
        pixels.x = pixels.x ^ 0x00FFFFFF;
        pixels.y = pixels.y ^ 0x00FFFFFF;
        pixels.z = pixels.z ^ 0x00FFFFFF;
        pixels.w = pixels.w ^ 0x00FFFFFF;
        image4[tid] = pixels;
    }
    if (tid < (N & 3)) {
        int idx = N - (N & 3) + tid;
        image1[idx] ^= 0x00FFFFFF;
    }
}

// image_input, image_output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(unsigned char* image, int width, int height) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (width * height + threadsPerBlock - 1) / threadsPerBlock;
    blocksPerGrid = (blocksPerGrid + 3) / 4;

    int N = width * height;
    invert_kernel<<<blocksPerGrid, threadsPerBlock>>>(image, N);
    cudaDeviceSynchronize();
}