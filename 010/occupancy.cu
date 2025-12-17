#include <iostream>

__global__ void Kernel1(int* d, int* a, int* b) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    d[idx] = a[idx] * b[idx];
}

__global__ void Kernel2(int* d, int* a, int* b) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    __shared__ int tile[4096];
    tile[idx] = a[idx] * b[idx];

    d[idx] = tile[idx * idx % 4096];
}

__global__ void Kernel3(int* d, int* a, int* b) {
    constexpr int NREGS = 16;
    int nums[NREGS];
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    idx *= NREGS;

    // use there registers, avoid compiler optimize them out
    for (int i = 0; i < NREGS; i++) {
        nums[i] = a[idx + i] * b[idx + i];
    }
    for (int i = 0; i < NREGS; i++) {
        for (int j = i; j < NREGS; j++) {
            if (nums[j] > nums[j]) {
                int t = nums[j];
                nums[j] = nums[i];
                nums[i] = t;
            }
        }
    }
    for (int i = 0; i < NREGS; i++) {
        d[idx + i] = nums[i];
    }
}

__global__ void Kernel4(int* d, int* a, int* b) {
    constexpr int NREGS = 32;
    int nums[NREGS];
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    idx *= NREGS;

    for (int i = 0; i < NREGS; i++) {
        nums[i] = a[idx + i] * b[idx + i];
    }
    for (int i = 0; i < NREGS; i++) {
        for (int j = i; j < NREGS; j++) {
            if (nums[j] > nums[j]) {
                int t = nums[j];
                nums[j] = nums[i];
                nums[i] = t;
            }
        }
    }
    for (int i = 0; i < NREGS; i++) {
        d[idx + i] = nums[i];
    }
}


__global__ void Kernel5(int* d, int* a, int* b) {
    constexpr int NREGS = 64;
    int nums[NREGS];
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    idx *= NREGS;

    for (int i = 0; i < NREGS; i++) {
        nums[i] = a[idx + i] * b[idx + i];
    }
    for (int i = 0; i < NREGS; i++) {
        for (int j = i; j < NREGS; j++) {
            if (nums[j] > nums[j]) {
                int t = nums[j];
                nums[j] = nums[i];
                nums[i] = t;
            }
        }
    }
    for (int i = 0; i < NREGS; i++) {
        d[idx + i] = nums[i];
    }
}


__global__ __maxnreg__(32) void Kernel6(int* d, int* a, int* b) {
    constexpr int NREGS = 128;
    int nums[NREGS];
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    idx *= NREGS;

    for (int i = 0; i < NREGS; i++) {
        nums[i] = a[idx + i] * b[idx + i];
    }
    for (int i = 0; i < NREGS; i++) {
        for (int j = i; j < NREGS; j++) {
            if (nums[j] > nums[j]) {
                int t = nums[j];
                nums[j] = nums[i];
                nums[i] = t;
            }
        }
    }
    for (int i = 0; i < NREGS; i++) {
        d[idx + i] = nums[i];
    }
}

void device_query() {
    int device;
    cudaDeviceProp prop;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);

    std::cout << prop.name << std::endl;
    std::cout << "SMs count:               " << prop.multiProcessorCount << std::endl;
    std::cout << "max blocks per sm:       " << prop.maxBlocksPerMultiProcessor << std::endl;
    std::cout << "max threads per sm:      " << prop.maxThreadsPerMultiProcessor << std::endl;
    std::cout << "max threads per block:   " << prop.maxThreadsPerBlock << std::endl;
    std::cout << "max warp per sm:         " << prop.maxThreadsPerMultiProcessor / prop.warpSize << std::endl;
    std::cout << "shared memory per sm:    " << prop.sharedMemPerMultiprocessor << std::endl;
    std::cout << "shared memory per block: " << prop.sharedMemPerBlock << std::endl;
    std::cout << "registers per sm:        " << prop.regsPerMultiprocessor << std::endl;
    std::cout << "registers per block:     " << prop.regsPerBlock << std::endl;
}

void calculate_occupancy(const void* kernel, const int block_size) {
    int blocks;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, kernel, block_size, 0);

    int device;
    cudaDeviceProp prop{};
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);

    int max_warps = prop.maxThreadsPerMultiProcessor / prop.warpSize;
    int active_warps = blocks * block_size / prop.warpSize;

    double occupancy = static_cast<double>(active_warps) / max_warps * 100;

    std::cout << "block_size:" << block_size << " blocks:" << blocks << " occupancy: " << occupancy << "%" << std::endl;
}

int main() {
    device_query();


    int i = 0;
    for (const auto kernel: {Kernel1, Kernel2, Kernel3, Kernel4, Kernel5, Kernel6}) {
        std::cout << "kernel " << i++ << ":" << std::endl;
        for (auto block_size : {64, 256, 512, 1024}) {
            calculate_occupancy(reinterpret_cast<const void*>(kernel), block_size);
        }
    }
    return 0;
}
