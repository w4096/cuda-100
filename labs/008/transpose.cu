#include "utils/utils.h"
#include <cstdio>
#include <cuda_runtime.h>
#include <vector>

#include <random>

__global__ void transpose_naive(const float* in, float* out, int m, int n) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (y < m && x < n) {
        // out[x][y] = in[y][x];
        out[x * m + y] = in[y * n + x];
    }
}

__global__ void transpose_coalesced(const float* in, float* out, int m, int n) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    y *= 4;

    float4 a;
    a.x = in[y * n + x];
    a.y = in[(y+1) * n + x];
    a.z = in[(y+2) * n + x];
    a.w = in[(y+3) * n + x];
    *reinterpret_cast<float4 *>(&out[x * m + y]) = a;
}

template<int TILE_DIM>
__global__ void transpose_tiling(const float* in, float* out, int m, int n) {
    __shared__ float tile[TILE_DIM][TILE_DIM];
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    if (y < m && x < n) {
        tile[threadIdx.y][threadIdx.x] = in[y * n + x];
    }
    __syncthreads();

    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;
    if (y < n && x < m) {
        out[y * m + x] = tile[threadIdx.x][threadIdx.y];
    }
}

template<int TILE_DIM>
__global__ void transpose_tiling_no_bank_conflict(const float* in, float* out, int m, int n) {
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    if (y < m && x < n) {
        tile[threadIdx.y][threadIdx.x] = in[y * n + x];
    }
    __syncthreads();

    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;
    if (y < n && x < m) {
        out[y * m + x] = tile[threadIdx.x][threadIdx.y];
    }
}

template<int TILE_DIM>
__global__ void transpose_tiling_swizzle(const float* input, float* output, int rows, int cols) {
    __shared__ float tile[TILE_DIM][TILE_DIM];
    int bx = blockIdx.x * TILE_DIM;
    int by = blockIdx.y * TILE_DIM;

    int x = bx + threadIdx.x;
    int y = by + threadIdx.y;
    if (y < rows && x < cols) {
        tile[threadIdx.y][threadIdx.x^threadIdx.y] = input[y * cols + x];
    }
    __syncthreads();

    x = by + threadIdx.x;
    y = bx + threadIdx.y;
    const int t_rows = cols;
    const int t_cols = rows;
    if (y < t_rows && x < t_cols) {
        output[y * t_cols + x] = tile[threadIdx.x][threadIdx.y^threadIdx.x];
    }
}

template<int TILE_DIM, int BLOCK_SIZE>
__global__ void transpose_tiling_multi_elements(const float* in, float* out, int m, int n) {
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];
    int tx = blockIdx.x * TILE_DIM;
    int ty = blockIdx.y * TILE_DIM;

    const int tid = threadIdx.y * blockDim.x + threadIdx.x;

#pragma unroll
    for (int i = 0; i < TILE_DIM * TILE_DIM; i += BLOCK_SIZE) {
        int idx = i + tid;
        int x = idx % TILE_DIM;
        int y = idx / TILE_DIM;
        int gx = tx + x;
        int gy = ty + y;
        if (gx < n && gy < m) {
            tile[y][x] = in[gy * n + gx];
        }
    }
    __syncthreads();

    tx = blockIdx.y * TILE_DIM;
    ty = blockIdx.x * TILE_DIM;
#pragma unroll
    for (int i = 0; i < TILE_DIM * TILE_DIM; i += BLOCK_SIZE) {
        int idx = i + tid;
        int x = idx % TILE_DIM;
        int y = idx / TILE_DIM;
        int gx = tx + x;
        int gy = ty + y;
        if (gx < m && gy < n) {
            out[gy * m + gx] = tile[x][y];
        }
    }
}

#include <cublas_v2.h>

void transpose_cublas(const float* in, float* out, int m, int n) {
    cublasHandle_t handle;
    cublasCreate_v2(&handle);
    float alpha = 1.0f;
    float beta = 0.0f;
    cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, &alpha, in, n, &beta, nullptr, m, out, m);
}

__global__ void checksum_kernel(const unsigned int* a, int n, unsigned int* sum) {
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x < n) {
        atomicAdd(sum, a[x] * x);
    }
}

unsigned int checksum(const float* a, int n) {
    unsigned int* sum;
    cudaMalloc(&sum, sizeof(unsigned int));
    unsigned int cs = 0;
    cudaMemcpy(sum, &cs, sizeof(unsigned int), cudaMemcpyHostToDevice);
    checksum_kernel<<<cdiv(n, 256), 256>>>(reinterpret_cast<const unsigned int*>(a), n, sum);
    CHECK_CUDA_ERR(cudaGetLastError());
    CHECK_CUDA_ERR(cudaDeviceSynchronize());
    cudaMemcpy(&cs, sum, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaFree(sum);
    return cs;
}

int main() {
    constexpr int WIDTH = 4096;
    constexpr int HEIGHT = 8192;
    std::vector<float> h_in(HEIGHT * WIDTH);
    std::default_random_engine generator;
    std::uniform_real_distribution<float> dist(-1000, 1000);
    for (int i = 0; i < HEIGHT * WIDTH; ++i) {
        h_in[i] = dist(generator);
    }

    const size_t bytes = h_in.size() * sizeof(float);
    float *d_in = nullptr, *d_out = nullptr;
    CHECK_CUDA_ERR(cudaMalloc(&d_in, bytes));
    CHECK_CUDA_ERR(cudaMalloc(&d_out, bytes));
    CHECK_CUDA_ERR(cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice));

    {
        Timer t("naive block=32x32");

        dim3 block(32, 32);
        dim3 grid((WIDTH + block.x - 1) / block.x, (HEIGHT + block.y - 1) / block.y);

        transpose_naive<<<grid, block>>>(d_in, d_out, HEIGHT, WIDTH);
        CHECK_CUDA_ERR(cudaGetLastError());
        CHECK_CUDA_ERR(cudaDeviceSynchronize());
    }
    unsigned int cs = checksum(d_out, HEIGHT * WIDTH);

    {
        Timer t("transpose_coalesced");

        dim3 block(32, 8);
        dim3 grid((WIDTH + block.x - 1) / block.x, (HEIGHT + block.y - 1) / block.y);
        grid.y /= 4;

        transpose_coalesced<<<grid, block>>>(d_in, d_out, HEIGHT, WIDTH);
        CHECK_CUDA_ERR(cudaGetLastError());
        CHECK_CUDA_ERR(cudaDeviceSynchronize());
    }
    ASSERT(cs == checksum(d_out, HEIGHT * WIDTH));


    {
        Timer t("naive block=16x16");

        dim3 block(16, 16);
        dim3 grid((WIDTH + block.x - 1) / block.x, (HEIGHT + block.y - 1) / block.y);

        transpose_naive<<<grid, block>>>(d_in, d_out, HEIGHT, WIDTH);
        CHECK_CUDA_ERR(cudaGetLastError());
        CHECK_CUDA_ERR(cudaDeviceSynchronize());
    }
    ASSERT(cs == checksum(d_out, HEIGHT * WIDTH));

    {
        Timer t("naive block=8x32");

        dim3 block(8, 32);
        dim3 grid((WIDTH + block.x - 1) / block.x, (HEIGHT + block.y - 1) / block.y);

        transpose_naive<<<grid, block>>>(d_in, d_out, HEIGHT, WIDTH);
        CHECK_CUDA_ERR(cudaGetLastError());
        CHECK_CUDA_ERR(cudaDeviceSynchronize());
    }
    ASSERT(cs == checksum(d_out, HEIGHT * WIDTH));

    {
        Timer t("transpose_tiling");

        dim3 block(32, 32);
        dim3 grid((WIDTH + block.x - 1) / block.x, (HEIGHT + block.y - 1) / block.y);

        transpose_tiling<32><<<grid, block>>>(d_in, d_out, HEIGHT, WIDTH);
        CHECK_CUDA_ERR(cudaGetLastError());
        CHECK_CUDA_ERR(cudaDeviceSynchronize());
    }
    ASSERT(cs == checksum(d_out, HEIGHT * WIDTH));

    {
        Timer t("transpose_tiling_no_bank_conflict");

        dim3 block(32, 32);
        dim3 grid((WIDTH + block.x - 1) / block.x, (HEIGHT + block.y - 1) / block.y);

        transpose_tiling_no_bank_conflict<32><<<grid, block>>>(d_in, d_out, HEIGHT, WIDTH);
        CHECK_CUDA_ERR(cudaGetLastError());
        CHECK_CUDA_ERR(cudaDeviceSynchronize());
    }
    ASSERT(cs == checksum(d_out, HEIGHT * WIDTH));

    {
        Timer t("transpose_tiling_swizzle");

        dim3 block(16, 16);
        dim3 grid((WIDTH + block.x - 1) / block.x, (HEIGHT + block.y - 1) / block.y);

        transpose_tiling_swizzle<16><<<grid, block>>>(d_in, d_out, HEIGHT, WIDTH);
        CHECK_CUDA_ERR(cudaGetLastError());
        CHECK_CUDA_ERR(cudaDeviceSynchronize());
    }
    ASSERT(cs == checksum(d_out, HEIGHT * WIDTH));

    {
        Timer t("transpose_tiling_multi_elements(16)");
        constexpr int TILE_DIM = 64;
        dim3 block(16, 16);
        dim3 grid((WIDTH + TILE_DIM - 1) / TILE_DIM, (HEIGHT + TILE_DIM - 1) / TILE_DIM);

        transpose_tiling_multi_elements<TILE_DIM, 256><<<grid, block>>>(d_in, d_out, HEIGHT, WIDTH);
        CHECK_CUDA_ERR(cudaGetLastError());
        CHECK_CUDA_ERR(cudaDeviceSynchronize());
    }
    {
        Timer t("transpose_tiling_multi_elements(4)");
        constexpr int TILE_DIM = 32;
        dim3 block(16, 16);
        dim3 grid((WIDTH + TILE_DIM - 1) / TILE_DIM, (HEIGHT + TILE_DIM - 1) / TILE_DIM);

        transpose_tiling_multi_elements<TILE_DIM, 256><<<grid, block>>>(d_in, d_out, HEIGHT, WIDTH);
        CHECK_CUDA_ERR(cudaGetLastError());
        CHECK_CUDA_ERR(cudaDeviceSynchronize());
    }
    ASSERT(cs == checksum(d_out, HEIGHT * WIDTH));

    {
        Timer t("cublas");
        transpose_cublas(d_in, d_out, HEIGHT, WIDTH);
    }
    ASSERT(cs == checksum(d_out, HEIGHT * WIDTH));

    CHECK_CUDA_ERR(cudaFree(d_in));
    CHECK_CUDA_ERR(cudaFree(d_out));
    return 0;
}
