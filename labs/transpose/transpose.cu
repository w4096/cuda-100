#include <cstdio>
#include <cuda_runtime.h>
#include <vector>

#include <random>

__global__ void transpose_naive_kernel(const float* in, float* out, int m, int n) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (y < m && x < n) {
        out[x * m + y] = in[y * n + x];
    }
}

__global__ void transpose_coalesced_kernel(const float* in, float* out, int m, int n) {
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
__global__ void transpose_tiling_kernel(const float* in, float* out, int m, int n) {
    __shared__ float tile[TILE_DIM][TILE_DIM];
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    if (y < m && x < n) {
        // write to a row, no bank conflict
        tile[threadIdx.y][threadIdx.x] = in[y * n + x];
    }
    __syncthreads();

    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;
    if (y < n && x < m) {
        // read from a column, have bank conflict
        out[y * m + x] = tile[threadIdx.x][threadIdx.y];
    }
}

template<int TILE_DIM>
__global__ void transpose_tiling_no_bank_conflict_kernel(const float* in, float* out, int m, int n) {
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
        // read from a column, no bank conflict since we added padding
        out[y * m + x] = tile[threadIdx.x][threadIdx.y];
    }
}

template<int TILE_DIM>
__global__ void transpose_tiling_swizzle_kernel(const float* input, float* output, int rows, int cols) {
    __shared__ float tile[TILE_DIM][TILE_DIM];
    int bx = blockIdx.x * TILE_DIM;
    int by = blockIdx.y * TILE_DIM;

    int x = bx + threadIdx.x;
    int y = by + threadIdx.y;
    int swizzled_x = threadIdx.x ^ threadIdx.y;
    if (y < rows && x < cols) {
        tile[threadIdx.y][swizzled_x] = input[y * cols + x];
    }
    __syncthreads();

    x = by + threadIdx.x;
    y = bx + threadIdx.y;
    const int t_rows = cols;
    const int t_cols = rows;
    if (y < t_rows && x < t_cols) {
        // read from different columns of different rows to avoid bank conflict
        output[y * t_cols + x] = tile[threadIdx.x][swizzled_x];
    }
}

template<int TILE_DIM, int BLOCK_SIZE>
__global__ void transpose_tiling_multi_elements_kernel(const float* in, float* out, int m, int n) {
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


extern "C" {

void transpose_cublas(const float* in, float* out, int m, int n) {
    cublasHandle_t handle;
    cublasCreate_v2(&handle);
    float alpha = 1.0f;
    float beta = 0.0f;
    cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, &alpha, in, n, &beta, nullptr, m, out, m);
}

void transpose_naive(const float* in, float* out, int m, int n) {
    dim3 block(32, 32);
    dim3 grid((n + block.x - 1) / block.x, (m + block.y - 1) / block.y);
    transpose_naive_kernel<<<grid, block>>>(in, out, m, n);
}

void transpose_coalesced(const float* in, float* out, int m, int n) {
    dim3 block(32, 8);
    dim3 grid((n + block.x - 1) / block.x, (m + block.y * 4 - 1) / (block.y * 4));
    transpose_coalesced_kernel<<<grid, block>>>(in, out, m, n);
}

void transpose_tiling(const float* in, float* out, int m, int n) {
    constexpr int TILE_DIM = 32;
    dim3 block(TILE_DIM, TILE_DIM);
    dim3 grid((n + TILE_DIM - 1) / TILE_DIM, (m + TILE_DIM - 1) / TILE_DIM);
    transpose_tiling_kernel<TILE_DIM><<<grid, block>>>(in, out, m, n);
}

void transpose_tiling_no_bank_conflict(const float* in, float* out, int m, int n) {
    constexpr int TILE_DIM = 32;
    dim3 block(TILE_DIM, TILE_DIM);
    dim3 grid((n + TILE_DIM - 1) / TILE_DIM, (m + TILE_DIM - 1) / TILE_DIM);
    transpose_tiling_no_bank_conflict_kernel<TILE_DIM><<<grid, block>>>(in, out, m, n);
}

void transpose_tiling_swizzle(const float* in, float* out, int m, int n) {
    constexpr int TILE_DIM = 16;
    dim3 block(TILE_DIM, TILE_DIM);
    dim3 grid((n + TILE_DIM - 1) / TILE_DIM, (m + TILE_DIM - 1) / TILE_DIM);
    transpose_tiling_swizzle_kernel<TILE_DIM><<<grid, block>>>(in, out, m, n);
}

void transpose_tiling_multi_elements(const float* in, float* out, int m, int n) {
    constexpr int TILE_DIM = 64;
    constexpr int BLOCK_SIZE = 256;
    dim3 block(16, 16);
    dim3 grid((n + TILE_DIM - 1) / TILE_DIM, (m + TILE_DIM - 1) / TILE_DIM);
    transpose_tiling_multi_elements_kernel<TILE_DIM, BLOCK_SIZE><<<grid, block>>>(in, out, m, n);
}



}
