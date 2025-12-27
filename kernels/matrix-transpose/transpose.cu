#include <cuda_runtime.h>

template<int TILE_DIM>
__global__ void transpose_kernel(const float* input, float* output, int rows, int cols) {
    __shared__ float tile[TILE_DIM][TILE_DIM];
    int bx = blockIdx.x * TILE_DIM;
    int by = blockIdx.y * TILE_DIM;

    int x = bx + threadIdx.x;
    int y = by + threadIdx.y;
    if (y < rows && x < cols) {
        tile[threadIdx.y][threadIdx.x ^ threadIdx.y] = input[y * cols + x];
    }
    __syncthreads();

    x = by + threadIdx.x;
    y = bx + threadIdx.y;
    const int t_rows = cols;
    const int t_cols = rows;
    if (y < t_rows && x < t_cols) {
        output[y * t_cols + x] = tile[threadIdx.x][threadIdx.y ^ threadIdx.x];
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int rows, int cols) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    transpose_kernel<16><<<blocksPerGrid, threadsPerBlock>>>(input, output, rows, cols);
    cudaDeviceSynchronize();
}
