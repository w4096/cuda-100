#include <cuda_runtime.h>

__global__ void conv2d_naive_kernel(const float* __restrict__ input, const float* __restrict__ kernel, float* output, int input_rows,
                                    int input_cols, int kernel_rows, int kernel_cols) {
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;

    if (row >= input_rows || col >= input_cols) {
        return;
    }

    int row_radius = kernel_rows / 2;
    int col_radius = kernel_cols / 2;

    float sum = 0;
    for (int kr = -row_radius; kr <= row_radius; ++kr) {
        for (int kc = -col_radius; kc <= col_radius; ++kc) {
            int in_row = row + kr;
            int in_col = col + kc;
            if (in_row >= 0 && in_row < input_rows && in_col >= 0 && in_col < input_cols) {
                sum +=
                    input[in_row * input_cols + in_col] * kernel[(kr + row_radius) * kernel_cols + (kc + col_radius)];
            }
        }
    }
    output[row * input_cols + col] = sum;
}

template<int TILE_WIDTH, int TILE_HEIGHT, int BLOCK_SIZE>
__global__ void conv2d_tiled_kernel_1(const float* __restrict__ input, const float* __restrict__ kernel, float* output, int input_rows,
                                    int input_cols, int kernel_rows, int kernel_cols) {
    __shared__ float tile[TILE_HEIGHT][TILE_WIDTH];

    // left top corner of the tile in the input matrix
    int tile_offset_x = blockIdx.x * TILE_WIDTH;
    int tile_offset_y = blockIdx.y * TILE_HEIGHT;

    // Load input tile into shared memory
    for (int i = 0; i < TILE_HEIGHT * TILE_WIDTH; i += BLOCK_SIZE) {
        int idx = i + threadIdx.x;
        int tile_y = idx / TILE_WIDTH;
        int tile_x = idx % TILE_WIDTH;
        int in_y = tile_offset_y + tile_y;
        int in_x = tile_offset_x + tile_x;
        if (in_y < input_rows && in_x < input_cols) {
            tile[tile_y][tile_x] = input[in_y * input_cols + in_x];
        } else {
            tile[tile_y][tile_x] = 0.0f;
        }
    }
    __syncthreads();

    int kernel_y_radius = kernel_rows / 2;
    int kernel_x_radius = kernel_cols / 2;

    for (int i = 0; i < TILE_HEIGHT * TILE_WIDTH; i += BLOCK_SIZE) {
        int idx = i + threadIdx.x;
        int tile_y = idx / TILE_WIDTH;
        int tile_x = idx % TILE_WIDTH;

        int out_y = tile_offset_y + tile_y;
        int out_x = tile_offset_x + tile_x;

        if (out_y >= input_rows || out_x >= input_cols) {
            continue;
        }

        float sum = 0;
        for (int kr = -kernel_y_radius; kr <= kernel_y_radius; ++kr) {
            for (int kc = -kernel_x_radius; kc <= kernel_x_radius; ++kc) {
                int k_x = kernel_x_radius + kc;
                int k_y = kernel_y_radius + kr;
                float k_value = kernel[k_y * kernel_cols + k_x];

                int smem_x = tile_x + kc;
                int smem_y = tile_y + kr;
                if (smem_x >= 0 && smem_x < TILE_WIDTH && smem_y >= 0 && smem_y < TILE_HEIGHT) {
                    sum += tile[smem_y][smem_x] * k_value;
                } else {
                    int in_y = out_y + kr;
                    int in_x = out_x + kc;
                    if (in_y >= 0 && in_y < input_rows && in_x >= 0 && in_x < input_cols) {
                        sum += input[in_y * input_cols + in_x] * k_value;
                    }
                }
            }
        }
        output[out_y * input_cols + out_x] = sum;
    }
}


template<int OUTPUT_TILE_WIDTH, int OUTPUT_TILE_HEIGHT>
__global__ void conv2d_tiled_kernel_2(const float* __restrict__ input, const float* __restrict__ kernel, float* output, int input_rows,
                                    int input_cols, int kernel_rows, int kernel_cols, int input_tile_width, int input_tile_height) {

    extern __shared__ float tile[];

    // left top corner of the tile in the input matrix
    int out_tile_offset_x = blockIdx.x * OUTPUT_TILE_WIDTH;
    int out_tile_offset_y = blockIdx.y * OUTPUT_TILE_HEIGHT;

    int kernel_x_radius = kernel_cols / 2;
    int kernel_y_radius = kernel_rows / 2;

    int in_tile_offset_x = out_tile_offset_x - kernel_x_radius;
    int in_tile_offset_y = out_tile_offset_y - kernel_y_radius;


    // Load input tile into shared memory
    for (int i = threadIdx.y; i < input_tile_height; i += blockDim.y) {
        int tile_y = i;
        int in_y = in_tile_offset_y + tile_y;

        for (int j = threadIdx.x; j < input_tile_width; j += blockDim.x) {
            int tile_x = j;
            int in_x = in_tile_offset_x + tile_x;

            if (in_y >= 0 && in_y < input_rows && in_x >= 0 && in_x < input_cols) {
                tile[tile_y * input_tile_width + tile_x] = input[in_y * input_cols + in_x];
            } else {
                tile[tile_y * input_tile_width + tile_x] = 0.0f;
            }
        }
    }
    __syncthreads();

    for (int i = 0; i < OUTPUT_TILE_WIDTH * OUTPUT_TILE_HEIGHT; i += blockDim.x * blockDim.y) {
        int idx = i + threadIdx.x + threadIdx.y * blockDim.x;
        int out_tile_y = idx / OUTPUT_TILE_WIDTH;
        int out_tile_x = idx % OUTPUT_TILE_WIDTH;

        int out_y = out_tile_offset_y + out_tile_y;
        int out_x = out_tile_offset_x + out_tile_x;

        if (out_y >= input_rows || out_x >= input_cols) {
            continue;
        }

        float sum = 0;
        for (int kr = -kernel_y_radius; kr <= kernel_y_radius; ++kr) {
            for (int kc = -kernel_x_radius; kc <= kernel_x_radius; ++kc) {
                int k_x = kernel_x_radius + kc;
                int k_y = kernel_y_radius + kr;
                float k_value = kernel[k_y * kernel_cols + k_x];

                int in_tile_x = out_tile_x + kc + kernel_x_radius;
                int in_tile_y = out_tile_y + kr + kernel_y_radius;
                sum += tile[in_tile_y * input_tile_width + in_tile_x] * k_value;
            }
        }
        output[out_y * input_cols + out_x] = sum;
    }
}

static unsigned cdiv(unsigned a, unsigned b) {
    return (a + b - 1) / b;
}


extern "C" void solve_naive(const float* input, const float* kernel, float* output, int input_rows, int input_cols,
                            int kernel_rows, int kernel_cols) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid(cdiv(input_cols, threadsPerBlock.x), cdiv(input_rows, threadsPerBlock.y));

    conv2d_naive_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, kernel, output, input_rows, input_cols, kernel_rows,
                                                            kernel_cols);
    cudaDeviceSynchronize();
}


extern "C" void solve(const float* input, const float* kernel, float* output, int input_rows, int input_cols,
                      int kernel_rows, int kernel_cols) {
    constexpr int TILE_WIDTH = 32;
    constexpr int TILE_HEIGHT = 32;
    constexpr int BLOCK_SIZE = 512;
    dim3 threadsPerBlock(BLOCK_SIZE);
    dim3 blocksPerGrid(cdiv(input_cols, TILE_WIDTH), cdiv(input_rows, TILE_HEIGHT));

    conv2d_tiled_kernel_1<TILE_WIDTH, TILE_HEIGHT, BLOCK_SIZE>
        <<<blocksPerGrid, threadsPerBlock>>>(input, kernel, output, input_rows, input_cols, kernel_rows, kernel_cols);
    cudaDeviceSynchronize();
}

extern "C" void solve2(const float* input, const float* kernel, float* output, int input_rows, int input_cols,
                      int kernel_rows, int kernel_cols) {
    constexpr int TILE_WIDTH = 32;
    constexpr int TILE_HEIGHT = 32;
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid(cdiv(input_cols, TILE_WIDTH), cdiv(input_rows, TILE_HEIGHT));

    int input_tile_with = TILE_WIDTH + kernel_cols - 1;
    int input_tile_height = TILE_HEIGHT + kernel_rows - 1;

    int smem_size = sizeof(float) * input_tile_with * input_tile_height;

    conv2d_tiled_kernel_2<TILE_WIDTH, TILE_HEIGHT>
        <<<blocksPerGrid, threadsPerBlock, smem_size>>>(input, kernel, output, input_rows, input_cols, kernel_rows, kernel_cols,
                                            input_tile_with, input_tile_height);
    cudaDeviceSynchronize();
}
