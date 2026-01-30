#include <cassert>
#include <cmath>
#include <cstdio>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <string>

struct SM75_U16x8_LDSM {
    __device__ static void copy(const uint32_t *smem_src, uint32_t& dst0, uint32_t& dst1, uint32_t& dst2, uint32_t& dst3) {
        uint32_t smem_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_src));
        asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                     : "=r"(dst0), "=r"(dst1), "=r"(dst2), "=r"(dst3)
                     : "r"(smem_int_ptr));
    }
};

struct SM75_U16x4_LDSM_T {
    __device__ static void copy(const uint32_t *smem_src, uint32_t& dst0, uint32_t& dst1) {
        uint32_t smem_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_src));
        asm volatile("ldmatrix.sync.aligned.x2.m8n8.shared.trans.b16 {%0, %1}, [%2];\n"
                     : "=r"(dst0), "=r"(dst1)
                     : "r"(smem_int_ptr));
    }
};

struct SM90_U16x8_STSM {
  __device__ static void
  copy(uint32_t const& src0, uint32_t const& src1, uint32_t const& src2, uint32_t const& src3,
       const uint32_t* smem_dst)
  {
    uint32_t smem_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_dst));
    asm volatile ("stmatrix.sync.aligned.x4.m8n8.shared.b16 [%0], {%1, %2, %3, %4};\n"
        :: "r"(smem_int_ptr),
          "r"(src0), "r"(src1), "r"(src2), "r"(src3));
  }
};


__device__ void print_matrix(const int16_t matrix[16][16]) {
    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 16; j++) {
            printf("%03d ", matrix[i][j]);
            if (j == 7) {
                printf("| ");
            }
        }
        printf("\n");
        if (i == 7) {
            printf("-----------------------------------------------------------------\n");
        }
    }
    printf("\n");
}


__global__ void ldmatrix_kernel() {
    __shared__  int16_t smem_a[16][16];
    for (int i = threadIdx.x; i < 16 * 16; i += 32) {
        int row = i / 16;
        int col = i % 16;
        smem_a[row][col] = i;
    }

    __shared__ int16_t smem_b[16][16];

    __syncthreads();



    if (threadIdx.x == 0) {
        printf("Shared memory matrix A:\n");
        print_matrix(smem_a);
    }


    uint32_t a_regs[4];

    int row = threadIdx.x % 16;
    int col = threadIdx.x / 16 * 8;
    unsigned int *smem_int_ptr = reinterpret_cast<unsigned int *>(__cvta_generic_to_shared(&smem_a[row][col]));
    SM75_U16x8_LDSM::copy(smem_int_ptr, a_regs[0], a_regs[1], a_regs[2], a_regs[3]);

    for (int i = 0; i < 32; i++) {
        if (threadIdx.x == i) {
            uint16_t *a16 = reinterpret_cast<uint16_t *>(a_regs);
            printf("Thread %2d loaded: %03d %03d %03d %03d %03d %03d %03d %03d\n", threadIdx.x,
                    a16[0], a16[1], a16[2], a16[3], a16[4], a16[5], a16[6], a16[7]);
        }
    }

    SM90_U16x8_STSM::copy(a_regs[0], a_regs[1], a_regs[2], a_regs[3], reinterpret_cast<uint32_t*>(&smem_b[row][col]));

    __syncthreads();

    if (threadIdx.x == 0) {
        printf("Shared memory matrix B:\n");
        print_matrix(smem_b);
    }
}

int main() {
    dim3 block(32);
    dim3 grid(1);
    ldmatrix_kernel<<<grid, block>>>();
    cudaDeviceSynchronize();
    return 0;
}


