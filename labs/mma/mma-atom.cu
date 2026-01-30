#include <cassert>
#include <cmath>
#include <cstdio>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cute/atom/mma_traits.hpp>


struct SM80_16x8x16_F32F16F16F32_TN
{
    using DRegisters = float[4];
    using ARegisters = uint32_t[4];
    using BRegisters = uint32_t[2];
    using CRegisters = float[4];

    __device__ static void
    fma(float         & d0, float         & d1, float         & d2, float         & d3,
        uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
        uint32_t const& b0, uint32_t const& b1,
        float const   & c0, float const   & c1, float const   & c2, float const   & c3)
    {
        asm volatile(
          "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
          "{%0,  %1,  %2,  %3},"
          "{%4,  %5,  %6,  %7},"
          "{%8,  %9},"
          "{%10, %11, %12, %13};\n"
          : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
          :  "r"(a0),  "r"(a1),  "r"(a2),  "r"(a3),
             "r"(b0),  "r"(b1),
             "f"(c0),  "f"(c1),  "f"(c2),  "f"(c3));
    }
};


__global__ void mma_16x8x16_f32f16f16f32_kernel(const half* a, const half* b, float* c) {
    __shared__  half smem_a[16][16];
    __shared__  half smem_b[16][8];
    __shared__ float smem_c[16][8];

    int lane = threadIdx.x;
    for (int i = lane; i < 16 * 16; i += 32) {
        smem_a[i / 16][i % 16] = a[i];
    }
    for (int i = lane; i < 16 * 8; i += 32) {
        smem_b[i / 8][i % 8] = b[i];
    }
    __syncthreads();
    

    uint32_t a_regs[4];
    uint32_t b_regs[2];
    float c_regs[4] = {0.0f};

    int a_row = lane / 4;
    int a_col = lane % 4 * 2;
    a_regs[0] = *reinterpret_cast<uint32_t*>(&smem_a[a_row][a_col]);
    a_regs[1] = *reinterpret_cast<uint32_t*>(&smem_a[a_row + 8][a_col]);
    a_regs[2] = *reinterpret_cast<uint32_t*>(&smem_a[a_row][a_col + 8]);
    a_regs[3] = *reinterpret_cast<uint32_t*>(&smem_a[a_row + 8][a_col + 8]);
    
    
    int b_row = (lane % 4) * 2;
    int b_col = lane / 4;
    half2 b0 = {smem_b[b_row][b_col], smem_b[b_row + 1][b_col]};
    half2 b1 = {smem_b[b_row + 8][b_col], smem_b[b_row + 9][b_col]};
    b_regs[0] = *reinterpret_cast<uint32_t*>(&b0);
    b_regs[1] = *reinterpret_cast<uint32_t*>(&b1);


    // Perform MMA operation
    SM80_16x8x16_F32F16F16F32_TN::fma(
        c_regs[0], c_regs[1], c_regs[2], c_regs[3],
        a_regs[0], a_regs[1], a_regs[2], a_regs[3],
        b_regs[0], b_regs[1],
        c_regs[0], c_regs[1], c_regs[2], c_regs[3]
    );

    int c_row = lane / 4;
    int c_col = (lane % 4) * 2;
    smem_c[c_row][c_col] = c_regs[0];
    smem_c[c_row][c_col + 1] = c_regs[1];
    smem_c[c_row + 8][c_col] = c_regs[2];
    smem_c[c_row + 8][c_col + 1] = c_regs[3];

    __syncthreads();
    for (int i = lane; i < 16 * 8; i += 32) {
        c[i] = smem_c[i / 8][i % 8];
    }
}

extern "C" void gemm_m16n8k16(const half* a, const half* b, float* c, int M, int N, int K) {
    assert(M == 16 && N == 8 && K == 16);
    dim3 block(32);
    dim3 grid(1);
    mma_16x8x16_f32f16f16f32_kernel<<<grid, block>>>(a, b, c);
}
