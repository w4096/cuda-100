#include <cassert>
#include <cmath>
#include <cstdio>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <random>

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


template<int kBlockM, int kBlockN, int kBlockK, int kWarps>
__global__ void mma_gemm_kernel(const half* a, const half* b, float* c, int M, int N, int K, float alpha, float beta) {
    __shared__  half smem_a[kBlockM][kBlockK];
    __shared__  half smem_b[kBlockK][kBlockN];
    __shared__ float smem_c[kBlockM][kBlockN];

    const int wrap_id = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;

    // initialize shared memory C
    for (int i = 0; i < kBlockM * kBlockN; i += blockDim.x * blockDim.y) {
        int idx = i + threadIdx.x;
        int row = idx / kBlockN;
        int col = idx % kBlockN;
        smem_c[row][col] = 0.0f;
    }
    __syncthreads();

    for (int k = 0; k < K; k += kBlockK) {
        // load A and B to shared memory
        for (int i = 0; i < kBlockM * kBlockK; i += blockDim.x * blockDim.y) {
            int idx = i + threadIdx.x;
            int row = idx / kBlockK;
            int col = idx % kBlockK;

            int a_row = blockIdx.y * kBlockM + row;
            int a_col = k + col;
            if (a_row < M && a_col < K) {
                smem_a[row][col] = a[a_row * K + a_col];
            } else {
                smem_a[row][col] = __float2half(0.0f);
            }
        }

        for (int i = 0; i < kBlockK * kBlockN; i += blockDim.x * blockDim.y) {
            int idx = i + threadIdx.x;
            int row = idx / kBlockN;
            int col = idx % kBlockN;

            int b_row = k + row;
            int b_col = blockIdx.x * kBlockN + col;
            if (b_row < K && b_col < N) {
                smem_b[row][col] = b[b_row * N + b_col];
            } else {
                smem_b[row][col] = __float2half(0.0f);
            }
        }

        __syncthreads();

        // compute MMA
        constexpr int m_fragments = kBlockM / 16;
        constexpr int n_fragments = kBlockN / 8;

        // loop over m x n small 16x8 MMA tiles
        for (int idx = wrap_id; idx < m_fragments * n_fragments; idx += kWarps) {
            const int m = idx / n_fragments;
            const int n = idx % n_fragments;

            // the left top corner of C tile in shared memory
            const int row = m * 16;
            const int col = n * 8;

            float c_regs[4] = {0.0f};

            constexpr int k_fragments = kBlockK / 16;
            for (int l = 0; l < k_fragments; ++l) {
                // load A and B from shared memory
                uint32_t a_regs[4];
                uint32_t b_regs[2];

                // the address within 16x16 tile of A matrix
                int a_row = row + (lane % 16);
                int a_col = l * 16 + (lane / 16) * 8;

                const auto a_addr = reinterpret_cast<uint32_t*>(&smem_a[a_row][a_col]);
                SM75_U16x8_LDSM::copy(a_addr, a_regs[0], a_regs[1], a_regs[2], a_regs[3]);

                // the address within 16x8 tile of B matrix
                int b_row = l * 16 + lane;
                int b_col = col;
                const auto b_addr = reinterpret_cast<uint32_t*>(&smem_b[b_row][b_col]);
                SM75_U16x4_LDSM_T::copy(b_addr, b_regs[0], b_regs[1]);

                // Perform MMA operation
                SM80_16x8x16_F32F16F16F32_TN::fma(
                    c_regs[0], c_regs[1], c_regs[2], c_regs[3],
                    a_regs[0], a_regs[1], a_regs[2], a_regs[3],
                    b_regs[0], b_regs[1],
                    c_regs[0], c_regs[1], c_regs[2], c_regs[3]
                );
            }

            // store C registers to shared memory
            const int c_col = col + (lane % 4) * 2;
            smem_c[row + lane / 4][c_col + 0] += c_regs[0];
            smem_c[row + lane / 4][c_col + 1] += c_regs[1];
            smem_c[row + lane / 4 + 8][c_col + 0] += c_regs[2];
            smem_c[row + lane / 4 + 8][c_col + 1] += c_regs[3];
        }
    }

    __syncthreads();

    // write back C from shared memory to global memory
    for (int i = 0; i < kBlockM * kBlockN; i += blockDim.x * blockDim.y) {
        int idx = i + threadIdx.x;
        int row = idx / kBlockN;
        int col = idx % kBlockN;
        int c_row = blockIdx.y * kBlockM + row;
        int c_col = blockIdx.x * kBlockN + col;
        if (c_row < M && c_col < N) {
            float c_value = smem_c[row][col];
            c[c_row * N + c_col] = alpha * c_value + beta * c[c_row * N + c_col];
        }
    }
}

extern "C" void gemm(const half* a, const half* b, float* c, int M, int N, int K) {
    constexpr int kBlockM = 64;
    constexpr int kBlockN = 64;
    constexpr int kBlockK = 16;
    
    constexpr int kWarps = 8;

    dim3 grid((N + kBlockN - 1) / kBlockN, (M + kBlockM - 1) / kBlockM);
    dim3 block(32 * kWarps);

    mma_gemm_kernel<kBlockM, kBlockN, kBlockK, kWarps><<<grid, block>>>(a, b, c, M, N, K, 1.0, 0.0);
    cudaDeviceSynchronize();
}

