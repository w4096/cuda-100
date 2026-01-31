#include <cuda/barrier>
#include <cuda/ptx>

using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace ptx = cuda::ptx;

template <unsigned int BUFFER_SIZE>
__global__ void copy_kernel(void *dst, void *src, int N) {
    __shared__ alignas(16) int smem[BUFFER_SIZE];
    int offset = blockIdx.x * BUFFER_SIZE;

    #pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ barrier bar;
    if (threadIdx.x == 0) {
        // init barrier using thread 0
        init(&bar, blockDim.x);
        // issue fence to ensure barrier is visible to other threads
        ptx::fence_proxy_async(ptx::space_shared);
    }
    __syncthreads();

    int bytes = min(BUFFER_SIZE, max(0, N - offset));

    if (threadIdx.x == 0) {
        // issue async copy from global to shared memory
        cuda::memcpy_async(
            smem,
            static_cast<char*>(src) + offset,
            bytes,
            bar
        );
    }

    barrier::arrival_token token = bar.arrive();
    
    // wait for the copy to complete
    bar.wait(std::move(token));


    int tail = bytes % 16;
    bytes = bytes - tail;

    if (threadIdx.x == 0 && bytes > 0) {
        asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                     :
                     : "l"(__cvta_generic_to_global(static_cast<char*>(dst) + offset)),
                       "r"(static_cast<uint32_t>(__cvta_generic_to_shared(smem))),
                       "r"(bytes)
                     : "memory");

        asm volatile("cp.async.bulk.commit_group;" : : :);

        asm volatile("cp.async.bulk.wait_group %0;" : : "n"(0) : "memory");
    }

    if (tail > 0) {
        if (threadIdx.x < tail) {
            static_cast<char*>(dst)[offset + bytes + threadIdx.x] = 
                static_cast<char*>(src)[offset + bytes + threadIdx.x];
        
        }
    }
}

extern "C" void copy(void *dst, void *src, int N) {
    constexpr unsigned int BUFFER_SIZE = 1024;
    int blocks = (N + BUFFER_SIZE - 1) / BUFFER_SIZE;
    int threads = 32;
    copy_kernel<BUFFER_SIZE><<<blocks, threads>>>(dst, src, N);
}
