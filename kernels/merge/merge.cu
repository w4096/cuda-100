#include <cuda_runtime.h>

template<typename T>
int __device__ co_rank(int k, const T* a, int m, const T* b, int n) {
    int lo = max(0, k - n);
    int hi = min(k, m)+1;

    while (lo < hi) {
        int i = lo + (hi - lo) / 2;
        int j = k - i;
        if (i < m && j > 0 && a[i] < b[j - 1]) {
            lo = i + 1;
        } else {
            hi = i;
        }
    }
    return lo;
}

template <typename T, int ELEMENTS_PER_THREAD>
__global__ void merge(const T* a, int m, const T* b, int n, T* c) {
    int c_idx = (blockDim.x * blockIdx.x + threadIdx.x) * ELEMENTS_PER_THREAD;
    int len = m + n;
    if (c_idx >= len) {
        return;
    }

    int a_idx = co_rank<T>(c_idx, a, m, b, n);
    int b_idx = c_idx - a_idx;

    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        if (c_idx + i >= len) {
            return;
        }
        T val;
        if (a_idx >= m) {
            val = b[b_idx++];
        } else if (b_idx >= n) {
            val = a[a_idx++];
        } else if (a[a_idx] <= b[b_idx]) {
            val = a[a_idx++];
        } else {
            val = b[b_idx++];
        }
        c[c_idx + i] = val;
    }
}

template <typename T, int ELEMENTS_PER_THREAD>
__global__ void merge_block_local(const T* a, int m, const T* b, int n, T* c) {
    int block_c_idx = blockDim.x * blockIdx.x * ELEMENTS_PER_THREAD;
    int len = m + n;
    if (block_c_idx >= len) {
        return;
    }

    __shared__ T block_a_idx;

    if (threadIdx.x == 0) {
        block_a_idx = co_rank<T>(block_c_idx, a, m, b, n);
    }
    __syncthreads();

    int block_b_idx = block_c_idx - block_a_idx;
    a = a + block_a_idx;
    b = b + block_b_idx;
    m = max(m - block_a_idx, 0);
    n = max(n - block_b_idx, 0);

    int thread_c_idx = threadIdx.x * ELEMENTS_PER_THREAD;
    int a_idx = co_rank<T>(thread_c_idx, a, m, b, n);
    int b_idx = thread_c_idx - a_idx;

    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        int c_idx = block_c_idx + thread_c_idx + i;
        if (c_idx >= len) {
            return;
        }
        T val;
        if (a_idx >= m) {
            val = b[b_idx++];
        } else if (b_idx >= n) {
            val = a[a_idx++];
        } else if (a[a_idx] <= b[b_idx]) {
            val = a[a_idx++];
        } else {
            val = b[b_idx++];
        }
        c[c_idx] = val;
    }
}


extern "C" void solve(const int* a, int m, const int* b, int n, int* c) {
    int total_size = m + n;
    int threadsPerBlock = 256;
    constexpr int ELEMENTS_PER_THREAD = 16;
    int blocksPerGrid = (total_size + threadsPerBlock * ELEMENTS_PER_THREAD - 1) / (threadsPerBlock * ELEMENTS_PER_THREAD);

    merge<int, ELEMENTS_PER_THREAD><<<blocksPerGrid, threadsPerBlock>>>(a, m, b, n, c);
    cudaDeviceSynchronize();
}