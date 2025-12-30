#include <cuda/std/numeric>
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


template<typename T, int ELEMENTS_PER_THREAD>
__global__ void merge(T* out, const T* input, int width, int N) {
    /* merge two sorted subarrays of size 'width'
     * each thread merges ELEMENTS_PER_THREAD elements */

    // calculate the left index of the two subarrays to be merged

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // each thread handles ELEMENTS_PER_THREAD elements
    int idx = tid * ELEMENTS_PER_THREAD;

    if (idx >= N) {
        return;
    }

    // the left index of the two subarrays
    int left = idx / (2 * width) * (2 * width);

    int a_len = min(width, N - left);
    int b_len = min(width, N - (left + a_len));

    // a and b are the starting pointers of the two subarrays to be merged
    const T *a = input + left;
    const T *b = a + a_len;

    // output pointer
    T *o = out + left;
    int o_idx = idx - left;


    int i = co_rank<T>(o_idx, a, a_len, b, b_len);
    int j = o_idx - i;


    for (int e = 0; e < ELEMENTS_PER_THREAD; e++) {
        if (left + o_idx + e >= N) {
            return;
        }
        T val;
        if (i >= a_len) {
            val = b[j++];
        } else if (j >= b_len) {
            val = a[i++];
        } else if (a[i] <= b[j]) {
            val = a[i++];
        } else {
            val = b[j++];
        }
        o[o_idx + e] = val;
    }
}

template <typename T>
__device__ void swap(T &a, T &b) {
    T temp = a;
    a = b;
    b = temp;
}

template<typename T, int ELEMENTS_PER_THREAD>
__global__  void thread_sort(T *data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    idx *= ELEMENTS_PER_THREAD;
    if (idx >= N) {
        return;
    }

    T local_data[ELEMENTS_PER_THREAD];
    // load data to local array
    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        int index = idx + i;
        if (index < N) {
            local_data[i] = data[index];
        } else {
            local_data[i] = cuda::std::numeric_limits<T>::max();
        }
    }

    // select sort for small local array
    for (int i = 0; i < ELEMENTS_PER_THREAD - 1; i++) {
        for (int j = i + 1; j < ELEMENTS_PER_THREAD; j++) {
            if (local_data[j] < local_data[i]) {
                swap(local_data[i], local_data[j]);
            }
        }
    }

    // store sorted local array back to global memory
    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        int index = idx + i;
        if (index < N) {
            data[index] = local_data[i];
        }
    }
}


static unsigned int cdiv(unsigned int a, unsigned int b) {
    return (a + b - 1) / b;
}

extern "C" void merge_sort(float* data, int N) {
    constexpr int ELEMENTS_PER_THREAD = 16;

    float* tmp;
    cudaMalloc(&tmp, N * sizeof(float));
    int threadsPerBlock = 256;
    int blocks = (cdiv(N, ELEMENTS_PER_THREAD) + threadsPerBlock - 1) / threadsPerBlock;


    /* Each thread sorts a small block of data
     * After this step, every ELEMENTS_PER_THREAD elements are sorted */
    thread_sort<float, ELEMENTS_PER_THREAD><<<blocks, threadsPerBlock>>>(data, N);

    // Iteratively merge sorted blocks
    float* input = data;
    float* out = tmp;
    for (int width = ELEMENTS_PER_THREAD; width < N; width *= 2) {
        merge<float, ELEMENTS_PER_THREAD><<<blocks, threadsPerBlock>>>(out, input, width, N);
        std::swap(input, out);
    }

    // If the final result is in tmp, copy it back to data
    if (input != data) {
        cudaMemcpy(data, input, N *sizeof(float), cudaMemcpyDeviceToDevice);
    }

    cudaFree(tmp);
    cudaDeviceSynchronize();
}
