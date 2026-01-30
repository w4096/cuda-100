#include <cute/tensor.hpp>

using namespace cute;

__global__ void copy_kernel(int* dst, const int* src, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = 0; i < N; i += gridDim.x * blockDim.x) {
        size_t index = idx + i;
        if (index < N) {
            dst[index] = src[index];
        }
    }
}


__global__ void copy_0(const int* src, size_t N) {
    __shared__ int smem[512];
    const int *data = src + blockIdx.x * 512;

    for (int i = 0; i < N; i += blockDim.x) {
        size_t index = i + threadIdx.x;
        if (index < N) {
            smem[index] = data[index];
        }
    }

    __syncthreads();

}


__global__ void copy_1(const int* src, size_t N) {
    int A[24];

    Tensor a = make_tensor(&A[0], make_layout(make_shape(4,6)));
    // ... fill A ...


    int B[24];
    Tensor b = make_tensor(&B[0], make_layout(make_shape(24)));

    // Copy from a to b
    copy(a, b);


}


__global__ void copy_g2s_s2g(int* dst, const int* src, size_t N) {
    __shared__ int smem[512];

    Layout layout = make_layout(make_shape(blockDim.x, 512 / blockDim.x), make_stride(1, blockDim.x));

    Tensor global_src_tensor = make_tensor(make_gmem_ptr<int>(src + blockIdx.x * 512), layout);
    Tensor gloabl_dst_tensor = make_tensor(make_gmem_ptr<int>(dst + blockIdx.x * 512), layout);
    Tensor smem_tensor = make_tensor(make_smem_ptr<int>(smem), layout);

    Tensor thread_src = global_src_tensor(threadIdx.x, _);
    Tensor thread_smem = smem_tensor(threadIdx.x, _);
    Tensor thread_dst = gloabl_dst_tensor(threadIdx.x, _);

    if (cute::thread0()) {
        print(global_src_tensor.layout());
        print(thread_src.layout());
    }

    copy(thread_src, thread_smem);
    
    __syncthreads();

    copy(thread_smem, thread_dst);
}

template <int kTileM, int kTileN>
__global__ void copy_g2s_s2g_2(int* dst, const int* input, size_t N) {
  int tid = threadIdx.x;
  
  __shared__ int shm[kTileM * kTileN];
  
  Tensor g_input_tile = make_tensor(make_gmem_ptr((int *)input), 
                               make_shape(Int<kTileM>{}, Int<kTileN>{}),
                               make_stride(Int<kTileN>{}, Int<1>{})); 
                               // (kTileM, kTileN)
                               
  Tensor s_tensor = make_tensor(make_smem_ptr((int *)shm), 
                               make_shape(Int<kTileM>{}, Int<kTileN>{}),
                               make_stride(Int<kTileN>{}, Int<1>{}));
                               // (kTileM, kTileN)
  
  using g2s_copy_op = UniversalCopy<int>;
  using g2s_copy_traits = Copy_Traits<g2s_copy_op>;
  using g2s_copy_atom = Copy_Atom<g2s_copy_traits, int>;

  Layout thr_layout = make_layout(make_shape(Int<4>{}, Int<8>{}),
                                  make_stride(Int<8>{}, Int<1>{}));
  Layout val_layout = make_layout(make_shape(Int<2>{}));

  auto tiled_copy_g2s = make_tiled_copy(g2s_copy_atom{}, thr_layout, val_layout);

  if (thread0()) {
    print(tiled_copy_g2s);
    print_latex(tiled_copy_g2s);
    printf("\n");
  }

  auto thr_copy_g2s = tiled_copy_g2s.get_slice(tid);
  auto tgA_g2s = thr_copy_g2s.partition_S(g_input_tile);
  auto tsA_g2s = thr_copy_g2s.partition_D(s_tensor);

  if (thread0()) {
    print(thr_copy_g2s);
    print(tgA_g2s); printf("\n");
    print(tsA_g2s); printf("\n");
  }

  copy(tiled_copy_g2s, tgA_g2s, tsA_g2s);
  
  __syncthreads();

  for (int i = 0; i < N; i += blockDim.x) {
      size_t index = i + tid;
      if (index < N) {
          dst[blockIdx.x * kTileM * kTileN + index] = shm[index];
      }
  }
}

void copy_example() {
    using namespace cute;

    std::array<int, 16> data_a;
    for (int i = 0; i < 16; i++) {
        data_a[i] = i;
    }

    Layout<Shape<_4, _4>, Stride<_4, _1>> layout;
    Tensor A = make_tensor(make_gmem_ptr<int>(data_a.data()), layout);
    print_tensor(A);

    Tensor B = make_tensor<int>(layout);

    copy(A, B);

    print_tensor(B);
}



int main(int argc, char** argv) {
    int *src;
    int *dst;
    cudaMallocManaged(&src, 1024 * sizeof(int));
    cudaMallocManaged(&dst, 1024 * sizeof(int));
    for (int i = 0; i < 1024; i++) {
        src[i] = i;
        dst[i] = 0;
    }
    
    // copy_g2s_s2g<<<2, 32>>>(dst, src, 1024);
    copy_g2s_s2g_2<8, 64><<<1, 32>>>(dst, src, 1024);
    cudaDeviceSynchronize();

    Tensor dst_tensor = make_tensor(make_gmem_ptr<int>(dst), make_layout(make_shape(Int<8>{}, Int<64>{}), make_stride(Int<64>{}, Int<1>{})));
    print_tensor(dst_tensor);
    printf("\n");
}
