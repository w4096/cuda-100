#include <cute/layout.hpp>
#include <cute/tensor.hpp>

template <class TensorS, class TensorD, class Tiled_Copy>
__global__ void copy_kernel_vectorized(TensorS S, TensorD D, Tiled_Copy tiled_copy)
{
    using namespace cute;

    // Slice the tensors to obtain a view into each tile.
    Tensor tile_S = S(make_coord(_, _), blockIdx.x, blockIdx.y);  // (BlockShape_M, BlockShape_N)
    Tensor tile_D = D(make_coord(_, _), blockIdx.x, blockIdx.y);  // (BlockShape_M, BlockShape_N)

    // Construct a Tensor corresponding to each thread's slice.
    ThrCopy thr_copy = tiled_copy.get_thread_slice(threadIdx.x);

    Tensor thr_tile_S = thr_copy.partition_S(tile_S);             // (CopyOp, CopyM, CopyN)
    Tensor thr_tile_D = thr_copy.partition_D(tile_D);             // (CopyOp, CopyM, CopyN)

    // Construct a register-backed Tensor with the same shape as each thread's partition
    // Use make_fragment because the first mode is the instruction-local mode
    Tensor fragment = make_fragment_like(thr_tile_D);             // (CopyOp, CopyM, CopyN)

    // Copy from GMEM to RMEM and from RMEM to GMEM
    copy(tiled_copy, thr_tile_S, fragment);
    copy(tiled_copy, fragment, thr_tile_D);
}


int main() {
    using namespace cute;
    using Element = float;

    auto shape = make_shape(256, 512);

    Element* h_src = new Element[size(shape)];
    for (size_t i = 0; i < size(shape); ++i) {
        h_src[i] = static_cast<Element>(i);
    }

    Element* src;
    cudaMalloc(&src, sizeof(Element) * size(shape));
    cudaMemcpy(src, h_src, sizeof(Element) * size(shape), cudaMemcpyHostToDevice);

    Element* dst;
    cudaMalloc(&dst, sizeof(Element) * size(shape));


    Tensor tensor_S = make_tensor(make_gmem_ptr(src), make_layout(shape));
    Tensor tensor_D = make_tensor(make_gmem_ptr(dst), make_layout(shape));

    auto block_shape = make_shape(Int<128>{}, Int<64>{});

    Tensor tiled_tensor_S = tiled_divide(tensor_S, block_shape);      // ((M, N), m', n')
    Tensor tiled_tensor_D = tiled_divide(tensor_D, block_shape);      // ((M, N), m', n')

    Layout thr_layout = make_layout(make_shape(Int<32>{}, Int<8>{}));  // (32,8) -> thr_idx
    Layout val_layout = make_layout(make_shape(Int<4>{}, Int<1>{}));   // (4,1) -> val_idx

    using CopyOp = UniversalCopy<uint_byte_t<sizeof(Element) * size(val_layout)>>;

    using Atom = Copy_Atom<CopyOp, Element>;

    TiledCopy tiled_copy = make_tiled_copy(Atom{},             // Access strategy
                                         thr_layout,         // thread layout (e.g. 32x4 Col-Major)
                                         val_layout);


    dim3 gridDim (size<1>(tiled_tensor_D), size<2>(tiled_tensor_D));   // Grid shape corresponds to modes m' and n'
    dim3 blockDim(size(thr_layout));


    copy_kernel_vectorized<<< gridDim, blockDim >>>(
        tiled_tensor_S,
        tiled_tensor_D,
        tiled_copy);

}