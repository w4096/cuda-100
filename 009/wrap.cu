#include <torch/types.h>
#include <torch/extension.h>
#include "softmax.cuh"

torch::Tensor softmax(torch::Tensor in) {
    const auto rows = in.size(0);
    const auto dim = in.size(1);

    auto out = torch::empty_like(in);

    if (dim <= 1024) {
        dim3 threads = 256;
        dim3 blocks = rows / (threads.x / 32);
        softmax_warp_reduce<<<blocks, threads>>>(in.data_ptr<float>(), out.data_ptr<float>(), rows, dim);
    } else {
        dim3 threads = 256;
        dim3 blocks = rows;
        softmax_block_reduce<<<blocks, threads>>>(in.data_ptr<float>(), out.data_ptr<float>(), rows, dim);
    }

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("softmax", torch::wrap_pybind_function(softmax), "softmax");
}