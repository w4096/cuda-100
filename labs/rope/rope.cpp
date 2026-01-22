#include <torch/all.h>
#include <torch/library.h>

#include "rope.h"

TORCH_LIBRARY(_C, ops) {
    ops.def(
    "rotary_embedding(Tensor positions, Tensor! query,"
    "                 Tensor!? key, int head_size,"
    "                 Tensor cos_sin_cache, bool NeoX) -> ()");
    ops.impl("rotary_embedding", torch::kCUDA, &rotary_embedding);
}
