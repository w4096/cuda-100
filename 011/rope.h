#pragma once

#include <torch/all.h>

void rotary_embedding(const torch::Tensor& positions, const torch::Tensor& query,
                      const std::optional<torch::Tensor>& key, int64_t head_size, const torch::Tensor& cos_sin_cache,
                      bool NeoX);
