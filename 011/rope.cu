#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>

template<typename scalar_t, bool NeoX>
__device__ void apply_token_rotary_embedding(scalar_t* __restrict__ arr, const scalar_t* __restrict__ cos_ptr,
                                             const scalar_t* __restrict__ sin_ptr, int rotary_offset, int embed_dim) {
    int x1_index, x2_index;
    if constexpr (NeoX) {
        x1_index = rotary_offset;
        x2_index = rotary_offset + embed_dim;
    } else {
        x1_index = 2 * rotary_offset;
        x2_index = 2 * rotary_offset + 1;
    }
    scalar_t cos = cos_ptr[rotary_offset];
    scalar_t sin = sin_ptr[rotary_offset];

    const scalar_t x1 = arr[x1_index];
    const scalar_t x2 = arr[x2_index];
    arr[x1_index] = x1 * cos - x2 * sin;
    arr[x2_index] = x2 * cos + x1 * sin;
}

template<typename scalar_t, bool NeoX>
__device__ void apply_rotary_embedding(scalar_t* __restrict__ input, const scalar_t* cache_ptr, const int head_size,
                                       const int num_heads, const int rotary_dim, const int token_idx,
                                       const int64_t token_stride, const int64_t head_stride) {
    const int embed_dim = rotary_dim / 2;
    const scalar_t* cos_ptr = cache_ptr;
    const scalar_t* sin_ptr = cache_ptr + embed_dim;

    const int nd = num_heads * embed_dim;
    for (int i = threadIdx.x; i < nd; i += blockDim.x) {
        const int head_idx = i / embed_dim;
        const int64_t token_head = token_idx * token_stride + head_idx * head_stride;
        const int offset = i % embed_dim;
        apply_token_rotary_embedding<scalar_t, NeoX>(input + token_head, cos_ptr, sin_ptr, offset, embed_dim);
    }
}

/**
 * @tparam scalar_t the data type of query, key and cos_sin_cache
 * @param positions shape: [batch_size, seq_len] or [num_tokens]
 * @param query  shape: [num_tokens, num_heads, head_size]
 * @param key  shape: nullptr or [num_tokens, num_kv_heads, head_size]
 * @param cos_sin_cache  shape: [max_position, 2, rot_dim // 2]
 * @param rotary_dim  the rotary embedding dimension
 * @param query_stride  the stride between two tokens in query
 * @param key_stride   the stride between two tokens in key
 * @param head_stride  the stride between two heads in query and key
 * @param num_heads  the number of heads in query
 * @param num_kv_heads  the number of heads in key
 * @param head_size  the head size of query and key
 */
template<typename scalar_t, bool NeoX>
__global__ void rotary_embedding_kernel(const int64_t* __restrict__ positions, scalar_t* __restrict__ query,
                                        scalar_t* __restrict__ key, const scalar_t* __restrict__ cos_sin_cache,
                                        const int rotary_dim, const int64_t query_stride, const int64_t key_stride,
                                        const int64_t head_stride, const int num_heads, const int num_kv_heads,
                                        const int head_size) {
    // Each thread block is responsible for one token.
    const int token_idx = blockIdx.x;
    int64_t pos = positions[token_idx];
    // pointer to the cosine and sine cache for this position, shape: [rotary_dim]
    const scalar_t* cos_sin_cache_ptr = cos_sin_cache + pos * rotary_dim;

    apply_rotary_embedding<scalar_t, NeoX>(query, cos_sin_cache_ptr, head_size, num_heads, rotary_dim, token_idx,
                                           query_stride, head_stride);

    if (key != nullptr) {
        apply_rotary_embedding<scalar_t, NeoX>(key, cos_sin_cache_ptr, head_size, num_kv_heads, rotary_dim, token_idx,
                                               key_stride, head_stride);
    }
}

/**
 * @param positions the positions of each token, shape: [batch_size, seq_len] or [num_tokens]
 * @param query the query tensor, the head_size dimension must be the last dimension,
 *              shape: [batch_size, seq_len, num_heads * head_size] or
 *              [num_tokens, num_heads * head_size] or
 *              [batch_size, seq_len, num_heads, head_size] or
 *              [num_tokens, num_heads, head_size]
 * @param key the key tensor, can be null, the shape is similar to query, except num_heads is replaced by
 * num_kv_heads
 * @param head_size the head size of query and key
 * @param cos_sin_cache the precomputed cosine and sine cache, shape: [max_position, rotary_dim]
 * @param NeoX whether to use NeoX style rotary embedding
 */
void rotary_embedding(const torch::Tensor& positions, const torch::Tensor& query,
                      const std::optional<torch::Tensor>& key, int64_t head_size, const torch::Tensor& cos_sin_cache,
                      bool NeoX) {
    auto tokens = positions.numel();
    int positions_dim = positions.dim();

    TORCH_CHECK(positions_dim == 1 || positions_dim == 2,
                "the shape of positions must be [batch_size, seq_len] or [seq_len]");

    if (positions_dim == 1) {
        TORCH_CHECK(positions.size(0) == query.size(0), "query and positions must have the same number of tokens");
        if (key) {
            TORCH_CHECK(key.value().size(0) == positions.size(0),
                        "key and positions must have the same number of tokens");
        }
    }
    if (positions_dim == 2) {
        TORCH_CHECK(positions.size(0) == query.size(0) && positions.size(1) == query.size(1),
                    "query and positions must have the same batch size and seq len");
        if (key) {
            TORCH_CHECK(key.value().size(0) == query.size(0) && key.value().size(1) == query.size(1),
                        "key and query must have the same batch size and seq len");
        }
    }

    auto query_hidden_size = query.numel() / tokens;
    auto num_heads = query_hidden_size / head_size;
    TORCH_CHECK(query_hidden_size % head_size == 0, "query hidden size must be divisible by head size");

    int key_hidden_size = key ? key.value().numel() / tokens : 0;
    int num_kv_heads = key ? key_hidden_size / head_size : num_heads;
    TORCH_CHECK(num_heads % num_kv_heads == 0, "num_heads must be divisible by num_kv_heads");

    int rotary_dim = cos_sin_cache.size(1);
    TORCH_CHECK(rotary_dim == head_size, "rotary_dim must be equal to head_size");
    /* if positions is 2D, which shape is [batch_size, seq_len], the seq dimension is at dim 1
     * if positions is 1D, which shape is [seq_len], the seq dimension is at dim 0 */
    int seq_dim_idx = positions_dim - 1;

    int64_t query_stride = query.stride(seq_dim_idx);
    int64_t key_stride = key ? key.value().stride(seq_dim_idx) : 0;

    int query_ndim = query.dim();
    /** if the query shape is [*, num_heads, head_size], use stride of last dim.
     * if the query shape is [*, num_heads * head_size], the head_stride is head_size */
    int64_t head_stride = (query_ndim == positions_dim + 2) ? query.stride(-2) : head_size;

    dim3 grid(tokens);
    dim3 block(std::min<int64_t>(num_heads * rotary_dim / 2, 512));
    const c10::cuda::OptionalCUDAGuard device_guard(query.device());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::BFloat16, at::ScalarType::Half, query.scalar_type(), "rotary_embedding", [&] {
            if (NeoX) {
                rotary_embedding_kernel<scalar_t, true><<<grid, block, 0, stream>>>(
                    positions.data_ptr<int64_t>(), query.data_ptr<scalar_t>(),
                    key ? key.value().data_ptr<scalar_t>() : nullptr, cos_sin_cache.data_ptr<scalar_t>(), rotary_dim,
                    query_stride, key_stride, head_stride, num_heads, num_kv_heads, head_size);
            } else {
                rotary_embedding_kernel<scalar_t, false><<<grid, block, 0, stream>>>(
                    positions.data_ptr<int64_t>(), query.data_ptr<scalar_t>(),
                    key ? key.value().data_ptr<scalar_t>() : nullptr, cos_sin_cache.data_ptr<scalar_t>(), rotary_dim,
                    query_stride, key_stride, head_stride, num_heads, num_kv_heads, head_size);
            }
        });
}
