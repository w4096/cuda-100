import torch
from rope import RotaryEmbedding


def benckmark():
    head_size = 128
    rotary_dim = 128
    max_position_embeddings = 4096
    base = 10000.0
    dtype = torch.float
    seqlen = 4096
    num_heads = 32
    num_kv_heads = 16

    rotary_emb = RotaryEmbedding(head_size, rotary_dim, max_position_embeddings, base, dtype).cuda()

    positions = torch.arange(0, seqlen, dtype=torch.long).cuda()
    query = torch.randn(seqlen, num_heads, head_size, dtype=dtype).cuda()
    key = torch.randn(seqlen, num_kv_heads, head_size, dtype=dtype).cuda()
    query_copied = query.clone()
    key_copied = key.clone()

    # warm up
    rotary_emb.forward_torch(positions, query_copied, key_copied)
    rotary_emb.forward_cuda(positions, query, key)

    print("Profiling torch implementation of RoPE")
    with torch.profiler.profile() as prof:
        query_rotated_torch, key_rotated_torch = rotary_emb.forward_torch(positions, query_copied, key_copied)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    print("Profiling CUDA implementation of RoPE")
    with torch.profiler.profile() as prof:
        query_rotated, key_rotated = rotary_emb.forward_cuda(positions, query, key)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))


def test():
    head_size = 64
    rotary_dim = 64
    max_position_embeddings = 512
    base = 10000.0
    dtype = torch.float
    seqlen = 64
    num_heads = 16
    num_kv_heads = 8

    rotary_emb = RotaryEmbedding(head_size, rotary_dim, max_position_embeddings, base, dtype).cuda()

    positions = torch.arange(0, seqlen, dtype=torch.long).cuda()
    query = torch.randn(seqlen, num_heads, head_size, dtype=dtype).cuda()
    key = torch.randn(seqlen, num_kv_heads, head_size, dtype=dtype).cuda()
    query_copied = query.clone()
    key_copied = key.clone()

    query_rotated, key_rotated = rotary_emb.forward_cuda(positions, query, key)
    assert query_rotated.shape == query.shape
    assert key_rotated.shape == key.shape

    query_rotated_torch, key_rotated_torch = rotary_emb.forward_torch(positions, query_copied, key_copied)
    assert torch.allclose(query_rotated, query_rotated_torch, atol=1e-5)
    assert torch.allclose(key_rotated, key_rotated_torch, atol=1e-5)


if __name__ == "__main__":
    benckmark()