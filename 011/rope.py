import torch
import torch.nn as nn

torch.ops.load_library("./build/librope.so")

def rotary_embedding(
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None,
        head_size: int,
        cos_sin_cache: torch.Tensor,
        neox_style: bool = False,
) -> None:
    torch.ops._C.rotary_embedding(positions, query, key, head_size, cos_sin_cache, neox_style)


class RotaryEmbedding(nn.Module):
    def __init__(
            self,
            head_size: int,
            rotary_dim: int,
            max_position_embeddings: int,
            base: float,
            dtype: torch.dtype
    ) -> None:
        super().__init__()
        assert head_size == rotary_dim
        self.head_size = head_size
        self.rotary_dim = rotary_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.dtype = dtype

        cache = self._compute_cos_sin_cache()
        self.cos_sin_cache: torch.Tensor
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    def _compute_inv_freq(self, base: float) -> torch.Tensor:
        inv_freq = 1.0 / (base ** (torch.arange(0, self.rotary_dim, 2, dtype=torch.float) / self.rotary_dim))
        return inv_freq

    def _compute_cos_sin_cache(self) -> torch.Tensor:
        inv_freq = self._compute_inv_freq(self.base)
        t = torch.arange(self.max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1)
        return cache

    @torch.compile
    def forward_torch(
            self,
            positions: torch.Tensor,
            query: torch.Tensor,
            key: torch.Tensor | None = None,
    ):
        assert positions.size(0) == query.size(0)
        assert positions.size(0) == key.size(0)

        cos_sin = self.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        if query.dim() == 3:
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)
        if query.dim() == 4:
            cos = cos.unsqueeze(1).unsqueeze(1)
            sin = sin.unsqueeze(1).unsqueeze(1)

        def apply_rotary_embedding(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
            x1, x2 = torch.chunk(x.float(), 2, dim=-1)
            y1 = x1 * cos - x2 * sin
            y2 = x2 * cos + x1 * sin
            return torch.cat((y1, y2), dim=-1).to(x.dtype)

        query = apply_rotary_embedding(query, cos, sin)
        if key is not None:
            key = apply_rotary_embedding(key, cos, sin)
        return query, key

    def forward_cuda(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        rotary_embedding(positions, query, key, self.head_size, self.cos_sin_cache, neox_style=True)
        return query, key
