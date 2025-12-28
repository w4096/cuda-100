import ctypes
import torch
from common.runner import RunnerBase

class Runner(RunnerBase):

    def signature(self) -> dict:
        return {
            "default": [
                ctypes.POINTER(ctypes.c_uint16),
                ctypes.POINTER(ctypes.c_uint16),
                ctypes.POINTER(ctypes.c_uint16),
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_float,
                ctypes.c_float,
            ]
        }

    def build_test_cases(self):
        # for M, N, K in [(256, 512, 1024), (8192, 6144, 4096)]:
        for M, N, K in [(16, 8, 8)]:
            yield {
                "A": torch.empty(M, K, device="cuda", dtype=torch.float16).uniform_(-1.0, 1.0),
                "B": torch.empty(K, N, device="cuda", dtype=torch.float16).uniform_(-1.0, 1.0),
                "C": torch.ones(M, N, device="cuda", dtype=torch.float16),
                "M": M,
                "N": N,
                "K": K,
                "alpha": 1.0,
                "beta": 1.0,
            }

    def check(self, A, B, C, alpha, beta, M, N, **kwargs):
        expected = alpha * (A @ B) + beta * torch.ones(M, N, device="cuda", dtype=A.dtype)

        assert torch.allclose(C, expected, atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    Runner()()

