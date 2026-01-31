import ctypes
import torch
from common.runner import RunnerBase

class Runner(RunnerBase):
    def signature(self) -> dict:
        return {
            "default": [
                ctypes.POINTER(ctypes.c_uint16),
                ctypes.POINTER(ctypes.c_uint16),
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int
            ]
        }

    def build_test_cases(self):
        M, N, K = 16, 8, 16
        yield {
            "function": "gemm_m16n8k16",
            "argv": {
                "A": torch.randn(M, K, device="cuda", dtype=torch.half),
                "B": torch.randn(K, N, device="cuda", dtype=torch.half),
                "C": torch.zeros(M, N, device="cuda", dtype=torch.float),
                "M": M,
                "N": N,
                "K": K
            }
        }
        
        M, N, K = 64, 64, 256
        yield {
            "function": "gemm",
            "argv": {
                "A": torch.randn(M, K, device="cuda", dtype=torch.half),
                "B": torch.randn(K, N, device="cuda", dtype=torch.half),
                "C": torch.zeros(M, N, device="cuda", dtype=torch.float),
                "M": M,
                "N": N,
                "K": K
            }
        }

    def check(self, A, B, C, **kwargs):
        D = torch.matmul(A.to(torch.float), B.to(torch.float))
        assert torch.allclose(C, D, atol=1e-4), "Result mismatch"


if __name__ == "__main__":
    Runner()()
