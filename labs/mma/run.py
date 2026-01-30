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
                "A": torch.randn(M, K, device="cuda", dtype=torch.half).uniform_(-1, 1),
                "B": torch.randn(K, N, device="cuda", dtype=torch.half).uniform_(-1, 1),
                "C": torch.zeros(M, N, device="cuda", dtype=torch.float),
                "M": M,
                "N": N,
                "K": K
            }
        }
        
        
        M, N, K = 256, 256, 256
        yield {
            "function": "gemm",
            "argv": {
                "A": torch.randn(M, K, device="cuda", dtype=torch.half).uniform_(-1, 1),
                "B": torch.randn(K, N, device="cuda", dtype=torch.half).uniform_(-1, 1),
                "C": torch.zeros(M, N, device="cuda", dtype=torch.float),
                "M": M,
                "N": N,
                "K": K
            }
        }

    def check(self, A, B, C, **kwargs):
        C_ref = torch.matmul(A.to(torch.float32), B.to(torch.float32))
        assert torch.allclose(C, C_ref, atol=1e-5), "Result mismatch in gemm function"


if __name__ == "__main__":
    Runner()()
