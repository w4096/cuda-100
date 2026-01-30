import ctypes
import torch
from common.runner import RunnerBase

class Runner(RunnerBase):
    def signature(self) -> dict:
        return {
            "default": [
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_float,
                ctypes.POINTER(ctypes.c_int16),
                ctypes.POINTER(ctypes.c_int16),
                ctypes.c_float,
                ctypes.POINTER(ctypes.c_float),
            ]
        }

    def build_test_cases(self):
        dtype = torch.half
        functions = [
            "gemm_cutlass",
            "gemm_naive",
            "gemm_tile_16x16", "gemm_tile_32x32",
            "gemm_tile_B16x16_T32x32", "gemm_tile_m64n64k16", "gemm_tile_m64n64k16_reg",
            "gemm_wmma_16x16x16", "gemm_wmma_16x16x16_smem",
            "gemm_wmma_16x16x16_smem_double_buffer",
        ]
        
        for M, N, K in [(4096, 4096, 2048)]:
            A = torch.empty(M, K, device="cuda", dtype=dtype).uniform_(-0.1, 0.1)
            B = torch.empty(K, N, device="cuda", dtype=dtype).uniform_(-0.1, 0.1)

            for func in functions:
                yield {
                    "function": func,
                    "argv": {
                        "M": M,
                        "N": N,
                        "K": K,
                        "alpha": 1.0,
                        "A": A,
                        "B": B,
                        "beta": 0.0,
                        "C": torch.zeros(M, N, device="cuda", dtype=torch.float),
                    }
                }

    def check(self, A, B, C, **kwargs):
        expect = torch.matmul(A.to(torch.float), B.to(torch.float))
        equal = torch.allclose(C, expect, atol=0.01, rtol=0.01)
        if not equal:
            print("abs max diff:", torch.max(torch.abs(C - expect)).item())
            print("max C:", torch.max(C).item(), "max expect:", torch.max(expect).item())
        assert equal


if __name__ == "__main__":
    Runner()()
