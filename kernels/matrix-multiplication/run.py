import ctypes
import torch
from common.runner import RunnerBase

class Runner(RunnerBase):

    def signature(self) -> dict:
        return {
            "solve": [
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
            ]
        }

    def build_test_cases(self):
        dtype = torch.float32
        cases = []
        for M, N, K in [(64, 128, 256), (128, 256, 512), (256, 512, 1024), (8192, 6144, 4096)]:
            cases.append({
                "A": torch.empty(M, N, device="cuda", dtype=dtype).uniform_(-10.0, 100.0),
                "B": torch.empty(N, K, device="cuda", dtype=dtype).uniform_(-10.0, 100.0),
                "C": torch.empty(M, K, device="cuda", dtype=dtype),
                "M": M,
                "N": N,
                "K": K,
            })
        return cases

    def check(self, A, B, C, **kwargs):
        assert torch.allclose(C, A @ B, atol=1e-4, rtol=1e-4)


if __name__ == "__main__":
    Runner()()
