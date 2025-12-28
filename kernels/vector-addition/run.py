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
            ]
        }

    def build_test_cases(self):
        dtype = torch.float32
        N = 409600

        return [
            {
                "A": torch.randn(N, device="cuda", dtype=dtype),
                "B": torch.randn(N, device="cuda", dtype=dtype),
                "C": torch.empty(N, device="cuda", dtype=dtype),
                "N": N,
            }
        ]

    def check(self, A, B, C, **kwargs):
        assert torch.allclose(C, A + B)


if __name__ == "__main__":
    Runner()()
