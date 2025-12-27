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

    def check(self, case):
        A = case["A"]
        B = case["B"]
        alpha = case["alpha"]
        beta = case["beta"]
        M = case["M"]
        N = case["N"]

        print(case["C"].sum(dim=-1))

        C = alpha * (A @ B) + beta * torch.ones(M, N, device="cuda", dtype=A.dtype)

        print(C.sum(dim=-1))
        assert torch.allclose(case["C"], C, atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    Runner()()

