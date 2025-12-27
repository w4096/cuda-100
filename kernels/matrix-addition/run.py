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
        N = 4096

        return [
            {
                "A": torch.empty(N, N, device="cuda", dtype=dtype).uniform_(-1000.0, 1000.0),
                "B": torch.empty(N, N, device="cuda", dtype=dtype).uniform_(-1000.0, 1000.0),
                "C": torch.zeros(N, N, device="cuda", dtype=dtype),
                "N": N,
            }
        ]

    def check(self, case):
        assert torch.allclose(
            case["C"],
            case["A"] + case["B"]
        )


if __name__ == "__main__":
    Runner()()
