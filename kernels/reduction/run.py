import ctypes
import torch
from common.runner import RunnerBase

class Runner(RunnerBase):

    def signature(self) -> dict:
        return {
            "solve": [
                ctypes.POINTER(ctypes.c_int32),
                ctypes.POINTER(ctypes.c_int32),
                ctypes.c_int,
            ]
        }

    def build_test_cases(self):
        dtype = torch.float32
        N = 409600
        return [
            {
                "input": torch.randn(N, device="cuda", dtype=dtype),
                "output": torch.empty(1, device="cuda", dtype=dtype),
                "N": N,
            },
        ]

    def check(self, case):
        assert torch.allclose(
            case["output"],
            torch.sum(case["input"]).unsqueeze(0)
        )



if __name__ == "__main__":
    Runner()()
    
