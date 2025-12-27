import ctypes
import torch
import torch.nn.functional as F
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
                "input": torch.empty(N, device="cuda", dtype=dtype).uniform_(-32.0, 32.0),
                "output": torch.empty(N, device="cuda", dtype=dtype),
                "N": N,
            },
        ]

    def check(self, case):
        assert torch.allclose(
            case["output"],
            F.silu(case["input"])
        )



if __name__ == "__main__":
    Runner()()
    
