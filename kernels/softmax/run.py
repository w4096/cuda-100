import ctypes
import torch
import sys
import os
from common.runner import RunnerBase

class Runner(RunnerBase):

    def signature(self) -> dict:
        return {
            "solve": [
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
                "input": torch.tensor([1000.0, 1.0, 2.0, 3.0], device="cuda", dtype=dtype),
                "output": torch.empty(4, device="cuda", dtype=dtype),
                "N": 4,
            },
            {
                "input": torch.empty(N, device="cuda", dtype=dtype).uniform_(-10.0, 10.0),
                "output": torch.empty(N, device="cuda", dtype=dtype),
                "N": N,
            },
        ]

    def check(self, input, output, **kwargs):
        assert torch.allclose(
            output,
            torch.softmax(input, dim=0),
            atol=1e-05,
            rtol=1e-05
        )



if __name__ == "__main__":
    Runner()()
    
