import ctypes
import torch
import sys
import os
from common.runner import RunnerBase

class Runner(RunnerBase):
    def signature(self) -> dict:
        return {
            "solve": [
                ctypes.POINTER(ctypes.c_int32),
                ctypes.POINTER(ctypes.c_int32),
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
            ]
        }

    def build_test_cases(self):
        dtype = torch.int32
        return [
            {
                "input": torch.randint(1, 32, (10000, 10000), device="cuda", dtype=dtype),
                "output": torch.empty(1, device="cuda", dtype=dtype),
                "N": 10000,
                "M": 10000,
                "K": 16,
            },
        ]

    def check(self, case):
        assert case["output"].item() == (case["input"] == case["K"]).sum().item()



if __name__ == "__main__":
    Runner()()
    
