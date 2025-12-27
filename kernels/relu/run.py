import ctypes
import torch
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
        return [
            {
                "input": torch.randn(10240, device='cuda', dtype=torch.float32),
                "output": torch.empty(10240, device='cuda', dtype=torch.float32),
                "N": 10240,
            },
            {
                "input": torch.empty(20480000, device="cuda", dtype=torch.float32).uniform_(-100.0, 100.0),
                "output": torch.zeros(20480000, device="cuda", dtype=torch.float32),
                "N": 20480000,
            }
        ]

    def check(self, case):
        assert torch.allclose(case["output"], case["input"].relu())



if __name__ == "__main__":
    Runner()()
    
