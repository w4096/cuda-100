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
                ctypes.c_int,
            ]
        }

    def build_test_cases(self):
        return [
            {
                "input": torch.randn(1000, 512, device='cuda', dtype=torch.float32),
                "output": torch.empty(512, 1000, device='cuda', dtype=torch.float32),
                "rows": 1000,
                "cols": 512,
            },
            {
                "input": torch.randn(256, 256, device='cuda', dtype=torch.float32),
                "output": torch.empty(256, 256, device='cuda', dtype=torch.float32),
                "rows": 256,
                "cols": 256,
            },
            {
                "input": torch.randn(123, 456, device='cuda', dtype=torch.float32),
                "output": torch.empty(456, 123, device='cuda', dtype=torch.float32),
                "rows": 123,
                "cols": 456,
            },
        ]

    def check(self, case):
        assert torch.allclose(case["output"], case["input"].t())



if __name__ == "__main__":
    Runner()()
    
