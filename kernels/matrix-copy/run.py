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
        dtype = torch.float32
        return [
            {
                "A": torch.empty(4096, 4096, device="cuda", dtype=dtype).uniform_(-1000.0, 1000.0),
                "B": torch.empty(4096, 4096, device="cuda", dtype=dtype),
                "N": 4096,
            },
            {
                "A": torch.tensor([[42]], device="cuda", dtype=dtype),
                "B": torch.empty(1, 1, device="cuda", dtype=dtype),
                "N": 1,
            }
        ]

    def check(self, case):
        assert torch.allclose(case["A"], case["B"])



if __name__ == "__main__":
    Runner()()
    
