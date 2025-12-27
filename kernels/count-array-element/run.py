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
                ctypes.c_int,
            ]
        }

    def build_test_cases(self):
        return [
            {
                "input": torch.randint(1, 102400, (100000000,), device="cuda", dtype=torch.int32),
                "output": torch.empty(1, device="cuda", dtype=torch.int32),
                "N": 100000000,
                "K": 4096,
            },
        ]

    def check(self, case):
        assert case["output"].item() == (case["input"] == case["K"]).sum().item()



if __name__ == "__main__":
    Runner()()
