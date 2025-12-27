import ctypes
import torch
from common.runner import RunnerBase

class Runner(RunnerBase):

    def signature(self) -> dict:
        return {
            "solve": [
                ctypes.POINTER(ctypes.c_int32),
                ctypes.POINTER(ctypes.c_uint32),
                ctypes.c_int,
            ]
        }

    def build_test_cases(self):
        return [
            {
                "input": torch.randint(0, 2147483647 + 1, (1024,), device="cuda", dtype=torch.int32),
                "output": torch.zeros(1024, device="cuda", dtype=torch.uint32),
                "N": 1024,
                "R": 50,
            },
        ]


    def check(self, case):
        pass



if __name__ == "__main__":
    Runner()()
    
