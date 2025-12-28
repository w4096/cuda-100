import ctypes
import torch
from common.runner import RunnerBase

class Runner(RunnerBase):

    def signature(self) -> dict:
        return {
            "default": [
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int),
                ctypes.c_int,
            ]
        }

    def build_test_cases(self):
        yield {
            "function": "radix_sort_cpu",
            "argv": {
                "input": torch.randint(0, 1000, (1000000,), device="cpu", dtype=torch.int32),
                "output": torch.empty(1000000, device="cpu", dtype=torch.int32),
                "N": 1000000,
            }
        }

    def check(self, input, output, **kwargs):
        assert torch.equal(output, torch.sort(input).values)


if __name__ == "__main__":
    Runner()()

