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
        N = 50000000
        yield {
            "function": "solve",
            "argv": {
                "input": torch.randint(0, 4294967296, (N,), device="cuda", dtype=torch.uint32),
                "output": torch.empty(N, device="cuda", dtype=torch.uint32),
                "N": N,
            }
        }

    def check(self, input, output, **kwargs):
        i64 = input.to(torch.int64)
        sorted_tensor = torch.sort(i64)[0]
        assert torch.equal(output, sorted_tensor.to(output.dtype))


if __name__ == "__main__":
    Runner()()

