import ctypes
import torch
from common.runner import RunnerBase

class Runner(RunnerBase):
    def signature(self) -> dict:
        # a mapping from function name to argument types
        return {
            "default": [
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_int,
            ],
        }

    def build_test_cases(self):
        N = 4096 * 1024 * 10
        return [
            {
                "function": "solve",
                "argv": {
                    "input": torch.ones(N, device='cuda', dtype=torch.float32),
                    "output": torch.empty(N, device='cuda', dtype=torch.float32),
                    "N": N,
                }
            },
            {
                "function": "solve2",
                "argv": {
                    "input": torch.ones(N, device='cuda', dtype=torch.float32),
                    "output": torch.empty(N, device='cuda', dtype=torch.float32),
                    "N": N,
                }
            }
        ]

    def check(self, case):
        args = case["argv"]
        expected = torch.cumsum(args["input"], dim=0)
        assert torch.allclose(args["output"], expected)



if __name__ == "__main__":
    Runner()()
