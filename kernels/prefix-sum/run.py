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
        N = 1024 * 1024 * 10
        return [
            {
                "function": "kogge_stone_scan",
                "argv": {
                    "input": torch.ones(N, device='cuda', dtype=torch.float32),
                    "output": torch.empty(N, device='cuda', dtype=torch.float32),
                    "N": N,
                }
            },
            {
                "function": "kogge_stone_scan_double_buffer",
                "argv": {
                    "input": torch.ones(N, device='cuda', dtype=torch.float32),
                    "output": torch.empty(N, device='cuda', dtype=torch.float32),
                    "N": N,
                }
            },
            {
                "function": "brent_kung_scan",
                "argv": {
                    "input": torch.ones(N, device='cuda', dtype=torch.float32),
                    "output": torch.empty(N, device='cuda', dtype=torch.float32),
                    "N": N,
                }
            },
            {
                "function": "brent_kung_scan_optimized",
                "argv": {
                    "input": torch.ones(N, device='cuda', dtype=torch.float32),
                    "output": torch.empty(N, device='cuda', dtype=torch.float32),
                    "N": N,
                }
            }
        ]

    def check(self, input, output, **kwargs):
        expected = torch.cumsum(input, dim=0)
        assert torch.allclose(output, expected)



if __name__ == "__main__":
    Runner()()
