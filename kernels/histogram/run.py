import ctypes
import torch
import torch.nn.functional as F
from common.runner import RunnerBase

class Runner(RunnerBase):

    def signature(self) -> dict:
        return {
            "default": [
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int),
                ctypes.c_int,
                ctypes.c_int
            ]
        }

    def build_test_cases(self):
        for n, bins in [(10240, 256), (20480000, 1024)]:
            input = torch.randint(0, bins, (n,), device='cuda', dtype=torch.int)

            yield {
                "function": "solve_naive",
                "argv": {
                    "input": input,
                    "histogram": torch.zeros(bins, device='cuda', dtype=torch.int),
                    "N": n,
                    "bins": bins
                }
            }

            yield {
                "function": "solve",
                "argv": {
                    "input": input,
                    "histogram": torch.zeros(bins, device='cuda', dtype=torch.int),
                    "N": n,
                    "bins": bins
                }
            }


    def check(self, input, histogram, bins, **kwargs):
        expected = torch.histc(input, bins=bins, min=0, max=bins-1).int()
        assert torch.equal(expected, histogram)


if __name__ == "__main__":
    Runner()()
