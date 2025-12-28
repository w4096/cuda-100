import ctypes
import torch
from common.runner import RunnerBase

class Runner(RunnerBase):
    def signature(self) -> dict:
        # a mapping from function name to argument types
        return {
            "default": [
                ctypes.POINTER(ctypes.c_int),
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_int),
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_int),
            ],
        }

    def build_test_cases(self):

        for m, n in [(1, 1023), (32, 1024), (64, 2048), (128, 4096), (409600, 409600)]:
            yield {
                "a": torch.arange(0, m, device='cuda', dtype=torch.int32),
                "m": m,
                "b": torch.arange(0, n, device='cuda', dtype=torch.int32),
                "n": n,
                "c": torch.empty(m + n, device='cuda', dtype=torch.int32),
            }

        for m, n in [(1, 1023), (32, 1024), (64, 2048), (128, 4096), (409600, 409600)]:
            yield {
                "a": torch.ones(m, device='cuda', dtype=torch.int32),
                "m": m,
                "b": torch.zeros(n, device='cuda', dtype=torch.int32),
                "n": n,
                "c": torch.empty(m + n, device='cuda', dtype=torch.int32),
            }

    def check(self, a, m, b, n, c, **kwargs):
        expected = torch.cat([a, b]).sort().values
        assert torch.equal(c, expected)


if __name__ == "__main__":
    Runner()()
