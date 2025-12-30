import ctypes
import torch
from common.runner import RunnerBase

class Runner(RunnerBase):
    def signature(self) -> dict:
        return {
            "default": [
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_int,
            ],
        }

    def build_test_cases(self):
        for n in [1, 3, 2047, 65536, 16385, 409600, 1048576]:
            data = torch.randn(n, device='cuda', dtype=torch.float32)
            yield {
                "function": "merge_sort",
                "clone": data.clone(),
                "argv": {
                    "data": data,
                    "n": n,
                }
            }

    def check(self, data, case, **kwargs):
        expected = torch.sort(case["clone"]).values
        assert torch.allclose(data, expected)


if __name__ == "__main__":
    Runner()()
