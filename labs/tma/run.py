import ctypes
import torch
from common.runner import RunnerBase

class Runner(RunnerBase):
    def signature(self) -> dict:
        return {
            "copy": [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_int
            ],
            "gemm": [
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int
            ]
        }

    def build_test_cases(self):
        yield {
            "function": "copy",
            "argv": {
                "dst": torch.empty(102401, device="cuda", dtype=torch.int32),
                "src": torch.arange(102401, device="cuda", dtype=torch.int32),
                "size": 102401 * ctypes.sizeof(ctypes.c_int32)
            }
        }

    def check(self, case, **kwargs):
        if case["function"] == "copy":
            dst = case["argv"]["dst"]
            src = case["argv"]["src"]
            assert torch.equal(dst, src), "Result mismatch in copy function"

        
        


if __name__ == "__main__":
    Runner()()
