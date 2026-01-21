import ctypes
import torch
from common.runner import RunnerBase
import os
import subprocess


class Runner(RunnerBase):
    def __init__(self) -> None:
        super().__init__()
        self.shared_lib_path = os.path.join(self.dir, "build", "libtranspose.so")
    
    def signature(self) -> dict:
        return {
            "default": [
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_int,
                ctypes.c_int
            ]
        }
        
    def build_shared_lib(self):
        compile_command = [
            "cmake",
            "--build",
            os.path.join(self.dir, "build"),
            "--config",
            "Release"
        ]
        subprocess.check_call(compile_command)

    def build_test_cases(self):
        M = 4096
        N = 2048
        nums = torch.rand((M, N), device="cuda", dtype=torch.float32)
        
        funcs = [
            "transpose_cublas",
            "transpose_naive",
            "transpose_coalesced",
            "transpose_tiling",
            "transpose_tiling_no_bank_conflict",
            "transpose_tiling_swizzle",
            "transpose_tiling_multi_elements"
        ]
        
        for func in funcs:
            yield {
                "function": func,
                "argv": {
                    "input": nums,
                    "output": torch.empty((N, M), device="cuda", dtype=torch.float32),
                    "M": M,
                    "N": N
                }
            }
    

    def check(self, input, output, **kwargs):
        assert torch.equal(output, input.t()), "Result mismatch in transpose function"


if __name__ == "__main__":
    Runner()()
