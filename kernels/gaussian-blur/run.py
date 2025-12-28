import ctypes
import torch
from common.runner import RunnerBase

class Runner(RunnerBase):

    def signature(self) -> dict:
        return {
            "default": [
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
            ]
        }

    def build_test_cases(self):
        for input_rows, input_cols, kernel_rows, kernel_cols in [(511, 512, 7, 7), (801, 609, 5, 5), (4096, 4048, 3, 27)]:
            yield {
                "function": "solve2",
                "argv": {
                    "input": torch.randint(low=-1, high=1, size=(input_rows * input_cols,), device="cuda").to(dtype=torch.float32),
                    "kernel": torch.randint(low=-1, high=2, size=(kernel_rows * kernel_cols,), device="cuda").to(dtype=torch.float32),
                    "output": torch.empty(input_rows * input_cols, device="cuda", dtype=torch.float32),
                    "input_rows": input_rows,
                    "input_cols": input_cols,
                    "kernel_rows": kernel_rows,
                    "kernel_cols": kernel_cols,
                }
            }
            yield {
                "function": "solve_naive",
                "argv": {
                    "input": torch.randint(low=-1, high=1, size=(input_rows * input_cols,), device="cuda").to(dtype=torch.float32),
                    "kernel": torch.randint(low=-1, high=2, size=(kernel_rows * kernel_cols,), device="cuda").to(dtype=torch.float32),
                    "output": torch.empty(input_rows * input_cols, device="cuda", dtype=torch.float32),
                    "input_rows": input_rows,
                    "input_cols": input_cols,
                    "kernel_rows": kernel_rows,
                    "kernel_cols": kernel_cols,
                }
            }

    def check(self, input, kernel, output, input_rows, input_cols, kernel_rows, kernel_cols, **kwargs):
        input = input.view(1, 1, input_rows, input_cols)
        kernel = kernel.view(1, 1, kernel_rows, kernel_cols)
        expected = torch.nn.functional.conv2d(input, kernel, padding='same').view(-1)
        assert torch.allclose(output, expected)

if __name__ == "__main__":
    Runner()()

