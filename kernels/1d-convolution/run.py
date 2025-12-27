import ctypes
import torch
from common.runner import RunnerBase


class Runner(RunnerBase):
    def signature(self) -> dict:
        return {
            "solve": [
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_int,
                ctypes.c_int,
            ]
        }

    def build_test_cases(self):
        dtype = torch.float32
        input_size, kernel_size = 150, 20
        output_size = input_size - kernel_size + 1

        return [
            {
                "input": torch.empty(input_size, device="cuda", dtype=dtype).uniform_(-1.0, 1.0),
                "kernel": torch.empty(kernel_size, device="cuda", dtype=dtype).uniform_(-1.0, 1.0),
                "output": torch.empty(output_size, device="cuda", dtype=dtype),
                "input_size": input_size,
                "kernel_size": kernel_size,
            }
        ]

    def check(self, case):
        expected_output = torch.nn.functional.conv1d(
            case["input"].unsqueeze(0).unsqueeze(0),
            case["kernel"].unsqueeze(0).unsqueeze(0)
        ).squeeze(0).squeeze(0)
        assert torch.allclose(case["output"], expected_output, atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    Runner()()
