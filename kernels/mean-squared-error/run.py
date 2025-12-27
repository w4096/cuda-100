import ctypes
import torch
from common.runner import RunnerBase

class Runner(RunnerBase):

    def signature(self) -> dict:
        return {
            "default": [
                ctypes.POINTER(ctypes.c_uint16),
                ctypes.POINTER(ctypes.c_uint16),
                ctypes.POINTER(ctypes.c_uint16),
                ctypes.c_int,
            ]
        }

    def build_test_cases(self):
        # mse test cases
        for N in [17, 255, 1023, 8192, 65536, 102400]:
            yield {
                "predictions": torch.randn(N, device="cuda", dtype=torch.float32),
                "targets": torch.randn(N, device="cuda", dtype=torch.float32),
                "mse": torch.zeros(1, device="cuda", dtype=torch.float32),
                "N": N,
            }

    def check(self, case):
        predictions = case["predictions"]
        targets = case["targets"]
        expected = torch.mean((predictions - targets) ** 2)
        assert torch.allclose(case["mse"], expected)


if __name__ == "__main__":
    Runner()()

