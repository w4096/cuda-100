import ctypes
import torch
from common.runner import RunnerBase

class Runner(RunnerBase):

    def signature(self) -> dict:
        # void solve(const float* input, float gamma, float beta, float* output, int N, float eps)
        return {
            "solve": [
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_float,
                ctypes.c_float,
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_int,
                ctypes.c_float,
            ]
        }

    def build_test_cases(self):
        for n in [1, 8, 15, 64, 127, 256, 511, 1024, 2048]:
            input = torch.randn(n, device='cuda', dtype=torch.float32)
            yield {
                "function": "solve",
                "argv": {
                    "input": input,
                    "gamma": 1.0,
                    "beta": 0.0,
                    "output": torch.empty_like(input),
                    "N": n,
                    "eps": 1e-5,
                }
            }

    def check(self, input, gamma, beta, output, N, eps, **kwargs):
        mean_square = torch.mean(input ** 2)
        rms = torch.sqrt(mean_square + eps)
        expected = gamma * input / rms + beta
        assert torch.allclose(output, expected)


if __name__ == "__main__":
    Runner()()

