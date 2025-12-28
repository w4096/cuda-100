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
            ]
        }

    def build_test_cases(self):
        for N in [17, 255, 1023, 8192, 65536, 102400]:
            A = torch.randn(N, device="cuda", dtype=torch.float32)
            B = torch.randn(N, device="cuda", dtype=torch.float32)
            yield {
                "function": "solve_naive",
                "argv": {
                    "A": A,
                    "B": B,
                    "C": torch.zeros(1, device="cuda", dtype=torch.float32),
                    "N": N,
                }
            }
            yield {
                "function": "solve_vectorize_warp_shuffle",
                "argv": {
                    "A": A,
                    "B": B,
                    "C": torch.zeros(1, device="cuda", dtype=torch.float32),
                    "N": N,
                }
            }


    def check(self, A, B, C, **kwargs):
        expected = torch.dot(A, B)
        assert torch.allclose(C, expected, atol=1e-3, rtol=1e-3)


if __name__ == "__main__":
    Runner()()

