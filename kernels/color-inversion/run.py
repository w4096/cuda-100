import ctypes
import torch
from common.runner import RunnerBase


class Runner(RunnerBase):
    def signature(self) -> dict:
        return {
            "solve": [
                ctypes.POINTER(ctypes.c_char),
                ctypes.c_int,
                ctypes.c_int,
            ]
        }

    def build_test_cases(self):
        width, height = 4096, 5120
        size = width * height * 4
        return [
            {
                "image": torch.randint(0, 256, (size,), device="cuda", dtype=torch.uint8),
                "width": width,
                "height": height,
            }
        ]

    def check(self, case):
        copied = case["image"].view(-1, 4)
        copied[:, :3] = 255.0 - copied[:, :3]
        copied = copied.view(-1)
        assert torch.allclose(copied, case["image"])



if __name__ == "__main__":
    Runner()()
