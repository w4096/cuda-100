import inspect
from abc import ABC, abstractmethod
import ctypes
import glob
import os
import torch
import time


class RunnerBase(ABC):
    def __init__(self) -> None:
        self.lib = None
        self.funcs = {}

        path = inspect.getfile(self.__class__)
        self.dir = os.path.dirname(os.path.abspath(path))
        self.shared_lib_path = os.path.join(self.dir, "kernel.so")


    @abstractmethod
    def signature(self) -> dict:
        """
        return a dictionary mapping function names to a list of ctypes argument types.
        If there is only one function, use the key "default".
        Example:
        {
            "default": [ctypes.POINTER(ctypes.c_float), ctypes.c_int],
            "other_function": [ctypes.c_int, ctypes.c_float],
        }
        """
        pass

    @abstractmethod
    def build_test_cases(self) -> list[dict]:
        """
        build and return a list of test cases. Each test case is represented as a dictionary containing argv and function name (optional).

        Example:
        [
            {
                "function": "solve",
                "argv": {
                    "input": torch.randn(1024, device='cuda', dtype=torch.float32),
                    "output": torch.empty(1024, device='cuda', dtype=torch.float32),
                    "N": 1024,
                }
            }
        ]

        or if there is only one function: (default function name is "solve")
        [
            {
                "input": torch.randn(1024, device='cuda', dtype=torch.float32),
                "output": torch.empty(1024, device='cuda', dtype=torch.float32),
                "N": 1024,
            }
        ]
        """
        pass

    @abstractmethod
    def check(self, case: dict):
        """
        using assertion check the output correctness for a test case
        """
        pass

    def compile(self):
        files = glob.glob(f"{self.dir}/*.cu") + glob.glob(f"{self.dir}/*.cpp")
        # if files is unchanged and soname exists, skip compilation
        if os.path.exists(self.shared_lib_path):
            soname_mtime = os.path.getmtime(self.shared_lib_path)
            files_mtime = max([os.path.getmtime(f) for f in files])
            if soname_mtime >= files_mtime:
                print(f"{self.shared_lib_path} is up to date, skipping compilation.")
                return

        compile_args = [
            "-O3",
            "--compiler-options",
            "-fPIC",
            "--shared",
            "-o", self.shared_lib_path,
        ]
        cmd = ["nvcc"] + files + compile_args
        print("Compiling with command:", " ".join(cmd))
        import subprocess
        subprocess.check_call(cmd)


    def call(self, func_name, arg_types, argv):
        if self.lib is None:
            self.compile()
            self.lib = ctypes.CDLL(self.shared_lib_path)

        if func_name not in self.funcs:
            func = self.lib[func_name]
            func.argtypes = arg_types
            func.restype = None
            self.funcs[func_name] = func

        func = self.funcs[func_name]
        func(*argv)

    def run_test_case(self, case: dict):
        if "function" in case:
            func_name = case["function"]
            argv = list(case["argv"].values())
        else:
            func_name = "solve"
            argv = list(case.values())

        signature = self.signature()
        if func_name in signature:
            arg_types = signature[func_name]
        else:
            arg_types = signature["default"]

        argv_casted = []
        for arg, argtype in zip(argv, arg_types):
            if isinstance(arg, torch.Tensor):
                argv_casted.append(ctypes.cast(arg.data_ptr(), argtype))
            else:
                argv_casted.append(argtype(arg))
        self.call(func_name, arg_types, argv_casted)


    def run(self):
        cases = self.build_test_cases()
        for i, case in enumerate(cases):
            with torch.profiler.profile() as prof:
                start = time.perf_counter()
                self.run_test_case(case)
                end = time.perf_counter()
            self.check(case)
            print("=" * 40 + f" Test case {i+1} " + "=" * 40)
            print(f"Test case passed in {(end - start) * 1000:.3f} ms.")
            print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
            print("\n")

        print(f"All test cases passed.")


    def __call__(self):
        self.run()
