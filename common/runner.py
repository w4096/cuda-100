import inspect
from abc import ABC, abstractmethod
import ctypes
import glob
import os
import torch
import time
from typing import Generator
from prettytable import PrettyTable


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
            "default": [
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_int
            ],
            "other_function": [
                ctypes.c_int,
                ctypes.c_float
            ],
        }
        """
        pass

    @abstractmethod
    def build_test_cases(self) -> Generator[dict, None, None]:
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

        or if there is only one function:, the default function name is "solve", and you can provide arguments directly:
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
    def check(self, **kwargs):
        """
        using assertion check the output correctness for a test case
        the case and arguments are passed as keyword arguments.
        """
        pass

    def compile_options(self) -> list[str]:
        options = []
        if os.environ.get("CUTLASS_PATH") is not None:
            cutlass_path = os.environ["CUTLASS_PATH"]
            include_path = os.path.join(cutlass_path, "include")
            options.append(f"-I{include_path}")

            util_path = os.path.join(cutlass_path, "tools", "util", "include")
            options.append(f"-I{util_path}")

        # add current directory to include path
        options.append(f"-I{self.dir}")

        cuda_home = "/usr/local/cuda"
        if os.environ.get("CUDA_HOME") is not None:
            cuda_home = os.environ["CUDA_HOME"]
        cuda_include = os.path.join(cuda_home, "include", "cccl")
        options.append(f"-I{cuda_include}")
        
        return options

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
            "-lineinfo",
            "--expt-relaxed-constexpr",
            "--shared",
            "-o", self.shared_lib_path,
        ]
        cmd = ["nvcc"] + files + compile_args + self.compile_options()
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
        if "argv" in case:
            argv = list(case["argv"].values())
        else:
            argv = list(case.values())

        if "function" in case:
            func_name = case["function"]
        else:
            func_name = "solve"

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
        performance_results = []
        for i, case in enumerate(cases):
            with torch.profiler.profile() as prof:
                self.run_test_case(case)

            print("=" * 40 + f" Test case {i+1} " + "=" * 40)
            case_summary = build_case_summary(case)
            print(case_summary)
            
            # get cuda total time
            
            key_averages = prof.key_averages()
            
            total_cpu_time = key_averages.self_cpu_time_total
            total_cuda_time = sum([item.self_device_time_total for item in key_averages])
            function_name = case.get("function", f"case {i+1}")
            performance_results.append({
                "function": function_name,
                "cpu_time_total": total_cpu_time,
                "cuda_time_total": total_cuda_time,
            })
            
            print(key_averages.table(sort_by="cuda_time_total", row_limit=10))

            if "argv" in case:
                argv = case["argv"]
            else:
                argv = case
            self.check(case=case, **argv)

            print("\n")

        print(f"All test cases passed.")
        
        
        # # show performance summary
        print("=" * 40 + " Performance Summary " + "=" * 40)
    
        table = PrettyTable()
        table.align = 'l'
        table.field_names = ["Test Case", "CPU Time (us)", "CUDA Time (us)"]
        for i, result in enumerate(performance_results):
            table.add_row([result["function"], f"{result['cpu_time_total']:.3f}", f"{result['cuda_time_total']:.3f}"])
        print(table)


    def __call__(self):
        self.run()


def build_case_summary(case: dict) -> str:
    summary = ""
    if "function" in case:
        summary += f"Function: {case['function']}\n"
        argv = case["argv"]
    else:
        argv = case

    summary += "Arguments:\n"
    for k, v in argv.items():
        if isinstance(v, torch.Tensor):
            summary += f"  {k}: shape={v.shape}, dtype={v.dtype}\n"
        else:
            summary += f"  {k}: {v}\n"
    return summary