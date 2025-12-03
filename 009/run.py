import torch
from torch.utils.cpp_extension import load

torch.set_grad_enabled(False)

# Load the CUDA kernel as a python module
my_softmax = load(name='my_softmax',
                  sources=['softmax.cu', 'wrap.cu'],
                  extra_cuda_cflags=[
                      "-O3",
                      "-U__CUDA_NO_HALF_OPERATORS__",
                      "-U__CUDA_NO_HALF_CONVERSIONS__",
                      "-U__CUDA_NO_HALF2_OPERATORS__",
                      "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                      "--expt-relaxed-constexpr",
                      "--expt-extended-lambda",
                      "--use_fast_math",
                  ],
                  extra_cflags=['-std=c++17'])

tokens = torch.rand((8092, 4096)).cuda()

print("Profiling torch.softmax")
with torch.profiler.profile() as prof:
    out1 = tokens.softmax(-1)
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

print("Profiling my_softmax")
with torch.profiler.profile() as prof:
    out2 = my_softmax.softmax(tokens)
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

assert torch.allclose(out1, out2)
