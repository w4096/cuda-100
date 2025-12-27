# 011  Rotary Embedding

## What I did?

- Read the [RoPE implementation](https://github.com/vllm-project/vllm/blob/v0.12.0/csrc/pos_encoding_kernels.cu) in vLLM and understand how it works.
- Implement the kernel by myself.
- Compare the performance of my implementation with PyTorch implementation.
- Understand how to bind custom CUDA kernel with PyTorch and call it from Python.

## Running the Example

1. Build cuda kernel as shared library

```bash
mkdir build
cd build
cmake ..
make
```

2. Run the example

```bash
python run.py
```


## Results

```
Profiling torch implementation of RoPE
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
## Call CompiledFxGraph fxyynixuixwn4a7n6oxppuiksidt...         0.00%       0.000us         0.00%       0.000us       0.000us     329.915us       100.72%     329.915us     329.915us             1  
                             Torch-Compiled Region: 0/0         1.77%      75.095us        92.82%       3.939ms       3.939ms       0.000us         0.00%     327.547us     327.547us             1  
## Call CompiledFxGraph fxyynixuixwn4a7n6oxppuiksidt...         3.44%     145.807us        90.71%       3.850ms       3.850ms       0.000us         0.00%     327.547us     327.547us             1  
triton_poi_fused_add_cat_index_mul_split_sub_unsquee...         0.58%      24.602us         1.08%      45.904us      45.904us     214.397us        65.46%     214.397us     214.397us             1  
triton_poi_fused_add_cat_index_mul_split_sub_unsquee...         0.00%       0.000us         0.00%       0.000us       0.000us     214.397us        65.46%     214.397us     214.397us             1  
triton_poi_fused_add_cat_index_mul_split_sub_unsquee...         0.17%       7.032us         0.27%      11.308us      11.308us     113.150us        34.54%     113.150us     113.150us             1  
triton_poi_fused_add_cat_index_mul_split_sub_unsquee...         0.00%       0.000us         0.00%       0.000us       0.000us     113.150us        34.54%     113.150us     113.150us             1  
                               TorchDynamo Cache Lookup         0.67%      28.522us         0.67%      28.522us      28.522us       0.000us         0.00%       0.000us       0.000us             1  
                                      Pregraph bytecode         0.11%       4.594us         0.11%       4.594us       4.594us       0.000us         0.00%       0.000us       0.000us             1  
                 AOTDispatcher Runtime Wrapper Prologue         0.23%       9.838us         0.23%       9.838us       9.838us       0.000us         0.00%       0.000us       0.000us             1  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 4.244ms
Self CUDA time total: 327.547us

Profiling CUDA implementation of RoPE
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                   _C::rotary_embedding         2.12%      29.457us        71.86%     998.904us     998.904us     371.099us       100.00%     742.198us     742.198us             1  
                                Activity Buffer Request        68.39%     950.715us        68.39%     950.715us     950.715us     371.099us       100.00%     371.099us     371.099us             1  
void rotary_embedding_kernel<float, true>(long const...         0.00%       0.000us         0.00%       0.000us       0.000us     371.099us       100.00%     371.099us     371.099us             1  
                                       cudaLaunchKernel         1.35%      18.732us         1.35%      18.732us      18.732us       0.000us         0.00%       0.000us       0.000us             1  
                                  cudaDeviceSynchronize        28.14%     391.240us        28.14%     391.240us     391.240us       0.000us         0.00%       0.000us       0.000us             1  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 1.390ms
Self CUDA time total: 371.099us
```

PyTorch implementation and CUDA implementation have similar performance. Why we implement a custom CUDA kernel then?
