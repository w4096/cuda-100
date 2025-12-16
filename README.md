## CUDA-100

The CUDA code I write when I was learning CUDA programming.

### 001 - hello world

- Get familiar with CUDA programming.

### 002 - understanding the cuda programming model

### 003 - hardware 

### 004 - memory model

### 005 - element-wise operation

- Implement element-wise operation with CUDA kernel.

### 006 - reduction

Implement the kernel mentioned in this [Optimizing Parallel Reduction in CUDA](https://developer.download.nvidia.cn/assets/cuda/files/reduction.pdf).

### 007 - matrix multiplication

- Implement naive matrix multiplication.
- Optimize the kernel with tiling and shared memory.
- Using register to reduce shared memory access.

### 008 - matrix transpose

- Implement naive matrix transpose.
- Optimize the warp read/write pattern to increase memory coalescing.
- Optimize the kernel with tiling and shared memory.

### 009 - softmax

- Implement softmax with warp reduction and block reduction.
- Implement online softmax.


