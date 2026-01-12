## GEMM


This lab implements various versions of General Matrix-Matrix Multiplication (GEMM) using CUDA. The goal is to explore different optimization techniques to improve the performance of matrix multiplication on NVIDIA GPUs.

Here are the test results for multiplying two 4096x2048 matrices (A and B) to produce a 4096x2048 matrix (C) using half-precision floating-point numbers (FP16). The performance is measured in microseconds (us) for both CPU and CUDA implementations.

```
+-------------------------+---------------+----------------+
| Test Case               | CPU Time (us) | CUDA Time (us) |
+-------------------------+---------------+----------------+
| gemm_cutlass            | 5122.510      | 4070.789       |
| gemm_naive              | 39379.288     | 38243.149      |
| gemm_tile_16x16         | 27621.196     | 26592.759      |
| gemm_tile_32x32         | 29934.456     | 28816.136      |
| gemm_tile_B16x16_T32x32 | 20333.678     | 18997.707      |
| gemm_tile_m64n64k16     | 8600.327      | 7411.761       |
| gemm_tile_m64n64k16_reg | 9557.479      | 8365.461       |
| gemm_wmma_16x16x16      | 5754.062      | 4427.610       |
| gemm_wmma_32x32x16_smem | 3576.325      | 2942.586       |
+-------------------------+---------------+----------------+
```