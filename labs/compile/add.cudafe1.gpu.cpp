typedef char __nv_bool;

struct CUstream_st;

typedef unsigned long size_t;
#include "crt/device_runtime.h"

typedef unsigned long _ZSt6size_t;

___device__(extern __no_sc__) unsigned __cudaPushCallConfiguration(struct dim3, struct dim3, size_t,
                                                                   struct CUstream_st*);

__global__ __var_used__ extern void _Z10vector_addPKfS0_Pfi(const float*, const float*, float*, int);
#include "common_functions.h"

__global__ __var_used__ void _Z10vector_addPKfS0_Pfi(

    const float* A,

    const float* B,

    float* C,

    int N) {
    {
        int __cuda_local_var_36667_9_non_const_idx;

        __cuda_local_var_36667_9_non_const_idx = ((int)(((blockIdx.x) * (blockDim.x)) + (threadIdx.x)));

        if (__cuda_local_var_36667_9_non_const_idx < N)

        {
            (C[__cuda_local_var_36667_9_non_const_idx]) =
                ((A[__cuda_local_var_36667_9_non_const_idx]) + (B[__cuda_local_var_36667_9_non_const_idx]));
        }
    }
}
