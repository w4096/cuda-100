#pragma once


__global__ void softmax_warp_reduce(const float * __restrict__ in, float *out, int rows, int dim);

__global__ void online_softmax_warp_reduce(const float * __restrict__ in, float *out, int rows, int dim);

__global__ void softmax_block_reduce(const float * __restrict__ in, float *out, int rows, int dim);

__global__ void online_softmax_block_reduce(const float * __restrict__ in, float *out, int rows, int dim);