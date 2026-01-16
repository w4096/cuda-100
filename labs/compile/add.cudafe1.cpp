__attribute__((visibility("hidden"))) void vector_add(const float* A, const float* B, float* C, int N);
#if 0
{
int idx = ((__device_builtin_variable_blockIdx.x) * (__device_builtin_variable_blockDim.x)) + (__device_builtin_variable_threadIdx.x);
if (idx < N) {
(C[idx]) = ((A[idx]) + (B[idx]));
}
}
#endif
int main() {
    const int N = 1024;
    float *A, *B, *C;
    cudaMallocManaged(&A, N * sizeof(float));
    cudaMallocManaged(&B, N * sizeof(float));
    cudaMallocManaged(&C, N * sizeof(float));
    (__cudaPushCallConfiguration((N + 255) / 256, 256)) ? (void)0 : vector_add(A, B, C, N);
    cudaDeviceSynchronize();
    return 0;
}

#include "add.cudafe1.stub.c"