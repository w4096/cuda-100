__global__ void vector_add(const float *A, const float *B, float *C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    const int N = 1024;
    float *A, *B, *C;
    // Allocate and initialize host memory
    cudaMallocManaged(&A, N * sizeof(float));
    cudaMallocManaged(&B, N * sizeof(float));
    cudaMallocManaged(&C, N * sizeof(float));

    vector_add<<<(N + 255) / 256, 256>>>(A, B, C, N);

    cudaDeviceSynchronize();
    
    return 0;
}
