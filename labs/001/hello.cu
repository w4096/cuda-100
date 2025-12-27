#include <stdio.h>

__global__ void hello() {
    printf("Hello, I am from block %d thread %d\n", blockIdx.x, threadIdx.x);
}

int main() {
    hello<<<2, 2>>>();
    printf("kernel launched\n");
    return 0;
}