#pragma once

#include <chrono>
#include <cmath>
#include <cuda_runtime.h>
#include <iostream>
#include <string>

inline unsigned int cdiv(unsigned int a, unsigned int b) {
    return (a + b - 1) / b;
}

#define CHECK_CUDA_ERR(err)                                                                                            \
    do {                                                                                                               \
        cudaError_t e = err;                                                                                           \
        if (e != cudaSuccess) {                                                                                        \
            fprintf(stderr, "CUDA Error: %s (%d) at line %d\n", cudaGetErrorString(e), e, __LINE__);                   \
            exit(EXIT_FAILURE);                                                                                        \
        }                                                                                                              \
    } while (0)

static void _assert(const char *cond, const char *file, int line, int panic) {
    fprintf( stderr,"assert '%s' failed @ (%s:%d)\n", cond, file, line);
    if (panic) {
        exit(EXIT_FAILURE);
    }
}

#define ASSERT(x)                                                                                                      \
    do {                                                                                                               \
        if (!(x)) {                                                                                                    \
            _assert(#x, __FILE__, __LINE__, 1);                                                                        \
        }                                                                                                              \
    } while (0)

class Timer {
 public:
    explicit Timer(std::string name) {
        name_ = std::move(name);
        start_ = std::chrono::high_resolution_clock::now();
    }
    ~Timer() {
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float, std::milli> elapsed = end - start_;
        std::cout << name_ << ": cost: " << elapsed.count() << "ms" << std::endl;
    }

 private:
    std::string name_{};
    std::chrono::high_resolution_clock::time_point start_{};
};
