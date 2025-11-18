#pragma once

#include <chrono>
#include <cmath>
#include <cuda_runtime.h>
#include <iostream>
#include <string>

#define CHECK_CUDA_ERR(err)                                                                                            \
    do {                                                                                                               \
        cudaError_t e = err;                                                                                           \
        if (e != cudaSuccess) {                                                                                        \
            fprintf(stderr, "CUDA Error: %s (%d) at line %d\n", cudaGetErrorString(e), e, __LINE__);                \
            exit(EXIT_FAILURE);                                                                                        \
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
        std::cout << name_ << " cost: " << elapsed.count() << "ms" << std::endl;
    }

 private:
    std::string name_{};
    std::chrono::high_resolution_clock::time_point start_{};
};
