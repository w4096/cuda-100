cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(cuda_007 LANGUAGES CXX CUDA)

include(../cmake/common.cmake)

include_directories(..)

set(CMAKE_BUILD_TYPE "Release")

add_executable(gemm main.cu gemm.cuh gemm.cu)
find_package(CUDAToolkit REQUIRED)

target_link_libraries(gemm PRIVATE
        CUDA::cudart
        CUDA::cublas
)