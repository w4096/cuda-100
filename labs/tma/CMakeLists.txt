cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(tma LANGUAGES CXX CUDA)

include(../cmake/common.cmake)
include(../cmake/cutlass.cmake)

include_directories(..)
include_directories(/usr/local/cuda/include/cccl/)

set(CMAKE_BUILD_TYPE "Release")

file(GLOB SRC_FILES
    "*.cu"
    "*.cpp"
)

add_executable(tma ${SRC_FILES})

find_package(CUDAToolkit REQUIRED)
target_link_libraries(gemm PRIVATE
        CUDA::cudart
        CUDA::cublas
)