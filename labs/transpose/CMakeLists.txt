cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(cuda LANGUAGES CXX CUDA)

include(../cmake/common.cmake)

include_directories(..)

set(CMAKE_BUILD_TYPE "Release")

add_library(transpose SHARED transpose.cu)
find_package(CUDAToolkit REQUIRED)

target_link_libraries(transpose PRIVATE
        CUDA::cudart
        CUDA::cublas
)
