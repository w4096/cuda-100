cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(cute LANGUAGES CXX CUDA)

include(../cmake/common.cmake)
include(../cmake/cutlass.cmake)

include_directories(..)
include_directories(/usr/local/cuda/include/cccl/)

set(CMAKE_BUILD_TYPE "Release")


add_executable(tensor tensor.cu)

add_executable(copy copy.cu)

add_executable(layout layout.cu)

add_executable(layout_algebra layout_algebra.cu)

add_executable(tile tile.cu)

add_executable(mma mma.cu)
