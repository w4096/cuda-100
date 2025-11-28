cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(cuda_006 LANGUAGES CXX CUDA)

include(../cmake/common.cmake)
include(../cmake/torch.cmake)

include_directories(..)

set(CMAKE_BUILD_TYPE "Release")

add_executable(006_reduce reduce.cu)
