cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(cuda LANGUAGES CXX CUDA)

include(../cmake/common.cmake)
include(../cmake/torch.cmake)

include_directories(..)

set(CMAKE_BUILD_TYPE "Release")

add_executable(elementwise elementwise.cu)
