cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(cuda LANGUAGES CXX CUDA)

include(../cmake/common.cmake)

include_directories(..)

add_executable(matrix_zero_002 matrix_zero.cu)
