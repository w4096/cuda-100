cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(cuda LANGUAGES CXX CUDA)

include(../cmake/common.cmake)

include_directories(..)

add_executable(sum sum.cu)
