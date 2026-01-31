cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

include(../cmake/common.cmake)

project(mma LANGUAGES CXX CUDA)

set(CMAKE_BUILD_TYPE "Release")

file(GLOB SRC_FILES
    "*.cu"
    "*.cpp"
)

add_executable(mma ${SRC_FILES})
