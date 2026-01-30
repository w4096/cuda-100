cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(cuda LANGUAGES CXX CUDA)

include(../cmake/common.cmake)

include_directories(..)

set(CMAKE_BUILD_TYPE "Release")

file(GLOB SRC_FILES
    "*.cu"
    "*.cpp"
)

add_executable(elementwise ${SRC_FILES})
