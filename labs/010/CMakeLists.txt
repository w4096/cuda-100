cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(cuda LANGUAGES CXX CUDA)

include(../cmake/common.cmake)

set(CMAKE_BUILD_TYPE "Release")

add_executable(occupancy occupancy.cu)

