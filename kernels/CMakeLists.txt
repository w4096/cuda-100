cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

include(../labs/cmake/common.cmake)

# for CLion cuda language support
file(GLOB_RECURSE  KERNEL_SOURCES "./*/*.cu" "./*/*.cuh")
add_library(kernels STATIC ${KERNEL_SOURCES})