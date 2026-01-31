cmake_minimum_required(VERSION 3.18)

file(GLOB_RECURSE KERNEL_SOURCES "./*/*.cu" "./*/*.cuh")
add_library(kernels STATIC ${KERNEL_SOURCES})
