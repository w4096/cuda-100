file(GLOB_RECURSE KERNEL_SOURCES "./*/*.cu" "./*/*.cuh")
add_library(labs STATIC ${KERNEL_SOURCES})
