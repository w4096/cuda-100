set(Torch_DIR ~/code/libtorch-cuda-12.8/libtorch/share/cmake/Torch)
find_package(Torch REQUIRED)

find_package(Python3 REQUIRED Interpreter Development)
include_directories(${Python3_INCLUDE_DIRS})
include_directories(${TORCH_INCLUDE_DIRS})

link_libraries(${Python3_LIBRARIES})
link_libraries(${TORCH_LIBRARIES})