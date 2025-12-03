set(Torch_DIR ~/Downloads/libtorch-shared-with-deps-2.9.1+cu130/libtorch/share/cmake/Torch)
find_package(Torch REQUIRED)

find_package(Python3 REQUIRED Interpreter Development)
include_directories(${Python3_INCLUDE_DIRS})
include_directories(${TORCH_INCLUDE_DIRS})

link_libraries(${Python3_LIBRARIES})
link_libraries(${TORCH_LIBRARIES})