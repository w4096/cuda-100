cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(cuda)

include(cmake/common.cmake)


file(GLOB_RECURSE files ${CMAKE_SOURCE_DIR}/*/*.cu ${CMAKE_SOURCE_DIR}/*/*.cpp ${CMAKE_SOURCE_DIR}/*/*.h)

add_subdirectory(001)
add_subdirectory(002)
add_subdirectory(005)
add_subdirectory(006)
add_subdirectory(007)
add_subdirectory(008)
