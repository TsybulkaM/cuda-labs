cmake_minimum_required(VERSION 3.18)
project(CudaCommonLibrary)

#  .h: INTERFACE, .cpp: STATIC
add_library(cuda_common INTERFACE)

target_include_directories(cuda_common INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)