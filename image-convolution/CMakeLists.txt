cmake_minimum_required(VERSION 3.27)
project(ImageConvolutionLANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES 75)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(FetchContent)
FetchContent_Declare(
    stb
    GIT_REPOSITORY https://github.com/nothings/stb.git
    GIT_TAG        master
)

FetchContent_MakeAvailable(stb)

file(GLOB_RECURSE COMMON_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu"
)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../cuda_common ${CMAKE_CURRENT_BINARY_DIR}/cuda_common)

add_library(objs OBJECT ${COMMON_SOURCES})
target_include_directories(objs PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include ${stb_SOURCE_DIR})
target_link_libraries(objs PRIVATE cuda_common)

set(TARGET_NAME "image_convolution")

add_executable(${TARGET_NAME}
    ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
    $<TARGET_OBJECTS:objs>
)

target_link_libraries(${TARGET_NAME} PRIVATE cuda_common)
target_link_options(${TARGET_NAME} PRIVATE -static-libstdc++ -static-libgcc)
target_include_directories(${TARGET_NAME} PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/include 
    ${stb_SOURCE_DIR}
)