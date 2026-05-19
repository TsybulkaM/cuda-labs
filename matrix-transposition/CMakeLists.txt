cmake_minimum_required(VERSION 3.18)
project(CudaProject LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

add_executable(cuda_app
    main.cpp 
    src/kernel.cu 
    include/kernel.cuh
)

target_include_directories(cuda_app PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)