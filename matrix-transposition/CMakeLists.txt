cmake_minimum_required(VERSION 3.18)
project(MatrixTransposition LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES 75)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

file(GLOB_RECURSE COMMON_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu"
)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../cuda_common ${CMAKE_CURRENT_BINARY_DIR}/cuda_common)

add_library(objs OBJECT ${COMMON_SOURCES})
target_include_directories(objs PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(objs PRIVATE cuda_common)

set(M_LIST 1024 2048 4096 8192)
set(N_LIST 1024 2048 4096 8192)

foreach(M_VAL IN LISTS M_LIST)
    foreach(N_VAL IN LISTS N_LIST)
        set(TARGET_NAME "mt_${M_VAL}x${N_VAL}")
        
        add_executable(${TARGET_NAME}
            ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
            $<TARGET_OBJECTS:objs>
        )
        
        target_link_libraries(${TARGET_NAME} PRIVATE cuda_common)
        target_compile_definitions(${TARGET_NAME} PRIVATE 
            M=${M_VAL}
            N=${N_VAL}
        )

        target_link_options(${TARGET_NAME} PRIVATE -static-libstdc++ -static-libgcc)
        target_include_directories(${TARGET_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    endforeach()
endforeach()