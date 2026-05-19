cmake_minimum_required(VERSION 3.18)
project(MatrixTransposition LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES 75)

file(GLOB_RECURSE MT_COMMON_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu"
)

add_library(mt_objs OBJECT ${MT_COMMON_SOURCES})
target_include_directories(mt_objs PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

set(M_LIST 1024 2048 4096 8192)
set(N_LIST 1024 2048 4096 8192)

foreach(M_VAL IN LISTS M_LIST)
    foreach(N_VAL IN LISTS N_LIST)
        set(TARGET_NAME "mt_${M_VAL}x${N_VAL}")
        
        add_executable(${TARGET_NAME}
            ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
            $<TARGET_OBJECTS:mt_objs>
        )
        
        target_compile_definitions(${TARGET_NAME} PRIVATE 
            M=${M_VAL}
            N=${N_VAL}
        )

        target_link_options(${TARGET_NAME} PRIVATE -static-libstdc++ -static-libgcc)
        target_include_directories(${TARGET_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    endforeach()
endforeach()