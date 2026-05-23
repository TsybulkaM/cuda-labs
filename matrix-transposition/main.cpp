#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <cstring>
#include <cuda_runtime.h>

#include "cuda_check.cuh"
#include "matrix.h"
#include "transposition.h"
#include "transposition.cuh"

#ifndef N
#define N 2048
#endif

#ifndef M
#define M 1024
#endif

int main() {
    printf("Matrix Transposition - CPU vs GPU Comparison\n");
    printf("Matrix size: %d x %d (%.2f MB)\n\n", N, M, (N * M * sizeof(float)) / (1024.0 * 1024.0));
    
    // ==== Host ====

    float* h_matrix = (float*)malloc(N * M * sizeof(float));
    float* h_transposed = (float*)malloc(M * N * sizeof(float));
    
    printf("Initializing matrix with random values...\n");
    fill_random(h_matrix, N, M);
    
    printf("\nCPU Transpose:\n");
    auto cpu_start = std::chrono::high_resolution_clock::now();
    transpose_cpu(h_matrix, h_transposed, N, M);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_time_ms = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();
    printf("  Time: %.4f ms\n", cpu_time_ms);
    
    printf("\nGPU Transpose (Naive):\n");

    // ==== Device ====

    float* d_input;
    float* d_output;
    cudaMalloc(&d_input, N * M * sizeof(float));
    cudaMalloc(&d_output, M * N * sizeof(float));
    
    cudaMemcpy(d_input, h_matrix, N * M * sizeof(float), cudaMemcpyHostToDevice);
    
    auto gpu_start = std::chrono::high_resolution_clock::now();
    transpose_gpu_naive(d_input, d_output, N, M);
    auto gpu_end = std::chrono::high_resolution_clock::now();
    double gpu_time_ms = std::chrono::duration<double, std::milli>(gpu_end - gpu_start).count();
    printf("  Time: %.4f ms\n", gpu_time_ms);
    printf("  Speedup: %.2fx\n", cpu_time_ms / gpu_time_ms);
    
    // Copy result back
    cudaMemcpy(h_transposed, d_output, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Verify correctness
    if (verify_transpose(h_matrix, h_transposed, N, M)) {
        printf("  Correctness: PASS\n");
    } else {
        printf("  Correctness: FAIL\n");
    }

    printf("\nGPU Transpose (Optimized, Unified Memory):\n");
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    int block_size = 16;
    if (prop.multiProcessorCount >= 40) {
        block_size = 32;
    } else if (prop.multiProcessorCount <= 10) {
        block_size = 8;
    }

    float* um_input = nullptr;
    float* um_output = nullptr;
    size_t matrix_bytes = N * M * sizeof(float);
    CUDA_CHECK(cudaMallocManaged(reinterpret_cast<void**>(&um_input), matrix_bytes));
    CUDA_CHECK(cudaMallocManaged(reinterpret_cast<void**>(&um_output), matrix_bytes));

    memcpy(um_input, h_matrix, matrix_bytes);

    int device_id = 0;
    CUDA_CHECK(cudaGetDevice(&device_id));
    CUDA_CHECK(cudaMemPrefetchAsync(um_input, matrix_bytes, device_id));
    CUDA_CHECK(cudaMemPrefetchAsync(um_output, matrix_bytes, device_id));
    CUDA_CHECK(cudaDeviceSynchronize());

    auto gpu_opt_start = std::chrono::high_resolution_clock::now();
    transpose_gpu_optimized(um_input, um_output, N, M, block_size);
    auto gpu_opt_end = std::chrono::high_resolution_clock::now();
    double gpu_opt_time_ms = std::chrono::duration<double, std::milli>(gpu_opt_end - gpu_opt_start).count();

    CUDA_CHECK(cudaMemPrefetchAsync(um_output, matrix_bytes, cudaCpuDeviceId));
    CUDA_CHECK(cudaDeviceSynchronize());

    memcpy(h_transposed, um_output, matrix_bytes);
    bool optimized_ok = verify_transpose(h_matrix, h_transposed, N, M);
    printf("  Selected block size (SM occupancy): %d x %d\n", block_size, block_size);
    printf("  Time: %.4f ms\n", gpu_opt_time_ms);
    printf("  Speedup vs CPU: %.2fx\n", cpu_time_ms / gpu_opt_time_ms);
    printf("  Improvement vs naive: %.2fx\n", gpu_time_ms / gpu_opt_time_ms);
    printf("  Correctness: %s\n", optimized_ok ? "PASS" : "FAIL");
    
    CUDA_CHECK(cudaFree(um_input));
    CUDA_CHECK(cudaFree(um_output));
    cudaFree(d_input);
    cudaFree(d_output);
    free(h_matrix);
    free(h_transposed);
    return 0;
}
