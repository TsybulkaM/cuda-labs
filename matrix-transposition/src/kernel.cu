#include <iostream>
#include <cuda_runtime.h>
#include "kernel.cuh"

__global__ void helloFromGPU() {
    printf("Hello World from GPU thread %d!\n", threadIdx.x);
}

void run_cuda_kernel() {
    // Запуск 1 блока с 5 потоками
    helloFromGPU<<<1, 5>>>();
    cudaDeviceSynchronize();
}