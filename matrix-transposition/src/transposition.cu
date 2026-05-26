#include <cuda_runtime.h>
#include "cuda_check.cuh"
#include "cuda_const.cuh"
#include "transposition.cuh"

__global__ void transpose_kernel_naive(const float* input, float* output, int rows, int cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row < rows && col < cols) {
        output[col * rows + row] = input[row * cols + col];
    }
}

void transpose_gpu_naive(const float* d_input, float* d_output, int rows, int cols) {
    dim3 block_dim(CUDA_BLOCK_SIZE, CUDA_BLOCK_SIZE);
    dim3 grid_dim((cols + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE, (rows + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE);
    
    transpose_kernel_naive<<<grid_dim, block_dim>>>(d_input, d_output, rows, cols);
    CUDA_CHECK(cudaDeviceSynchronize());
}


__global__ void transpose_kernel_optimized(const float* input, float* output, int rows, int cols) {
    extern __shared__ float smem[];
    
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int local_x = threadIdx.x;
    int local_y = threadIdx.y;
    
    if (row < rows && col < cols) {
        smem[local_y * (blockDim.x + 1) + local_x] = input[row * cols + col];
    }
    
    __syncthreads();
    
    int new_col = blockIdx.y * blockDim.y + local_x;
    int new_row = blockIdx.x * blockDim.x + local_y;
    
    if (new_row < cols && new_col < rows) {
        output[new_row * rows + new_col] = smem[local_x * (blockDim.x + 1) + local_y];
    }
}

void transpose_gpu_optimized(const float* d_input, float* d_output, int rows, int cols) {
    dim3 block_dim(CUDA_BLOCK_SIZE, CUDA_BLOCK_SIZE);
    dim3 grid_dim((cols + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE, (rows + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE);
    
    size_t smem_size = (CUDA_BLOCK_SIZE * (CUDA_BLOCK_SIZE + 1)) * sizeof(float);
    
    transpose_kernel_optimized<<<grid_dim, block_dim, smem_size>>>(d_input, d_output, rows, cols);
    CUDA_CHECK(cudaDeviceSynchronize());
}
