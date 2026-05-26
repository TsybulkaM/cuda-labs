#pragma once

void transpose_gpu_naive(const float* d_input, float* d_output, int rows, int cols);
void transpose_gpu_optimized(const float* d_input, float* d_output, int rows, int cols);
