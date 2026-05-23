#pragma once

void transpose_cpu(const float* input, float* output, int rows, int cols);
bool verify_transpose(const float* original, const float* transposed, int rows, int cols);
