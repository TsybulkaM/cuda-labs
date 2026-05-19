#include "transposition.h"
#include <cmath>

void transpose_cpu(const float* input, float* output, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            output[j * rows + i] = input[i * cols + j];
        }
    }
}

bool verify_transpose(const float* original, const float* transposed, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            float expected = original[i * cols + j];
            float actual = transposed[j * rows + i];
            if (expected != actual) {
                return false;
            }
        }
    }
    return true;
}
