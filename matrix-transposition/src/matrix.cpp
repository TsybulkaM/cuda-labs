#include "matrix.h"
#include <bits/stdc++.h>

void fill_random(float *matrix, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        matrix[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}