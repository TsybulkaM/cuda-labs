#pragma once

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

static inline void CUDA_CHECK(cudaError_t result) {
	if (result != cudaSuccess) {
		fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
		exit(EXIT_FAILURE);
	}
}