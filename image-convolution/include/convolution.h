#pragma once
#include "kernels.h"
#include <span>

// CPU implementation of 2D convolution
void convolutionCPU(const Image *input, Image *output, const std::span<const float> filter);