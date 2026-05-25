#pragma once
#include "kernels.h"
#include <span>

void convolutionGPU(const Image *input, Image *output,
                    const std::span<const float> filter);
void convolutionGPUShared(const Image *input, Image *output,
                          const std::span<const float> filter);
void convolutionGPUSeparable(const Image *input, Image *output,
                             const std::span<const float> filter);
