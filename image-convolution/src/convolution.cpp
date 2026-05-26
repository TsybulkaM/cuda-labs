#include "convolution.h"
#include "kernels.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <span>
#include <stdint.h>

void convolutionCPU(const Image *input, Image *output,
                    std::span<const float> filter) {
  int filterWidth = static_cast<int>(std::sqrt(filter.size()));
  int pad = filterWidth / 2;

  output->width = input->width;
  output->height = input->height;
  output->channels = input->channels;

  std::free(output->data);
  output->data =
      (uint8_t *)std::malloc(output->width * output->height * output->channels);

  for (int y = 0; y < input->height; y++) {
    for (int x = 0; x < input->width; x++) {
      for (int c = 0; c < input->channels; c++) {
        float sum = 0.0f;
        for (int j = -pad; j <= pad; j++) {
          for (int i = -pad; i <= pad; i++) {
            int imgX = std::clamp(x + i, 0, input->width - 1);
            int imgY = std::clamp(y + j, 0, input->height - 1);
            sum +=
                input
                    ->data[(imgY * input->width + imgX) * input->channels + c] *
                filter[(j + pad) * filterWidth + (i + pad)];
          }
        }
        output->data[(y * output->width + x) * output->channels + c] =
            static_cast<uint8_t>(std::clamp(sum, 0.0f, 255.0f));
      }
    }
  }
}