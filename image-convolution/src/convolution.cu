#include "convolution.cuh"
#include "cuda_check.cuh"
#include "cuda_const.cuh"
#include "kernels.h"
#include <cmath>
#include <cstdlib>
#include <span>
#include <stdint.h>

__constant__ float d_filter[81]; // Max 9x9 filter

__global__ void convolutionKernelNaive(const uint8_t *input, uint8_t *output,
                                       const int width, const int height,
                                       const int channels,
                                       const int filterWidth) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  int halfFilter = filterWidth / 2;
  float sum[4] = {0};

  for (int j = -halfFilter; j <= halfFilter; j++) {
    for (int i = -halfFilter; i <= halfFilter; i++) {
      int imgX = min(max(x + i, 0), width - 1);
      int imgY = min(max(y + j, 0), height - 1);
      int imgIdx = (imgY * width + imgX) * channels;
      int filterIdx = (j + halfFilter) * filterWidth + (i + halfFilter);
      for (int c = 0; c < channels; c++)
        sum[c] += input[imgIdx + c] * d_filter[filterIdx];
    }
  }

  int outIdx = (y * width + x) * channels;
  for (int c = 0; c < channels; c++)
    output[outIdx + c] = (uint8_t)min(max((int)sum[c], 0), 255);
}

__global__ void convolutionKernelShared(const uint8_t *input, uint8_t *output,
                                        const int width, const int height,
                                        const int channels,
                                        const int filterWidth) {
  extern __shared__ uint8_t tile[];

  const int halfFilter = filterWidth / 2;
  const int sharedW = blockDim.x + 2 * halfFilter;
  const int sharedH = blockDim.y + 2 * halfFilter;
  const int sharedWSafe = 25;

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int outX = blockIdx.x * blockDim.x + tx;
  const int outY = blockIdx.y * blockDim.y + ty;

  const int startX = (int)(blockIdx.x * blockDim.x) - halfFilter;
  const int startY = (int)(blockIdx.y * blockDim.y) - halfFilter;
  const int tid = ty * blockDim.x + tx;
  const int numThreads = blockDim.x * blockDim.y;

  size_t tileIdx;
  for (int i = tid; i < sharedW * sharedH; i += numThreads) {
    int sy = i / sharedWSafe; // Use padded width for indexing
    int sx = i % sharedWSafe; // Use padded width for indexing
    int imgX = min(max(startX + sx, 0), width - 1);
    int imgY = min(max(startY + sy, 0), height - 1);
    for (int c = 0; c < channels; c++) {
      tileIdx = (sy * sharedWSafe + sx) * channels + c;
      tile[tileIdx] = input[(imgY * width + imgX) * channels + c];
    }
  }

  __syncthreads();

  if (outX >= width || outY >= height)
    return;

  float sum[4] = {0};
  size_t sy, sx;
  for (int j = -halfFilter; j <= halfFilter; j++) {
    for (int i = -halfFilter; i <= halfFilter; i++) {
      int filterIdx = (j + halfFilter) * filterWidth + (i + halfFilter);
      for (size_t c = 0; c < channels; c++) {
        sy = ty + halfFilter + j;
        sx = tx + halfFilter + i;
        tileIdx = (sy * sharedWSafe + sx) * channels + c;

        sum[c] += tile[tileIdx] * d_filter[filterIdx];
      }
    }
  }

  int outIdx = (outY * width + outX) * channels;
  for (int c = 0; c < channels; c++)
    output[outIdx + c] = (uint8_t)min(max((int)sum[c], 0), 255);
}

__global__ void convolutionKernelSeparableHorizontal(
    const uint8_t *input, float *output, const int width, const int height,
    const int channels, const int filterWidth) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  int halfFilter = filterWidth / 2;
  float sum[4] = {0};

  for (int i = -halfFilter; i <= halfFilter; i++) {
    int imgX = min(max(x + i, 0), width - 1);
    int imgIdx = (y * width + imgX) * channels;
    int filterIdx = i + halfFilter;
    for (int c = 0; c < channels; c++)
      sum[c] += input[imgIdx + c] * d_filter[filterIdx];
  }

  int outIdx = (y * width + x) * channels;
  for (int c = 0; c < channels; c++)
    output[outIdx + c] = sum[c];
}

// Separable filter: vertical pass (1D filter applied down columns)
__global__ void
convolutionKernelSeparableVertical(const float *input, uint8_t *output,
                                   const int width, const int height,
                                   const int channels, const int filterWidth) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  int halfFilter = filterWidth / 2;
  float sum[4] = {0};

  for (int j = -halfFilter; j <= halfFilter; j++) {
    int imgY = min(max(y + j, 0), height - 1);
    int imgIdx = (imgY * width + x) * channels;
    int filterIdx = j + halfFilter;
    for (int c = 0; c < channels; c++)
      sum[c] += input[imgIdx + c] * d_filter[filterIdx];
  }

  int outIdx = (y * width + x) * channels;
  for (int c = 0; c < channels; c++)
    output[outIdx + c] = (uint8_t)min(max((int)sum[c], 0), 255);
}

void convolutionGPU(const Image *input, Image *output,
                    const std::span<const float> filter) {
  size_t imageSize = input->width * input->height * input->channels;
  int filterWidth = (int)std::sqrt((double)filter.size());

  uint8_t *d_input, *d_output;
  CUDA_CHECK(cudaMalloc(&d_input, imageSize));
  CUDA_CHECK(cudaMalloc(&d_output, imageSize));
  CUDA_CHECK(
      cudaMemcpy(d_input, input->data, imageSize, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpyToSymbol(d_filter, filter.data(),
                                filter.size() * sizeof(float)));

  std::free(output->data);
  output->data = (uint8_t *)std::malloc(imageSize);
  output->width = input->width;
  output->height = input->height;
  output->channels = input->channels;

  dim3 block(CUDA_BLOCK_SIZE, CUDA_BLOCK_SIZE);
  dim3 grid((input->width + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE,
            (input->height + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE);
  convolutionKernelNaive<<<grid, block>>>(d_input, d_output, input->width,
                                          input->height, input->channels,
                                          filterWidth);
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(
      cudaMemcpy(output->data, d_output, imageSize, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_input));
  CUDA_CHECK(cudaFree(d_output));
}

void convolutionGPUShared(const Image *input, Image *output,
                          const std::span<const float> filter) {
  size_t imageSize = input->width * input->height * input->channels;
  int filterWidth = (int)std::sqrt((double)filter.size());

  uint8_t *d_input, *d_output;
  CUDA_CHECK(cudaMalloc(&d_input, imageSize));
  CUDA_CHECK(cudaMalloc(&d_output, imageSize));
  CUDA_CHECK(
      cudaMemcpy(d_input, input->data, imageSize, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpyToSymbol(d_filter, filter.data(),
                                filter.size() * sizeof(float)));

  std::free(output->data);
  output->data = (uint8_t *)std::malloc(imageSize);
  output->width = input->width;
  output->height = input->height;
  output->channels = input->channels;

  dim3 block(CUDA_BLOCK_SIZE, CUDA_BLOCK_SIZE);
  dim3 grid((input->width + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE,
            (input->height + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE);

  int halfFilter = filterWidth / 2;
  int sharedH = CUDA_BLOCK_SIZE + 2 * halfFilter;
  int sharedW = CUDA_BLOCK_SIZE + 2 * halfFilter;
  int sharedWSafe = sharedW + 1;

  size_t sharedMemSize =
      sharedH * sharedWSafe * input->channels * sizeof(uint8_t);
  convolutionKernelShared<<<grid, block, sharedMemSize>>>(
      d_input, d_output, input->width, input->height, input->channels,
      filterWidth);
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(
      cudaMemcpy(output->data, d_output, imageSize, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_input));
  CUDA_CHECK(cudaFree(d_output));
}

void convolutionGPUSeparable(const Image *input, Image *output,
                             const std::span<const float> filter) {
  size_t imageSize = input->width * input->height * input->channels;
  int filterWidth = (int)std::sqrt((double)filter.size());

  uint8_t *d_input, *d_output;
  float *d_intermediate;

  CUDA_CHECK(cudaMalloc(&d_input, imageSize));
  CUDA_CHECK(cudaMalloc(&d_output, imageSize));
  CUDA_CHECK(
      cudaMalloc(&d_intermediate, imageSize * sizeof(float) / sizeof(uint8_t)));
  CUDA_CHECK(
      cudaMemcpy(d_input, input->data, imageSize, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpyToSymbol(d_filter, filter.data(),
                                filter.size() * sizeof(float)));

  std::free(output->data);
  output->data = (uint8_t *)std::malloc(imageSize);
  output->width = input->width;
  output->height = input->height;
  output->channels = input->channels;

  dim3 block(CUDA_BLOCK_SIZE, CUDA_BLOCK_SIZE);
  dim3 grid((input->width + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE,
            (input->height + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE);

  // Horizontal pass
  convolutionKernelSeparableHorizontal<<<grid, block>>>(
      d_input, d_intermediate, input->width, input->height, input->channels,
      filterWidth);
  CUDA_CHECK(cudaGetLastError());

  // Vertical pass
  convolutionKernelSeparableVertical<<<grid, block>>>(
      d_intermediate, d_output, input->width, input->height, input->channels,
      filterWidth);
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(
      cudaMemcpy(output->data, d_output, imageSize, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_input));
  CUDA_CHECK(cudaFree(d_intermediate));
  CUDA_CHECK(cudaFree(d_output));
}
