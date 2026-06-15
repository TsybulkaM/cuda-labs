#pragma once

#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include "filter_utils.h"

namespace cuda_filter
{

void applyFilterGPU(const cv::Mat &input, cv::Mat &output, const cv::Mat &kernel);
void applyFilterCPU(const cv::Mat &input, cv::Mat &output, const cv::Mat &kernel);
void applyHDRTonemapGPU(const cv::Mat &input, cv::Mat &output, const HdrParams &params);

// Stream-aware variants: operate on pre-allocated device buffers (no host transfers)
void applyConvolutionOnStream(
    const unsigned char* d_input, unsigned char* d_output,
    const float* d_kernel, int width, int height, int channels, int kernelSize,
    cudaStream_t stream);

void applyHDROnStream(
    const unsigned char* d_input, unsigned char* d_output,
    int width, int height, int channels,
    const HdrParams& params, cudaStream_t stream);

    namespace cuda
    {
// CUDA-specific type declarations and helper functions
#ifdef __CUDACC__
        // These will only be visible to CUDA compiler
        __host__ __device__ inline int divUp(int a, int b)
        {
            return (a + b - 1) / b;
        }
#endif
    }

} // namespace cuda_filter
