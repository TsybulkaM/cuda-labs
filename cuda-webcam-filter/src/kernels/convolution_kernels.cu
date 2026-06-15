#include "kernels.h"
#include <cuda_runtime.h>
#include <plog/Log.h>

namespace cuda_filter
{

// CUDA error checking
#define CHECK_CUDA_ERROR(call)                                                          \
    {                                                                                   \
        cudaError_t err = call;                                                         \
        if (err != cudaSuccess)                                                         \
        {                                                                               \
            PLOG_ERROR << "CUDA error in " << #call << ": " << cudaGetErrorString(err); \
            return;                                                                     \
        }                                                                               \
    }

    // CUDA kernel for 2D convolution
    __global__ void convolutionKernel(const unsigned char *input, unsigned char *output,
                                      const float *kernel, int width, int height,
                                      int channels, int kernelSize)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x >= width || y >= height)
            return;

        int radius = kernelSize / 2;

        for (int c = 0; c < channels; c++)
        {
            float sum = 0.0f;

            for (int ky = -radius; ky <= radius; ky++)
            {
                for (int kx = -radius; kx <= radius; kx++)
                {
                    int ix = min(max(x + kx, 0), width - 1);
                    int iy = min(max(y + ky, 0), height - 1);

                    float kernelValue = kernel[(ky + radius) * kernelSize + (kx + radius)];
                    float pixelValue = input[(iy * width + ix) * channels + c];

                    sum += pixelValue * kernelValue;
                }
            }

            // Clamp the result to [0, 255]
            output[(y * width + x) * channels + c] = static_cast<unsigned char>(min(max(sum, 0.0f), 255.0f));
        }
    }

    void applyFilterGPU(const cv::Mat &input, cv::Mat &output, const cv::Mat &kernel)
    {
        if (input.empty() || kernel.empty())
        {
            PLOG_ERROR << "Input image or kernel is empty";
            return;
        }

        // Ensure output has the same size and type as input
        output.create(input.size(), input.type());

        // Get image dimensions
        int width = input.cols;
        int height = input.rows;
        int channels = input.channels();
        int kernelSize = kernel.rows;

        // Allocate device memory
        unsigned char *d_input = nullptr;
        unsigned char *d_output = nullptr;
        float *d_kernel = nullptr;

        size_t imageSize = width * height * channels * sizeof(unsigned char);
        size_t kernelSize_bytes = kernelSize * kernelSize * sizeof(float);

        // Copy kernel to CPU float array
        float *h_kernel = new float[kernelSize * kernelSize];
        for (int i = 0; i < kernelSize; i++)
        {
            for (int j = 0; j < kernelSize; j++)
            {
                h_kernel[i * kernelSize + j] = kernel.at<float>(i, j);
            }
        }

        // Allocate device memory
        CHECK_CUDA_ERROR(cudaMalloc(&d_input, imageSize));
        CHECK_CUDA_ERROR(cudaMalloc(&d_output, imageSize));
        CHECK_CUDA_ERROR(cudaMalloc(&d_kernel, kernelSize_bytes));

        // Copy data to device
        CHECK_CUDA_ERROR(cudaMemcpy(d_input, input.data, imageSize, cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMemcpy(d_kernel, h_kernel, kernelSize_bytes, cudaMemcpyHostToDevice));

        // Define block and grid dimensions
        dim3 blockDim(16, 16);
        dim3 gridDim(cuda::divUp(width, blockDim.x), cuda::divUp(height, blockDim.y));

        // Launch kernel
        convolutionKernel<<<gridDim, blockDim>>>(d_input, d_output, d_kernel, width, height, channels, kernelSize);

        // Check for kernel launch errors
        CHECK_CUDA_ERROR(cudaGetLastError());

        // Synchronize to ensure kernel execution is complete
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        // Copy result back to host
        CHECK_CUDA_ERROR(cudaMemcpy(output.data, d_output, imageSize, cudaMemcpyDeviceToHost));

        // Free device memory
        cudaFree(d_input);
        cudaFree(d_output);
        cudaFree(d_kernel);

        // Free host memory
        delete[] h_kernel;
    }

    void applyFilterCPU(const cv::Mat &input, cv::Mat &output, const cv::Mat &kernel)
    {
        if (input.empty() || kernel.empty())
        {
            PLOG_ERROR << "Input image or kernel is empty";
            return;
        }

        // Ensure output has the same size and type as input
        output.create(input.size(), input.type());

        // Get image dimensions
        int width = input.cols;
        int height = input.rows;
        int channels = input.channels();
        int kernelSize = kernel.rows;
        int radius = kernelSize / 2;

        // Convert kernel to float array for faster access
        float *h_kernel = new float[kernelSize * kernelSize];
        for (int i = 0; i < kernelSize; i++)
        {
            for (int j = 0; j < kernelSize; j++)
            {
                h_kernel[i * kernelSize + j] = kernel.at<float>(i, j);
            }
        }

        // Process each pixel
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                for (int c = 0; c < channels; c++)
                {
                    float sum = 0.0f;

                    // Apply kernel
                    for (int ky = -radius; ky <= radius; ky++)
                    {
                        for (int kx = -radius; kx <= radius; kx++)
                        {
                            int ix = std::min(std::max(x + kx, 0), width - 1);
                            int iy = std::min(std::max(y + ky, 0), height - 1);

                            float kernelValue = h_kernel[(ky + radius) * kernelSize + (kx + radius)];
                            float pixelValue = input.at<cv::Vec3b>(iy, ix)[c];

                            sum += pixelValue * kernelValue;
                        }
                    }

                    // Clamp the result to [0, 255]
                    output.at<cv::Vec3b>(y, x)[c] = static_cast<unsigned char>(std::min(std::max(sum, 0.0f), 255.0f));
                }
            }
        }

        delete[] h_kernel;
    }

    __device__ static float BT_709(float3 c)
    {
        return 0.0722f * c.x + 0.7152f * c.y + 0.2126f * c.z; // BGR luma
    }

    __device__ static float3 hdrSaturation(float3 c, float s)
    {
        float gray = BT_709(c);
        return make_float3(
            fmaxf(0.0f, gray + s * (c.x - gray)),
            fmaxf(0.0f, gray + s * (c.y - gray)),
            fmaxf(0.0f, gray + s * (c.z - gray)));
    }

    __device__ static float3 hdrTonemap(float3 c, int algo)
    {
        float L = BT_709(c);
        if (L < 1e-6f) return c;

        float Ld = 0.0f;
        switch (algo)
        {
        case 1: // Drago logarithmic
            Ld = logf(1.0f + L) /
                 logf(2.0f + 8.0f * powf(L, logf(0.85f) / logf(0.5f)));
            break;
        case 2: // Mantiuk simplified perceptual
            Ld = powf(L, 0.7f) / (1.0f + powf(L, 0.7f));
            break;
        default: // Reinhard
            Ld = L / (1.0f + L);
            break;
        }
        float scale = Ld / L;
        return make_float3(c.x * scale, c.y * scale, c.z * scale);
    }

    __global__ void hdrTonemapKernel(const unsigned char *input, unsigned char *output,
                                     int width, int height, int channels,
                                     float exposure, float gamma,
                                     float saturation, int algorithm)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (x >= width || y >= height) return;

        int idx = (y * width + x) * channels;
        float3 c = make_float3(input[idx]     / 255.0f * exposure,
                               input[idx + 1] / 255.0f * exposure,
                               input[idx + 2] / 255.0f * exposure);

        c = hdrSaturation(c, saturation);
        c = hdrTonemap(c, algorithm);

        float inv = 1.0f / fmaxf(gamma, 0.01f);
        c.x = powf(fmaxf(c.x, 0.0f), inv);
        c.y = powf(fmaxf(c.y, 0.0f), inv);
        c.z = powf(fmaxf(c.z, 0.0f), inv);

        output[idx]     = (unsigned char)fminf(c.x * 255.0f, 255.0f);
        output[idx + 1] = (unsigned char)fminf(c.y * 255.0f, 255.0f);
        output[idx + 2] = (unsigned char)fminf(c.z * 255.0f, 255.0f);
    }

    void applyHDRTonemapGPU(const cv::Mat &input, cv::Mat &output, const HdrParams &p)
    {
        if (input.empty()) { PLOG_ERROR << "Input image is empty"; return; }

        output.create(input.rows, input.cols, input.type());

        int width = input.cols, height = input.rows, channels = input.channels();
        size_t sz = width * height * channels * sizeof(unsigned char);

        unsigned char *d_in = nullptr, *d_out = nullptr;
        CHECK_CUDA_ERROR(cudaMalloc(&d_in,  sz));
        CHECK_CUDA_ERROR(cudaMalloc(&d_out, sz));
        CHECK_CUDA_ERROR(cudaMemcpy(d_in, input.data, sz, cudaMemcpyHostToDevice));

        dim3 block(16, 16);
        dim3 grid(cuda::divUp(width, 16), cuda::divUp(height, 16));
        hdrTonemapKernel<<<grid, block>>>(d_in, d_out, width, height, channels,
                                          p.exposure, p.gamma, p.saturation, p.algorithm);

        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        CHECK_CUDA_ERROR(cudaMemcpy(output.data, d_out, sz, cudaMemcpyDeviceToHost));

        cudaFree(d_in);
        cudaFree(d_out);
    }

    // ---- stream-aware wrappers (no host memory management) ------------------

    void applyConvolutionOnStream(
        const unsigned char* d_input, unsigned char* d_output,
        const float* d_kernel, int width, int height, int channels, int kernelSize,
        cudaStream_t stream)
    {
        dim3 block(16, 16);
        dim3 grid(cuda::divUp(width, 16), cuda::divUp(height, 16));
        convolutionKernel<<<grid, block, 0, stream>>>(
            d_input, d_output, d_kernel, width, height, channels, kernelSize);
    }

    void applyHDROnStream(
        const unsigned char* d_input, unsigned char* d_output,
        int width, int height, int channels,
        const HdrParams& params, cudaStream_t stream)
    {
        dim3 block(16, 16);
        dim3 grid(cuda::divUp(width, 16), cuda::divUp(height, 16));
        hdrTonemapKernel<<<grid, block, 0, stream>>>(
            d_input, d_output, width, height, channels,
            params.exposure, params.gamma, params.saturation, params.algorithm);
    }

} // namespace cuda_filter
