#include "wipe_transition.h"
#include <cuda_runtime.h>
#include <plog/Log.h>
#include <algorithm>

namespace cuda_filter {

#define CHK(call)                                                                \
    do {                                                                         \
        cudaError_t _e = (call);                                                 \
        if (_e != cudaSuccess)                                                   \
            PLOG_ERROR << "CUDA " #call ": " << cudaGetErrorString(_e);          \
    } while (0)

// ---- kernel -----------------------------------------------------------------

__global__ static void wipeKernel(
    const unsigned char* __restrict__ imgA,
    const unsigned char* __restrict__ imgB,
    unsigned char* __restrict__ output,
    int width, int height, int channels, int wipeX)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = (y * width + x) * channels;

    // Draw a 2-pixel white boundary line
    if (x == wipeX || x == wipeX - 1) {
        for (int c = 0; c < channels; ++c) output[idx + c] = 255;
    } else {
        const unsigned char* src = (x < wipeX) ? imgA : imgB;
        for (int c = 0; c < channels; ++c) output[idx + c] = src[idx + c];
    }
}

// ---- class implementation --------------------------------------------------

WipeTransition::WipeTransition()  = default;
WipeTransition::~WipeTransition() { deallocate(); }

void WipeTransition::deallocate()
{
    if (m_dA)   cudaFree(m_dA);
    if (m_dB)   cudaFree(m_dB);
    if (m_dOut) cudaFree(m_dOut);
    m_dA = m_dB = m_dOut = nullptr;
}

void WipeTransition::allocate(int w, int h, int ch)
{
    deallocate();
    m_width = w;  m_height = h;  m_channels = ch;
    const size_t sz = static_cast<size_t>(w) * h * ch;
    CHK(cudaMalloc(&m_dA,   sz));
    CHK(cudaMalloc(&m_dB,   sz));
    CHK(cudaMalloc(&m_dOut, sz));
}

void WipeTransition::apply(const cv::Mat& imgA, const cv::Mat& imgB,
                            cv::Mat& output, float progress)
{
    const int w  = imgA.cols, h = imgA.rows, ch = imgA.channels();
    const size_t sz = static_cast<size_t>(w) * h * ch;

    if (w != m_width || h != m_height || ch != m_channels)
        allocate(w, h, ch);

    output.create(h, w, imgA.type());

    CHK(cudaMemcpy(m_dA, imgA.data, sz, cudaMemcpyHostToDevice));
    CHK(cudaMemcpy(m_dB, imgB.data, sz, cudaMemcpyHostToDevice));

    const float  p     = std::clamp(progress, 0.0f, 1.0f);
    const int    wipeX = static_cast<int>(p * w);

    dim3 block(16, 16);
    dim3 grid((w + 15) / 16, (h + 15) / 16);
    wipeKernel<<<grid, block>>>(m_dA, m_dB, m_dOut, w, h, ch, wipeX);

    CHK(cudaGetLastError());
    CHK(cudaDeviceSynchronize());
    CHK(cudaMemcpy(output.data, m_dOut, sz, cudaMemcpyDeviceToHost));
}

} // namespace cuda_filter
