#pragma once

#include <opencv2/opencv.hpp>

namespace cuda_filter {

// Left-to-right wipe between two images on the GPU.
// progress=0 → full imgA; progress=1 → full imgB.
// A 2-pixel white line marks the current boundary.
class WipeTransition {
public:
    WipeTransition();
    ~WipeTransition();

    WipeTransition(const WipeTransition&)            = delete;
    WipeTransition& operator=(const WipeTransition&) = delete;

    void apply(const cv::Mat& imgA, const cv::Mat& imgB,
               cv::Mat& output, float progress);

private:
    unsigned char* m_dA   = nullptr;
    unsigned char* m_dB   = nullptr;
    unsigned char* m_dOut = nullptr;
    int    m_width = 0, m_height = 0, m_channels = 0;

    void allocate(int w, int h, int ch);
    void deallocate();
};

} // namespace cuda_filter
