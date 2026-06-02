#pragma once

#include <opencv2/opencv.hpp>
#include <string>

namespace cuda_filter
{

    enum class FilterType
    {
        BLUR,
        SHARPEN,
        EDGE_DETECTION,
        EMBOSS,
        IDENTITY,
        HDR_TONEMAPPING
    };

    struct HdrParams
    {
        float exposure   = 1.0f;
        float gamma      = 1.0f;
        float saturation = 1.0f;
        int   algorithm  = 0;    // 0=Reinhard  1=Drago  2=Mantiuk
    };

    class FilterUtils
    {
    public:
        static FilterType stringToFilterType(const std::string &filterName);
        static cv::Mat createFilterKernel(FilterType type, int kernelSize, float intensity = 1.0f);

        static void applyFilterCPU(const cv::Mat &input, cv::Mat &output, const cv::Mat &kernel);
        static void applyHDRTonemapCPU(const cv::Mat &input, cv::Mat &output, const HdrParams &params);
    };

} // namespace cuda_filter
