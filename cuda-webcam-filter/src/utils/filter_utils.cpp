#include "filter_utils.h"
#include <plog/Log.h>
#include <algorithm>
#include <cmath>

namespace cuda_filter
{

    FilterType FilterUtils::stringToFilterType(const std::string &filterName)
    {
        if (filterName == "blur")
            return FilterType::BLUR;
        if (filterName == "sharpen")
            return FilterType::SHARPEN;
        if (filterName == "edge")
            return FilterType::EDGE_DETECTION;
        if (filterName == "emboss")
            return FilterType::EMBOSS;
        if (filterName == "hdr")
            return FilterType::HDR_TONEMAPPING;

        PLOG_WARNING << "Unknown filter type: " << filterName << ". Using default blur filter.";
        return FilterType::BLUR;
    }

    cv::Mat FilterUtils::createFilterKernel(FilterType type, int kernelSize, float intensity)
    {
        // Ensure kernel size is odd
        if (kernelSize % 2 == 0)
        {
            kernelSize++;
        }

        cv::Mat kernel;

        switch (type)
        {
        case FilterType::BLUR:
            kernel = cv::Mat::ones(kernelSize, kernelSize, CV_32F) / (float)(kernelSize * kernelSize);
            break;

        case FilterType::SHARPEN:
            kernel = cv::Mat::zeros(kernelSize, kernelSize, CV_32F);
            kernel.at<float>(kernelSize / 2, kernelSize / 2) = 1.0f + 4.0f * intensity;
            if (kernelSize >= 3)
            {
                kernel.at<float>(kernelSize / 2 - 1, kernelSize / 2) = -intensity;
                kernel.at<float>(kernelSize / 2 + 1, kernelSize / 2) = -intensity;
                kernel.at<float>(kernelSize / 2, kernelSize / 2 - 1) = -intensity;
                kernel.at<float>(kernelSize / 2, kernelSize / 2 + 1) = -intensity;
            }
            break;

        case FilterType::EDGE_DETECTION:
            kernel = cv::Mat::zeros(kernelSize, kernelSize, CV_32F);
            if (kernelSize >= 3)
            {
                kernel.at<float>(0, 0) = -intensity;
                kernel.at<float>(0, 1) = -intensity;
                kernel.at<float>(0, 2) = -intensity;
                kernel.at<float>(1, 0) = -intensity;
                kernel.at<float>(1, 1) = 8.0f * intensity;
                kernel.at<float>(1, 2) = -intensity;
                kernel.at<float>(2, 0) = -intensity;
                kernel.at<float>(2, 1) = -intensity;
                kernel.at<float>(2, 2) = -intensity;
            }
            break;

        case FilterType::EMBOSS:
            kernel = cv::Mat::zeros(kernelSize, kernelSize, CV_32F);
            if (kernelSize >= 3)
            {
                kernel.at<float>(0, 0) = -2.0f * intensity;
                kernel.at<float>(0, 1) = -intensity;
                kernel.at<float>(0, 2) = 0.0f;
                kernel.at<float>(1, 0) = -intensity;
                kernel.at<float>(1, 1) = 1.0f;
                kernel.at<float>(1, 2) = intensity;
                kernel.at<float>(2, 0) = 0.0f;
                kernel.at<float>(2, 1) = intensity;
                kernel.at<float>(2, 2) = 2.0f * intensity;
            }
            break;

        case FilterType::HDR_TONEMAPPING:
            // Not a convolution filter — return a 1×1 identity placeholder
            kernel = cv::Mat::eye(1, 1, CV_32F);
            break;

        case FilterType::IDENTITY:
        default:
            kernel = cv::Mat::zeros(kernelSize, kernelSize, CV_32F);
            kernel.at<float>(kernelSize / 2, kernelSize / 2) = 1.0f;
            break;
        }

        return kernel;
    }

    void FilterUtils::applyFilterCPU(const cv::Mat &input, cv::Mat &output, const cv::Mat &kernel)
    {
        cv::filter2D(input, output, -1, kernel);
    }

    void FilterUtils::applyHDRTonemapCPU(const cv::Mat &input, cv::Mat &output, const HdrParams &p)
    {
        if (input.empty()) { PLOG_ERROR << "Input image is empty"; return; }

        output.create(input.rows, input.cols, input.type());

        for (int y = 0; y < input.rows; y++)
        {
            for (int x = 0; x < input.cols; x++)
            {
                const cv::Vec3b bgr = input.at<cv::Vec3b>(y, x);

                float b = bgr[0] / 255.0f * p.exposure;
                float g = bgr[1] / 255.0f * p.exposure;
                float r = bgr[2] / 255.0f * p.exposure;

                // Luminance-preserving saturation
                float gray = 0.2126f * r + 0.7152f * g + 0.0722f * b;
                r = std::max(0.0f, gray + p.saturation * (r - gray));
                g = std::max(0.0f, gray + p.saturation * (g - gray));
                b = std::max(0.0f, gray + p.saturation * (b - gray));

                // Tonemapping operator applied to luminance, colours scaled proportionally
                float L = 0.2126f * r + 0.7152f * g + 0.0722f * b;
                if (L > 1e-6f)
                {
                    float Ld = 0.0f;
                    switch (p.algorithm)
                    {
                    case 1: // Drago logarithmic
                        Ld = std::log(1.0f + L) /
                             std::log(2.0f + 8.0f * std::pow(L, std::log(0.85f) / std::log(0.5f)));
                        break;
                    case 2: // Mantiuk (simplified perceptual)
                        Ld = std::pow(L, 0.7f) / (1.0f + std::pow(L, 0.7f));
                        break;
                    default: // Reinhard
                        Ld = L / (1.0f + L);
                        break;
                    }
                    float scale = Ld / L;
                    r *= scale; g *= scale; b *= scale;
                }

                // Gamma correction
                float inv = 1.0f / std::max(p.gamma, 0.01f);
                r = std::pow(std::max(r, 0.0f), inv);
                g = std::pow(std::max(g, 0.0f), inv);
                b = std::pow(std::max(b, 0.0f), inv);

                output.at<cv::Vec3b>(y, x) = cv::Vec3b(
                    cv::saturate_cast<uchar>(b * 255.0f),
                    cv::saturate_cast<uchar>(g * 255.0f),
                    cv::saturate_cast<uchar>(r * 255.0f));
            }
        }
    }

} // namespace cuda_filter
