#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace cuda_filter {

// Draw a real-time bar-chart overlay showing per-stage pipeline timings.
//
// The panel appears in the top-right corner of `frame` and contains:
//   - stream mode label (Multi-stream / Single-stream)
//   - one horizontal bar per stage, width proportional to its kernel time
//   - per-stage numeric timing label
//   - total pipeline time
void drawTimingOverlay(cv::Mat& frame,
                       const std::vector<float>& timingsMs,
                       const std::vector<std::string>& labels,
                       const std::string& streamMode);

} // namespace cuda_filter
