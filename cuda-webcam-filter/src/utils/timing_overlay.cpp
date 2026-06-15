#include "timing_overlay.h"
#include <algorithm>
#include <cstdio>
#include <numeric>

namespace cuda_filter {

void drawTimingOverlay(cv::Mat& frame,
                       const std::vector<float>& timingsMs,
                       const std::vector<std::string>& labels,
                       const std::string& streamMode)
{
    if (timingsMs.empty()) return;

    const int N = static_cast<int>(timingsMs.size());

    // Layout constants
    const int BAR_MAX_W = 180;
    const int BAR_H     = 14;
    const int LABEL_W   = 90;   // left label column width
    const int VALUE_W   = 58;   // right numeric label width
    const int MARGIN    = 6;
    const int ROW_H     = BAR_H + MARGIN;
    const int HEADER_H  = ROW_H + MARGIN;
    const int FOOTER_H  = ROW_H + MARGIN;

    const int panelW = MARGIN + LABEL_W + MARGIN + BAR_MAX_W + MARGIN + VALUE_W + MARGIN;
    const int panelH = HEADER_H + N * ROW_H + FOOTER_H;

    const int px = frame.cols - panelW - 10;
    const int py = 10;

    if (px < 0 || py < 0) return;

    // Semi-transparent background (darken the region first)
    cv::Mat roi = frame(cv::Rect(px, py, panelW, panelH));
    roi *= 0.4;

    // Border
    cv::rectangle(frame, cv::Point(px, py),
                  cv::Point(px + panelW, py + panelH),
                  cv::Scalar(180, 180, 180), 1);

    // Header: stream mode
    cv::putText(frame, streamMode,
                cv::Point(px + MARGIN, py + HEADER_H - MARGIN),
                cv::FONT_HERSHEY_SIMPLEX, 0.40,
                cv::Scalar(255, 220, 60), 1, cv::LINE_AA);

    const float maxT = *std::max_element(timingsMs.begin(), timingsMs.end());

    for (int i = 0; i < N; ++i) {
        const int rowY = py + HEADER_H + i * ROW_H;

        // Stage label (truncated to fit)
        std::string lbl = (i < static_cast<int>(labels.size())) ? labels[i] : "stage";
        if (lbl.size() > 11) lbl = lbl.substr(0, 10) + "~";
        cv::putText(frame, lbl,
                    cv::Point(px + MARGIN, rowY + BAR_H - 1),
                    cv::FONT_HERSHEY_SIMPLEX, 0.33,
                    cv::Scalar(200, 200, 200), 1, cv::LINE_AA);

        // Timing bar
        const int barLen = (maxT > 0.0f)
            ? static_cast<int>(timingsMs[i] / maxT * BAR_MAX_W) : 0;
        const int bx = px + MARGIN + LABEL_W + MARGIN;
        cv::rectangle(frame,
                      cv::Point(bx, rowY + 2),
                      cv::Point(bx + barLen, rowY + BAR_H - 2),
                      cv::Scalar(0, 210, 120), cv::FILLED);

        // Numeric value
        char buf[24];
        std::snprintf(buf, sizeof(buf), "%.2f ms", timingsMs[i]);
        cv::putText(frame, buf,
                    cv::Point(bx + BAR_MAX_W + MARGIN, rowY + BAR_H - 1),
                    cv::FONT_HERSHEY_SIMPLEX, 0.33,
                    cv::Scalar(255, 255, 100), 1, cv::LINE_AA);
    }

    // Footer: total
    const float total = std::accumulate(timingsMs.begin(), timingsMs.end(), 0.0f);
    char tbuf[40];
    std::snprintf(tbuf, sizeof(tbuf), "Total: %.2f ms", total);
    const int footerY = py + HEADER_H + N * ROW_H + ROW_H - MARGIN;
    cv::putText(frame, tbuf,
                cv::Point(px + MARGIN, footerY),
                cv::FONT_HERSHEY_SIMPLEX, 0.40,
                cv::Scalar(255, 200, 60), 1, cv::LINE_AA);
}

} // namespace cuda_filter
