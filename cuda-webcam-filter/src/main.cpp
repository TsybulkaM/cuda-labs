#include "input_args_parser/input_args_parser.h"
#include "kernels/kernels.h"
#include "pipeline/filter_pipeline.h"
#include "pipeline/wipe_transition.h"
#include "utils/filter_utils.h"
#include "utils/input_handler.h"
#include "utils/timing_overlay.h"
#include <plog/Appenders/ColorConsoleAppender.h>
#include <plog/Formatters/TxtFormatter.h>
#include <plog/Initializers/RollingFileInitializer.h>
#include <plog/Log.h>
#include <algorithm>
#include <sstream>
#include <string>
#include <vector>

namespace {

// ---------------------------------------------------------------------------
// Parse pipeline spec string into FilterStage list.
// Format: "filterType[:kernelSize[:intensity]],..."
// HDR stages re-use the global HdrParams.
// ---------------------------------------------------------------------------
std::vector<cuda_filter::FilterStage>
parsePipelineSpec(const std::string& spec,
                  const cuda_filter::HdrParams& hdrParams)
{
    std::vector<cuda_filter::FilterStage> stages;
    std::istringstream ss(spec);
    std::string token;

    while (std::getline(ss, token, ',')) {
        if (token.empty()) continue;

        // Split token by ':'
        std::vector<std::string> parts;
        std::istringstream ts(token);
        std::string part;
        while (std::getline(ts, part, ':')) parts.push_back(part);

        const std::string& typeName = parts[0];
        const int   ksz       = (parts.size() > 1) ? std::stoi(parts[1]) : 3;
        const float intensity = (parts.size() > 2) ? std::stof(parts[2]) : 1.0f;

        cuda_filter::FilterType ft =
            cuda_filter::FilterUtils::stringToFilterType(typeName);

        cuda_filter::FilterStage stage;
        stage.type      = ft;
        stage.hdrParams = hdrParams;
        stage.name      = typeName + ":" + std::to_string(ksz);
        stage.kernel    = cuda_filter::FilterUtils::createFilterKernel(ft, ksz, intensity);

        stages.push_back(std::move(stage));
        PLOG_INFO << "Pipeline stage: " << typeName
                  << " ks=" << ksz << " int=" << intensity;
    }
    return stages;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
static std::string fpsText(const std::string& prefix, double fps, double ms)
{
    return prefix + " FPS: " + std::to_string(static_cast<int>(fps))
         + " Time: " + std::to_string(ms * 1000.0).substr(0, 4) + "ms";
}

} // anonymous namespace

// ===========================================================================
int main(int argc, char** argv)
{
    plog::ConsoleAppender<plog::TxtFormatter> consoleAppender;
    plog::init(plog::info, &consoleAppender);

    cuda_filter::InputArgsParser parser(argc, argv);
    cuda_filter::FilterOptions options = parser.parseArgs();

    cuda_filter::InputHandler inputHandler(options);
    if (!inputHandler.isOpened()) {
        PLOG_ERROR << "Failed to initialize input source";
        return -1;
    }

    const cuda_filter::HdrParams hdrParams{
        options.exposure, options.gamma, options.saturation, options.tonemapAlgo};

    // =========================================================================
    //  PIPELINE MODE
    // =========================================================================
    if (!options.pipelineSpec.empty()) {
        PLOG_INFO << "Pipeline mode — spec: " << options.pipelineSpec;
        PLOG_INFO << "Stream mode: "
                  << (options.singleStream ? "single" : "multi");
        PLOG_INFO << "Keys: ESC=quit  S=toggle stream mode  T=toggle timings  "
                     "W=toggle wipe  </>=wipe speed";

        auto stages = parsePipelineSpec(options.pipelineSpec, hdrParams);
        if (stages.empty()) {
            PLOG_ERROR << "No valid pipeline stages parsed from spec";
            return -1;
        }

        cuda_filter::FilterPipeline pipeline;
        for (auto& s : stages) pipeline.addStage(std::move(s));

        cuda_filter::WipeTransition wipe;

        // Build stage label list for the overlay
        std::vector<std::string> stageLabels;
        for (size_t i = 0; i < pipeline.stageCount(); ++i)
            stageLabels.push_back(pipeline.getStage(i).name);

        bool multiStream     = !options.singleStream;
        bool showTimings     = options.showTimings;
        bool wipeEnabled     = options.wipeTransition;
        float wipeSpeed      = std::clamp(options.wipeSpeed, 0.01f, 5.0f);
        float wipeProgress   = 0.0f;
        int   wipeDir        = 1;   // 1=advancing, -1=retreating (auto-bounce)

        cv::Mat frame, pipelineOut, displayFrame;
        double fps   = 0.0;
        int    fCnt  = 0;
        double tStart = static_cast<double>(cv::getTickCount());
        std::vector<float> lastTimings;

        cv::VideoWriter writer;
        bool writerInit = false;

        const bool headless =
            options.saveOutput &&
            (options.inputSource == cuda_filter::InputSource::IMAGE ||
             options.inputSource == cuda_filter::InputSource::VIDEO ||
             options.inputSource == cuda_filter::InputSource::SYNTHETIC);

        while (true) {
            if (!inputHandler.readFrame(frame)) {
                PLOG_ERROR << "Failed to read frame";
                break;
            }

            // Run pipeline
            lastTimings = pipeline.apply(frame, pipelineOut, multiStream);

            // Wipe transition: blend original and pipeline output
            if (wipeEnabled && !lastTimings.empty()) {
                // Advance wipe (auto-bounce)
                const float dt = (fps > 0.0) ? (1.0f / static_cast<float>(fps)) : 0.033f;
                wipeProgress += wipeSpeed * dt * static_cast<float>(wipeDir);
                if (wipeProgress >= 1.0f) { wipeProgress = 1.0f; wipeDir = -1; }
                if (wipeProgress <= 0.0f) { wipeProgress = 0.0f; wipeDir =  1; }
                wipe.apply(frame, pipelineOut, displayFrame, wipeProgress);
            } else {
                displayFrame = pipelineOut;
            }

            // FPS counter
            const double now = static_cast<double>(cv::getTickCount());
            ++fCnt;
            if ((now - tStart) / cv::getTickFrequency() >= 1.0) {
                fps    = fCnt;
                fCnt   = 0;
                tStart = now;
            }

            // Overlay: FPS + stream mode
            const std::string modeStr = multiStream ? "Multi-stream" : "Single-stream";
            cv::putText(displayFrame,
                        "FPS: " + std::to_string(static_cast<int>(fps))
                            + "  [" + modeStr + "]",
                        cv::Point(10, 28),
                        cv::FONT_HERSHEY_SIMPLEX, 0.65,
                        cv::Scalar(255, 220, 0), 2, cv::LINE_AA);

            if (wipeEnabled) {
                cv::putText(displayFrame,
                            "Wipe: " + std::to_string(static_cast<int>(wipeProgress * 100)) + "%",
                            cv::Point(10, 56),
                            cv::FONT_HERSHEY_SIMPLEX, 0.55,
                            cv::Scalar(0, 220, 255), 1, cv::LINE_AA);
            }

            // Timing overlay (bar chart)
            if (showTimings && !lastTimings.empty()) {
                cuda_filter::drawTimingOverlay(
                    displayFrame, lastTimings, stageLabels, modeStr);
            }

            if (!headless) inputHandler.displayFrame(displayFrame);

            // Save output
            if (options.saveOutput) {
                // Detect image-format output extensions — save a single frame.
                auto isImageExt = [](const std::string& p) {
                    auto ext = p.substr(p.rfind('.') + 1);
                    for (auto& c : ext) c = static_cast<char>(std::tolower(c));
                    return ext == "png" || ext == "jpg" || ext == "jpeg"
                        || ext == "bmp" || ext == "tiff" || ext == "tif";
                };

                const bool saveAsImage =
                    options.inputSource == cuda_filter::InputSource::IMAGE ||
                    (!options.outputPath.empty() && isImageExt(options.outputPath));

                if (saveAsImage) {
                    const std::string p = options.outputPath.empty()
                        ? (options.inputPath + "_pipeline.png")
                        : options.outputPath;
                    cv::imwrite(p, displayFrame);
                    PLOG_INFO << "Saved: " << p;
                    break;
                }
                if (!writerInit) {
                    cv::Size sz = inputHandler.getFrameSize();
                    if (sz.width == 0) sz = cv::Size(displayFrame.cols, displayFrame.rows);
                    const std::string p = options.outputPath.empty()
                        ? "pipeline_output.mp4" : options.outputPath;
                    const double vfps = inputHandler.getFPS();
                    writer.open(p, cv::VideoWriter::fourcc('m','p','4','v'),
                                vfps > 0 ? vfps : 30.0, sz, true);
                    if (writer.isOpened()) {
                        PLOG_INFO << "Recording: " << p;
                        writerInit = true;
                    }
                }
                if (writerInit) writer.write(displayFrame);
            }

            if (!headless) {
                const int key = cv::waitKey(1);
                if (key == 27) break;                   // ESC – exit
                if (key == 's' || key == 'S') {         // S – toggle stream mode
                    multiStream = !multiStream;
                    PLOG_INFO << "Switched to "
                              << (multiStream ? "multi" : "single") << "-stream";
                }
                if (key == 't' || key == 'T')           // T – toggle timing overlay
                    showTimings = !showTimings;
                if (key == 'w' || key == 'W')           // W – toggle wipe
                    wipeEnabled = !wipeEnabled;
                if (key == '<' || key == ',')            // < – slower wipe
                    wipeSpeed = std::max(0.01f, wipeSpeed * 0.8f);
                if (key == '>' || key == '.')            // > – faster wipe
                    wipeSpeed = std::min(5.0f, wipeSpeed * 1.25f);
            }
        }

        PLOG_INFO << "Pipeline terminated";
        return 0;
    }

    // =========================================================================
    //  ORIGINAL SINGLE-FILTER MODE (unchanged)
    // =========================================================================
    cuda_filter::FilterType filterType =
        cuda_filter::FilterUtils::stringToFilterType(options.filterType);
    cv::Mat kernel = cuda_filter::FilterUtils::createFilterKernel(
        filterType, options.kernelSize, options.intensity);

    const bool isHDR = (filterType == cuda_filter::FilterType::HDR_TONEMAPPING);

    if (isHDR)
        PLOG_INFO << "Filter: hdr, Tonemap: " << options.tonemapAlgo
                  << ", Exposure: " << options.exposure
                  << ", Gamma: " << options.gamma
                  << ", Saturation: " << options.saturation;
    else
        PLOG_INFO << "Filter: " << options.filterType
                  << ", Kernel size: " << options.kernelSize
                  << ", Intensity: " << options.intensity;

    cv::Mat frame, filteredCPU, filteredGPU;
    cv::VideoWriter writer;
    bool writerInitialized = false;
    double fpsCPU = 0.0, fpsGPU = 0.0;
    int frameCountCPU = 0, frameCountGPU = 0;
    double startTimeCPU = static_cast<double>(cv::getTickCount());
    double startTimeGPU = static_cast<double>(cv::getTickCount());

    PLOG_INFO << "Press 'ESC' to exit";

    while (true) {
        if (!inputHandler.readFrame(frame)) {
            PLOG_ERROR << "Failed to read frame";
            break;
        }

        const double cpuStart = static_cast<double>(cv::getTickCount());
        if (isHDR)
            cuda_filter::FilterUtils::applyHDRTonemapCPU(frame, filteredCPU, hdrParams);
        else
            cuda_filter::applyFilterCPU(frame, filteredCPU, kernel);
        const double cpuEnd  = static_cast<double>(cv::getTickCount());
        const double cpuTime = (cpuEnd - cpuStart) / cv::getTickFrequency();
        ++frameCountCPU;
        if ((cpuEnd - startTimeCPU) / cv::getTickFrequency() >= 1.0) {
            fpsCPU = frameCountCPU; frameCountCPU = 0; startTimeCPU = cpuEnd;
        }

        const double gpuStart = static_cast<double>(cv::getTickCount());
        if (isHDR)
            cuda_filter::applyHDRTonemapGPU(frame, filteredGPU, hdrParams);
        else
            cuda_filter::applyFilterGPU(frame, filteredGPU, kernel);
        const double gpuEnd  = static_cast<double>(cv::getTickCount());
        const double gpuTime = (gpuEnd - gpuStart) / cv::getTickFrequency();
        ++frameCountGPU;
        if ((gpuEnd - startTimeGPU) / cv::getTickFrequency() >= 1.0) {
            fpsGPU = frameCountGPU; frameCountGPU = 0; startTimeGPU = gpuEnd;
        }

        cv::putText(filteredCPU, fpsText("CPU", fpsCPU, cpuTime),
                    cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7,
                    cv::Scalar(255, 255, 0), 2);
        cv::putText(filteredGPU, fpsText("GPU", fpsGPU, gpuTime),
                    cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7,
                    cv::Scalar(255, 255, 0), 2);

        cv::Mat combined;
        cv::hconcat(filteredCPU, filteredGPU, combined);
        cv::putText(combined, "CPU Version",
                    cv::Point(10, combined.rows - 10),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 0), 2);
        cv::putText(combined, "GPU Version",
                    cv::Point(combined.cols / 2 + 10, combined.rows - 10),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 0), 2);

        const bool headless =
            options.saveOutput &&
            (options.inputSource == cuda_filter::InputSource::IMAGE ||
             options.inputSource == cuda_filter::InputSource::VIDEO ||
             options.inputSource == cuda_filter::InputSource::SYNTHETIC);
        if (!headless) {
            if (options.preview)
                inputHandler.displaySideBySide(frame, combined);
            else
                inputHandler.displayFrame(combined);
        }

        if (options.saveOutput) {
            if (options.inputSource == cuda_filter::InputSource::IMAGE) {
                const std::string p = options.outputPath.empty()
                    ? (options.inputPath + "_out.png") : options.outputPath;
                if (cv::imwrite(p, combined)) PLOG_INFO << "Saved: " << p;
                else PLOG_ERROR << "Failed to save: " << p;
                break;
            }
            if (!writerInitialized) {
                cv::Size sz = inputHandler.getFrameSize();
                const double fps_v = inputHandler.getFPS();
                if (sz.width == 0) sz = cv::Size(combined.cols, combined.rows);
                std::string p;
                if (!options.outputPath.empty()) p = options.outputPath;
                else if (options.inputSource == cuda_filter::InputSource::VIDEO)
                    p = options.inputPath + "_out.mp4";
                else if (options.inputSource == cuda_filter::InputSource::WEBCAM)
                    p = "webcam_output.mp4";
                else p = "output.mp4";

                const int fourcc = cv::VideoWriter::fourcc('m','p','4','v');
                writer.open(p, fourcc, fps_v > 0 ? fps_v : 30.0, sz, true);
                if (!writer.isOpened()) PLOG_ERROR << "Failed to open VideoWriter: " << p;
                else { PLOG_INFO << "Recording: " << p; writerInitialized = true; }
            }
            if (writerInitialized) writer.write(combined);
        }

        if (!headless && cv::waitKey(1) == 27) break;
    }

    PLOG_INFO << "Application terminated";
    return 0;
}
