#include "input_args_parser/input_args_parser.h"
#include "kernels/kernels.h"
#include "utils/filter_utils.h"
#include "utils/input_handler.h"
#include <plog/Appenders/ColorConsoleAppender.h>
#include <plog/Formatters/TxtFormatter.h>
#include <plog/Initializers/RollingFileInitializer.h>
#include <plog/Log.h>

int main(int argc, char **argv) {
  // Initialize logger
  plog::ConsoleAppender<plog::TxtFormatter> consoleAppender;
  plog::init(plog::info, &consoleAppender);

  // Parse command line arguments
  cuda_filter::InputArgsParser parser(argc, argv);
  cuda_filter::FilterOptions options = parser.parseArgs();

  // Initialize input handler
  cuda_filter::InputHandler inputHandler(options);
  if (!inputHandler.isOpened()) {
    PLOG_ERROR << "Failed to initialize input source";
    return -1;
  }

  // Create filter kernel
  cuda_filter::FilterType filterType =
      cuda_filter::FilterUtils::stringToFilterType(options.filterType);
  cv::Mat kernel = cuda_filter::FilterUtils::createFilterKernel(
      filterType, options.kernelSize, options.intensity);

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
    // Capture frame
    if (!inputHandler.readFrame(frame)) {
      PLOG_ERROR << "Failed to read frame";
      break;
    }

    // Apply filter using CPU
    const double cpuStart = static_cast<double>(cv::getTickCount());
    cuda_filter::applyFilterCPU(frame, filteredCPU, kernel);
    const double cpuEnd = static_cast<double>(cv::getTickCount());
    const double cpuTime = (cpuEnd - cpuStart) / cv::getTickFrequency();
    frameCountCPU++;
    if ((cpuEnd - startTimeCPU) / cv::getTickFrequency() >= 1.0) {
      fpsCPU = frameCountCPU;
      frameCountCPU = 0;
      startTimeCPU = cpuEnd;
    }

    // Apply filter using GPU
    const double gpuStart = static_cast<double>(cv::getTickCount());
    cuda_filter::applyFilterGPU(frame, filteredGPU, kernel);
    const double gpuEnd = static_cast<double>(cv::getTickCount());
    const double gpuTime = (gpuEnd - gpuStart) / cv::getTickFrequency();
    frameCountGPU++;
    if ((gpuEnd - startTimeGPU) / cv::getTickFrequency() >= 1.0) {
      fpsGPU = frameCountGPU;
      frameCountGPU = 0;
      startTimeGPU = gpuEnd;
    }

    // Add FPS and processing time text to the frames
    std::string cpuText =
        "CPU FPS: " + std::to_string(static_cast<int>(fpsCPU)) +
        " Time: " + std::to_string(cpuTime * 1000).substr(0, 4) + "ms";
    std::string gpuText =
        "GPU FPS: " + std::to_string(static_cast<int>(fpsGPU)) +
        " Time: " + std::to_string(gpuTime * 1000).substr(0, 4) + "ms";

    cv::putText(filteredCPU, cpuText, cv::Point(10, 30),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 0), 2);
    cv::putText(filteredGPU, gpuText, cv::Point(10, 30),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 0), 2);

    // Create a combined image showing both results
    cv::Mat combined;
    cv::hconcat(filteredCPU, filteredGPU, combined);
    cv::putText(combined, "CPU Version", cv::Point(10, combined.rows - 10),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 0), 2);
    cv::putText(combined, "GPU Version",
                cv::Point(combined.cols / 2 + 10, combined.rows - 10),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 0), 2);

    // Display the combined result
    if (options.preview) {
      inputHandler.displaySideBySide(frame, combined);
    } else {
      inputHandler.displayFrame(combined);
    }

    // Handle saving output if requested
    if (options.saveOutput) {
      if (options.inputSource == cuda_filter::InputSource::IMAGE) {
        std::string outPath = options.outputPath.empty()
                                  ? (options.inputPath + "_out.png")
                                  : options.outputPath;
        if (cv::imwrite(outPath, combined)) {
          PLOG_INFO << "Saved output image: " << outPath;
        } else {
          PLOG_ERROR << "Failed to save output image: " << outPath;
        }
        break; // only one frame for image
      }

      // For video/webcam/synthetic, initialize writer on first frame
      if (!writerInitialized) {
        cv::Size sz = inputHandler.getFrameSize();
        double fps = inputHandler.getFPS();
        if (sz.width == 0 || sz.height == 0) {
          sz = cv::Size(combined.cols, combined.rows);
        }
        std::string outPath;
        if (!options.outputPath.empty()) {
          outPath = options.outputPath;
        } else {
          if (options.inputSource == cuda_filter::InputSource::VIDEO) {
            outPath = options.inputPath + "_out.mp4";
          } else if (options.inputSource == cuda_filter::InputSource::WEBCAM) {
            outPath = "webcam_output.mp4";
          } else {
            outPath = "output.mp4";
          }
        }

        int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
        writer.open(outPath, fourcc, fps > 0.0 ? fps : 30.0, sz, true);
        if (!writer.isOpened()) {
          PLOG_ERROR << "Failed to open VideoWriter for: " << outPath;
        } else {
          PLOG_INFO << "Recording output to: " << outPath;
          writerInitialized = true;
        }
      }

      if (writerInitialized) {
        writer.write(combined);
      }
    }
    // Exit on ESC key
    if (cv::waitKey(1) == 27) {
      break;
    }
  }

  PLOG_INFO << "Application terminated";
  return 0;
}
