#pragma once

#include <cxxopts.hpp>
#include <string>

namespace cuda_filter {

enum class InputSource { WEBCAM, IMAGE, VIDEO, SYNTHETIC };

enum class SyntheticPattern { CHECKERBOARD, GRADIENT, NOISE };

struct FilterOptions {
  InputSource inputSource;
  std::string inputPath;
  SyntheticPattern syntheticPattern;
  int deviceId;
  std::string filterType;
  int kernelSize;
  float sigma;
  float intensity;
  bool preview;
  bool saveOutput;
  std::string outputPath;
  // HDR tonemapping parameters
  float exposure   = 1.0f;
  float gamma      = 1.0f;
  float saturation = 1.0f;
  int   tonemapAlgo = 0;   // 0=reinhard  1=drago  2=mantiuk

  // ---- Pipeline mode -------------------------------------------------------
  // When pipelineSpec is non-empty the app runs in pipeline mode.
  // Format: "filterType[:kernelSize[:intensity]],..."
  // e.g.  "blur:3:1.0,sharpen:5:2.0,edge:3:1.5"
  // HDR stages inherit the global exposure/gamma/saturation/tonemapAlgo values.
  std::string pipelineSpec;
  bool singleStream = false;  // force single CUDA stream (for comparison)
  bool showTimings  = false;  // draw per-stage timing overlay

  // ---- Wipe transition (pipeline mode only) --------------------------------
  bool  wipeTransition = false;
  float wipeSpeed      = 0.3f;  // full left-to-right sweep per second
};

class InputArgsParser {
public:
  InputArgsParser(int argc, char **argv);

  FilterOptions parseArgs();

private:
  int m_argc;
  char **m_argv;

  void setupOptions(cxxopts::Options &options);
  InputSource stringToInputSource(const std::string &str);
  SyntheticPattern stringToSyntheticPattern(const std::string &str);
};

} // namespace cuda_filter
