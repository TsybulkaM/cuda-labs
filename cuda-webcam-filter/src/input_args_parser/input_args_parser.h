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
