#pragma once

#include "filter_utils.h"
#include <cuda_runtime.h>
#include <string>
#include <vector>

namespace cuda_filter {

struct FilterStage {
    FilterType  type;
    cv::Mat     kernel;     // float32; unused for HDR stages
    HdrParams   hdrParams;  // used only when type == HDR_TONEMAPPING
    std::string name;
};

// Multi-stage GPU filter pipeline with CUDA stream support.
//
// apply() uses one dedicated CUDA stream per stage (multiStream=true) or a
// single stream for all stages (multiStream=false).  CUDA events enforce
// data-dependency ordering across streams while giving the GPU scheduler
// visibility into the full pipeline — enabling overlap between stage kernels
// and asynchronous memory transfers.
class FilterPipeline {
public:
    FilterPipeline();
    ~FilterPipeline();

    FilterPipeline(const FilterPipeline&)            = delete;
    FilterPipeline& operator=(const FilterPipeline&) = delete;

    void   addStage(FilterStage stage);
    void   removeStage(size_t index);
    void   clearStages();
    size_t stageCount() const { return m_stages.size(); }
    const FilterStage& getStage(size_t i) const { return m_stages[i]; }

    // Process input through all stages in order.
    // Returns per-stage kernel execution times in ms (empty when no stages).
    std::vector<float> apply(const cv::Mat& input, cv::Mat& output, bool multiStream);

private:
    std::vector<FilterStage>    m_stages;

    // Ping-pong device image buffers:
    //   m_dBufs[0]   = copy of host input
    //   m_dBufs[i+1] = output of stage i
    std::vector<unsigned char*> m_dBufs;
    std::vector<float*>         m_dKernels;  // null for HDR stages

    // One stream & two events per stage
    std::vector<cudaStream_t>   m_streams;
    std::vector<cudaEvent_t>    m_evStart;
    std::vector<cudaEvent_t>    m_evDone;

    int    m_width = 0, m_height = 0, m_channels = 0;
    size_t m_allocStages = 0;

    void allocate(int w, int h, int ch);
    void deallocate();
};

} // namespace cuda_filter
