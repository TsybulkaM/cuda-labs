#include "filter_pipeline.h"
#include "kernels.h"
#include <plog/Log.h>
#include <vector>

namespace cuda_filter {

#define CHK(call)                                                                \
    do {                                                                         \
        cudaError_t _e = (call);                                                 \
        if (_e != cudaSuccess)                                                   \
            PLOG_ERROR << "CUDA " #call ": " << cudaGetErrorString(_e);          \
    } while (0)

FilterPipeline::FilterPipeline()  = default;
FilterPipeline::~FilterPipeline() { deallocate(); }

void FilterPipeline::addStage(FilterStage stage)
{
    m_stages.push_back(std::move(stage));
}

void FilterPipeline::removeStage(size_t index)
{
    if (index < m_stages.size())
        m_stages.erase(m_stages.begin() + static_cast<ptrdiff_t>(index));
}

void FilterPipeline::clearStages() { m_stages.clear(); }

// ---- GPU resource management -----------------------------------------------

void FilterPipeline::deallocate()
{
    for (auto* p : m_dBufs)    if (p) cudaFree(p);
    for (auto* p : m_dKernels) if (p) cudaFree(p);
    for (auto  s : m_streams)  cudaStreamDestroy(s);
    for (auto  e : m_evStart)  cudaEventDestroy(e);
    for (auto  e : m_evDone)   cudaEventDestroy(e);
    m_dBufs.clear();    m_dKernels.clear();
    m_streams.clear();  m_evStart.clear();  m_evDone.clear();
    m_allocStages = 0;
}

void FilterPipeline::allocate(int w, int h, int ch)
{
    deallocate();
    m_width = w;  m_height = h;  m_channels = ch;
    const size_t N     = m_stages.size();
    const size_t imgSz = static_cast<size_t>(w) * h * ch;

    // N+1 image buffers: [0]=input copy, [1..N]=stage outputs
    m_dBufs.assign(N + 1, nullptr);
    for (auto& p : m_dBufs) CHK(cudaMalloc(&p, imgSz));

    m_dKernels.assign(N, nullptr);
    m_streams.resize(N);
    m_evStart.resize(N);
    m_evDone.resize(N);

    for (size_t i = 0; i < N; ++i) {
        CHK(cudaStreamCreate(&m_streams[i]));
        CHK(cudaEventCreate(&m_evStart[i]));
        CHK(cudaEventCreate(&m_evDone[i]));

        if (m_stages[i].type != FilterType::HDR_TONEMAPPING) {
            const int ksz = m_stages[i].kernel.rows;
            CHK(cudaMalloc(&m_dKernels[i], static_cast<size_t>(ksz) * ksz * sizeof(float)));
            std::vector<float> tmp(static_cast<size_t>(ksz) * ksz);
            for (int r = 0; r < ksz; r++)
                for (int c = 0; c < ksz; c++)
                    tmp[r * ksz + c] = m_stages[i].kernel.at<float>(r, c);
            CHK(cudaMemcpy(m_dKernels[i], tmp.data(),
                           static_cast<size_t>(ksz) * ksz * sizeof(float),
                           cudaMemcpyHostToDevice));
        }
    }
    m_allocStages = N;
}

// ---- pipeline execution ----------------------------------------------------

std::vector<float> FilterPipeline::apply(const cv::Mat& input,
                                          cv::Mat&       output,
                                          bool           multiStream)
{
    if (m_stages.empty()) {
        input.copyTo(output);
        return {};
    }

    const int    w  = input.cols, h = input.rows, ch = input.channels();
    const size_t N  = m_stages.size();
    const size_t sz = static_cast<size_t>(w) * h * ch;

    if (m_allocStages != N || w != m_width || h != m_height || ch != m_channels)
        allocate(w, h, ch);

    output.create(h, w, input.type());

    // In single-stream mode every operation goes on streams[0].
    auto S = [&](size_t i) -> cudaStream_t {
        return multiStream ? m_streams[i] : m_streams[0];
    };

    // Upload input on stream 0 (stage 0 also uses stream 0 so no event needed)
    CHK(cudaMemcpyAsync(m_dBufs[0], input.data, sz,
                        cudaMemcpyHostToDevice, S(0)));

    // Launch each stage
    for (size_t i = 0; i < N; ++i) {
        cudaStream_t s = S(i);

        // Multi-stream: wait for the previous stage to finish writing its output
        if (multiStream && i > 0)
            CHK(cudaStreamWaitEvent(m_streams[i], m_evDone[i - 1], 0));

        CHK(cudaEventRecord(m_evStart[i], s));

        if (m_stages[i].type == FilterType::HDR_TONEMAPPING) {
            applyHDROnStream(m_dBufs[i], m_dBufs[i + 1],
                             w, h, ch, m_stages[i].hdrParams, s);
        } else {
            applyConvolutionOnStream(m_dBufs[i], m_dBufs[i + 1],
                                     m_dKernels[i], w, h, ch,
                                     m_stages[i].kernel.rows, s);
        }

        CHK(cudaEventRecord(m_evDone[i], s));
    }

    // Download result on stream 0; if multi-stream, wait for the last stage first
    if (multiStream && N > 1)
        CHK(cudaStreamWaitEvent(m_streams[0], m_evDone[N - 1], 0));

    CHK(cudaMemcpyAsync(output.data, m_dBufs[N], sz,
                        cudaMemcpyDeviceToHost, S(0)));

    // Synchronize all active streams
    const size_t usedStreams = multiStream ? N : 1;
    for (size_t i = 0; i < usedStreams; ++i)
        CHK(cudaStreamSynchronize(m_streams[i]));

    // Collect per-stage kernel timings
    std::vector<float> timings(N, 0.0f);
    for (size_t i = 0; i < N; ++i)
        cudaEventElapsedTime(&timings[i], m_evStart[i], m_evDone[i]);

    return timings;
}

} // namespace cuda_filter
