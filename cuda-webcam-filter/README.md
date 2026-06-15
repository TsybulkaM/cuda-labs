# CUDA Webcam Filter

## Purpose

Real-time GPU-accelerated image filter application demonstrating CUDA convolution,
HDR tonemapping, a multi-stage **filter pipeline** with CUDA stream concurrency,
a GPU **wipe transition**, and live **performance instrumentation**.

---

## Features

- Real-time webcam video capture
- Multiple convolution filters: blur, sharpen, edge detection, emboss
- HDR tonemapping with three algorithms (Reinhard, Drago, Mantiuk)
- **Filter pipeline** — chain any number of filters applied sequentially
- **Multi-stream pipeline** — one CUDA stream per stage + event-based inter-stage synchronisation
- **Wipe transition** — GPU kernel that sweeps between original and filtered output
- **Real-time timing overlay** — per-stage bar chart rendered on the live frame
- Runtime switching between single-stream and multi-stream mode (press `S`)
- Side-by-side CPU vs. GPU comparison (original single-filter mode)
- Headless operation for cloud / container environments

---

## Quick Start

```bash
# Build
mkdir build && cd build
cmake ..
cmake --build . -j $(nproc)

# Single filter — classic CPU/GPU side-by-side
./cuda-webcam-filter --filter blur

# Three-stage pipeline on webcam
./cuda-webcam-filter --pipeline "blur:3:1.0,sharpen:5:2.0,edge:3:1.5" --show-timings

# Pipeline with wipe transition (wipes between original and filtered)
./cuda-webcam-filter --pipeline "blur:3,sharpen:5" --wipe --wipe-speed 0.4

# Pipeline + compare single vs multi-stream interactively
./cuda-webcam-filter --pipeline "blur:3,edge:3,emboss:3" --show-timings
# Then press S to toggle single/multi-stream and observe timing changes
```

---

## Pipeline Mode

### Activating the pipeline

Pass `--pipeline` with a comma-separated filter specification.  The original
single-filter CPU/GPU comparison mode remains active when `--pipeline` is omitted.

```
--pipeline "blur:3:1.0,sharpen:5:2.0,edge:3:1.5,hdr"
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
           each element: filterType[:kernelSize[:intensity]]
```

| Field | Default | Notes |
|-------|---------|-------|
| `filterType` | — | `blur` `sharpen` `edge` `emboss` `hdr` `identity` |
| `kernelSize` | 3 | Odd integer; ignored for `hdr` |
| `intensity` | 1.0 | Strength multiplier; ignored for `hdr` |

`hdr` stages inherit the global `--exposure`, `--gamma`, `--saturation`,
`--tonemap` values.

### Pipeline examples

```bash
# Blur → sharpen → edge detection
./cuda-webcam-filter --pipeline "blur:5:1.0,sharpen:3:2.0,edge:3:1.5"

# HDR → emboss
./cuda-webcam-filter --pipeline "hdr,emboss:3:1.0" --tonemap drago --exposure 1.5

# Process a video file and save the pipeline output
./cuda-webcam-filter \
    --input video --path clip.mp4 \
    --pipeline "blur:3,sharpen:5,edge:3" \
    --show-timings --save --out pipeline_out.mp4

# Synthetic input — no webcam required
./cuda-webcam-filter-portable \
    --input synthetic --synthetic gradient \
    --pipeline "blur:3,emboss:5:2.0" \
    --wipe --show-timings --save --out pipeline_synthetic.png
```

### Pipeline keyboard controls

| Key | Action |
|-----|--------|
| `ESC` | Quit |
| `S` | Toggle **single-stream ↔ multi-stream** (live performance comparison) |
| `T` | Toggle per-stage timing overlay |
| `W` | Toggle wipe transition on/off |
| `<` / `,` | Decrease wipe speed |
| `>` / `.` | Increase wipe speed |

---

## Wipe Transition

`--wipe` enables a GPU-accelerated left-to-right wipe between the **original
frame** (left side) and the **pipeline output** (right side).  A 2-pixel white
line marks the current boundary.

The wipe auto-bounces: it advances to the right then retreats to the left,
repeating continuously.

```bash
# Wipe at 0.3 screen-widths per second (default)
./cuda-webcam-filter --pipeline "sharpen:5:2.0,edge:3" --wipe

# Faster wipe
./cuda-webcam-filter --pipeline "blur:7,emboss:3" --wipe --wipe-speed 0.8
```

| Option | Default | Description |
|--------|---------|-------------|
| `--wipe` | off | Enable wipe transition |
| `--wipe-speed` | 0.3 | Full-width sweeps per second (0.01 – 5.0) |

---

## Performance Analysis

### Instrumentation

Every pipeline stage is timed with CUDA events (`cudaEventRecord` before and
after the kernel launch).  `cudaEventElapsedTime` extracts the exact GPU
execution time for each stage independently.

### Real-time overlay (`--show-timings`)

A bar chart drawn on the live frame shows:

- Mode label: **Multi-stream** or **Single-stream**
- One horizontal bar per stage, width proportional to its GPU time
- Numeric `ms` value beside each bar
- **Total pipeline time** at the bottom

### Single-stream vs multi-stream

The app runs in **multi-stream** mode by default.  Press `S` (or pass
`--single-stream`) to switch.

| Mode | Behaviour |
|------|-----------|
| Multi-stream | One `cudaStream_t` per stage; `cudaStreamWaitEvent` enforces data-dependencies while exposing the full pipeline graph to the GPU scheduler |
| Single-stream | All operations serialised on `stream[0]`; no inter-stage events needed |

**What to observe when switching:**

- For computation-heavy stages (large kernels, many stages) multi-stream is
  marginally faster — the GPU scheduler can fill bubbles between stages from
  different streams.
- For memory-bound stages the PCIe overlap benefit dominates; multi-stream
  allows the host-to-device copy of the next frame to overlap with earlier
  stage kernels from the current frame (visible in Nsight Systems).
- The timing overlay updates every frame, so toggle `S` and watch the per-stage
  bars and total time change in real time.

### Profiling with NVIDIA Nsight Systems

```bash
nsys profile --trace=cuda,osrt \
    ./cuda-webcam-filter \
    --input synthetic --synthetic gradient \
    --pipeline "blur:3,sharpen:5,edge:3" \
    --show-timings --save --out /dev/null
```

Open the generated `.nsys-rep` in the Nsight Systems GUI to see:
- Separate stream lanes for each pipeline stage
- `cudaStreamWaitEvent` stall markers showing inter-stage dependencies
- PCIe transfer / kernel overlap across frames

---

## All Options

```
  -i, --input arg        Input source: webcam | image | video | synthetic (default: webcam)
  -p, --path arg         Path to input image/video (default: test_image.jpg)
  -s, --synthetic arg    Synthetic pattern: checkerboard | gradient | noise (default: checkerboard)
  -d, --device arg       Camera device ID (default: 0)

  -f, --filter arg       Single filter: blur | sharpen | edge | emboss | hdr (default: blur)
  -k, --kernel-size arg  Kernel size (default: 3)
      --intensity arg    Filter intensity (default: 1.0)
      --sigma arg        Sigma for Gaussian blur (default: 1.0)

      --exposure arg     HDR exposure multiplier (default: 1.0)
      --gamma arg        HDR gamma correction (default: 1.0)
      --saturation arg   HDR saturation scale (default: 1.0)
      --tonemap arg      HDR algorithm: reinhard | drago | mantiuk (default: reinhard)

      --pipeline arg     Pipeline spec: filter[:ks[:intensity]],... (activates pipeline mode)
      --single-stream    Force single CUDA stream for pipeline (comparison baseline)
      --show-timings     Draw real-time per-stage timing bar chart overlay
      --wipe             Wipe transition between original and pipeline output
      --wipe-speed arg   Wipe speed in full-widths per second (default: 0.3)

      --preview          Show original alongside filtered (single-filter mode)
      --save             Save output to file
      --out arg          Output file path (auto-generated if omitted)
  -h, --help             Print usage
  -v, --version          Print version
```

---

## Architecture

```
src/
├── main.cpp                         # Entry point; dispatches pipeline vs single-filter mode
├── kernels/
│   ├── convolution_kernels.cu       # convolutionKernel, hdrTonemapKernel,
│   │                                #   applyFilterGPU, applyHDRTonemapGPU,
│   │                                #   applyConvolutionOnStream, applyHDROnStream
│   └── kernels.h                    # Declarations incl. stream-aware variants
├── pipeline/
│   ├── filter_pipeline.h/.cu        # FilterPipeline — multi-stage, multi-stream execution
│   ├── wipe_transition.h/.cu        # WipeTransition — GPU left-to-right wipe kernel
│   └── ...
├── utils/
│   ├── filter_utils.h/.cpp          # FilterType, kernel creation, CPU apply
│   ├── input_handler.h/.cpp         # Webcam / image / video / synthetic input
│   ├── timing_overlay.h/.cpp        # Real-time bar-chart overlay (OpenCV draw calls)
│   └── version.h.in
└── input_args_parser/
    ├── input_args_parser.h/.cpp     # CLI parsing (cxxopts)
```

### FilterPipeline internals

```
┌─────────────────────────────────────────────────────────────────────┐
│  FilterPipeline::apply()                                            │
│                                                                     │
│  Buffers:  dBuf[0]   dBuf[1]   dBuf[2]  ...  dBuf[N]               │
│             input  → stage-0 → stage-1  ...  output                │
│                                                                     │
│  Multi-stream mode:                                                 │
│                                                                     │
│  stream[0]: [H2D]──[kernel-0]──────────────[D2H]                   │
│                         ↓ event                                     │
│  stream[1]:         [wait]──[kernel-1]                              │
│                                  ↓ event                            │
│  stream[2]:               [wait]──[kernel-2]                        │
│                                       ↓ event                       │
│  stream[0]:                       [wait D2H ──────────]            │
│                                                                     │
│  Single-stream mode: all operations on stream[0], no events.       │
└─────────────────────────────────────────────────────────────────────┘
```

### WipeTransition kernel

```cuda
// For each pixel (x, y):
//   if x < wipeX  → copy from imgA (original)
//   if x >= wipeX → copy from imgB (filtered)
//   if x == wipeX → draw white boundary line
```

---

## Adding New Filters

### Convolution filter

1. Add entry to `FilterType` enum in `filter_utils.h`
2. Add string mapping in `FilterUtils::stringToFilterType()`
3. Add kernel matrix in `FilterUtils::createFilterKernel()`
4. GPU and pipeline support are automatic — no other changes needed

### Pixel-wise filter (like HDR)

1. Add `FilterType` entry and parameter struct
2. Implement CPU version in `filter_utils.cpp`
3. Implement `__global__` kernel and `applyXOnStream()` wrapper in `convolution_kernels.cu`
4. Declare `applyXOnStream()` in `kernels.h`
5. Wire into `FilterPipeline::apply()` (alongside the existing HDR branch)
6. Add CLI options in `input_args_parser.cpp`

---

## Hardware Requirements

- CUDA-capable GPU (compute capability ≥ 7.5; adjust `CMAKE_CUDA_ARCHITECTURES` in CMakeLists.txt)
- Linux or Windows

## Dependencies

| Library | Version | How to get |
|---------|---------|------------|
| CUDA Toolkit | ≥ 12.0 | NVIDIA installer |
| OpenCV | ≥ 4.5 | See below |
| CMake | ≥ 3.27 | Package manager / kitware |
| cxxopts | 3.3.1 | Fetched by CMake |
| plog | 1.1.11 | Fetched by CMake |
| Google Test | 1.17.0 | Fetched by CMake |

---

## OpenCV Installation

### Linux — package manager (quickest)

```bash
sudo apt-get update && sudo apt-get install libopencv-dev
```

### Linux — build from source with CUDA support

```bash
# Prerequisites
sudo apt-get install build-essential cmake git pkg-config \
    libgtk-3-dev libavcodec-dev libavformat-dev libswscale-dev \
    libv4l-dev libjpeg-dev libpng-dev libtiff-dev

# Download
wget -O opencv.tar.gz https://github.com/opencv/opencv/archive/refs/tags/4.11.0.tar.gz
tar -xzf opencv.tar.gz && cd opencv-4.11.0

mkdir build && cd build
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DWITH_CUDA=ON -DWITH_CUBLAS=ON \
  -DCUDA_ARCH_BIN="7.5" \          # match your GPU compute capability
  -DBUILD_EXAMPLES=OFF -DBUILD_TESTS=OFF
cmake --build . -j $(nproc)
sudo make install && sudo ldconfig
```

### Windows — pre-built binaries

1. Download `opencv-4.11.0-windows.exe` from the [OpenCV releases page](https://github.com/opencv/opencv/releases/tag/4.11.0).
2. Extract to e.g. `C:\opencv`.
3. Add `C:\opencv\build\x64\vc16\bin` to `PATH`.
4. Configure: `cmake .. -DOpenCV_DIR="C:\opencv\build"`

---

## Build

> Adjust `CMAKE_CUDA_ARCHITECTURES` in `CMakeLists.txt` to match your GPU
> (75 = Turing/RTX 20xx, 86 = Ampere/RTX 30xx, 89 = Ada/RTX 40xx).

### Linux

```bash
mkdir build && cd build
cmake ..
cmake --build . -j $(nproc)
```

### Windows

```powershell
mkdir build; cd build
cmake .. -G "Visual Studio 17 2022" -A x64
cmake --build . --config Release
```

---

## Portable Binary (Linux)

```bash
sudo apt install patchelf && pip install staticx

mkdir build && cd build && cmake .. && cmake --build . -j $(nproc)
# Portable binary: build/cuda-webcam-filter-portable
```

---

## Testing

```bash
cmake .. -DRUN_UNIT_TESTS=ON
cmake --build . -j $(nproc)
cd tests/unit_tests && ctest
```
