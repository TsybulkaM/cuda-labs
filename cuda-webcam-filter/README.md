# CUDA Webcam Filter

## Purpose
This program applies various convolution filters to a webcam feed in real-time using CUDA for GPU acceleration. The application demonstrates how to utilize GPU computing to process video streams efficiently.

## Features
- Real-time webcam video capture
- Multiple convolution filter options (blur, sharpen, edge detection, emboss)
- HDR tonemapping with three algorithms (Reinhard, Drago, Mantiuk)
- GPU-accelerated processing using CUDA (both convolution and HDR filters)
- Command-line options for filter selection and parameters
- Single portable binary with bundled dependencies (via staticx)
- Headless operation for cloud/container environments

## Usage
```
  cuda-webcam-filter [OPTION...]
```

### List of options
```
  -i, --input arg        Input source: 'webcam', 'image', 'video', or 'synthetic' (default: webcam)
  -p, --path arg         Path to input image or video file (default: test_image.jpg)
  -s, --synthetic arg    Synthetic pattern type: 'checkerboard', 'gradient', 'noise' (default: checkerboard)
  -d, --device arg       Camera device ID (default: 0)
  -f, --filter arg       Filter type: blur, sharpen, edge, emboss, hdr (default: blur)
  -k, --kernel-size arg  Kernel size for filters (default: 3)
      --sigma arg        Sigma value for Gaussian blur (default: 1.0)
      --intensity arg    Filter intensity (default: 1.0)
      --exposure arg     Exposure value for HDR (default: 1.0)
      --gamma arg        Gamma value for HDR (default: 1.0)
      --saturation arg   Saturation for HDR (default: 1.0)
      --tonemap arg      HDR tonemap algorithm: reinhard, drago, mantiuk (default: reinhard)
      --preview          Show original video alongside filtered
      --save             Save output to file
      --out arg          Output filename (default: auto-generated)
  -h, --help             Print usage
  -v, --version          Print version information
```

## Running

Press **ESC** to exit the application.

### Linux

```bash
# Webcam with blur filter
./build/cuda-webcam-filter

# Edge detection on webcam with side-by-side preview
./build/cuda-webcam-filter --filter edge --preview

# HDR tonemapping with default Reinhard algorithm
./build/cuda-webcam-filter --filter hdr

# HDR with Drago logarithmic algorithm, increased exposure and saturation
./build/cuda-webcam-filter --filter hdr --tonemap drago --exposure 1.5 --saturation 1.2

# HDR with Mantiuk perceptual algorithm and gamma correction
./build/cuda-webcam-filter --filter hdr --tonemap mantiuk --gamma 2.2

# Synthetic input (no webcam required)
./build/cuda-webcam-filter --input synthetic --synthetic checkerboard --filter sharpen --preview

# Process an image file
./build/cuda-webcam-filter --input image --path cat.jpeg --filter emboss --save --out result.jpeg

# Process image with HDR and save output
./build/cuda-webcam-filter --input image --path photo.jpg --filter hdr --exposure 2.0 --save
```

### Windows

The OpenCV DLL directory must be on `PATH` before launching. Adjust the path below if you installed OpenCV elsewhere.

**PowerShell:**
```powershell
$env:PATH = "C:\opencv\build\x64\vc16\bin;" + $env:PATH

# Webcam with blur filter
.\build\Release\cuda-webcam-filter.exe

# Edge detection on webcam with side-by-side preview
.\build\Release\cuda-webcam-filter.exe --filter edge --preview

# HDR tonemapping with Drago algorithm
.\build\Release\cuda-webcam-filter.exe --filter hdr --tonemap drago --exposure 1.5

# Synthetic input (no webcam required)
.\build\Release\cuda-webcam-filter.exe --input synthetic --synthetic checkerboard --filter sharpen --preview

# Process an image file with HDR
.\build\Release\cuda-webcam-filter.exe --input image --path photo.jpg --filter hdr --save
```

**Command Prompt:**
```cmd
set PATH=C:\opencv\build\x64\vc16\bin;%PATH%
.\build\Release\cuda-webcam-filter.exe --filter edge --preview
.\build\Release\cuda-webcam-filter.exe --filter hdr --tonemap drago --exposure 1.5
```

## HDR Tonemapping Filters

The HDR filter applies tone mapping operators to enhance image dynamic range. Three algorithms are available:

- **Reinhard**: Simple operator, good for general-purpose tone mapping (default)
- **Drago**: Logarithmic operator, preserves local contrast
- **Mantiuk**: Perceptual operator, good for visually appealing results

### HDR Parameters

- `--exposure` (0.1–3.0, default: 1.0): Multiplier for input brightness
- `--gamma` (0.5–3.0, default: 1.0): Gamma correction (< 1.0 brightens, > 1.0 darkens)
- `--saturation` (0.0–2.0, default: 1.0): Color saturation (0 = grayscale, 1 = normal, > 1 = more vivid)
- `--tonemap` (reinhard|drago|mantiuk, default: reinhard): Tone mapping algorithm

### Example HDR Usage

```bash
# Low-light image enhancement
./cuda-webcam-filter --input image --path dark.jpg --filter hdr --exposure 2.0 --gamma 0.8 --save

# High contrast, vibrant colors
./cuda-webcam-filter --filter hdr --tonemap mantiuk --saturation 1.5

# Preserve local contrast
./cuda-webcam-filter --filter hdr --tonemap drago --gamma 2.2
```

## Hardware requirements
Requires a CUDA-enabled GPU.

## Dependencies
- OpenCV (>= 4.5.0) — **install separately, see below**
- CUDA Toolkit (>= 12.0)
- cxxopts (fetched by CMake)
- plog (fetched by CMake)
- Google Test (fetched by CMake)
- CMake (>= 3.28)

## OpenCV Installation

OpenCV is **not bundled** in this repository. Install it before building.

### Linux

#### Option A — Package manager (quickest, no CUDA support in OpenCV itself)
```bash
sudo apt-get update
sudo apt-get install libopencv-dev
```
The CUDA kernel in this app runs independently of OpenCV's CUDA build, so this is sufficient for most use cases.

#### Option B — Build from source with CUDA support
Required when you want OpenCV's own GPU-accelerated functions (e.g. `cv::cuda::*`).

**1. Install prerequisites**
```bash
sudo apt-get update
sudo apt-get install build-essential cmake git pkg-config \
    libgtk-3-dev libavcodec-dev libavformat-dev libswscale-dev \
    libv4l-dev libxvidcore-dev libx264-dev libjpeg-dev libpng-dev \
    libtiff-dev gfortran openexr libatlas-base-dev python3-dev \
    python3-numpy libtbb-dev libdc1394-dev
```

**2. Download OpenCV 4.11.0**
```bash
wget -O opencv.tar.gz https://github.com/opencv/opencv/archive/refs/tags/4.11.0.tar.gz
wget -O opencv_contrib.tar.gz https://github.com/opencv/opencv_contrib/archive/refs/tags/4.11.0.tar.gz
tar -xzf opencv.tar.gz
tar -xzf opencv_contrib.tar.gz
```

**3. Build and install**
```bash
cd opencv-4.11.0
mkdir build && cd build
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DOPENCV_EXTRA_MODULES_PATH=../../opencv_contrib-4.11.0/modules \
  -DWITH_CUDA=ON \
  -DWITH_CUBLAS=ON \
  -DCUDA_ARCH_BIN="8.9" \
  -DCUDA_FAST_MATH=ON \
  -DBUILD_opencv_python2=OFF \
  -DBUILD_opencv_python3=OFF \
  -DBUILD_EXAMPLES=OFF \
  -DBUILD_TESTS=OFF \
  -DBUILD_PERF_TESTS=OFF \
  -DBUILD_DOCS=OFF
cmake --build . -j $(nproc)
sudo make install
sudo ldconfig
```
> Adjust `CUDA_ARCH_BIN` to match your GPU's compute capability (e.g. `"8.6"` for RTX 3070, `"7.5"` for RTX 2080).

**4. Point CMake to the installation (if installed to a non-default prefix)**
```bash
cmake .. -DOpenCV_DIR=/path/to/opencv/build
```

---

### Windows

#### Option A — Pre-built binaries (quickest, no CUDA support in OpenCV itself)

1. Download the Windows installer from the [OpenCV releases page](https://github.com/opencv/opencv/releases/tag/4.11.0) — pick `opencv-4.11.0-windows.exe`.
2. Run the installer and extract to e.g. `C:\opencv`.
3. Add `C:\opencv\build\x64\vc16\bin` to your `PATH` environment variable.
4. Pass the OpenCV CMake dir when configuring:
```powershell
cmake .. -G "Visual Studio 17 2022" -A x64 -DOpenCV_DIR="C:\opencv\build"
```

#### Option B — Build from source with CUDA support

**Prerequisites:** Visual Studio 2022, CUDA Toolkit, CMake 3.28+, Git.

**1. Download sources**
```powershell
Invoke-WebRequest -Uri https://github.com/opencv/opencv/archive/refs/tags/4.11.0.zip -OutFile opencv.zip
Invoke-WebRequest -Uri https://github.com/opencv/opencv_contrib/archive/refs/tags/4.11.0.zip -OutFile opencv_contrib.zip
Expand-Archive opencv.zip -DestinationPath .
Expand-Archive opencv_contrib.zip -DestinationPath .
```

**2. Build and install**
```powershell
cd opencv-4.11.0
mkdir build; cd build
cmake .. `
  -G "Visual Studio 17 2022" -A x64 `
  -DCMAKE_BUILD_TYPE=Release `
  -DOPENCV_EXTRA_MODULES_PATH="..\..\opencv_contrib-4.11.0\modules" `
  -DWITH_CUDA=ON `
  -DWITH_CUBLAS=ON `
  -DCUDA_ARCH_BIN="8.9" `
  -DCUDA_FAST_MATH=ON `
  -DBUILD_opencv_python2=OFF `
  -DBUILD_opencv_python3=OFF `
  -DBUILD_EXAMPLES=OFF `
  -DBUILD_TESTS=OFF `
  -DBUILD_PERF_TESTS=OFF `
  -DBUILD_DOCS=OFF `
  -DCMAKE_INSTALL_PREFIX="C:\opencv-cuda"
cmake --build . --config Release -j
cmake --install . --config Release
```

**3. Point CMake to the installation**
```powershell
cmake .. -G "Visual Studio 17 2022" -A x64 -DOpenCV_DIR="C:\opencv-cuda"
```

---

### CMake Installation (Linux)
```bash
# For Ubuntu/Debian
sudo apt-get update
sudo apt-get install cmake=3.28.*
# If not available in default repositories, add Kitware's repository:
wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | sudo tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null
echo "deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/kitware.list >/dev/null
sudo apt-get update
sudo apt-get install cmake=3.28.*
```

```bash
sudo apt-get update
sudo apt-get install gcc-12 g++-12
sudo apt-get install libgtk-3-dev pkg-config
```

## Build

TODO: Adjust the CUDA architecture in CMakeLists.txt (CMAKE_CUDA_ARCHITECTURES)

### Build on Linux
```bash
mkdir build && cd build
cmake ..
cmake --build . -j $(nproc)
```

### Build on Windows
```powershell
mkdir build
cd build
cmake .. -G "Visual Studio 17 2022" -A x64
cmake --build . --config Release
```

## Portable Binary (Linux)

The CMakeLists.txt includes support for creating a self-contained binary using `staticx`, which bundles all dependencies (OpenCV, CUDA runtime, libstdc++) into a single executable.

### Prerequisites
```bash
sudo apt install patchelf
pip install staticx
```

### Build Portable Binary
```bash
mkdir build && cd build
cmake ..
cmake --build . -j $(nproc)
# The portable binary will be at: cuda-webcam-filter-portable
```

The resulting `cuda-webcam-filter-portable` binary can be distributed to other Linux systems without requiring OpenCV, CUDA, or specific libstdc++ versions to be pre-installed.

### Headless Mode

The binary supports headless operation (no display server required) for cloud/container environments:

```bash
# Process image without display
./cuda-webcam-filter-portable --input image --path photo.jpg --filter hdr --save

# Process video to file without display
./cuda-webcam-filter-portable --input video --path input.mp4 --filter emboss --save --out output.mp4
```

No display output is shown when using `--save` with image/video/synthetic inputs.

## Testing
The project includes unit tests and functional tests which can be enabled during the build:
```bash
cmake .. -DRUN_UNIT_TESTS=ON
make
cd tests/unit_tests/
ctest
```

## Project Structure
```
cuda-webcam-filter/
├── CMakeLists.txt           # Main build configuration
├── README.md                # Project documentation
├── src/
│   ├── main.cpp             # Application entry point
│   ├── kernels/
│   │   ├── convolution_kernels.cu  # CUDA implementation
│   │   └── kernels.h        # Kernel interfaces
│   ├── utils/
│   │   ├── input_handler.cpp  # Input/output handling
│   │   ├── input_handler.h
│   │   ├── filter_utils.cpp    # Filter creation utilities
│   │   ├── filter_utils.h
│   │   └── version.h.in        # Version template
│   └── input_args_parser/
│       ├── input_args_parser.cpp  # Command line argument parsing
│       └── input_args_parser.h
├── tests/
│   ├── unit_tests/
│   │   ├── CMakeLists.txt
│   │   ├── test_convolution.cpp
│   │   └── test_utils.cpp
│   └── functional_tests/
│       ├── CMakeLists.txt
│       └── test_filters.cpp
```

## Example: Adding a New Filter

### Adding a Convolution Filter

To add a new convolution-based filter:

1. Add a new filter type to the `FilterType` enum in `filter_utils.h`
2. Add the filter mapping in `stringToFilterType()` in `filter_utils.cpp`
3. Implement the kernel creation in `createFilterKernel()` in `filter_utils.cpp`
4. Both CPU and GPU implementations will automatically use the kernel

### Adding a Pixel-wise Filter (like HDR)

To add a new pixel-wise filter (e.g., color grading, tone mapping):

1. Add a new filter type and parameter struct in `filter_utils.h`
2. Add CPU implementation (e.g., `applyFilterNameCPU()`) in `filter_utils.cpp`
3. Add GPU kernel in `convolution_kernels.cu` with corresponding `applyFilterNameGPU()`
4. Add to `FilterType` enum and `stringToFilterType()` mapping
5. Update command-line parser in `input_args_parser.cpp` to handle new parameters
6. Update `main.cpp` to route to new filter when selected

## Performance Considerations

### Convolution Filters
- The provided convolution kernel is optimized for readability but can be further optimized:
  - Consider using shared memory to reduce global memory accesses
  - Explore using texture memory for input images
  - Implement separable convolution for certain filters (like Gaussian blur)

### HDR Tonemapping
- Current implementation uses per-pixel operations with global memory access
- Potential optimizations:
  - Use shared memory for intermediate calculations
  - Vectorize operations using CUDA intrinsics
  - Consider reduced precision (fp16) for faster computation on compatible GPUs

### Profiling
- Use NVIDIA's profiling tools to identify bottlenecks:
  - `nvidia-smi`: Monitor GPU memory and utilization
  - `nvprof` or `nsys`: Detailed kernel execution profiling
  - Check for memory bandwidth limitations vs. compute limitations
