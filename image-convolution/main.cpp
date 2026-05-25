#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include "convolution.h"
#include "convolution.cuh"
#include "kernels.h"

int main()
{
    Image original_image = {}, tmp_image = {};

    if (loadImage("images/cat.jpeg", original_image)) {
        printf("Loaded image: %d x %d, channels: %d\n", original_image.width, original_image.height, original_image.channels);
    } else {
        fprintf(stderr, "Failed to load image\n");
        return EXIT_FAILURE;
    }

    clock_t start_cpu, stop_cpu;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float cpu_time_ms = 0;
    float gpu_time_ms = 0;
    float gpu_opt_time_ms = 0;

    // === BOX BLUR 3x3 ===
    printf("Box Blur 3x3\n");

    start_cpu = clock();
    convolutionCPU(&original_image, &tmp_image, boxBlur3x3);
    stop_cpu = clock();
    cpu_time_ms = ((double)(stop_cpu - start_cpu)) / CLOCKS_PER_SEC * 1000;
    printf("    CPU Convolution (Box Blur 3x3) Time: %.4f ms\n", cpu_time_ms);
    saveImage("images/cat_blurred.png", tmp_image);

    cudaEventRecord(start);
    convolutionGPU(&original_image, &tmp_image, boxBlur3x3);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_time_ms, start, stop);
    printf("    GPU Naive (Box Blur 3x3) Time: %.4f ms\n", gpu_time_ms);
    saveImage("images/cat_blurred_gpu.png", tmp_image);

    cudaEventRecord(start);
    convolutionGPUShared(&original_image, &tmp_image, boxBlur3x3);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_opt_time_ms, start, stop);
    printf("    GPU Shared (Box Blur 3x3) Time: %.4f ms\n", gpu_opt_time_ms);
    saveImage("images/cat_blurred_gpu_shared.png", tmp_image);

    printf("    CPU/GPU Naive Speedup: %.2fx | CPU/GPU Shared Speedup: %.2fx | Shared vs Naive: %.2fx\n",
           cpu_time_ms / gpu_time_ms, cpu_time_ms / gpu_opt_time_ms, gpu_time_ms / gpu_opt_time_ms);

    // === GAUSSIAN BLUR 5x5 ===
    printf("Gaussian Blur 5x5\n");

    start_cpu = clock();
    convolutionCPU(&original_image, &tmp_image, gaussianBlur5x5);
    stop_cpu = clock();
    cpu_time_ms = ((double)(stop_cpu - start_cpu)) / CLOCKS_PER_SEC * 1000;
    printf("    CPU Convolution (Gaussian Blur 5x5) Time: %.4f ms\n", cpu_time_ms);
    saveImage("images/cat_gaussian_blurred.png", tmp_image);

    cudaEventRecord(start);
    convolutionGPU(&original_image, &tmp_image, gaussianBlur5x5);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_time_ms, start, stop);
    printf("    GPU Naive (Gaussian Blur 5x5) Time: %.4f ms\n", gpu_time_ms);
    saveImage("images/cat_gaussian_blurred_gpu.png", tmp_image);

    cudaEventRecord(start);
    convolutionGPUShared(&original_image, &tmp_image, gaussianBlur5x5);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_opt_time_ms, start, stop);
    printf("    GPU Shared (Gaussian Blur 5x5) Time: %.4f ms\n", gpu_opt_time_ms);
    saveImage("images/cat_gaussian_blurred_gpu_shared.png", tmp_image);

    printf("    CPU/GPU Naive Speedup: %.2fx | CPU/GPU Shared Speedup: %.2fx | Shared vs Naive: %.2fx\n",
           cpu_time_ms / gpu_time_ms, cpu_time_ms / gpu_opt_time_ms, gpu_time_ms / gpu_opt_time_ms);

    cudaEventRecord(start);
    convolutionGPUSeparable(&original_image, &tmp_image, gaussianBlur5x5);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_opt_time_ms, start, stop);
    printf("    GPU Separable (Gaussian Blur 5x5) Time: %.4f ms\n", gpu_opt_time_ms);
    saveImage("images/cat_gaussian_blurred_gpu_separable.png", tmp_image);

    printf("    CPU/GPU Naive Speedup: %.2fx | CPU/GPU Shared Speedup: %.2fx | Shared vs Naive: %.2fx\n",
           cpu_time_ms / gpu_time_ms, cpu_time_ms / gpu_opt_time_ms, gpu_time_ms / gpu_opt_time_ms);

    // === SOBEL X ===
    printf("Sobel X\n");

    start_cpu = clock();
    convolutionCPU(&original_image, &tmp_image, sobelX);
    stop_cpu = clock();
    cpu_time_ms = ((double)(stop_cpu - start_cpu)) / CLOCKS_PER_SEC * 1000;
    printf("    CPU Convolution (Sobel X) Time: %.4f ms\n", cpu_time_ms);
    saveImage("images/cat_sobel_x.png", tmp_image);

    cudaEventRecord(start);
    convolutionGPU(&original_image, &tmp_image, sobelX);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_time_ms, start, stop);
    printf("    GPU Naive (Sobel X) Time: %.4f ms\n", gpu_time_ms);
    saveImage("images/cat_sobel_x_gpu.png", tmp_image);

    cudaEventRecord(start);
    convolutionGPUShared(&original_image, &tmp_image, sobelX);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_opt_time_ms, start, stop);
    printf("    GPU Shared (Sobel X) Time: %.4f ms\n", gpu_opt_time_ms);
    saveImage("images/cat_sobel_x_gpu_shared.png", tmp_image);

    printf("    CPU/GPU Naive Speedup: %.2fx | CPU/GPU Shared Speedup: %.2fx | Shared vs Naive: %.2fx\n",
           cpu_time_ms / gpu_time_ms, cpu_time_ms / gpu_opt_time_ms, gpu_time_ms / gpu_opt_time_ms);

    // === SOBEL Y ===
    printf("Sobel Y\n");

    start_cpu = clock();
    convolutionCPU(&original_image, &tmp_image, sobelY);
    stop_cpu = clock();
    cpu_time_ms = ((double)(stop_cpu - start_cpu)) / CLOCKS_PER_SEC * 1000;
    printf("    CPU Convolution (Sobel Y) Time: %.4f ms\n", cpu_time_ms);
    saveImage("images/cat_sobel_y.png", tmp_image);

    cudaEventRecord(start);
    convolutionGPU(&original_image, &tmp_image, sobelY);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_time_ms, start, stop);
    printf("    GPU Naive (Sobel Y) Time: %.4f ms\n", gpu_time_ms);
    saveImage("images/cat_sobel_y_gpu.png", tmp_image);

    cudaEventRecord(start);
    convolutionGPUShared(&original_image, &tmp_image, sobelY);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_opt_time_ms, start, stop);
    printf("    GPU Shared (Sobel Y) Time: %.4f ms\n", gpu_opt_time_ms);
    saveImage("images/cat_sobel_y_gpu_shared.png", tmp_image);

    printf("    CPU/GPU Naive Speedup: %.2fx | CPU/GPU Shared Speedup: %.2fx | Shared vs Naive: %.2fx\n",
           cpu_time_ms / gpu_time_ms, cpu_time_ms / gpu_opt_time_ms, gpu_time_ms / gpu_opt_time_ms);

    // === SHARPEN ===
    printf("Sharpen\n");

    start_cpu = clock();
    convolutionCPU(&original_image, &tmp_image, sharpen);
    stop_cpu = clock();
    cpu_time_ms = ((double)(stop_cpu - start_cpu)) / CLOCKS_PER_SEC * 1000;
    printf("    CPU Convolution (Sharpen) Time: %.4f ms\n", cpu_time_ms);
    saveImage("images/cat_sharpened.png", tmp_image);

    cudaEventRecord(start);
    convolutionGPU(&original_image, &tmp_image, sharpen);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_time_ms, start, stop);
    printf("    GPU Naive (Sharpen) Time: %.4f ms\n", gpu_time_ms);
    saveImage("images/cat_sharpened_gpu.png", tmp_image);

    cudaEventRecord(start);
    convolutionGPUShared(&original_image, &tmp_image, sharpen);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_opt_time_ms, start, stop);
    printf("    GPU Shared (Sharpen) Time: %.4f ms\n", gpu_opt_time_ms);
    saveImage("images/cat_sharpened_gpu_shared.png", tmp_image);

    printf("    CPU/GPU Naive Speedup: %.2fx | CPU/GPU Shared Speedup: %.2fx | Shared vs Naive: %.2fx\n",
           cpu_time_ms / gpu_time_ms, cpu_time_ms / gpu_opt_time_ms, gpu_time_ms / gpu_opt_time_ms);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    freeImage(original_image);
    freeImage(tmp_image);

    return 0;
}
