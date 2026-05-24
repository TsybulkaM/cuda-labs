#pragma once

typedef struct
{
    unsigned char *data;
    int width;
    int height;
    int channels; // 1 for grayscale, 3 for RGB, 4 for RGBA
} Image;

bool loadImage(const char* filename, Image& img);
void saveImage(const char* filename, const Image& img);
void freeImage(Image& img);

// Box blur (3x3)
const float boxBlur3x3[9] = {
    1 / 9.0f, 1 / 9.0f, 1 / 9.0f,
    1 / 9.0f, 1 / 9.0f, 1 / 9.0f,
    1 / 9.0f, 1 / 9.0f, 1 / 9.0f
};

// Gaussian blur (5x5)
const float gaussianBlur5x5[25] = {
    1 / 273.0f, 4 / 273.0f, 7 / 273.0f, 4 / 273.0f, 1 / 273.0f,
    4 / 273.0f, 16 / 273.0f, 26 / 273.0f, 16 / 273.0f, 4 / 273.0f,
    7 / 273.0f, 26 / 273.0f, 41 / 273.0f, 26 / 273.0f, 7 / 273.0f,
    4 / 273.0f, 16 / 273.0f, 26 / 273.0f, 16 / 273.0f, 4 / 273.0f,
    1 / 273.0f, 4 / 273.0f, 7 / 273.0f, 4 / 273.0f, 1 / 273.0f
};

// Sobel edge detection (horizontal)
const float sobelX[9] = {
    -1, 0, 1,
    -2, 0, 2,
    -1, 0, 1
};

// Sobel edge detection (vertical)
const float sobelY[9] = {
    -1, -2, -1,
    0, 0, 0,
    1, 2, 1
};

// Sharpen filter
const float sharpen[9] = {
    0, -1, 0,
    -1, 5, -1,
    0, -1, 0
};
