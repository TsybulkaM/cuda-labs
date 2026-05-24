#include "kernels.h"
#include <stdio.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

bool loadImage(const char* filename, Image& img) {
    img.data = stbi_load(filename, &img.width, &img.height, &img.channels, 0);
    
    if (img.data == nullptr) {
        fprintf(stderr, "Error loading image %s\n", filename);
        return false;
    }
    
    return true;
}

void saveImage(const char* filename, const Image& img) {
    stbi_write_png(filename, img.width, img.height, img.channels, img.data, img.width * img.channels);
}

void freeImage(Image& img) {
    if (img.data != nullptr) {
        stbi_image_free(img.data);
        img.data = nullptr;
    }
}