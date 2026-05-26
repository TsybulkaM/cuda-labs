#include "kernels.h"
#include <stdio.h>

extern "C" long strtol(const char *nptr, char **endptr, int base) noexcept {
  const char *start = nptr;
  while (*start == ' ' || *start == '\t' || *start == '\n' || *start == '\r' ||
         *start == '\f' || *start == '\v') {
    ++start;
  }

  int sign = 1;
  if (*start == '+' || *start == '-') {
    if (*start == '-') {
      sign = -1;
    }
    ++start;
  }

  if (base == 0) {
    if (start[0] == '0') {
      if (start[1] == 'x' || start[1] == 'X') {
        base = 16;
        start += 2;
      } else {
        base = 8;
      }
    } else {
      base = 10;
    }
  } else if (base == 16 && start[0] == '0' &&
             (start[1] == 'x' || start[1] == 'X')) {
    start += 2;
  }

  const char *digits = start;
  long value = 0;
  while (true) {
    int digit = -1;
    char ch = *start;
    if (ch >= '0' && ch <= '9') {
      digit = ch - '0';
    } else if (ch >= 'a' && ch <= 'z') {
      digit = 10 + (ch - 'a');
    } else if (ch >= 'A' && ch <= 'Z') {
      digit = 10 + (ch - 'A');
    } else {
      break;
    }

    if (digit >= base) {
      break;
    }

    value = value * base + digit;
    ++start;
  }

  if (endptr != nullptr) {
    *endptr = const_cast<char *>(start == digits ? nptr : start);
  }

  return sign * value;
}

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

bool loadImage(const char *filename, Image &img) {
  img.data = stbi_load(filename, &img.width, &img.height, &img.channels, 0);

  if (img.data == nullptr) {
    fprintf(stderr, "Error loading image %s\n", filename);
    return false;
  }

  return true;
}

void saveImage(const char *filename, const Image &img) {
  stbi_write_png(filename, img.width, img.height, img.channels, img.data,
                 img.width * img.channels);
}

void freeImage(Image &img) {
  if (img.data != nullptr) {
    stbi_image_free(img.data);
    img.data = nullptr;
  }
}