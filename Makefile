.PHONY: build clean build-mt clean-mt build-ic clean-ic

BUILD_DIR=build
CMAKE_FLAGS= -G Ninja

build: build-mt build-ic

clean: clean-mt clean-ic

# ==== Matrix Transposition ====

build-mt:
		cmake ${CMAKE_FLAGS} -S matrix-transposition -B matrix-transposition/${BUILD_DIR} && cmake --build matrix-transposition/${BUILD_DIR}

clean-mt:
	rm -rf matrix-transposition/${BUILD_DIR}

# ==== Image Convolution ====

build-ic:
	cmake ${CMAKE_FLAGS} -S image-convolution -B image-convolution/${BUILD_DIR} && cmake --build image-convolution/${BUILD_DIR}

clean-ic:
	rm -rf image-convolution/${BUILD_DIR}