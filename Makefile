.PHONY: build clean build-mt clean-mt build-ic clean-ic

CMAKE_FLAGS= -G Ninja

build: build-mt build-ic

clean: clean-mt clean-ic

# ==== Matrix Transposition ====

build-mt:
	cd matrix-transposition && cmake --build build

clean-mt:
	rm -rf matrix-transposition/build

# ==== Image Convolution ====

build-ic:
	cd image-convolution && cmake --build build

clean-ic:
	rm -rf image-convolution/build