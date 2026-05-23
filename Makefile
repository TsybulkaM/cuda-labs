.PHONY: build clean build-mt clean-mt build-ic clean-ic

CMAKE_FLAGS= -G Ninja

build: build-mt build-ic

clean: clean-mt clean-ic

# ==== Matrix Transposition ====

build-mt:
	cd matrix-transposition && mkdir -p build && cd build && cmake ${CMAKE_FLAGS} .. && ninja

clean-mt:
	rm -rf matrix-transposition/build

# ==== Image Convolution ====

build-ic:
	cd image-convolution && mkdir -p build && cd build && cmake ${CMAKE_FLAGS} .. && ninja

clean-ic:
	rm -rf image-convolution/build