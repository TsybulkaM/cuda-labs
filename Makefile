
build: build-mt

clean: clean-mt

# ==== Matrix Transposition ====

build-mt:
	cd matrix-transposition && mkdir -p build && cd build && cmake .. && make

clean-mt:
	rm -rf matrix-transposition/build
