#include <iostream>
#include "kernel.cuh"

int main() {
    std::cout << "Running on CPU..." << std::endl;
    run_cuda_kernel();
    std::cout << "Done!" << std::endl;
    return 0;
}