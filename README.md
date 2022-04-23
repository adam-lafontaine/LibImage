# LibImage
A Basic image processing libray for C++
* C++17
* Interface inspired by boost::GIL
* Uses stb_image to read, write and resize images (https://github.com/nothings/stb)
* image_t owns the memory, view_t points to image_t memory with custom iterator
* Forward iterator begin() and end() to allow using the STL algorithms
* Histogram, mean, standard deviation
* Copy, conversion, binarization, alpha blending, edge detection
* Settings macros defined in /libimage/defines.hpp
* /libimage_compact for version with fewer files

Windows
* Sequential and parallel processing (std::execution)
* SIMD available for Intel 128 bit, Intel 256 bit, ARM 128 bit

Raspberry Pi 3B+
* Sequential and parallel processing (std::execution)
* SIMD ARM Neon 128 bit
* See /RPiTests/rpi_tests_main.cpp for demonstration
* "make setup" to create build directory
* "make build" or "make run"

CUDA
* Typesafe wrapper for preallocating memory on the GPU
* Developed on the Jetson Nano.  C++14
* See /CudaTests/cuda_test_main.cpp for demonstration
* Makefile in /CudaTests/
* "make setup" to create build directory
* "make build" or "make run"