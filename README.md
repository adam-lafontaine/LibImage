# LibImage
A Basic image processing libray for C++

## Basic Implementation - /libimage/
* C++17
* Interface inspired by boost::GIL
* Uses stb_image to read, write and resize images (https://github.com/nothings/stb)
* image_t owns the memory, view_t points to image_t memory with custom iterator
* Forward iterator begin() and end() to allow using the STL algorithms
* Histogram, mean, standard deviation
* Copy, conversion, binarization, alpha blending, edge detection
* Settings macros defined in /defines.hpp

## Another Implementation - /libimage_parallel/
* C++17
* Settings macros defined in /defines.hpp
* A more C-style api than the basic implementation
* Image object owns the memory
* Processing is done on a View object
* Processes images by dividing rows by per number of specified threads
* Limited SIMD support provided
* Visual Studio project for Windows
* Makefiles for Ubuntu and Raspberry Pi
* "make setup" to create build and output directories
* "make build" or "make run"

## Latest Implementation - /libimage_planar
* C++17
* Settings macros defined in /defines.hpp
* Images are converted to planar 32 bit float channels for presumably better compiler optimizations
* RGB/HSV conversion
* Similar C-style api
* Simple MemoryBuffer class for managing memory instead of having various make/destroy overloads
* Visual Studio project for Windows
* Makefiles for Ubuntu and Raspberry Pi
* "make setup" to create build and output directories
* "make build" or "make run"

## CUDA
* C++17 for host code
* C++14 for device code
* Developed on the Jetson Nano (C++14)
* Images are converted to planar 32 bit float channels
* MemoryBuffer classes for managing memory on the host and GPU
* Makefile in /CudaNanoTests/planar
* "make setup" to create build and output directories
* "make build" or "make run"