# LibImage
A Basic image processing libray for C++

* C++17
* Interface inspired by boost::GIL
* Uses stb_image to read, write and resize images (https://github.com/nothings/stb)
* image_t owns the memory, view_t points to image_t memory with custom iterator
* Forward iterator begin() and end() to allow using the STL algorithms
* Preprocessor flags available in libimage.hpp to enable/disable features
* Histogram, mean, standard deviation
* Copy, conversion, binarization, alpha blending, edge detection
* Sequential and parallel processing available
* Widespread abuse of lambdas
* See libimage_tests.cpp for demonstration