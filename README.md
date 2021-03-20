# LibImage
A Basic image processing libray for C++ using stb_image (https://github.com/nothings/stb)

* C++17
* Similar interface to that of boost::GIL
* image_t owns the memory, view_t points to image_t memory with custom iterator
* Forward iterator begin() and end() to allow using the STL algorithms
* Read write and modify images in code
* Preprocessor flags available in libimage.hpp enable/disable features
* No OpenCV type functionality yet