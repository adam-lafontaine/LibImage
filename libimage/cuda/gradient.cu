/*

Copyright (c) 2021 Adam Lafontaine

*/
#ifndef LIBIMAGE_NO_GRAYSCALE

#include "verify.hpp"
#include "../proc/verify.hpp"
#include "convolve.cuh"

#include <cassert>
#include <array>

constexpr int THREADS_PER_BLOCK = 1024;

constexpr u32 GRAD_3X3_SIZE = 9;

constexpr std::array<r32, GRAD_3X3_SIZE> GRAD_X_3X3
{
    1.0f, 0.0f, -1.0f,
    2.0f, 0.0f, -2.0f,
    1.0f, 0.0f, -1.0f,
};

constexpr std::array<r32, 9> GRAD_Y_3X3
{
    1.0f,  2.0f,  1.0f,
    0.0f,  0.0f,  0.0f,
    -1.0f, -2.0f, -1.0f,
};

#endif // !LIBIMAGE_NO_GRAYSCALE