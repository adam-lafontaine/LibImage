#pragma once

#include <cstddef>

#define LIBIMAGE_PNG
#define LIBIMAGE_BMP

//#define LIBIMAGE_NO_COLOR
//#define LIBIMAGE_NO_GRAYSCALE
//#define LIBIMAGE_NO_WRITE
//#define LIBIMAGE_NO_RESIZE
//#define LIBIMAGE_NO_PARALLEL
//#define LIBIMAGE_NO_FILESYSTEM

#define LIBIMAGE_NO_CPP17

#define LIBIMAGE_NO_SIMD

//#define INTEL_CPU
//#define RPI_3B_PLUS
//#define JETSON_NANO


#ifndef LIBIMAGE_NO_SIMD

#ifdef INTEL_CPU

#define SIMD_INTEL_128
//#define SIMD_INTEL_256

#endif // INTEL_CPU

#ifdef RPI_3B_PLUS

#define SIMD_ARM_NEON
#define LIBIMAGE_NO_CPP17

#endif // RPI_3B_PLUS

#ifdef JETSON_NANO

#define SIMD_ARM_NEON
#define LIBIMAGE_NO_CPP17

#endif // JETSON_NANO

#endif // !LIBIMAGE_NO_SIMD


#ifdef LIBIMAGE_NO_CPP17

#define LIBIMAGE_NO_PARALLEL
#define LIBIMAGE_NO_FILESYSTEM

#endif // LIBIMAGE_NO_CPP17


/*  types.hpp  */

using u8 = uint8_t;
using u32 = uint32_t;
using u64 = uint64_t;
using r32 = float;
using r64 = double;
using i32 = int32_t;


#ifdef LIBIMAGE_NO_PARALLEL

constexpr u32 N_THREADS = 1;

#else

constexpr u32 N_THREADS = 8;

#endif //!LIBIMAGE_NO_PARALLEL