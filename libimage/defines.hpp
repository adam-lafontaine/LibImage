#pragma once

#define LIBIMAGE_PNG
#define LIBIMAGE_BMP

//#define LIBIMAGE_NO_COLOR
//#define LIBIMAGE_NO_GRAYSCALE
//#define LIBIMAGE_NO_WRITE
//#define LIBIMAGE_NO_RESIZE
//#define LIBIMAGE_NO_PARALLEL
//#define LIBIMAGE_NO_FILESYSTEM


// jetson nano
//#define LIBIMAGE_NO_CPP17
#define LIBIMAGE_NO_SIMD


#ifdef LIBIMAGE_NO_CPP17

#define LIBIMAGE_NO_PARALLEL
#define LIBIMAGE_NO_FILESYSTEM

#endif // LIBIMAGE_NO_CPP17


#ifndef LIBIMAGE_NO_SIMD

//#define SIMD_INTEL_128
//#define SIMD_INTEL_256
#define SIMD_ARM_NEON

#endif // !LIBIMAGE_NO_SIMD
