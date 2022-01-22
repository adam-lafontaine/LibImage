#pragma once

#include "../process.hpp"


#define SIMD_INTEL


#ifdef  SIMD_INTEL

#include <xmmintrin.h>
#include <immintrin.h>

constexpr u32 VEC_LEN = 4;


using vec_t = __m128;


class PixelPlanar
{
public:
	r32 red[4]   = { 0 };
	r32 green[4] = { 0 };
	r32 blue[4]  = { 0 };
	r32 alpha[4] = { 0 };
};


static inline vec_t simd_load_broadcast(const r32* a)
{
	return _mm_load_ps1(a);
}


static inline vec_t simd_load(const r32* a)
{
	return _mm_load_ps(a);
}


static inline void simd_store(r32* dst, vec_t& a)
{
	_mm_store_ps(dst, a);
}


static inline vec_t simd_multiply(vec_t& a, vec_t& b)
{
	return _mm_mul_ps(a, b);
}


static inline vec_t simd_fmadd(vec_t& a, vec_t& b, vec_t& c)
{
	return _mm_fmadd_ps(a, b, c);
}


#endif //  SIMD_INTEL


static inline void copy_vec_len(libimage::pixel_t* src, PixelPlanar& dst)
{
	for (u32 i = 0; i < VEC_LEN; ++i)
	{
		dst.red[i] = src[i].red;
		dst.green[i] = src[i].green;
		dst.blue[i] = src[i].blue;
		dst.alpha[i] = src[i].alpha;
	}
}


template <typename SRC_T, typename DST_T>
static inline void copy_vec_len(SRC_T* src, DST_T* dst)
{
	for (u32 i = 0; i < VEC_LEN; ++i)
	{
		dst[i] = (DST_T)src[i];
	}
}

