#pragma once

#include "../process.hpp"


//#define SIMD_INTEL_128
#define SIMD_INTEL_256


#ifdef  SIMD_INTEL_128

#include <xmmintrin.h>
#include <immintrin.h>

constexpr u32 VEC_LEN = 4;


using vec_t = __m128;


class PixelPlanar
{
public:
	r32 red[VEC_LEN]   = { 0 };
	r32 green[VEC_LEN] = { 0 };
	r32 blue[VEC_LEN]  = { 0 };
	r32 alpha[VEC_LEN] = { 0 };
};


static inline vec_t simd_load_broadcast(const r32* a)
{
	return _mm_load_ps1(a);
}


static inline vec_t simd_load(const r32* a)
{
	return _mm_load_ps(a);
}


static inline void simd_store(r32* dst, vec_t const& a)
{
	_mm_store_ps(dst, a);
}


static inline vec_t simd_add(vec_t const& a, vec_t const& b)
{
	return _mm_add_ps(a, b);
}


static inline vec_t simd_subtract(vec_t const& a, vec_t const& b)
{
	return _mm_sub_ps(a, b);
}


static inline vec_t simd_multiply(vec_t const& a, vec_t const& b)
{
	return _mm_mul_ps(a, b);
}


static inline vec_t simd_divide(vec_t const& a, vec_t const& b)
{
	return _mm_div_ps(a, b);
}


static inline vec_t simd_fmadd(vec_t const& a, vec_t const& b, vec_t const& c)
{
	return _mm_fmadd_ps(a, b, c);
}


#endif // SIMD_INTEL_128

#ifdef SIMD_INTEL_256

#include <xmmintrin.h>
#include <immintrin.h>

constexpr u32 VEC_LEN = 8;


using vec_t = __m256;


class PixelPlanar
{
public:
	r32 red[VEC_LEN]   = { 0 };
	r32 green[VEC_LEN] = { 0 };
	r32 blue[VEC_LEN]  = { 0 };
	r32 alpha[VEC_LEN] = { 0 };
};


static inline vec_t simd_load_broadcast(const r32* a)
{
	return _mm256_broadcast_ss(a);
}


static inline vec_t simd_load(const r32* a)
{
	return _mm256_load_ps(a);
}


static inline void simd_store(r32* dst, vec_t const& a)
{
	_mm256_store_ps(dst, a);
}


static inline vec_t simd_add(vec_t const& a, vec_t const& b)
{
	return _mm256_add_ps(a, b);
}


static inline vec_t simd_subtract(vec_t const& a, vec_t const& b)
{
	return _mm256_sub_ps(a, b);
}


static inline vec_t simd_multiply(vec_t const& a, vec_t const& b)
{
	return _mm256_mul_ps(a, b);
}


static inline vec_t simd_divide(vec_t const& a, vec_t const& b)
{
	return _mm256_div_ps(a, b);
}


static inline vec_t simd_fmadd(vec_t const& a, vec_t const& b, vec_t const& c)
{
	return _mm256_fmadd_ps(a, b, c);
}


#endif // SIMD_INTEL_256







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

