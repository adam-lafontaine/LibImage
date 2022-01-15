#include "process.hpp"
#include "verify.hpp"
#include "index_range.hpp"

#ifndef LIBIMAGE_NO_SIMD
#include <xmmintrin.h>
#include <immintrin.h>
#endif // !LIBIMAGE_NO_SIMD




#ifndef LIBIMAGE_NO_GRAYSCALE

static constexpr u8 lerp_clamp(u8 src_low, u8 src_high, u8 dst_low, u8 dst_high, u8 val)
{
	if (val < src_low)
	{
		return dst_low;
	}
	else if (val > src_high)
	{
		return dst_high;
	}

	auto const ratio = (static_cast<r64>(val) - src_low) / (src_high - src_low);

	assert(ratio >= 0.0);
	assert(ratio <= 1.0);

	auto const diff = ratio * (dst_high - dst_low);

	return dst_low + static_cast<u8>(diff);
}


#endif // !LIBIMAGE_NO_GRAYSCALE


namespace libimage
{
#ifndef LIBIMAGE_NO_PARALLEL

#ifndef LIBIMAGE_NO_GRAYSCALE

	void contrast(gray::image_t const& src, gray::image_t const& dst, u8 src_low, u8 src_high)
	{
		assert(src_low < src_high);
		constexpr u8 dst_low = 0;
		constexpr u8 dst_high = 255;
		auto const conv = [&](u8 p) { return lerp_clamp(src_low, src_high, dst_low, dst_high, p); };
		transform(src, dst, conv);
	}


	void contrast(gray::image_t const& src, gray::view_t const& dst, u8 src_low, u8 src_high)
	{
		assert(src_low < src_high);
		constexpr u8 dst_low = 0;
		constexpr u8 dst_high = 255;
		auto const conv = [&](u8 p) { return lerp_clamp(src_low, src_high, dst_low, dst_high, p); };
		transform(src, dst, conv);
	}


	void contrast(gray::view_t const& src, gray::image_t const& dst, u8 src_low, u8 src_high)
	{
		assert(src_low < src_high);
		constexpr u8 dst_low = 0;
		constexpr u8 dst_high = 255;
		auto const conv = [&](u8 p) { return lerp_clamp(src_low, src_high, dst_low, dst_high, p); };
		transform(src, dst, conv);
	}



	void contrast(gray::view_t const& src, gray::view_t const& dst, u8 src_low, u8 src_high)
	{
		assert(src_low < src_high);
		constexpr u8 dst_low = 0;
		constexpr u8 dst_high = 255;
		auto const conv = [&](u8 p) { return lerp_clamp(src_low, src_high, dst_low, dst_high, p); };
		transform(src, dst, conv);
	}


	void contrast_self(gray::image_t const& src_dst, u8 src_low, u8 src_high)
	{
		assert(src_low < src_high);
		constexpr u8 dst_low = 0;
		constexpr u8 dst_high = 255;
		auto const conv = [&](u8 p) { return lerp_clamp(src_low, src_high, dst_low, dst_high, p); };
		transform_self(src_dst, conv);
	}


	void contrast_self(gray::view_t const& src_dst, u8 src_low, u8 src_high)
	{
		assert(src_low < src_high);
		constexpr u8 dst_low = 0;
		constexpr u8 dst_high = 255;
		auto const conv = [&](u8 p) { return lerp_clamp(src_low, src_high, dst_low, dst_high, p); };
		transform_self(src_dst, conv);
	}

#endif // !LIBIMAGE_NO_GRAYSCALE

#endif // !LIBIMAGE_NO_PARALLEL



	namespace seq
	{
#ifndef LIBIMAGE_NO_GRAYSCALE

		void contrast(gray::image_t const& src, gray::image_t const& dst, u8 src_low, u8 src_high)
		{
			assert(src_low < src_high);
			u8 dst_low = 0;
			u8 dst_high = 255;
			auto const conv = [&](u8 p) { return lerp_clamp(src_low, src_high, dst_low, dst_high, p); };
			seq::transform(src, dst, conv);
		}


		void contrast(gray::image_t const& src, gray::view_t const& dst, u8 src_low, u8 src_high)
		{
			assert(src_low < src_high);
			u8 dst_low = 0;
			u8 dst_high = 255;
			auto const conv = [&](u8 p) { return lerp_clamp(src_low, src_high, dst_low, dst_high, p); };
			seq::transform(src, dst, conv);
		}


		void contrast(gray::view_t const& src, gray::image_t const& dst, u8 src_low, u8 src_high)
		{
			assert(src_low < src_high);
			u8 dst_low = 0;
			u8 dst_high = 255;
			auto const conv = [&](u8 p) { return lerp_clamp(src_low, src_high, dst_low, dst_high, p); };
			seq::transform(src, dst, conv);
		}


		void contrast(gray::view_t const& src, gray::view_t const& dst, u8 src_low, u8 src_high)
		{
			assert(src_low < src_high);
			u8 dst_low = 0;
			u8 dst_high = 255;
			auto const conv = [&](u8 p) { return lerp_clamp(src_low, src_high, dst_low, dst_high, p); };
			seq::transform(src, dst, conv);
		}


		void contrast_self(gray::image_t const& src_dst, u8 src_low, u8 src_high)
		{
			assert(src_low < src_high);
			u8 dst_low = 0;
			u8 dst_high = 255;
			auto const conv = [&](u8 p) { return lerp_clamp(src_low, src_high, dst_low, dst_high, p); };
			seq::transform_self(src_dst, conv);
		}


		void contrast_self(gray::view_t const& src_dst, u8 src_low, u8 src_high)
		{
			assert(src_low < src_high);
			u8 dst_low = 0;
			u8 dst_high = 255;
			auto const conv = [&](u8 p) { return lerp_clamp(src_low, src_high, dst_low, dst_high, p); };
			seq::transform_self(src_dst, conv);
		}


#endif // !LIBIMAGE_NO_GRAYSCALE
	}


#ifndef LIBIMAGE_NO_SIMD

	namespace simd
	{
#ifndef LIBIMAGE_NO_GRAYSCALE

		//static void contrast_row(u8* src_begin, u8* dst_begin, u32 length, u8 src_low, u8 src_high)
		//{
		//	constexpr u32 N = 4;
		//	constexpr u32 STEP = N;
		//	r32 memory[N];

		//	r32 low = src_low;
		//	r32 high = src_high;

		//	auto low_vec = _mm_load1_ps(&low);
		//	auto high_vec = _mm_load1_ps(&high);

		//	auto const do_simd = [&](u32 i) 
		//	{
		//		auto dst_vec = _mm_setzero_ps();

		//		for (u32 n = 0; n < N; ++n)
		//		{
		//			memory[n] = static_cast<r32>(src_begin[i + n]);
		//		}

		//		auto val_vec = _mm_load_ps(memory);

		//		_mm_cmple_ps(val_vec, low_vec);

		//		_mm_cmpge_ps(val_vec, high_vec);

		//		// not done

		//		_mm_store_ps(memory, dst_vec);
		//		for (u32 n = 0; n < N; ++n)
		//		{
		//			dst_begin[i + n] = static_cast<u8>(memory[n]);
		//		}
		//	};


		//	for (u32 i = 0; i < length - STEP; i += STEP)
		//	{
		//		do_simd(i);
		//	}

		//	do_simd(length - STEP);
		//}

#endif // !LIBIMAGE_NO_GRAYSCALE
	}

#endif // !LIBIMAGE_NO_SIMD
}