#include "process.hpp"
#include "verify.hpp"
#include "index_range.hpp"

constexpr u8 U8_MIN = 0;
constexpr u8 U8_MAX = 255;


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
		auto const conv = [&](u8 p) { return lerp_clamp(src_low, src_high, U8_MIN, U8_MAX, p); };
		transform(src, dst, conv);
	}


	void contrast(gray::image_t const& src, gray::view_t const& dst, u8 src_low, u8 src_high)
	{
		assert(src_low < src_high);
		auto const conv = [&](u8 p) { return lerp_clamp(src_low, src_high, U8_MIN, U8_MAX, p); };
		transform(src, dst, conv);
	}


	void contrast(gray::view_t const& src, gray::image_t const& dst, u8 src_low, u8 src_high)
	{
		assert(src_low < src_high);
		auto const conv = [&](u8 p) { return lerp_clamp(src_low, src_high, U8_MIN, U8_MAX, p); };
		transform(src, dst, conv);
	}



	void contrast(gray::view_t const& src, gray::view_t const& dst, u8 src_low, u8 src_high)
	{
		assert(src_low < src_high);
		auto const conv = [&](u8 p) { return lerp_clamp(src_low, src_high, U8_MIN, U8_MAX, p); };
		transform(src, dst, conv);
	}


	void contrast_self(gray::image_t const& src_dst, u8 src_low, u8 src_high)
	{
		assert(src_low < src_high);
		auto const conv = [&](u8 p) { return lerp_clamp(src_low, src_high, U8_MIN, U8_MAX, p); };
		transform_self(src_dst, conv);
	}


	void contrast_self(gray::view_t const& src_dst, u8 src_low, u8 src_high)
	{
		assert(src_low < src_high);
		auto const conv = [&](u8 p) { return lerp_clamp(src_low, src_high, U8_MIN, U8_MAX, p); };
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
			auto const conv = [&](u8 p) { return lerp_clamp(src_low, src_high, U8_MIN, U8_MAX, p); };
			seq::transform(src, dst, conv);
		}


		void contrast(gray::image_t const& src, gray::view_t const& dst, u8 src_low, u8 src_high)
		{
			assert(src_low < src_high);
			auto const conv = [&](u8 p) { return lerp_clamp(src_low, src_high, U8_MIN, U8_MAX, p); };
			seq::transform(src, dst, conv);
		}


		void contrast(gray::view_t const& src, gray::image_t const& dst, u8 src_low, u8 src_high)
		{
			assert(src_low < src_high);
			auto const conv = [&](u8 p) { return lerp_clamp(src_low, src_high, U8_MIN, U8_MAX, p); };
			seq::transform(src, dst, conv);
		}


		void contrast(gray::view_t const& src, gray::view_t const& dst, u8 src_low, u8 src_high)
		{
			assert(src_low < src_high);
			auto const conv = [&](u8 p) { return lerp_clamp(src_low, src_high, U8_MIN, U8_MAX, p); };
			seq::transform(src, dst, conv);
		}


		void contrast_self(gray::image_t const& src_dst, u8 src_low, u8 src_high)
		{
			assert(src_low < src_high);
			auto const conv = [&](u8 p) { return lerp_clamp(src_low, src_high, U8_MIN, U8_MAX, p); };
			seq::transform_self(src_dst, conv);
		}


		void contrast_self(gray::view_t const& src_dst, u8 src_low, u8 src_high)
		{
			assert(src_low < src_high);
			auto const conv = [&](u8 p) { return lerp_clamp(src_low, src_high, U8_MIN, U8_MAX, p); };
			seq::transform_self(src_dst, conv);
		}


#endif // !LIBIMAGE_NO_GRAYSCALE
	}

}

namespace libimage
{

	/*static constexpr r32 q_inv_sqrt(r32 n)
	{
		const float threehalfs = 1.5F;
		float y = n;

		long i = *(long*)&y;

		i = 0x5f3759df - (i >> 1);
		y = *(float*)&i;

		y = y * (threehalfs - ((n * 0.5F) * y * y));

		return y;
	}


	static r32 rms_contrast(gray::view_t const& view)
	{
		assert(verify(view));

		auto const norm = [](auto p) { return p / 255.0f; };

		auto total = std::accumulate(view.begin(), view.end(), 0.0f);
		auto mean = norm(total / (view.width * view.height));

		total = std::accumulate(view.begin(), view.end(), 0.0f, [&](r32 total, u8 p) { auto diff = norm(p) - mean; return diff * diff; });

		auto inv_mean = (view.width * view.height) / total;

		return q_inv_sqrt(inv_mean);
	}*/

}