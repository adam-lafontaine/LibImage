/*

Copyright (c) 2021 Adam Lafontaine

*/

#include "process.hpp"
#include "verify.hpp"
#include "index_range.hpp"

#include <algorithm>

#ifndef LIBIMAGE_NO_PARALLEL
#include <execution>
#endif // !LIBIMAGE_NO_PARALLEL


namespace libimage
{
	static constexpr u8 rgb_grayscale_standard(u8 red, u8 green, u8 blue)
	{
		return static_cast<u8>(0.299 * red + 0.587 * green + 0.114 * blue);
	}


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


#ifndef LIBIMAGE_NO_COLOR

	static constexpr u8 pixel_grayscale_standard(pixel_t const& p)
	{
		return rgb_grayscale_standard(p.red, p.green, p.blue);
	}

#endif // !LIBIMAGE_NO_COLOR

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

}


namespace libimage
{
#ifndef LIBIMAGE_NO_PARALLEL

#ifndef LIBIMAGE_NO_COLOR


	void transform(image_t const& src, image_t const& dst, pixel_to_pixel_f const& func)
	{
		assert(verify(src, dst));
		std::transform(std::execution::par, src.begin(), src.end(), dst.begin(), func);
	}


	void transform(image_t const& src, view_t const& dst, pixel_to_pixel_f const& func)
	{
		assert(verify(src, dst));
		std::transform(std::execution::par, src.begin(), src.end(), dst.begin(), func);
	}


	void transform(view_t const& src, image_t const& dst, pixel_to_pixel_f const& func)
	{
		assert(verify(src, dst));
		std::transform(std::execution::par, src.begin(), src.end(), dst.begin(), func);
	}


	void transform(view_t const& src, view_t const& dst, pixel_to_pixel_f const& func)
	{
		assert(verify(src, dst));
		std::transform(std::execution::par, src.begin(), src.end(), dst.begin(), func);
	}


	void transform_self(image_t const& src_dst, pixel_to_pixel_f const& func)
	{
		assert(verify(src_dst));
		auto const update = [&](pixel_t& p) { p = func(p); };
		std::for_each(std::execution::par, src_dst.begin(), src_dst.end(), update);
	}


	void transform_self(view_t const& src_dst, pixel_to_pixel_f const& func)
	{
		assert(verify(src_dst));
		auto const update = [&](pixel_t& p) { p = func(p); };
		std::for_each(std::execution::par, src_dst.begin(), src_dst.end(), update);
	}


	void transform_alpha(image_t const& src_dst, pixel_to_u8_f const& func)
	{
		assert(verify(src_dst));
		auto const update = [&](pixel_t& p) { p.alpha = func(p); };
		std::for_each(std::execution::par, src_dst.begin(), src_dst.end(), update);
	}


	void transform_alpha(view_t const& src_dst, pixel_to_u8_f const& func)
	{
		assert(verify(src_dst));
		auto const update = [&](pixel_t& p) { p.alpha = func(p); };
		std::for_each(std::execution::par, src_dst.begin(), src_dst.end(), update);
	}


	void transform_alpha_grayscale(image_t const& src)
	{
		transform_alpha(src, pixel_grayscale_standard);
	}


	void transform_alpha_grayscale(view_t const& src)
	{
		transform_alpha(src, pixel_grayscale_standard);
	}

#endif // !LIBIMAGE_NO_COLOR


#ifndef LIBIMAGE_NO_GRAYSCALE

	lookup_table_t to_lookup_table(u8_to_u8_f const& func)
	{
		std::array<u8, 256> lut = { 0 };

		u32_range_t ids(0u, 256u);

		std::for_each(std::execution::par, ids.begin(), ids.end(), [&](u32 id) { lut[id] = func(id); });

		return lut;
	}


	void transform(gray::image_t const& src, gray::image_t const& dst, lookup_table_t const& lut)
	{
		assert(verify(src, dst));
		auto const conv = [&lut](u8 p) { return lut[p]; };
		std::transform(std::execution::par, src.begin(), src.end(), dst.begin(), conv);
	}


	void transform(gray::image_t const& src, gray::view_t const& dst, lookup_table_t const& lut)
	{
		assert(verify(src, dst));
		auto const conv = [&lut](u8 p) { return lut[p]; };
		std::transform(std::execution::par, src.begin(), src.end(), dst.begin(), conv);
	}


	void transform(gray::view_t const& src, gray::image_t const& dst, lookup_table_t const& lut)
	{
		assert(verify(src, dst));
		auto const conv = [&lut](u8 p) { return lut[p]; };
		std::transform(std::execution::par, src.begin(), src.end(), dst.begin(), conv);
	}


	void transform(gray::view_t const& src, gray::view_t const& dst, lookup_table_t const& lut)
	{
		assert(verify(src, dst));
		auto const conv = [&lut](u8 p) { return lut[p]; };
		std::transform(std::execution::par, src.begin(), src.end(), dst.begin(), conv);
	}


	void transform(gray::image_t const& src, gray::image_t const& dst, u8_to_u8_f const& func)
	{
		assert(verify(src, dst));
		auto const lut = to_lookup_table(func);
		transform(src, dst, lut);
	}


	void transform(gray::image_t const& src, gray::view_t const& dst, u8_to_u8_f const& func)
	{
		assert(verify(src, dst));
		auto const lut = to_lookup_table(func);
		transform(src, dst, lut);
	}


	void transform(gray::view_t const& src, gray::image_t const& dst, u8_to_u8_f const& func)
	{
		assert(verify(src, dst));
		auto const lut = to_lookup_table(func);
		transform(src, dst, lut);
	}


	void transform(gray::view_t const& src, gray::view_t const& dst, u8_to_u8_f const& func)
	{
		assert(verify(src, dst));
		auto const lut = to_lookup_table(func);
		transform(src, dst, lut);
	}


	void transform_self(gray::image_t const& src_dst, lookup_table_t const& lut)
	{
		assert(verify(src_dst));
		auto const conv = [&lut](u8& p) { p = lut[p]; };
		std::for_each(std::execution::par, src_dst.begin(), src_dst.end(), conv);
	}

	void transform_self(gray::view_t const& src_dst, lookup_table_t const& lut)
	{
		assert(verify(src_dst));
		auto const conv = [&lut](u8& p) { p = lut[p]; };
		std::for_each(std::execution::par, src_dst.begin(), src_dst.end(), conv);
	}


	void transform_self(gray::image_t const& src_dst, u8_to_u8_f const& func)
	{
		assert(verify(src_dst));
		auto const lut = to_lookup_table(func);
		transform_self(src_dst, lut);
	}


	void transform_self(gray::view_t const& src_dst, u8_to_u8_f const& func)
	{
		assert(verify(src_dst));
		auto const lut = to_lookup_table(func);
		transform_self(src_dst, lut);
	}


	void transform_contrast(gray::image_t const& src, gray::image_t const& dst, u8 src_low, u8 src_high)
	{
		assert(src_low < src_high);
		constexpr u8 dst_low = 0;
		constexpr u8 dst_high = 255;
		auto const conv = [&](u8 p) { return lerp_clamp(src_low, src_high, dst_low, dst_high, p); };
		transform(src, dst, conv);
	}


	void transform_contrast(gray::image_t const& src, gray::view_t const& dst, u8 src_low, u8 src_high)
	{
		assert(src_low < src_high);
		constexpr u8 dst_low = 0;
		constexpr u8 dst_high = 255;
		auto const conv = [&](u8 p) { return lerp_clamp(src_low, src_high, dst_low, dst_high, p); };
		transform(src, dst, conv);
	}


	void transform_contrast(gray::view_t const& src, gray::image_t const& dst, u8 src_low, u8 src_high)
	{
		assert(src_low < src_high);
		constexpr u8 dst_low = 0;
		constexpr u8 dst_high = 255;
		auto const conv = [&](u8 p) { return lerp_clamp(src_low, src_high, dst_low, dst_high, p); };
		transform(src, dst, conv);
	}



	void transform_contrast(gray::view_t const& src, gray::view_t const& dst, u8 src_low, u8 src_high)
	{
		assert(src_low < src_high);
		constexpr u8 dst_low = 0;
		constexpr u8 dst_high = 255;
		auto const conv = [&](u8 p) { return lerp_clamp(src_low, src_high, dst_low, dst_high, p); };
		transform(src, dst, conv);
	}


	void transform_contrast_self(gray::image_t const& src_dst, u8 src_low, u8 src_high)
	{
		assert(src_low < src_high);
		constexpr u8 dst_low = 0;
		constexpr u8 dst_high = 255;
		auto const conv = [&](u8 p) { return lerp_clamp(src_low, src_high, dst_low, dst_high, p); };
		transform_self(src_dst, conv);
	}


	void transform_contrast_self(gray::view_t const& src_dst, u8 src_low, u8 src_high)
	{
		assert(src_low < src_high);
		constexpr u8 dst_low = 0;
		constexpr u8 dst_high = 255;
		auto const conv = [&](u8 p) { return lerp_clamp(src_low, src_high, dst_low, dst_high, p); };
		transform_self(src_dst, conv);
	}

#endif // !LIBIMAGE_NO_GRAYSCALE


#ifndef LIBIMAGE_NO_COLOR
#ifndef LIBIMAGE_NO_GRAYSCALE

	void transform(image_t const& src, gray::image_t const& dst, pixel_to_u8_f const& func)
	{
		assert(verify(src, dst));
		std::transform(std::execution::par, src.begin(), src.end(), dst.begin(), func);
	}


	void transform(image_t const& src, gray::view_t const& dst, pixel_to_u8_f const& func)
	{
		assert(verify(src, dst));
		std::transform(std::execution::par, src.begin(), src.end(), dst.begin(), func);
	}


	void transform(view_t const& src, gray::image_t const& dst, pixel_to_u8_f const& func)
	{
		assert(verify(src, dst));
		std::transform(std::execution::par, src.begin(), src.end(), dst.begin(), func);
	}


	void transform(view_t const& src, gray::view_t const& dst, pixel_to_u8_f const& func)
	{
		assert(verify(src, dst));
		std::transform(std::execution::par, src.begin(), src.end(), dst.begin(), func);
	}


	void transform_grayscale(image_t const& src, gray::image_t const& dst)
	{
		transform(src, dst, pixel_grayscale_standard);
	}


	void transform_grayscale(image_t const& src, gray::view_t const& dst)
	{
		transform(src, dst, pixel_grayscale_standard);
	}


	void transform_grayscale(view_t const& src, gray::image_t const& dst)
	{
		transform(src, dst, pixel_grayscale_standard);
	}


	void transform_grayscale(view_t const& src, gray::view_t const& dst)
	{
		transform(src, dst, pixel_grayscale_standard);
	}


#endif // !LIBIMAGE_NO_GRAYSCALE
#endif // !LIBIMAGE_NO_COLOR


#endif // !LIBIMAGE_NO_PARALLEL
	namespace seq
	{

#ifndef LIBIMAGE_NO_COLOR


		void transform(image_t const& src, image_t const& dst, pixel_to_pixel_f const& func)
		{
			assert(verify(src, dst));
			std::transform(src.begin(), src.end(), dst.begin(), func);
		}


		void transform(image_t const& src, view_t const& dst, pixel_to_pixel_f const& func)
		{
			assert(verify(src, dst));
			std::transform(src.begin(), src.end(), dst.begin(), func);
		}


		void transform(view_t const& src, image_t const& dst, pixel_to_pixel_f const& func)
		{
			assert(verify(src, dst));
			std::transform(src.begin(), src.end(), dst.begin(), func);
		}


		void transform(view_t const& src, view_t const& dst, pixel_to_pixel_f const& func)
		{
			assert(verify(src, dst));
			std::transform(src.begin(), src.end(), dst.begin(), func);
		}


		void transform_self(image_t const& src_dst, pixel_to_pixel_f const& func)
		{
			assert(verify(src_dst));
			auto const update = [&](pixel_t& p) { p = func(p); };
			std::for_each(src_dst.begin(), src_dst.end(), update);
		}


		void transform_self(view_t const& src_dst, pixel_to_pixel_f const& func)
		{
			assert(verify(src_dst));
			auto const update = [&](pixel_t& p) { p = func(p); };
			std::for_each(src_dst.begin(), src_dst.end(), update);
		}


		void transform_alpha(image_t const& src_dst, pixel_to_u8_f const& func)
		{
			assert(verify(src_dst));
			auto const conv = [&](pixel_t& p) { p.alpha = func(p); };
			std::for_each(src_dst.begin(), src_dst.end(), conv);
		}


		void transform_alpha(view_t const& src_dst, pixel_to_u8_f const& func)
		{
			assert(verify(src_dst));
			auto const conv = [&](pixel_t& p) { p.alpha = func(p); };
			std::for_each(src_dst.begin(), src_dst.end(), conv);
		}


		void transform_alpha_grayscale(image_t const& src_dst)
		{
			seq::transform_alpha(src_dst, pixel_grayscale_standard);
		}


		void transform_alpha_grayscale(view_t const& src_dst)
		{
			seq::transform_alpha(src_dst, pixel_grayscale_standard);
		}

#endif // !LIBIMAGE_NO_COLOR

#ifndef LIBIMAGE_NO_GRAYSCALE

		lookup_table_t to_lookup_table(u8_to_u8_f const& func)
		{
			std::array<u8, 256> lut = { 0 };

			u32_range_t ids(0u, 256u);

			std::for_each(ids.begin(), ids.end(), [&](u32 id) { lut[id] = func(id); });

			return lut;
		}


		void transform(gray::image_t const& src, gray::image_t const& dst, lookup_table_t const& lut)
		{
			assert(verify(src, dst));
			auto const conv = [&lut](u8 p) { return lut[p]; };
			std::transform(src.begin(), src.end(), dst.begin(), conv);
		}


		void transform(gray::image_t const& src, gray::view_t const& dst, lookup_table_t const& lut)
		{
			assert(verify(src, dst));
			auto const conv = [&lut](u8 p) { return lut[p]; };
			std::transform(src.begin(), src.end(), dst.begin(), conv);
		}


		void transform(gray::view_t const& src, gray::image_t const& dst, lookup_table_t const& lut)
		{
			assert(verify(src, dst));
			auto const conv = [&lut](u8 p) { return lut[p]; };
			std::transform(src.begin(), src.end(), dst.begin(), conv);
		}


		void transform(gray::view_t const& src, gray::view_t const& dst, lookup_table_t const& lut)
		{
			assert(verify(src, dst));
			auto const conv = [&lut](u8 p) { return lut[p]; };
			std::transform(src.begin(), src.end(), dst.begin(), conv);
		}


		void transform(gray::image_t const& src, gray::image_t const& dst, u8_to_u8_f const& func)
		{
			assert(verify(src, dst));
			auto const lut = seq::to_lookup_table(func);
			auto const conv = [&lut](u8 p) { return lut[p]; };
			std::transform(src.begin(), src.end(), dst.begin(), conv);
		}


		void transform(gray::image_t const& src, gray::view_t const& dst, u8_to_u8_f const& func)
		{
			assert(verify(src, dst));
			auto const lut = seq::to_lookup_table(func);
			auto const conv = [&lut](u8 p) { return lut[p]; };
			std::transform(src.begin(), src.end(), dst.begin(), conv);
		}


		void transform(gray::view_t const& src, gray::image_t const& dst, u8_to_u8_f const& func)
		{
			assert(verify(src, dst));
			auto const lut = seq::to_lookup_table(func);
			auto const conv = [&lut](u8 p) { return lut[p]; };
			std::transform(src.begin(), src.end(), dst.begin(), conv);
		}


		void transform(gray::view_t const& src, gray::view_t const& dst, u8_to_u8_f const& func)
		{
			assert(verify(src, dst));
			auto const lut = seq::to_lookup_table(func);
			auto const conv = [&lut](u8 p) { return lut[p]; };
			std::transform(src.begin(), src.end(), dst.begin(), conv);
		}


		void transform_self(gray::image_t const& src_dst, lookup_table_t const& lut)
		{
			assert(verify(src_dst));
			auto const conv = [&lut](u8& p) { p = lut[p]; };
			std::for_each(src_dst.begin(), src_dst.end(), conv);
		}

		void transform_self(gray::view_t const& src_dst, lookup_table_t const& lut)
		{
			assert(verify(src_dst));
			auto const conv = [&lut](u8& p) { p = lut[p]; };
			std::for_each(src_dst.begin(), src_dst.end(), conv);
		}


		void transform_self(gray::image_t const& src_dst, u8_to_u8_f const& func)
		{
			assert(verify(src_dst));
			auto const lut = to_lookup_table(func);
			seq::transform_self(src_dst, lut);
		}


		void transform_self(gray::view_t const& src_dst, u8_to_u8_f const& func)
		{
			assert(verify(src_dst));
			auto const lut = to_lookup_table(func);
			seq::transform_self(src_dst, lut);
		}


		void transform_contrast(gray::image_t const& src, gray::image_t const& dst, u8 src_low, u8 src_high)
		{
			assert(src_low < src_high);
			u8 dst_low = 0;
			u8 dst_high = 255;
			auto const conv = [&](u8 p) { return lerp_clamp(src_low, src_high, dst_low, dst_high, p); };
			seq::transform(src, dst, conv);
		}


		void transform_contrast(gray::image_t const& src, gray::view_t const& dst, u8 src_low, u8 src_high)
		{
			assert(src_low < src_high);
			u8 dst_low = 0;
			u8 dst_high = 255;
			auto const conv = [&](u8 p) { return lerp_clamp(src_low, src_high, dst_low, dst_high, p); };
			seq::transform(src, dst, conv);
		}


		void transform_contrast(gray::view_t const& src, gray::image_t const& dst, u8 src_low, u8 src_high)
		{
			assert(src_low < src_high);
			u8 dst_low = 0;
			u8 dst_high = 255;
			auto const conv = [&](u8 p) { return lerp_clamp(src_low, src_high, dst_low, dst_high, p); };
			seq::transform(src, dst, conv);
		}


		void transform_contrast(gray::view_t const& src, gray::view_t const& dst, u8 src_low, u8 src_high)
		{
			assert(src_low < src_high);
			u8 dst_low = 0;
			u8 dst_high = 255;
			auto const conv = [&](u8 p) { return lerp_clamp(src_low, src_high, dst_low, dst_high, p); };
			seq::transform(src, dst, conv);
		}


		void transform_contrast_self(gray::image_t const& src_dst, u8 src_low, u8 src_high)
		{
			assert(src_low < src_high);
			u8 dst_low = 0;
			u8 dst_high = 255;
			auto const conv = [&](u8 p) { return lerp_clamp(src_low, src_high, dst_low, dst_high, p); };
			seq::transform_self(src_dst, conv);
		}


		void transform_contrast_self(gray::view_t const& src_dst, u8 src_low, u8 src_high)
		{
			assert(src_low < src_high);
			u8 dst_low = 0;
			u8 dst_high = 255;
			auto const conv = [&](u8 p) { return lerp_clamp(src_low, src_high, dst_low, dst_high, p); };
			seq::transform_self(src_dst, conv);
		}

#endif // !LIBIMAGE_NO_GRAYSCALE


#ifndef LIBIMAGE_NO_COLOR
#ifndef LIBIMAGE_NO_GRAYSCALE


		void transform(image_t const& src, gray::image_t const& dst, pixel_to_u8_f const& func)
		{
			assert(verify(src, dst));
			std::transform(src.begin(), src.end(), dst.begin(), func);
		}


		void transform(image_t const& src, gray::view_t const& dst, pixel_to_u8_f const& func)
		{
			assert(verify(src, dst));
			std::transform(src.begin(), src.end(), dst.begin(), func);
		}


		void transform(view_t const& src, gray::image_t const& dst, pixel_to_u8_f const& func)
		{
			assert(verify(src, dst));
			std::transform(src.begin(), src.end(), dst.begin(), func);
		}


		void transform(view_t const& src, gray::view_t const& dst, pixel_to_u8_f const& func)
		{
			assert(verify(src, dst));
			std::transform(src.begin(), src.end(), dst.begin(), func);
		}


		void transform_grayscale(image_t const& src, gray::image_t const& dst)
		{
			seq::transform(src, dst, pixel_grayscale_standard);
		}


		void transform_grayscale(image_t const& src, gray::view_t const& dst)
		{
			seq::transform(src, dst, pixel_grayscale_standard);
		}


		void transform_grayscale(view_t const& src, gray::image_t const& dst)
		{
			seq::transform(src, dst, pixel_grayscale_standard);
		}


		void transform_grayscale(view_t const& src, gray::view_t const& dst)
		{
			seq::transform(src, dst, pixel_grayscale_standard);
		}

#endif // !LIBIMAGE_NO_GRAYSCALE
#endif // !LIBIMAGE_NO_COLOR
	}

}