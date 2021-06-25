#include "process.hpp"

#include <cassert>
#include <algorithm>
#include <execution>

namespace libimage
{
	static u8 rgb_grayscale_standard(u8 red, u8 green, u8 blue)
	{
		return static_cast<u8>(0.299 * red + 0.587 * green + 0.114 * blue);
	}


	static u8 pixel_grayscale_standard(pixel_t const& p)
	{
		return rgb_grayscale_standard(p.red, p.green, p.blue);
	}


	static u8 lerp_clamp(u8 src_low, u8 src_high, u8 dst_low, u8 dst_high, u8 val)
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


	static pixel_t alpha_blend_linear(pixel_t const& src, pixel_t const& current)
	{
		auto const a = static_cast<r32>(src.alpha) / 255.0f;

		auto const blend = [&](u8 s, u8 c)
		{
			auto sf = static_cast<r32>(s);
			auto cf = static_cast<r32>(c);

			auto blended = a * cf + (1.0f - a) * sf;

			return static_cast<u8>(blended);
		};

		auto red = blend(src.red, current.red);
		auto green = blend(src.green, current.green);
		auto blue = blend(src.blue, current.blue);

		return to_pixel(red, green, blue);
	}


	static bool verify(view_t const& view)
	{
		return view.image_data && view.width && view.height;
	}


	static bool verify(gray::view_t const& view)
	{
		return view.image_data && view.width && view.height;
	}


	static bool verify(view_t const& src, view_t const& dst)
	{
		return verify(src) && verify(dst) && dst.width == src.width && dst.height == src.height;
	}


	static bool verify(view_t const& src, gray::view_t const& dst)
	{
		return verify(src) && verify(dst) && dst.width == src.width && dst.height == src.height;
	}


	static bool verify(gray::view_t const& src, gray::view_t const& dst)
	{
		return verify(src) && verify(dst) && dst.width == src.width && dst.height == src.height;
	}


	void convert(view_t const& src, gray::view_t const& dst, pixel_to_u8_f const& func)
	{
		assert(verify(src, dst));

		std::transform(src.begin(), src.end(), dst.begin(), func);
	}


	void convert(gray::view_t const& src, gray::view_t const& dst, u8_to_u8_f const& func)
	{
		assert(verify(src, dst));

		std::transform(src.begin(), src.end(), dst.begin(), func);
	}


	void convert(gray::view_t const& src, u8_to_u8_f const& func)
	{
		assert(verify(src));

		auto const conv = [&](u8& p) { p = func(p); };
		std::for_each(src.begin(), src.end(), conv);
	}


	void convert_grayscale(view_t const& src, gray::view_t const& dst)
	{
		convert(src, dst, pixel_grayscale_standard);
	}


	void convert_alpha(view_t const& view, pixel_to_u8_f const& func)
	{
		assert(verify(view));

		auto const update = [&](pixel_t& p) { p.alpha = func(p); };
		std::for_each(view.begin(), view.end(), update);
	}


	void convert_alpha_grayscale(view_t const& view)
	{
		convert_alpha(view, pixel_grayscale_standard);
	}


	void adjust_contrast(gray::view_t const& src, gray::view_t const& dst, u8 src_low, u8 src_high)
	{
		assert(src_low < src_high);

		u8 dst_low = 0;
		u8 dst_high = 255;

		auto const conv = [&](gray::pixel_t const& p) { return lerp_clamp(src_low, src_high, dst_low, dst_high, p); };
		convert(src, dst, conv);
	}


	void adjust_contrast(gray::view_t const& src, u8 src_low, u8 src_high)
	{
		assert(src_low < src_high);

		u8 dst_low = 0;
		u8 dst_high = 255;

		auto const conv = [&](gray::pixel_t const& p) { return lerp_clamp(src_low, src_high, dst_low, dst_high, p); };
		convert(src, conv);
	}


	void binarize(gray::view_t const& src, gray::view_t const& dst, u8_to_bool_f const& func)
	{
		auto const conv = [&](gray::pixel_t const& p) { return func(p) ? 255 : 0; };
		convert(src, dst, conv);
	}


	void binarize(gray::view_t const& src, u8_to_bool_f const& func)
	{
		auto const conv = [&](gray::pixel_t const& p) { return func(p) ? 255 : 0; };
		convert(src, conv);
	}


	void alpha_blend(view_t const& src, view_t const& current, view_t const& dst)
	{
		assert(verify(src, current));
		assert(verify(src, dst));

		auto it_src = src.begin();
		auto it_current = current.begin();
		auto it_dst = dst.begin();
		for (;it_src != src.end(); ++it_src, ++it_current, ++it_dst)
		{
			*it_dst = alpha_blend_linear(*it_src, *it_current);
		}
	}


	void alpha_blend(view_t const& src, view_t const& current_dst)
	{
		assert(verify(src, current_dst));

		auto it_src = src.begin();
		auto it_current_dst = current_dst.begin();
		for (; it_src != src.end(); ++it_src, ++it_current_dst)
		{
			*it_current_dst = alpha_blend_linear(*it_src, *it_current_dst);
		}
	}


	namespace par
	{
		void convert(view_t const& src, gray::view_t const& dst, pixel_to_u8_f const& func)
		{
			assert(verify(src, dst));

			std::transform(std::execution::par, src.begin(), src.end(), dst.begin(), func);
		}


		void convert(gray::view_t const& src, gray::view_t const& dst, u8_to_u8_f const& func)
		{
			assert(verify(src, dst));

			std::transform(std::execution::par, src.begin(), src.end(), dst.begin(), func);
		}


		void convert(gray::view_t const& src, u8_to_u8_f const& func)
		{
			assert(verify(src));

			auto const conv = [&](u8& p) { p = func(p); };
			std::for_each(std::execution::par, src.begin(), src.end(), conv);
		}


		void convert_grayscale(view_t const& src, gray::view_t const& dst)
		{
			par::convert(src, dst, pixel_grayscale_standard);
		}


		void convert_alpha(view_t const& view, pixel_to_u8_f const& func)
		{
			assert(verify(view));

			auto const update = [&](pixel_t& p) { p.alpha = func(p); };
			std::for_each(view.begin(), view.end(), update);
		}


		void convert_alpha_grayscale(view_t const& view)
		{
			par::convert_alpha(view, pixel_grayscale_standard);
		}


		void adjust_contrast(gray::view_t const& src, gray::view_t const& dst, u8 src_low, u8 src_high)
		{
			assert(src_low < src_high);

			u8 dst_low = 0;
			u8 dst_high = 255;

			auto const conv = [&](gray::pixel_t const& p) { return lerp_clamp(src_low, src_high, dst_low, dst_high, p); };
			par::convert(src, dst, conv);
		}


		void adjust_contrast(gray::view_t const& src, u8 src_low, u8 src_high)
		{
			assert(src_low < src_high);

			u8 dst_low = 0;
			u8 dst_high = 255;

			auto const conv = [&](gray::pixel_t const& p) { return lerp_clamp(src_low, src_high, dst_low, dst_high, p); };
			par::convert(src, conv);
		}


		void binarize(gray::view_t const& src, gray::image_t const& dst, u8_to_bool_f const& func)
		{
			auto const conv = [&](gray::pixel_t const& p) { return func(p) ? 255 : 0; };
			par::convert(src, conv);
		}


		void binarize(gray::view_t const& src, gray::view_t const& dst, u8_to_bool_f const& func)
		{
			auto const conv = [&](gray::pixel_t const& p) { return func(p) ? 255 : 0; };
			par::convert(src, conv);
		}


		void binarize(gray::view_t const& src, u8_to_bool_f const& func)
		{
			auto const conv = [&](gray::pixel_t const& p) { return func(p) ? 255 : 0; };
			par::convert(src, conv);
		}

	}

}