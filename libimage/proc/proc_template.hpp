#pragma once

#include "../libimage.hpp"

#include <functional>
#include <algorithm>
#include <cassert>

// TODO: module


namespace libimage
{
	using pixel_to_u8_f = std::function<u8(pixel_t const& p)>;

	using u8_to_u8_f = std::function<u8(u8)>;

	using u8_to_bool_f = std::function<bool(u8)>;


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


	template<class Image>
	static bool verify(Image const& image)
	{
		return image.data && image.width && image.height;
	}


	template<class Image>
	static bool verify(Image const& src, Image const& dst)
	{
		return verify(src) && verify(dst) && dst.width == src.width && dst.height == src.height;
	}


	template<class ImageSrc, class ImageDst>
	static bool verify(ImageSrc const& src, ImageDst const& dst)
	{
		return verify(src) && verify(dst) && dst.width == src.width && dst.height == src.height;
	}


	template<class ColorImage, class GrayImage>
	void convert(ColorImage const& src, GrayImage const& dst, pixel_to_u8_f const& func)
	{
		assert(verify(src, dst));

		std::transform(src.begin(), src.end(), dst.begin(), func);
	}


	template<class ColorImage, class GrayImage>
	void convert_grayscale(ColorImage const& src, GrayImage const& dst)
	{
		convert(src, dst, pixel_grayscale_standard);
	}


	template<class ColorImage>
	void convert_alpha(ColorImage const& image, pixel_to_u8_f const& func)
	{
		assert(verify(image));

		auto const update = [&](pixel_t& p) { p.alpha = func(p); };
		std::for_each(image.begin(), image.end(), update);
	}


	template<class ColorImage>
	void convert_alpha_grayscale(ColorImage const& image)
	{
		convert_alpha(image, pixel_grayscale_standard);
	}
}