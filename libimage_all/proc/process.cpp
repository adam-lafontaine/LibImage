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


	void convert(image_t const& src, gray::image_t const& dst, std::function<u8(pixel_t const& p)> const& func)
	{
		assert(src.data);
		assert(src.width);
		assert(src.height);
		assert(dst.data);
		assert(dst.width == src.width);
		assert(dst.height == src.height);

		std::transform(std::execution::par, src.begin(), src.end(), dst.begin(), func);
	}


	void convert(image_t const& src, gray::view_t const& dst, std::function<u8(pixel_t const& p)> const& func)
	{
		assert(src.data);
		assert(src.width);
		assert(src.height);
		assert(dst.data);
		assert(dst.width == src.width);
		assert(dst.height == src.height);

		std::transform(std::execution::par, src.begin(), src.end(), dst.begin(), func);
	}


	void convert(view_t const& src, gray::image_t const& dst, std::function<u8(pixel_t const& p)> const& func)
	{
		assert(src.data);
		assert(src.width);
		assert(src.height);
		assert(dst.data);
		assert(dst.width == src.width);
		assert(dst.height == src.height);

		std::transform(std::execution::par, src.begin(), src.end(), dst.begin(), func);
	}


	void convert(view_t const& src, gray::view_t const& dst, std::function<u8(pixel_t const& p)> const& func)
	{
		assert(src.data);
		assert(src.width);
		assert(src.height);
		assert(dst.data);
		assert(dst.width == src.width);
		assert(dst.height == src.height);

		std::transform(std::execution::par, src.begin(), src.end(), dst.begin(), func);
	}


	void convert_grayscale(image_t const& src, gray::image_t const& dst)
	{
		convert(src, dst, pixel_grayscale_standard);
	}


	void convert_grayscale(image_t const& src, gray::view_t const& dst)
	{
		convert(src, dst, pixel_grayscale_standard);
	}


	void convert_grayscale(view_t const& src, gray::image_t const& dst)
	{
		convert(src, dst, pixel_grayscale_standard);
	}


	void convert_grayscale(view_t const& src, gray::view_t const& dst)
	{
		convert(src, dst, pixel_grayscale_standard);
	}


	void convert_alpha_grayscale(image_t const& image)
	{
		std::for_each(image.begin(), image.end(), pixel_grayscale_standard);
	}


	void convert_alpha_grayscale(view_t const& view)
	{
		std::for_each(view.begin(), view.end(), pixel_grayscale_standard);
	}


	void convert_alpha(image_t const& image, std::function<u8(pixel_t const& p)> const& func)
	{
		auto const update = [&](pixel_t& p) { p.alpha = func(p); };
		std::for_each(image.begin(), image.end(), update);
	}


	void convert_alpha(view_t const& view, std::function<u8(pixel_t const& p)> const& func)
	{
		auto const update = [&](pixel_t& p) { p.alpha = func(p); };
		std::for_each(view.begin(), view.end(), update);
	}
}