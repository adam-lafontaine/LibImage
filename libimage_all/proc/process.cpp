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


	static u8 lerp(u8 src_low, u8 src_high, u8 dst_low, u8 dst_high, u8 val)
	{
		auto const ratio = (static_cast<r64>(val) - src_low) / (src_high - src_low);

		assert(ratio >= 0.0);
		assert(ratio <= 1.0);

		auto const diff = ratio * (dst_high - dst_low);

		return dst_low + static_cast<u8>(diff);
	}


	static bool verify(image_t const& image)
	{
		return image.data && image.width && image.height;
	}


	static bool verify(view_t const& view)
	{
		return view.image_data && view.width && view.height;
	}


	static bool verify(gray::image_t const& image)
	{
		return image.data && image.width && image.height;
	}


	static bool verify(gray::view_t const& view)
	{
		return view.image_data && view.width && view.height;
	}


	static bool verify_src_dst(image_t const& src, image_t const& dst)
	{
		return verify(src) && verify(dst) && dst.width == src.width && dst.height == src.height;
	}


	static bool verify_src_dst(image_t const& src, view_t const& dst)
	{
		return verify(src) && verify(dst) && dst.width == src.width && dst.height == src.height;
	}


	static bool verify_src_dst(view_t const& src, view_t const& dst)
	{
		return verify(src) && verify(dst) && dst.width == src.width && dst.height == src.height;
	}


	static bool verify_src_dst(view_t const& src, image_t const& dst)
	{
		return verify(src) && verify(dst) && dst.width == src.width && dst.height == src.height;
	}


	static bool verify_src_dst(image_t const& src, gray::image_t const& dst)
	{
		return verify(src) && verify(dst) && dst.width == src.width && dst.height == src.height;
	}


	static bool verify_src_dst(image_t const& src, gray::view_t const& dst)
	{
		return verify(src) && verify(dst) && dst.width == src.width && dst.height == src.height;
	}


	static bool verify_src_dst(view_t const& src, gray::view_t const& dst)
	{
		return verify(src) && verify(dst) && dst.width == src.width && dst.height == src.height;
	}


	static bool verify_src_dst(view_t const& src, gray::image_t const& dst)
	{
		return verify(src) && verify(dst) && dst.width == src.width && dst.height == src.height;
	}


	static bool verify_src_dst(gray::image_t const& src, gray::image_t const& dst)
	{
		return verify(src) && verify(dst) && dst.width == src.width && dst.height == src.height;
	}


	static bool verify_src_dst(gray::image_t const& src, gray::view_t const& dst)
	{
		return verify(src) && verify(dst) && dst.width == src.width && dst.height == src.height;
	}


	static bool verify_src_dst(gray::view_t const& src, gray::view_t const& dst)
	{
		return verify(src) && verify(dst) && dst.width == src.width && dst.height == src.height;
	}


	static bool verify_src_dst(gray::view_t const& src, gray::image_t const& dst)
	{
		return verify(src) && verify(dst) && dst.width == src.width && dst.height == src.height;
	}


	void convert(image_t const& src, gray::image_t const& dst, pixel_to_u8_f const& func)
	{
		assert(verify_src_dst(src, dst));

		std::transform(src.begin(), src.end(), dst.begin(), func);
	}


	void convert(image_t const& src, gray::view_t const& dst, pixel_to_u8_f const& func)
	{
		assert(verify_src_dst(src, dst));

		std::transform(src.begin(), src.end(), dst.begin(), func);
	}


	void convert(view_t const& src, gray::image_t const& dst, pixel_to_u8_f const& func)
	{
		assert(verify_src_dst(src, dst));

		std::transform(src.begin(), src.end(), dst.begin(), func);
	}


	void convert(view_t const& src, gray::view_t const& dst, pixel_to_u8_f const& func)
	{
		assert(verify_src_dst(src, dst));

		std::transform(src.begin(), src.end(), dst.begin(), func);
	}


	void convert_grayscale(image_t const& src, gray::image_t const& dst)
	{
		assert(verify_src_dst(src, dst));

		std::transform(src.begin(), src.end(), dst.begin(), pixel_grayscale_standard);
	}


	void convert_grayscale(image_t const& src, gray::view_t const& dst)
	{
		assert(verify_src_dst(src, dst));

		std::transform(src.begin(), src.end(), dst.begin(), pixel_grayscale_standard);
	}


	void convert_grayscale(view_t const& src, gray::image_t const& dst)
	{
		assert(verify_src_dst(src, dst));

		std::transform(src.begin(), src.end(), dst.begin(), pixel_grayscale_standard);
	}


	void convert_grayscale(view_t const& src, gray::view_t const& dst)
	{
		assert(verify_src_dst(src, dst));

		std::transform(src.begin(), src.end(), dst.begin(), pixel_grayscale_standard);
	}


	void convert_alpha_grayscale(image_t const& image)
	{
		convert_alpha(image, pixel_grayscale_standard);
	}


	void convert_alpha_grayscale(view_t const& view)
	{
		convert_alpha(view, pixel_grayscale_standard);
	}


	void convert_alpha(image_t const& image, pixel_to_u8_f const& func)
	{
		assert(verify(image));

		auto const update = [&](pixel_t& p) { p.alpha = func(p); };
		std::for_each(image.begin(), image.end(), update);
	}


	void convert_alpha(view_t const& view, pixel_to_u8_f const& func)
	{
		assert(verify(view));

		auto const update = [&](pixel_t& p) { p.alpha = func(p); };
		std::for_each(view.begin(), view.end(), update);
	}


	void adjust_contrast(gray::image_t const& src, gray::image_t const& dst, u8 src_low, u8 src_high)
	{
		assert(verify_src_dst(src, dst));
		assert(src_low < src_high);

		u8 dst_low = 0;
		u8 dst_high = 255;

		auto const conv = [&](gray::pixel_t const& p) { return lerp(src_low, src_high, dst_low, dst_high, p); };
		std::transform(src.begin(), src.end(), dst.begin(), conv);
	}


	void adjust_contrast(gray::image_t const& src, u8 src_low, u8 src_high)
	{
		assert(verify(src));
		assert(src_low < src_high);

		u8 dst_low = 0;
		u8 dst_high = 255;

		auto const conv = [&](gray::pixel_t& p) { p = lerp(src_low, src_high, dst_low, dst_high, p); };
		std::for_each(src.begin(), src.end(), conv);
	}


	namespace par
	{
		void convert(image_t const& src, gray::image_t const& dst, pixel_to_u8_f const& func)
		{
			assert(verify_src_dst(src, dst));

			std::transform(std::execution::par, src.begin(), src.end(), dst.begin(), func);
		}


		void convert(image_t const& src, gray::view_t const& dst, pixel_to_u8_f const& func)
		{
			assert(verify_src_dst(src, dst));

			std::transform(std::execution::par, src.begin(), src.end(), dst.begin(), func);
		}


		void convert(view_t const& src, gray::image_t const& dst, pixel_to_u8_f const& func)
		{
			assert(verify_src_dst(src, dst));

			std::transform(std::execution::par, src.begin(), src.end(), dst.begin(), func);
		}


		void convert(view_t const& src, gray::view_t const& dst, pixel_to_u8_f const& func)
		{
			assert(verify_src_dst(src, dst));

			std::transform(std::execution::par, src.begin(), src.end(), dst.begin(), func);
		}


		void convert_grayscale(image_t const& src, gray::image_t const& dst)
		{
			assert(verify_src_dst(src, dst));

			std::transform(std::execution::par, src.begin(), src.end(), dst.begin(), pixel_grayscale_standard);
		}


		void convert_grayscale(image_t const& src, gray::view_t const& dst)
		{
			assert(verify_src_dst(src, dst));

			std::transform(std::execution::par, src.begin(), src.end(), dst.begin(), pixel_grayscale_standard);
		}


		void convert_grayscale(view_t const& src, gray::image_t const& dst)
		{
			assert(verify_src_dst(src, dst));

			std::transform(std::execution::par, src.begin(), src.end(), dst.begin(), pixel_grayscale_standard);
		}


		void convert_grayscale(view_t const& src, gray::view_t const& dst)
		{
			assert(verify_src_dst(src, dst));

			std::transform(std::execution::par, src.begin(), src.end(), dst.begin(), pixel_grayscale_standard);
		}


		void convert_alpha_grayscale(image_t const& image)
		{
			assert(verify(image));

			std::for_each(std::execution::par, image.begin(), image.end(), pixel_grayscale_standard);
		}


		void convert_alpha_grayscale(view_t const& view)
		{
			assert(verify(view));

			std::for_each(std::execution::par, view.begin(), view.end(), pixel_grayscale_standard);
		}


		void convert_alpha(image_t const& image, pixel_to_u8_f const& func)
		{
			assert(verify(image));

			auto const update = [&](pixel_t& p) { p.alpha = func(p); };
			std::for_each(std::execution::par, image.begin(), image.end(), update);
		}


		void convert_alpha(view_t const& view, pixel_to_u8_f const& func)
		{
			assert(verify(view));

			auto const update = [&](pixel_t& p) { p.alpha = func(p); };
			std::for_each(std::execution::par, view.begin(), view.end(), update);
		}


		void adjust_contrast(gray::image_t const& src, gray::image_t const& dst, u8 src_low, u8 src_high)
		{
			assert(verify_src_dst(src, dst));
			assert(src_low < src_high);

			u8 dst_low = 0;
			u8 dst_high = 255;

			auto const conv = [&](gray::pixel_t const& p) { return lerp(src_low, src_high, dst_low, dst_high, p); };
			std::transform(std::execution::par, src.begin(), src.end(), dst.begin(), conv);
		}


		void adjust_contrast(gray::image_t const& src, u8 src_low, u8 src_high)
		{
			assert(verify(src));
			assert(src_low < src_high);

			u8 dst_low = 0;
			u8 dst_high = 255;

			auto const conv = [&](gray::pixel_t& p) { p = lerp(src_low, src_high, dst_low, dst_high, p); };
			std::for_each(std::execution::par, src.begin(), src.end(), conv);
		}
	}



}