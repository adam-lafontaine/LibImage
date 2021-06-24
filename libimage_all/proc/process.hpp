#pragma once

#include "../libimage.hpp"

#include <functional>


namespace libimage
{
	void convert_grayscale(image_t const& src, gray::image_t const& dst);

	void convert_grayscale(image_t const& src, gray::view_t const& dst);

	void convert_grayscale(view_t const& src, gray::image_t const& dst);

	void convert_grayscale(view_t const& src, gray::view_t const& dst);

	void convert(image_t const& src, gray::image_t const& dst, std::function<u8(pixel_t const& p)> const& func);

	void convert(image_t const& src, gray::view_t const& dst, std::function<u8(pixel_t const& p)> const& func);

	void convert(view_t const& src, gray::image_t const& dst, std::function<u8(pixel_t const& p)> const& func);

	void convert(view_t const& src, gray::view_t const& dst, std::function<u8(pixel_t const& p)> const& func);

	void convert_alpha_grayscale(image_t const& src);

	void convert_alpha_grayscale(view_t const& src);

	void convert_alpha(image_t const& src, std::function<u8(pixel_t const& p)> const& func);

	void convert_alpha(view_t const& src, std::function<u8(pixel_t const& p)> const& func);
	
	void adjust_contrast(gray::image_t const& src, gray::image_t const& dst, u8 src_low, u8 src_high);

	void adjust_contrast(gray::image_t const& src, u8 src_low, u8 src_high);


	namespace par
	{
		void convert_grayscale(image_t const& src, gray::image_t const& dst);

		void convert_grayscale(image_t const& src, gray::view_t const& dst);

		void convert_grayscale(view_t const& src, gray::image_t const& dst);

		void convert_grayscale(view_t const& src, gray::view_t const& dst);

		void convert(image_t const& src, gray::image_t const& dst, std::function<u8(pixel_t const& p)> const& func);

		void convert(image_t const& src, gray::view_t const& dst, std::function<u8(pixel_t const& p)> const& func);

		void convert(view_t const& src, gray::image_t const& dst, std::function<u8(pixel_t const& p)> const& func);

		void convert(view_t const& src, gray::view_t const& dst, std::function<u8(pixel_t const& p)> const& func);

		void convert_alpha_grayscale(image_t const& src);

		void convert_alpha_grayscale(view_t const& src);

		void convert_alpha(image_t const& src, std::function<u8(pixel_t const& p)> const& func);

		void convert_alpha(view_t const& src, std::function<u8(pixel_t const& p)> const& func);

		void adjust_contrast(gray::image_t const& src, gray::image_t const& dst, u8 src_low, u8 src_high);

		void adjust_contrast(gray::image_t const& src, u8 src_low, u8 src_high);
	}
}


