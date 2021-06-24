#pragma once

#include "../libimage.hpp"

#include <functional>

// TODO: alpha blending
// TODO: binarize
// TODO: convert gray -> gray



namespace libimage
{
	using pixel_to_u8_f = std::function<u8(pixel_t const& p)>;

	using u8_to_u8_f = std::function<u8(u8)>;

	void convert_grayscale(image_t const& src, gray::image_t const& dst);

	void convert_grayscale(image_t const& src, gray::view_t const& dst);

	void convert_grayscale(view_t const& src, gray::image_t const& dst);

	void convert_grayscale(view_t const& src, gray::view_t const& dst);

	void convert(image_t const& src, gray::image_t const& dst, pixel_to_u8_f const& func);

	void convert(image_t const& src, gray::view_t const& dst, pixel_to_u8_f const& func);

	void convert(view_t const& src, gray::image_t const& dst, pixel_to_u8_f const& func);

	void convert(view_t const& src, gray::view_t const& dst, pixel_to_u8_f const& func);

	void convert(gray::image_t const& src, gray::image_t const& dst, u8_to_u8_f const& func);

	void convert(gray::image_t const& src, gray::view_t const& dst, u8_to_u8_f const& func);

	void convert(gray::view_t const& src, gray::image_t const& dst, u8_to_u8_f const& func);

	void convert(gray::view_t const& src, gray::view_t const& dst, u8_to_u8_f const& func);

	void convert_alpha_grayscale(image_t const& src);

	void convert_alpha_grayscale(view_t const& src);

	void convert_alpha(image_t const& src, pixel_to_u8_f const& func);

	void convert_alpha(view_t const& src, pixel_to_u8_f const& func);
	
	void adjust_contrast(gray::image_t const& src, gray::image_t const& dst, u8 src_low, u8 src_high);

	void adjust_contrast(gray::image_t const& src, u8 src_low, u8 src_high);


	namespace par
	{
		void convert_grayscale(image_t const& src, gray::image_t const& dst);

		void convert_grayscale(image_t const& src, gray::view_t const& dst);

		void convert_grayscale(view_t const& src, gray::image_t const& dst);

		void convert_grayscale(view_t const& src, gray::view_t const& dst);

		void convert(image_t const& src, gray::image_t const& dst, pixel_to_u8_f const& func);

		void convert(image_t const& src, gray::view_t const& dst, pixel_to_u8_f const& func);

		void convert(view_t const& src, gray::image_t const& dst, pixel_to_u8_f const& func);

		void convert(view_t const& src, gray::view_t const& dst, pixel_to_u8_f const& func);

		void convert(gray::image_t const& src, gray::image_t const& dst, u8_to_u8_f const& func);

		void convert(gray::image_t const& src, gray::view_t const& dst, u8_to_u8_f const& func);

		void convert(gray::view_t const& src, gray::image_t const& dst, u8_to_u8_f const& func);

		void convert(gray::view_t const& src, gray::view_t const& dst, u8_to_u8_f const& func);

		void convert_alpha_grayscale(image_t const& src);

		void convert_alpha_grayscale(view_t const& src);

		void convert_alpha(image_t const& src, pixel_to_u8_f const& func);

		void convert_alpha(view_t const& src, pixel_to_u8_f const& func);

		void adjust_contrast(gray::image_t const& src, gray::image_t const& dst, u8 src_low, u8 src_high);

		void adjust_contrast(gray::image_t const& src, u8 src_low, u8 src_high);
	}
}


