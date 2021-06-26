#pragma once

#include "../libimage.hpp"

#include <functional>



namespace libimage
{
	using pixel_to_u8_f = std::function<u8(pixel_t const& p)>;

	using u8_to_u8_f = std::function<u8(u8)>;

	using u8_to_bool_f = std::function<bool(u8)>;


	void convert(view_t const& src, gray::view_t const& dst, pixel_to_u8_f const& func);

	void convert(gray::view_t const& src, gray::view_t const& dst, u8_to_u8_f const& func);

	void convert(gray::view_t const& src, u8_to_u8_f const& func);

	void copy(view_t const& src, view_t const& dst);

	void copy(gray::view_t const& src, gray::view_t const& dst);

	void convert_grayscale(view_t const& src, gray::view_t const& dst);

	void convert_alpha(view_t const& src, pixel_to_u8_f const& func);

	void convert_alpha_grayscale(view_t const& src);

	void adjust_contrast(gray::view_t const& src, gray::view_t const& dst, u8 src_low, u8 src_high);

	void adjust_contrast(gray::view_t const& src, u8 src_low, u8 src_high);

	void binarize(gray::view_t const& src, gray::view_t const& dst, u8_to_bool_f const& func);

	void binarize(gray::view_t const& src, u8_to_bool_f const& func);

	void alpha_blend(view_t const& src, view_t const& current, view_t const& dst);

	void alpha_blend(view_t const& src, view_t const& current_dst);

	void blur(gray::view_t const& src, gray::view_t const& dst);


	namespace par
	{
		void convert(view_t const& src, gray::view_t const& dst, pixel_to_u8_f const& func);

		void convert(gray::view_t const& src, gray::view_t const& dst, u8_to_u8_f const& func);

		void convert(gray::view_t const& src, u8_to_u8_f const& func);

		void copy(view_t const& src, view_t const& dst);

		void copy(gray::view_t const& src, gray::view_t const& dst);

		void convert_grayscale(view_t const& src, gray::view_t const& dst);

		void convert_alpha(view_t const& src, pixel_to_u8_f const& func);

		void convert_alpha_grayscale(view_t const& src);

		void adjust_contrast(gray::view_t const& src, gray::view_t const& dst, u8 src_low, u8 src_high);

		void adjust_contrast(gray::view_t const& src, u8 src_low, u8 src_high);

		void binarize(gray::view_t const& src, gray::view_t const& dst, u8_to_bool_f const& func);

		void binarize(gray::view_t const& src, u8_to_bool_f const& func);

	}
}


