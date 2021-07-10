#pragma once
/*

Copyright (c) 2021 Adam Lafontaine

*/

#include "../libimage.hpp"

#include <functional>


namespace libimage
{
#ifndef LIBIMAGE_NO_COLOR

	using pixel_to_u8_f = std::function<u8(pixel_t const& p)>;
	

	void copy(view_t const& src, view_t const& dst);

	void transform_alpha(view_t const& src, pixel_to_u8_f const& func);

	void transform_alpha_grayscale(view_t const& src);

	void alpha_blend(view_t const& src, view_t const& current, view_t const& dst);

	void alpha_blend(view_t const& src, view_t const& current_dst);

#endif // !LIBIMAGE_NO_COLOR


#ifndef LIBIMAGE_NO_GRAYSCALE

	using u8_to_bool_f = std::function<bool(u8)>;

	using u8_to_u8_f = std::function<u8(u8)>;


	void transform(gray::view_t const& src, gray::view_t const& dst, u8_to_u8_f const& func);

	void transform(gray::view_t const& src, u8_to_u8_f const& func);

	void copy(gray::view_t const& src, gray::view_t const& dst);

	void transform_contrast(gray::view_t const& src, gray::view_t const& dst, u8 src_low, u8 src_high);

	void transform_contrast(gray::view_t const& src, u8 src_low, u8 src_high);

	void binarize(gray::view_t const& src, gray::view_t const& dst, u8 min_threashold);

	void binarize(gray::view_t const& src, u8 min_threashold);

	void binarize(gray::view_t const& src, gray::view_t const& dst, u8_to_bool_f const& func);

	void binarize(gray::view_t const& src, u8_to_bool_f const& func);

	void blur(gray::view_t const& src, gray::view_t const& dst);

	void edges(gray::view_t const& src, gray::view_t const& dst, u8 threshold);

	void gradient(gray::view_t const& src, gray::view_t const& dst);

#endif // !LIBIMAGE_NO_GRAYSCALE


#ifndef LIBIMAGE_NO_COLOR
#ifndef LIBIMAGE_NO_GRAYSCALE

	void transform(view_t const& src, gray::view_t const& dst, pixel_to_u8_f const& func);

	void transform_grayscale(view_t const& src, gray::view_t const& dst);

#endif // !LIBIMAGE_NO_GRAYSCALE
#endif // !LIBIMAGE_NO_COLOR	


	namespace seq
	{
#ifndef LIBIMAGE_NO_COLOR

		void copy(view_t const& src, view_t const& dst);

		void transform_alpha(view_t const& src, pixel_to_u8_f const& func);

		void transform_alpha_grayscale(view_t const& src);

		void alpha_blend(view_t const& src, view_t const& current, view_t const& dst);

		void alpha_blend(view_t const& src, view_t const& current_dst);

#endif // !LIBIMAGE_NO_COLOR


#ifndef LIBIMAGE_NO_GRAYSCALE

		void transform(gray::view_t const& src, gray::view_t const& dst, u8_to_u8_f const& func);

		void transform(gray::view_t const& src, u8_to_u8_f const& func);

		void copy(gray::view_t const& src, gray::view_t const& dst);

		void transform_contrast(gray::view_t const& src, gray::view_t const& dst, u8 src_low, u8 src_high);

		void transform_contrast(gray::view_t const& src, u8 src_low, u8 src_high);

		void binarize(gray::view_t const& src, gray::view_t const& dst, u8 min_threashold);

		void binarize(gray::view_t const& src, u8 min_threashold);

		void binarize(gray::view_t const& src, gray::view_t const& dst, u8_to_bool_f const& func);

		void binarize(gray::view_t const& src, u8_to_bool_f const& func);

		void blur(gray::view_t const& src, gray::view_t const& dst);

		void edges(gray::view_t const& src, gray::view_t const& dst, u8 threshold);

		void gradient(gray::view_t const& src, gray::view_t const& dst);

#endif // !LIBIMAGE_NO_GRAYSCALE


#ifndef LIBIMAGE_NO_COLOR
#ifndef LIBIMAGE_NO_GRAYSCALE

		void transform(view_t const& src, gray::view_t const& dst, pixel_to_u8_f const& func);

		void transform_grayscale(view_t const& src, gray::view_t const& dst);

#endif // !LIBIMAGE_NO_GRAYSCALE
#endif // !LIBIMAGE_NO_COLOR	

	}
}


