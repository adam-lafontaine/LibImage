#pragma once
/*

Copyright (c) 2021 Adam Lafontaine

*/

#include "../libimage.hpp"

#include <functional>
#include <array>


namespace libimage
{
	constexpr u32 VIEW_MIN_DIM = 5;


#ifndef LIBIMAGE_NO_COLOR

	using pixel_to_pixel_f = std::function<pixel_t(pixel_t const&)>;

	using pixel_to_u8_f = std::function<u8(pixel_t const& p)>;

#endif // !LIBIMAGE_NO_COLOR

#ifndef LIBIMAGE_NO_GRAYSCALE

	using u8_to_bool_f = std::function<bool(u8)>;

	using u8_to_u8_f = std::function<u8(u8)>;

	using lookup_table_t = std::array<u8, 256>;

#endif // !LIBIMAGE_NO_GRAYSCALE
}


namespace libimage
{	

#ifndef LIBIMAGE_NO_PARALLEL

#ifndef LIBIMAGE_NO_COLOR	

	void copy(image_t const& src, image_t const& dst);

	void copy(image_t const& src, view_t const& dst);

	void copy(view_t const& src, image_t const& dst);

	void copy(view_t const& src, view_t const& dst);


	void transform(image_t const& src, image_t const& dst, pixel_to_pixel_f const& func);

	void transform(image_t const& src, view_t const& dst, pixel_to_pixel_f const& func);

	void transform(view_t const& src, image_t const& dst, pixel_to_pixel_f const& func);

	void transform(view_t const& src, view_t const& dst, pixel_to_pixel_f const& func);


	void transform_self(image_t const& src_dst, pixel_to_pixel_f const& func);

	void transform_self(view_t const& src_dst, pixel_to_pixel_f const& func);


	void transform_alpha(image_t const& src_dst, pixel_to_u8_f const& func);

	void transform_alpha(view_t const& src_dst, pixel_to_u8_f const& func);


	void transform_alpha_grayscale(image_t const& src);

	void transform_alpha_grayscale(view_t const& src);


	void alpha_blend(image_t const& src, image_t const& current, image_t const& dst);

	void alpha_blend(image_t const& src, image_t const& current, view_t const& dst);

	void alpha_blend(image_t const& src, view_t const& current, image_t const& dst);

	void alpha_blend(image_t const& src, view_t const& current, view_t const& dst);

	void alpha_blend(view_t const& src, image_t const& current, image_t const& dst);

	void alpha_blend(view_t const& src, image_t const& current, view_t const& dst);

	void alpha_blend(view_t const& src, view_t const& current, image_t const& dst);

	void alpha_blend(view_t const& src, view_t const& current, view_t const& dst);


	void alpha_blend(image_t const& src, image_t const& current_dst);

	void alpha_blend(image_t const& src, view_t const& current_dst);

	void alpha_blend(view_t const& src, image_t const& current_dst);

	void alpha_blend(view_t const& src, view_t const& current_dst);

#endif // !LIBIMAGE_NO_COLOR


#ifndef LIBIMAGE_NO_GRAYSCALE

	lookup_table_t to_lookup_table(u8_to_u8_f const& func);


	void copy(gray::image_t const& src, gray::image_t const& dst);

	void copy(gray::image_t const& src, gray::view_t const& dst);

	void copy(gray::view_t const& src, gray::image_t const& dst);

	void copy(gray::view_t const& src, gray::view_t const& dst);


	void transform(gray::image_t const& src, gray::image_t const& dst, lookup_table_t const& lut);

	void transform(gray::image_t const& src, gray::view_t const& dst, lookup_table_t const& lut);

	void transform(gray::view_t const& src, gray::image_t const& dst, lookup_table_t const& lut);

	void transform(gray::view_t const& src, gray::view_t const& dst, lookup_table_t const& lut);


	void transform(gray::image_t const& src, gray::image_t const& dst, u8_to_u8_f const& func);

	void transform(gray::image_t const& src, gray::view_t const& dst, u8_to_u8_f const& func);

	void transform(gray::view_t const& src, gray::image_t const& dst, u8_to_u8_f const& func);

	void transform(gray::view_t const& src, gray::view_t const& dst, u8_to_u8_f const& func);


	void transform_self(gray::image_t const& src_dst, lookup_table_t const& lut);

	void transform_self(gray::view_t const& src_dst, lookup_table_t const& lut);


	void transform_self(gray::image_t const& src_dst, u8_to_u8_f const& func);

	void transform_self(gray::view_t const& src_dst, u8_to_u8_f const& func);


	void contrast(gray::image_t const& src, gray::image_t const& dst, u8 src_low, u8 src_high);

	void contrast(gray::image_t const& src, gray::view_t const& dst, u8 src_low, u8 src_high);

	void contrast(gray::view_t const& src, gray::image_t const& dst, u8 src_low, u8 src_high);

	void contrast(gray::view_t const& src, gray::view_t const& dst, u8 src_low, u8 src_high);


	void contrast_self(gray::image_t const& src_dst, u8 src_low, u8 src_high);

	void contrast_self(gray::view_t const& src_dst, u8 src_low, u8 src_high);


	void binarize(gray::image_t const& src, gray::image_t const& dst, u8_to_bool_f const& cond);

	void binarize(gray::image_t const& src, gray::view_t const& dst, u8_to_bool_f const& cond);

	void binarize(gray::view_t const& src, gray::image_t const& dst, u8_to_bool_f const& cond);

	void binarize(gray::view_t const& src, gray::view_t const& dst, u8_to_bool_f const& cond);


	void binarize_self(gray::image_t const& src_dst, u8_to_bool_f const& func);

	void binarize_self(gray::view_t const& src_dst, u8_to_bool_f const& func);


	void blur(gray::image_t const& src, gray::image_t const& dst);

	void blur(gray::image_t const& src, gray::view_t const& dst);

	void blur(gray::view_t const& src, gray::image_t const& dst);

	void blur(gray::view_t const& src, gray::view_t const& dst);


	void edges(gray::image_t const& src, gray::image_t const& dst, gray::image_t const& temp, u8_to_bool_f const& cond);

	void edges(gray::image_t const& src, gray::view_t const& dst, gray::image_t const& temp, u8_to_bool_f const& cond);

	void edges(gray::view_t const& src, gray::image_t const& dst, gray::image_t const& temp, u8_to_bool_f const& cond);

	void edges(gray::view_t const& src, gray::view_t const& dst, gray::image_t const& temp, u8_to_bool_f const& cond);


	void edges(gray::image_t const& src, gray::image_t const& dst, u8_to_bool_f const& cond);

	void edges(gray::image_t const& src, gray::view_t const& dst, u8_to_bool_f const& cond);

	void edges(gray::view_t const& src, gray::image_t const& dst, u8_to_bool_f const& cond);

	void edges(gray::view_t const& src, gray::view_t const& dst, u8_to_bool_f const& cond);


	void gradients(gray::image_t const& src, gray::image_t const& dst, gray::image_t const& temp);

	void gradients(gray::image_t const& src, gray::view_t const& dst, gray::image_t const& temp);

	void gradients(gray::view_t const& src, gray::image_t const& dst, gray::image_t const& temp);

	void gradients(gray::view_t const& src, gray::view_t const& dst, gray::image_t const& temp);


	void gradients(gray::image_t const& src, gray::image_t const& dst);

	void gradients(gray::image_t const& src, gray::view_t const& dst);

	void gradients(gray::view_t const& src, gray::image_t const& dst);

	void gradients(gray::view_t const& src, gray::view_t const& dst);

#endif // !LIBIMAGE_NO_GRAYSCALE


#ifndef LIBIMAGE_NO_COLOR
#ifndef LIBIMAGE_NO_GRAYSCALE

	void transform(image_t const& src, gray::image_t const& dst, pixel_to_u8_f const& func);

	void transform(image_t const& src, gray::view_t const& dst, pixel_to_u8_f const& func);

	void transform(view_t const& src, gray::image_t const& dst, pixel_to_u8_f const& func);

	void transform(view_t const& src, gray::view_t const& dst, pixel_to_u8_f const& func);


	void grayscale(image_t const& src, gray::image_t const& dst);

	void grayscale(image_t const& src, gray::view_t const& dst);

	void grayscale(view_t const& src, gray::image_t const& dst);

	void grayscale(view_t const& src, gray::view_t const& dst);

#endif // !LIBIMAGE_NO_GRAYSCALE
#endif // !LIBIMAGE_NO_COLOR

#endif // !LIBIMAGE_NO_PARALLEL

	namespace seq
	{
#ifndef LIBIMAGE_NO_COLOR

		void copy(image_t const& src, image_t const& dst);

		void copy(image_t const& src, view_t const& dst);

		void copy(view_t const& src, image_t const& dst);

		void copy(view_t const& src, view_t const& dst);


		void transform(image_t const& src, image_t const& dst, pixel_to_pixel_f const& func);

		void transform(image_t const& src, view_t const& dst, pixel_to_pixel_f const& func);

		void transform(view_t const& src, image_t const& dst, pixel_to_pixel_f const& func);

		void transform(view_t const& src, view_t const& dst, pixel_to_pixel_f const& func);


		void transform_self(image_t const& src_dst, pixel_to_pixel_f const& func);

		void transform_self(view_t const& src_dst, pixel_to_pixel_f const& func);


		void transform_alpha(image_t const& src_dst, pixel_to_u8_f const& func);

		void transform_alpha(view_t const& src_dst, pixel_to_u8_f const& func);


		void transform_alpha_grayscale(image_t const& src);

		void transform_alpha_grayscale(view_t const& src);


		void alpha_blend(image_t const& src, image_t const& current, image_t const& dst);

		void alpha_blend(image_t const& src, image_t const& current, view_t const& dst);

		void alpha_blend(image_t const& src, view_t const& current, image_t const& dst);

		void alpha_blend(image_t const& src, view_t const& current, view_t const& dst);

		void alpha_blend(view_t const& src, image_t const& current, image_t const& dst);

		void alpha_blend(view_t const& src, image_t const& current, view_t const& dst);

		void alpha_blend(view_t const& src, view_t const& current, image_t const& dst);

		void alpha_blend(view_t const& src, view_t const& current, view_t const& dst);


		void alpha_blend(image_t const& src, image_t const& current_dst);

		void alpha_blend(image_t const& src, view_t const& current_dst);

		void alpha_blend(view_t const& src, image_t const& current_dst);

		void alpha_blend(view_t const& src, view_t const& current_dst);

#endif // !LIBIMAGE_NO_COLOR


#ifndef LIBIMAGE_NO_GRAYSCALE

		lookup_table_t to_lookup_table(u8_to_u8_f const& func);


		void copy(gray::image_t const& src, gray::image_t const& dst);

		void copy(gray::image_t const& src, gray::view_t const& dst);

		void copy(gray::view_t const& src, gray::image_t const& dst);

		void copy(gray::view_t const& src, gray::view_t const& dst);


		void transform(gray::image_t const& src, gray::image_t const& dst, lookup_table_t const& lut);

		void transform(gray::image_t const& src, gray::view_t const& dst, lookup_table_t const& lut);

		void transform(gray::view_t const& src, gray::image_t const& dst, lookup_table_t const& lut);

		void transform(gray::view_t const& src, gray::view_t const& dst, lookup_table_t const& lut);


		void transform(gray::image_t const& src, gray::image_t const& dst, u8_to_u8_f const& func);

		void transform(gray::image_t const& src, gray::view_t const& dst, u8_to_u8_f const& func);

		void transform(gray::view_t const& src, gray::image_t const& dst, u8_to_u8_f const& func);

		void transform(gray::view_t const& src, gray::view_t const& dst, u8_to_u8_f const& func);


		void transform_self(gray::image_t const& src_dst, lookup_table_t const& lut);

		void transform_self(gray::view_t const& src_dst, lookup_table_t const& lut);


		void transform_self(gray::image_t const& src_dst, u8_to_u8_f const& func);

		void transform_self(gray::view_t const& src_dst, u8_to_u8_f const& func);


		void contrast(gray::image_t const& src, gray::image_t const& dst, u8 src_low, u8 src_high);

		void contrast(gray::image_t const& src, gray::view_t const& dst, u8 src_low, u8 src_high);

		void contrast(gray::view_t const& src, gray::image_t const& dst, u8 src_low, u8 src_high);

		void contrast(gray::view_t const& src, gray::view_t const& dst, u8 src_low, u8 src_high);


		void contrast_self(gray::image_t const& src_dst, u8 src_low, u8 src_high);

		void contrast_self(gray::view_t const& src_dst, u8 src_low, u8 src_high);


		void binarize(gray::image_t const& src, gray::image_t const& dst, u8_to_bool_f const& cond);

		void binarize(gray::image_t const& src, gray::view_t const& dst, u8_to_bool_f const& cond);

		void binarize(gray::view_t const& src, gray::image_t const& dst, u8_to_bool_f const& cond);

		void binarize(gray::view_t const& src, gray::view_t const& dst, u8_to_bool_f const& cond);


		void binarize_self(gray::image_t const& src_dst, u8_to_bool_f const& cond);

		void binarize_self(gray::view_t const& src_dst, u8_to_bool_f const& cond);


		void blur(gray::image_t const& src, gray::image_t const& dst);

		void blur(gray::image_t const& src, gray::view_t const& dst);

		void blur(gray::view_t const& src, gray::image_t const& dst);

		void blur(gray::view_t const& src, gray::view_t const& dst);


		void edges(gray::image_t const& src, gray::image_t const& dst, u8_to_bool_f const& cond);

		void edges(gray::image_t const& src, gray::view_t const& dst, u8_to_bool_f const& cond);

		void edges(gray::view_t const& src, gray::image_t const& dst, u8_to_bool_f const& cond);

		void edges(gray::view_t const& src, gray::view_t const& dst, u8_to_bool_f const& cond);


		void edges(gray::image_t const& src, gray::image_t const& dst, gray::image_t const& temp, u8_to_bool_f const& cond);

		void edges(gray::image_t const& src, gray::view_t const& dst, gray::image_t const& temp, u8_to_bool_f const& cond);

		void edges(gray::view_t const& src, gray::image_t const& dst, gray::image_t const& temp, u8_to_bool_f const& cond);

		void edges(gray::view_t const& src, gray::view_t const& dst, gray::image_t const& temp, u8_to_bool_f const& cond);


		void gradients(gray::image_t const& src, gray::image_t const& dst);

		void gradients(gray::image_t const& src, gray::view_t const& dst);

		void gradients(gray::view_t const& src, gray::image_t const& dst);

		void gradients(gray::view_t const& src, gray::view_t const& dst);


		void gradients(gray::image_t const& src, gray::image_t const& dst, gray::image_t const& temp);

		void gradients(gray::image_t const& src, gray::view_t const& dst, gray::image_t const& temp);

		void gradients(gray::view_t const& src, gray::image_t const& dst, gray::image_t const& temp);

		void gradients(gray::view_t const& src, gray::view_t const& dst, gray::image_t const& temp);

#endif // !LIBIMAGE_NO_GRAYSCALE


#ifndef LIBIMAGE_NO_COLOR
#ifndef LIBIMAGE_NO_GRAYSCALE

		void transform(image_t const& src, gray::image_t const& dst, pixel_to_u8_f const& func);

		void transform(image_t const& src, gray::view_t const& dst, pixel_to_u8_f const& func);

		void transform(view_t const& src, gray::image_t const& dst, pixel_to_u8_f const& func);

		void transform(view_t const& src, gray::view_t const& dst, pixel_to_u8_f const& func);


		void grayscale(image_t const& src, gray::image_t const& dst);

		void grayscale(image_t const& src, gray::view_t const& dst);

		void grayscale(view_t const& src, gray::image_t const& dst);

		void grayscale(view_t const& src, gray::view_t const& dst);

#endif // !LIBIMAGE_NO_GRAYSCALE
#endif // !LIBIMAGE_NO_COLOR	

	}


	namespace simd
	{
#ifndef LIBIMAGE_NO_GRAYSCALE


		void blur(gray::image_t const& src, gray::image_t const& dst);

		void blur(gray::image_t const& src, gray::view_t const& dst);

		void blur(gray::view_t const& src, gray::image_t const& dst);

		void blur(gray::view_t const& src, gray::view_t const& dst);


		void edges(gray::image_t const& src, gray::image_t const& dst, gray::image_t const& temp, u8_to_bool_f const& cond);

		void edges(gray::image_t const& src, gray::view_t const& dst, gray::image_t const& temp, u8_to_bool_f const& cond);

		void edges(gray::view_t const& src, gray::image_t const& dst, gray::image_t const& temp, u8_to_bool_f const& cond);

		void edges(gray::view_t const& src, gray::view_t const& dst, gray::image_t const& temp, u8_to_bool_f const& cond);

		void edges(gray::image_t const& src, gray::image_t const& dst, u8_to_bool_f const& cond);

		void edges(gray::image_t const& src, gray::view_t const& dst, u8_to_bool_f const& cond);

		void edges(gray::view_t const& src, gray::image_t const& dst, u8_to_bool_f const& cond);

		void edges(gray::view_t const& src, gray::view_t const& dst, u8_to_bool_f const& cond);


		void gradients(gray::image_t const& src, gray::image_t const& dst, gray::image_t const& temp);

		void gradients(gray::image_t const& src, gray::view_t const& dst, gray::image_t const& temp);

		void gradients(gray::view_t const& src, gray::image_t const& dst, gray::image_t const& temp);

		void gradients(gray::view_t const& src, gray::view_t const& dst, gray::image_t const& temp);

		void gradients(gray::image_t const& src, gray::image_t const& dst);

		void gradients(gray::image_t const& src, gray::view_t const& dst);

		void gradients(gray::view_t const& src, gray::image_t const& dst);

		void gradients(gray::view_t const& src, gray::view_t const& dst);


#endif // !LIBIMAGE_NO_GRAYSCALE

#ifndef LIBIMAGE_NO_COLOR
#ifndef LIBIMAGE_NO_GRAYSCALE


		void grayscale(image_t const& src, gray::image_t const& dst);

		void grayscale(image_t const& src, gray::view_t const& dst);

		void grayscale(view_t const& src, gray::image_t const& dst);

		void grayscale(view_t const& src, gray::view_t const& dst);


#endif // !LIBIMAGE_NO_GRAYSCALE
#endif // !LIBIMAGE_NO_COLOR
	}
}


