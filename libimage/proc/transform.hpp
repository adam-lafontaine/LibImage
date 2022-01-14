#pragma once

#include "proc_def.hpp"

#include <array>


namespace libimage
{


	/*** transform parallel ***/

#ifndef LIBIMAGE_NO_PARALLEL

#ifndef LIBIMAGE_NO_COLOR	

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

#endif // !LIBIMAGE_NO_COLOR

#ifndef LIBIMAGE_NO_GRAYSCALE

	lookup_table_t to_lookup_table(u8_to_u8_f const& func);

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


	/*** transform sequential ***/

	namespace seq
	{
#ifndef LIBIMAGE_NO_COLOR

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


#endif // !LIBIMAGE_NO_COLOR

#ifndef LIBIMAGE_NO_GRAYSCALE

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



	/*** transform simd **/

#ifndef LIBIMAGE_NO_SIMD

	namespace simd
	{
#ifndef LIBIMAGE_NO_COLOR
#ifndef LIBIMAGE_NO_GRAYSCALE

		void grayscale(image_t const& src, gray::image_t const& dst);

		void grayscale(image_t const& src, gray::view_t const& dst);

		void grayscale(view_t const& src, gray::image_t const& dst);

		void grayscale(view_t const& src, gray::view_t const& dst);

#endif // !LIBIMAGE_NO_GRAYSCALE
#endif // !LIBIMAGE_NO_COLOR
	}

#endif // !LIBIMAGE_NO_SIMD
}