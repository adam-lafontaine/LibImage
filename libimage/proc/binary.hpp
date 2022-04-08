#pragma once

#include "proc_def.hpp"


namespace libimage
{
#ifndef LIBIMAGE_NO_PARALLEL

#ifndef LIBIMAGE_NO_GRAYSCALE

	void binarize(gray::image_t const& src, gray::image_t const& dst, u8_to_bool_f const& cond);

	void binarize(gray::image_t const& src, gray::view_t const& dst, u8_to_bool_f const& cond);

	void binarize(gray::view_t const& src, gray::image_t const& dst, u8_to_bool_f const& cond);

	void binarize(gray::view_t const& src, gray::view_t const& dst, u8_to_bool_f const& cond);


	void binarize_self(gray::image_t const& src_dst, u8_to_bool_f const& func);

	void binarize_self(gray::view_t const& src_dst, u8_to_bool_f const& func);


#ifndef LIBIMAGE_NO_COLOR

	void binarize(image_t const& src, gray::image_t const& dst, pixel_to_bool_f const& cond);

	void binarize(image_t const& src, gray::view_t const& dst, pixel_to_bool_f const& cond);

	void binarize(view_t const& src, gray::image_t const& dst, pixel_to_bool_f const& cond);

	void binarize(view_t const& src, gray::view_t const& dst, pixel_to_bool_f const& cond);

#endif // !LIBIMAGE_NO_COLOR


	Point2Du32 centroid(gray::image_t const& src);

	Point2Du32 centroid(gray::image_t const& src, u8_to_bool_f const& func);


	Point2Du32 centroid(gray::view_t const& src);

	Point2Du32 centroid(gray::view_t const& src, u8_to_bool_f const& func);


#endif // !LIBIMAGE_NO_GRAYSCALE

#endif // !LIBIMAGE_NO_PARALLEL


	namespace seq
	{
#ifndef LIBIMAGE_NO_GRAYSCALE

		void binarize(gray::image_t const& src, gray::image_t const& dst, u8_to_bool_f const& cond);

		void binarize(gray::image_t const& src, gray::view_t const& dst, u8_to_bool_f const& cond);

		void binarize(gray::view_t const& src, gray::image_t const& dst, u8_to_bool_f const& cond);

		void binarize(gray::view_t const& src, gray::view_t const& dst, u8_to_bool_f const& cond);


		void binarize_self(gray::image_t const& src_dst, u8_to_bool_f const& cond);

		void binarize_self(gray::view_t const& src_dst, u8_to_bool_f const& cond);


#ifndef LIBIMAGE_NO_COLOR

		void binarize(image_t const& src, gray::image_t const& dst, pixel_to_bool_f const& cond);

		void binarize(image_t const& src, gray::view_t const& dst, pixel_to_bool_f const& cond);

		void binarize(view_t const& src, gray::image_t const& dst, pixel_to_bool_f const& cond);

		void binarize(view_t const& src, gray::view_t const& dst, pixel_to_bool_f const& cond);


#endif // !LIBIMAGE_NO_COLOR


		Point2Du32 centroid(gray::image_t const& src);

		Point2Du32 centroid(gray::image_t const& src, u8_to_bool_f const& func);


		Point2Du32 centroid(gray::view_t const& src);

		Point2Du32 centroid(gray::view_t const& src, u8_to_bool_f const& func);


		void thin_objects(gray::image_t const& src, gray::image_t const& dst, gray::image_t const& temp);

		void thin_objects(gray::image_t const& src, gray::view_t const& dst, gray::image_t const& temp);

		void thin_objects(gray::view_t const& src, gray::image_t const& dst, gray::image_t const& temp);

		void thin_objects(gray::view_t const& src, gray::view_t const& dst, gray::image_t const& temp);


#endif // !LIBIMAGE_NO_GRAYSCALE
	}

}