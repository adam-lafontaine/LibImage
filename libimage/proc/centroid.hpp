#pragma once

#include "proc_def.hpp"


namespace libimage
{
#ifndef LIBIMAGE_NO_PARALLEL


#ifndef LIBIMAGE_NO_GRAYSCALE

	Point2Du32 centroid(gray::image_t const& src);

	Point2Du32 centroid(gray::image_t const& src, u8_to_bool_f const& func);


	Point2Du32 centroid(gray::view_t const& src);

	Point2Du32 centroid(gray::view_t const& src, u8_to_bool_f const& func);

#endif // !LIBIMAGE_NO_GRAYSCALE


#endif // !LIBIMAGE_NO_PARALLEL

	namespace seq
	{
#ifndef LIBIMAGE_NO_GRAYSCALE

		Point2Du32 centroid(gray::image_t const& src);

		Point2Du32 centroid(gray::image_t const& src, u8_to_bool_f const& func);


		Point2Du32 centroid(gray::view_t const& src);

		Point2Du32 centroid(gray::view_t const& src, u8_to_bool_f const& func);

#endif // !LIBIMAGE_NO_GRAYSCALE
	}
}