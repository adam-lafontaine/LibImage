#pragma once

#include "proc_def.hpp"


namespace libimage
{
#ifndef LIBIMAGE_NO_COLOR
#ifndef LIBIMAGE_NO_GRAYSCALE


	void grayscale(image_soa const& src, gray::image_t const& dst);

	void grayscale(image_soa const& src, gray::view_t const& dst);


#endif // !LIBIMAGE_NO_GRAYSCALE
#endif // !LIBIMAGE_NO_COLOR


#ifndef LIBIMAGE_NO_PARALLEL

#ifndef LIBIMAGE_NO_COLOR
#ifndef LIBIMAGE_NO_GRAYSCALE


	void grayscale(image_t const& src, gray::image_t const& dst);

	void grayscale(image_t const& src, gray::view_t const& dst);

	void grayscale(view_t const& src, gray::image_t const& dst);

	void grayscale(view_t const& src, gray::view_t const& dst);


#endif // !LIBIMAGE_NO_GRAYSCALE
#endif // !LIBIMAGE_NO_COLOR


#ifndef LIBIMAGE_NO_COLOR

	void alpha_grayscale(image_t const& src);

	void alpha_grayscale(view_t const& src);

#endif // !LIBIMAGE_NO_COLOR



#endif // !LIBIMAGE_NO_PARALLEL


	namespace seq
	{
#ifndef LIBIMAGE_NO_COLOR
#ifndef LIBIMAGE_NO_GRAYSCALE

		void grayscale(image_t const& src, gray::image_t const& dst);

		void grayscale(image_t const& src, gray::view_t const& dst);

		void grayscale(view_t const& src, gray::image_t const& dst);

		void grayscale(view_t const& src, gray::view_t const& dst);

#endif // !LIBIMAGE_NO_GRAYSCALE
#endif // !LIBIMAGE_NO_COLOR

#ifndef LIBIMAGE_NO_COLOR

		void alpha_grayscale(image_t const& src);

		void alpha_grayscale(view_t const& src);

#endif // !LIBIMAGE_NO_COLOR

	}


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

#ifndef LIBIMAGE_NO_COLOR

		void alpha_grayscale(image_t const& src);

		void alpha_grayscale(view_t const& src);

#endif // !LIBIMAGE_NO_COLOR
	}

#endif // !LIBIMAGE_NO_SIMD
}