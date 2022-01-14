#pragma once

#include "proc_def.hpp"


namespace libimage
{
#ifndef LIBIMAGE_NO_PARALLEL

#ifndef LIBIMAGE_NO_GRAYSCALE


	void gradients(gray::image_t const& src, gray::image_t const& dst);

	void gradients(gray::image_t const& src, gray::view_t const& dst);

	void gradients(gray::view_t const& src, gray::image_t const& dst);

	void gradients(gray::view_t const& src, gray::view_t const& dst);


	void gradients(gray::image_t const& src, gray::image_t const& dst, gray::image_t const& temp);

	void gradients(gray::image_t const& src, gray::view_t const& dst, gray::image_t const& temp);

	void gradients(gray::view_t const& src, gray::image_t const& dst, gray::image_t const& temp);

	void gradients(gray::view_t const& src, gray::view_t const& dst, gray::image_t const& temp);


#endif // !LIBIMAGE_NO_GRAYSCALE

#endif // !LIBIMAGE_NO_PARALLEL


	namespace seq
	{
#ifndef LIBIMAGE_NO_GRAYSCALE

		void gradients(gray::image_t const& src, gray::image_t const& dst);

		void gradients(gray::image_t const& src, gray::view_t const& dst);

		void gradients(gray::view_t const& src, gray::image_t const& dst);

		void gradients(gray::view_t const& src, gray::view_t const& dst);


		void gradients(gray::image_t const& src, gray::image_t const& dst, gray::image_t const& temp);

		void gradients(gray::image_t const& src, gray::view_t const& dst, gray::image_t const& temp);

		void gradients(gray::view_t const& src, gray::image_t const& dst, gray::image_t const& temp);

		void gradients(gray::view_t const& src, gray::view_t const& dst, gray::image_t const& temp);

#endif // !LIBIMAGE_NO_GRAYSCALE
	}


#ifndef LIBIMAGE_NO_SIMD

	namespace simd
	{
#ifndef LIBIMAGE_NO_GRAYSCALE

		void gradients(gray::image_t const& src, gray::image_t const& dst, gray::image_t const& temp);

		void gradients(gray::image_t const& src, gray::view_t const& dst, gray::image_t const& temp);

		void gradients(gray::view_t const& src, gray::image_t const& dst, gray::image_t const& temp);

		void gradients(gray::view_t const& src, gray::view_t const& dst, gray::image_t const& temp);


		void gradients(gray::image_t const& src, gray::image_t const& dst);

		void gradients(gray::image_t const& src, gray::view_t const& dst);

		void gradients(gray::view_t const& src, gray::image_t const& dst);

		void gradients(gray::view_t const& src, gray::view_t const& dst);

#endif // !LIBIMAGE_NO_GRAYSCALE
	}

#endif // !LIBIMAGE_NO_SIMD
}