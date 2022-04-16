#pragma once

#ifndef LIBIMAGE_NO_GRAYSCALE
#ifndef LIBIMAGE_NO_COLOR

#include "proc_def.hpp"


namespace libimage
{
	void grayscale(image_soa const& src, gray::image_t const& dst);

	void grayscale(image_soa const& src, gray::view_t const& dst);


#ifndef LIBIMAGE_NO_PARALLEL

	void grayscale(image_t const& src, gray::image_t const& dst);

	void grayscale(image_t const& src, gray::view_t const& dst);

	void grayscale(view_t const& src, gray::image_t const& dst);

	void grayscale(view_t const& src, gray::view_t const& dst);


	void alpha_grayscale(image_t const& src);

	void alpha_grayscale(view_t const& src);

#endif // !LIBIMAGE_NO_PARALLEL

	namespace seq
	{
		void grayscale(image_t const& src, gray::image_t const& dst);

		void grayscale(image_t const& src, gray::view_t const& dst);

		void grayscale(view_t const& src, gray::image_t const& dst);

		void grayscale(view_t const& src, gray::view_t const& dst);

		void alpha_grayscale(image_t const& src);

		void alpha_grayscale(view_t const& src);
	}


#ifndef LIBIMAGE_NO_SIMD

	namespace simd
	{
		void grayscale(image_t const& src, gray::image_t const& dst);

		void grayscale(image_t const& src, gray::view_t const& dst);

		void grayscale(view_t const& src, gray::image_t const& dst);

		void grayscale(view_t const& src, gray::view_t const& dst);


		void grayscale(image_soa const& src, gray::image_t const& dst);

		void grayscale(image_soa const& src, gray::view_t const& dst);


		void alpha_grayscale(image_t const& src);

		void alpha_grayscale(view_t const& src);


	}

#endif // !LIBIMAGE_NO_SIMD
}

#endif // !LIBIMAGE_NO_COLOR
#endif // !LIBIMAGE_NO_GRAYSCALE