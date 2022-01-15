#pragma once

#include "proc_def.hpp"


namespace libimage
{
	/*** copy parallel ***/

#ifndef LIBIMAGE_NO_PARALLEL

#ifndef LIBIMAGE_NO_COLOR	

	void copy(image_t const& src, image_t const& dst);

	void copy(image_t const& src, view_t const& dst);

	void copy(view_t const& src, image_t const& dst);

	void copy(view_t const& src, view_t const& dst);

#endif // !LIBIMAGE_NO_COLOR

#ifndef LIBIMAGE_NO_GRAYSCALE

	lookup_table_t to_lookup_table(u8_to_u8_f const& func);


	void copy(gray::image_t const& src, gray::image_t const& dst);

	void copy(gray::image_t const& src, gray::view_t const& dst);

	void copy(gray::view_t const& src, gray::image_t const& dst);

	void copy(gray::view_t const& src, gray::view_t const& dst);


#endif // !LIBIMAGE_NO_GRAYSCALE

#endif // !LIBIMAGE_NO_PARALLEL


	/*** copy sequential ***/

	namespace seq
	{
#ifndef LIBIMAGE_NO_COLOR	

		void copy(image_t const& src, image_t const& dst);

		void copy(image_t const& src, view_t const& dst);

		void copy(view_t const& src, image_t const& dst);

		void copy(view_t const& src, view_t const& dst);

#endif // !LIBIMAGE_NO_COLOR

#ifndef LIBIMAGE_NO_GRAYSCALE

		void copy(gray::image_t const& src, gray::image_t const& dst);

		void copy(gray::image_t const& src, gray::view_t const& dst);

		void copy(gray::view_t const& src, gray::image_t const& dst);

		void copy(gray::view_t const& src, gray::view_t const& dst);


#endif // !LIBIMAGE_NO_GRAYSCALE
	}
}
