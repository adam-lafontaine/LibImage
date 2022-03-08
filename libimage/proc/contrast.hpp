#pragma once

#include "proc_def.hpp"


namespace libimage
{
#ifndef LIBIMAGE_NO_PARALLEL

#ifndef LIBIMAGE_NO_GRAYSCALE

	void contrast(gray::image_t const& src, gray::image_t const& dst, u8 src_low, u8 src_high);

	void contrast(gray::image_t const& src, gray::view_t const& dst, u8 src_low, u8 src_high);

	void contrast(gray::view_t const& src, gray::image_t const& dst, u8 src_low, u8 src_high);

	void contrast(gray::view_t const& src, gray::view_t const& dst, u8 src_low, u8 src_high);


	void contrast_self(gray::image_t const& src_dst, u8 src_low, u8 src_high);

	void contrast_self(gray::view_t const& src_dst, u8 src_low, u8 src_high);

#endif // !LIBIMAGE_NO_GRAYSCALE

#endif // !LIBIMAGE_NO_PARALLEL


	namespace seq
	{
#ifndef LIBIMAGE_NO_GRAYSCALE

		void contrast(gray::image_t const& src, gray::image_t const& dst, u8 src_low, u8 src_high);

		void contrast(gray::image_t const& src, gray::view_t const& dst, u8 src_low, u8 src_high);

		void contrast(gray::view_t const& src, gray::image_t const& dst, u8 src_low, u8 src_high);

		void contrast(gray::view_t const& src, gray::view_t const& dst, u8 src_low, u8 src_high);


		void contrast_self(gray::image_t const& src_dst, u8 src_low, u8 src_high);

		void contrast_self(gray::view_t const& src_dst, u8 src_low, u8 src_high);

#endif // !LIBIMAGE_NO_GRAYSCALE

	}


}


