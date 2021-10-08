#pragma once
/*

Copyright (c) 2021 Adam Lafontaine

*/

#include "../libimage.hpp"

namespace libimage
{
#ifndef LIBIMAGE_NO_COLOR


	inline bool verify(view_t const& view)
	{
		return view.image_data && view.width && view.height;
	}


	inline bool verify(image_t const& img)
	{
		return img.data && img.width && img.height;
	}


	inline bool verify(image_t const& src, image_t const& dst)
	{
		return verify(src) && verify(dst) && dst.width == src.width && dst.height == src.height;
	}


	inline bool verify(image_t const& src, view_t const& dst)
	{
		return verify(src) && verify(dst) && dst.width == src.width && dst.height == src.height;
	}


	inline bool verify(view_t const& src, image_t const& dst)
	{
		return verify(src) && verify(dst) && dst.width == src.width && dst.height == src.height;
	}


	inline bool verify(view_t const& src, view_t const& dst)
	{
		return verify(src) && verify(dst) && dst.width == src.width && dst.height == src.height;
	}


#endif // !LIBIMAGE_NO_COLOR


#ifndef LIBIMAGE_NO_GRAYSCALE

	inline bool verify(gray::image_t const& img)
	{
		return img.data && img.width && img.height;
	}


	inline bool verify(gray::view_t const& view)
	{
		return view.image_data && view.width && view.height;
	}


	inline bool verify(gray::image_t const& src, gray::image_t const& dst)
	{
		return verify(src) && verify(dst) && dst.width == src.width && dst.height == src.height;
	}


	inline bool verify(gray::image_t const& src, gray::view_t const& dst)
	{
		return verify(src) && verify(dst) && dst.width == src.width && dst.height == src.height;
	}


	inline bool verify(gray::view_t const& src, gray::image_t const& dst)
	{
		return verify(src) && verify(dst) && dst.width == src.width && dst.height == src.height;
	}


	inline bool verify(gray::view_t const& src, gray::view_t const& dst)
	{
		return verify(src) && verify(dst) && dst.width == src.width && dst.height == src.height;
	}


#endif // !LIBIMAGE_NO_GRAYSCALE


#ifndef LIBIMAGE_NO_COLOR
#ifndef LIBIMAGE_NO_GRAYSCALE

	inline bool verify(image_t const& src, gray::image_t const& dst)
	{
		return verify(src) && verify(dst) && dst.width == src.width && dst.height == src.height;
	}


	inline bool verify(image_t const& src, gray::view_t const& dst)
	{
		return verify(src) && verify(dst) && dst.width == src.width && dst.height == src.height;
	}


	inline bool verify(view_t const& src, gray::image_t const& dst)
	{
		return verify(src) && verify(dst) && dst.width == src.width && dst.height == src.height;
	}


	inline bool verify(view_t const& src, gray::view_t const& dst)
	{
		return verify(src) && verify(dst) && dst.width == src.width && dst.height == src.height;
	}

#endif // !LIBIMAGE_NO_GRAYSCALE
#endif // !LIBIMAGE_NO_COLOR
}

