#pragma once

#include "proc_def.hpp"


namespace libimage
{
#ifndef LIBIMAGE_NO_COLOR

	void rotate(image_t const& src, image_t const& dst, u32 origin_x, u32 origin_y, r32 theta);

	void rotate(image_t const& src, view_t const& dst, u32 origin_x, u32 origin_y, r32 theta);

	void rotate(view_t const& src, image_t const& dst, u32 origin_x, u32 origin_y, r32 theta);

	void rotate(view_t const& src, view_t const& dst, u32 origin_x, u32 origin_y, r32 theta);

#endif // !LIBIMAGE_NO_COLOR




	namespace seq
	{
#ifndef LIBIMAGE_NO_COLOR

		void rotate(image_t const& src, image_t const& dst, u32 origin_x, u32 origin_y, r32 theta);

		void rotate(image_t const& src, view_t const& dst, u32 origin_x, u32 origin_y, r32 theta);

		void rotate(view_t const& src, image_t const& dst, u32 origin_x, u32 origin_y, r32 theta);

		void rotate(view_t const& src, view_t const& dst, u32 origin_x, u32 origin_y, r32 theta);

#endif // !LIBIMAGE_NO_COLOR
	}
}