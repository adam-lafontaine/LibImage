#pragma once
/*

Copyright (c) 2021 Adam Lafontaine

*/

#include "image_view.hpp"

#include <cstdint>
#include <functional>

//#define LIBIMAGE_NO_WRITE
//#define LIBIMAGE_NO_RESIZE

using u8 = uint8_t;
using u32 = uint32_t;
using u64 = uint64_t;


namespace libimage
{
	void read_image_from_file(const char* img_path_src, image_t& image_dst);

	void make_image(image_t& image_dst, u32 width, u32 height);

	view_t make_view(image_t const& image);

	view_t sub_view(image_t const& image, pixel_range_t const& range);

	view_t sub_view(view_t const& view, pixel_range_t const& range);

	view_t row_view(image_t const& image, u32 y);

	view_t row_view(view_t const& view, u32 y);

	view_t column_view(image_t const& image, u32 x);

	view_t column_view(view_t const& view, u32 x);

	view_t row_view(image_t const& image, u32 x_begin, u32 x_end, u32 y);

	view_t row_view(view_t const& view, u32 x_begin, u32 x_end, u32 y);

	view_t column_view(image_t const& image, u32 y_begin, u32 y_end, u32 x);

	view_t column_view(view_t const& view, u32 y_begin, u32 y_end, u32 x);

#ifndef LIBIMAGE_NO_WRITE

	void write_image(image_t const& image_src, const char* file_path_dst);

	void write_view(view_t const& view_src, const char* file_path_dst);

#endif // !LIBIMAGE_NO_WRITE


#ifndef LIBIMAGE_NO_RESIZE

	void resize_image(image_t const& image_src, image_t& image_dst);

	view_t make_resized_view(image_t const& image_src, image_t& image_dst);

#endif // !LIBIMAGE_NO_RESIZE
	

	//======= GRAYSCALE OVERLOADS ================

	void read_image_from_file(const char* file_path_src, gray::image_t& image_dst);

	void make_image(gray::image_t& image_dst, u32 width, u32 height);

	gray::view_t make_view(gray::image_t const& image);

	gray::view_t sub_view(gray::image_t const& image, pixel_range_t const& range);

	gray::view_t sub_view(gray::view_t const& view, pixel_range_t const& range);

	gray::view_t row_view(gray::image_t const& image, u32 y);

	gray::view_t row_view(gray::view_t const& view, u32 y);

	gray::view_t column_view(gray::image_t const& image, u32 y);

	gray::view_t column_view(gray::view_t const& view, u32 y);

	gray::view_t row_view(gray::image_t const& image, u32 x_begin, u32 x_end, u32 y);

	gray::view_t row_view(gray::view_t const& view, u32 x_begin, u32 x_end, u32 y);

	gray::view_t column_view(gray::image_t const& image, u32 y_begin, u32 y_end, u32 x);

	gray::view_t column_view(gray::view_t const& view, u32 y_begin, u32 y_end, u32 x);

#ifndef LIBIMAGE_NO_WRITE

	void write_image(gray::image_t const& image_src, const char* file_path_dst);

	void write_view(gray::view_t const& view_src, const char* file_path_dst);

#endif // !LIBIMAGE_NO_WRITE

#ifndef LIBIMAGE_NO_RESIZE

	void resize_image(gray::image_t const& img_src, gray::image_t& img_dst);

	gray::view_t make_resized_view(gray::image_t const& image_src, gray::image_t& image_dst);

#endif // !LIBIMAGE_NO_RESIZE
}