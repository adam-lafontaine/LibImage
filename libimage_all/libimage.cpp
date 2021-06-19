/*

Copyright (c) 2021 Adam Lafontaine

*/

#include "libimage.hpp"
#include "stb_include.hpp"


namespace libimage
{

#ifndef LIBIMAGE_NO_COLOR

	void read_image_from_file(const char* img_path_src, image_t& image_dst)
	{
		int width = 0;
		int height = 0;
		int image_channels = 0;
		int desired_channels = 4;

		auto data = (rgba_pixel*)stbi_load(img_path_src, &width, &height, &image_channels, desired_channels);

		assert(data);
		assert(width);
		assert(height);

		image_dst.data = data;
		image_dst.width = width;
		image_dst.height = height;
	}


	void make_image(image_t& image_dst, u32 width, u32 height)
	{
		assert(width);
		assert(height);

		image_dst.width = width;
		image_dst.height = height;
		image_dst.data = (pixel_t*)malloc(sizeof(pixel_t) * width * height);

		assert(image_dst.data);
	}


	view_t make_view(image_t const& img)
	{
		assert(img.width);
		assert(img.height);
		assert(img.data);

		view_t view;

		view.image_data = img.data;
		view.image_width = img.width;
		view.x_begin = 0;
		view.y_begin = 0;
		view.x_end = img.width;
		view.y_end = img.height;
		view.width = img.width;
		view.height = img.height;

		return view;
	}


	view_t sub_view(image_t const& image, pixel_range_t const& range)
	{
		assert(image.width);
		assert(image.height);
		assert(image.data);

		view_t sub_view;

		sub_view.image_data = image.data;
		sub_view.image_width = image.width;
		sub_view.x_begin = range.x_begin;
		sub_view.y_begin = range.y_begin;
		sub_view.x_end = range.x_end;
		sub_view.y_end = range.y_end;
		sub_view.width = range.x_end - range.x_begin;
		sub_view.height = range.y_end - range.y_begin;

		assert(sub_view.width);
		assert(sub_view.height);

		return sub_view;
	}


	view_t sub_view(view_t const& view, pixel_range_t const& range)
	{
		assert(view.width);
		assert(view.height);
		assert(view.image_data);

		assert(range.x_begin >= view.x_begin);
		assert(range.x_end <= view.x_end);
		assert(range.y_begin >= view.y_begin);
		assert(range.y_end <= view.y_end);

		view_t sub_view;

		sub_view.image_data = view.image_data;
		sub_view.image_width = view.image_width;
		sub_view.x_begin = view.x_begin + range.x_begin;
		sub_view.y_begin = view.y_begin + range.y_begin;
		sub_view.x_end = view.x_begin + range.x_end;
		sub_view.y_end = view.y_begin + range.y_end;
		sub_view.width = range.x_end - range.x_begin;
		sub_view.height = range.y_end - range.y_begin;

		assert(sub_view.width);
		assert(sub_view.height);

		return sub_view;
	}


	view_t row_view(image_t const& image, u32 y)
	{
		pixel_range_t range;
		range.x_begin = 0;
		range.x_end = image.width;
		range.y_begin = y;
		range.y_end = y + 1;

		return sub_view(image, range);
	}


	view_t row_view(view_t const& view, u32 y)
	{
		pixel_range_t range;
		range.x_begin = 0;
		range.x_end = view.width;
		range.y_begin = y;
		range.y_end = y + 1;

		return sub_view(view, range);
	}


	view_t column_view(image_t const& image, u32 x)
	{
		pixel_range_t range;
		range.x_begin = x;
		range.x_end = x + 1;
		range.y_begin = 0;
		range.y_end = image.height;

		return sub_view(image, range);
	}


	view_t column_view(view_t const& view, u32 x)
	{
		pixel_range_t range;
		range.x_begin = x;
		range.x_end = x + 1;
		range.y_begin = 0;
		range.y_end = view.height;

		return sub_view(view, range);
	}


	view_t row_view(image_t const& image, u32 x_begin, u32 x_end, u32 y)
	{
		pixel_range_t range;
		range.x_begin = x_begin;
		range.x_end = x_end;
		range.y_begin = y;
		range.y_end = y + 1;

		return sub_view(image, range);
	}


	view_t row_view(view_t const& view, u32 x_begin, u32 x_end, u32 y)
	{
		pixel_range_t range;
		range.x_begin = x_begin;
		range.x_end = x_end;
		range.y_begin = y;
		range.y_end = y + 1;

		return sub_view(view, range);
	}


	view_t column_view(image_t const& image, u32 y_begin, u32 y_end, u32 x)
	{
		pixel_range_t range;
		range.x_begin = x;
		range.x_end = x + 1;
		range.y_begin = y_begin;
		range.y_end = y_end;

		return sub_view(image, range);
	}


	view_t column_view(view_t const& view, u32 y_begin, u32 y_end, u32 x)
	{
		pixel_range_t range;
		range.x_begin = x;
		range.x_end = x + 1;
		range.y_begin = y_begin;
		range.y_end = y_end;

		return sub_view(view, range);
	}



#ifndef LIBIMAGE_NO_WRITE

	void write_image(image_t const& image_src, const char* file_path_dst)
	{
		assert(image_src.width);
		assert(image_src.height);
		assert(image_src.data);

		int width = static_cast<int>(image_src.width);
		int height = static_cast<int>(image_src.height);
		int channels = static_cast<int>(RGBA_CHANNELS);
		auto const data = image_src.data;

		int result = 0;

		auto ext = fs::path(file_path_dst).extension();

		assert(ext == ".bmp" || ext == ".png");

		if (ext == ".bmp" || ext == ".BMP")
		{
			result = stbi_write_bmp(file_path_dst, width, height, channels, data);
		}
		else if (ext == ".png" || ext == ".PNG")
		{
			int stride_in_bytes = width * channels;

			result = stbi_write_png(file_path_dst, width, height, channels, data, stride_in_bytes);
		}
		else if (ext == ".jpg" || ext == ".jpeg" || ext == ".JPG" || ext == ".JPEG")
		{
			// TODO: quality?
			// stbi_write_jpg(char const *filename, int w, int h, int comp, const void *data, int quality);
		}

		assert(result);
	}


	static void make_image(view_t const& view, image_t& image_dst)
	{
		make_image(image_dst, view.width, view.height);

		std::transform(view.begin(), view.end(), image_dst.begin(), [&](auto p) { return p; });
	}


	void write_view(view_t const& view_src, const char* file_path_dst)
	{
		image_t image;
		make_image(view_src, image);

		write_image(image, file_path_dst);
	}

#endif // !LIBIMAGE_NO_WRITE


#ifndef LIBIMAGE_NO_RESIZE

	void resize_image(image_t const& image_src, image_t& image_dst)
	{
		assert(image_src.width);
		assert(image_src.height);
		assert(image_src.data);
		assert(image_dst.width);
		assert(image_dst.height);

		int channels = static_cast<int>(RGBA_CHANNELS);

		int width_src = static_cast<int>(image_src.width);
		int height_src = static_cast<int>(image_src.height);
		int stride_bytes_src = width_src * channels;

		int width_dst = static_cast<int>(image_dst.width);
		int height_dst = static_cast<int>(image_dst.height);
		int stride_bytes_dst = width_dst * channels;

		int result = 0;

		image_dst.data = (pixel_t*)malloc(sizeof(pixel_t) * image_dst.width * image_dst.height);

		result = stbir_resize_uint8(
			(u8*)image_src.data, width_src, height_src, stride_bytes_src,
			(u8*)image_dst.data, width_dst, height_dst, stride_bytes_dst,
			channels);

		assert(result);
	}


	view_t make_resized_view(image_t const& img_src, image_t& img_dst)
	{
		resize_image(img_src, img_dst);

		return make_view(img_dst);
	}

#endif // !LIBIMAGE_NO_RESIZE

#endif // !LIBIMAGE_NO_COLOR
	
#ifndef LIBIMAGE_NO_GRAYSCALE

	void read_image_from_file(const char* file_path_src, gray::image_t& image_dst)
	{
		int width = 0;
		int height = 0;
		int image_channels = 0;
		int desired_channels = 1;

		auto data = (gray::pixel_t*)stbi_load(file_path_src, &width, &height, &image_channels, desired_channels);

		assert(data);
		assert(width);
		assert(height);

		image_dst.data = data;
		image_dst.width = width;
		image_dst.height = height;
	}


	void make_image(gray::image_t& image_dst, u32 width, u32 height)
	{
		assert(width);
		assert(height);

		image_dst.width = width;
		image_dst.height = height;
		image_dst.data = (gray::pixel_t*)malloc(sizeof(gray::pixel_t) * width * height);

		assert(image_dst.data);
	}


	gray::view_t make_view(gray::image_t const& img)
	{
		assert(img.width);
		assert(img.height);
		assert(img.data);

		gray::view_t view;

		view.image_data = img.data;
		view.image_width = img.width;
		view.x_begin = 0;
		view.y_begin = 0;
		view.x_end = img.width;
		view.y_end = img.height;
		view.width = img.width;
		view.height = img.height;

		return view;
	}


	gray::view_t sub_view(gray::image_t const& image, pixel_range_t const& range)
	{
		assert(image.width);
		assert(image.height);
		assert(image.data);

		gray::view_t sub_view;

		sub_view.image_data = image.data;
		sub_view.image_width = image.width;
		sub_view.x_begin = range.x_begin;
		sub_view.y_begin = range.y_begin;
		sub_view.x_end = range.x_end;
		sub_view.y_end = range.y_end;
		sub_view.width = range.x_end - range.x_begin;
		sub_view.height = range.y_end - range.y_begin;

		assert(sub_view.width);
		assert(sub_view.height);

		return sub_view;
	}


	gray::view_t sub_view(gray::view_t const& view, pixel_range_t const& range)
	{
		assert(view.width);
		assert(view.height);
		assert(view.image_data);

		assert(range.x_begin >= view.x_begin);
		assert(range.x_end <= view.x_end);
		assert(range.y_begin >= view.y_begin);
		assert(range.y_end <= view.y_end);

		gray::view_t sub_view;

		sub_view.image_data = view.image_data;
		sub_view.image_width = view.image_width;
		sub_view.x_begin = view.x_begin + range.x_begin;
		sub_view.y_begin = view.y_begin + range.y_begin;
		sub_view.x_end = view.x_begin + range.x_end;
		sub_view.y_end = view.y_begin + range.y_end;
		sub_view.width = range.x_end - range.x_begin;
		sub_view.height = range.y_end - range.y_begin;

		assert(sub_view.width);
		assert(sub_view.height);

		return sub_view;
	}


	gray::view_t row_view(gray::image_t const& image, u32 y)
	{
		pixel_range_t range;
		range.x_begin = 0;
		range.x_end = image.width;
		range.y_begin = y;
		range.y_end = y + 1;

		return sub_view(image, range);
	}


	gray::view_t row_view(gray::view_t const& view, u32 y)
	{
		pixel_range_t range;
		range.x_begin = 0;
		range.x_end = view.width;
		range.y_begin = y;
		range.y_end = y + 1;

		return sub_view(view, range);
	}


	gray::view_t column_view(gray::image_t const& image, u32 x)
	{
		pixel_range_t range;
		range.x_begin = x;
		range.x_end = x + 1;
		range.y_begin = 0;
		range.y_end = image.height;

		return sub_view(image, range);
	}


	gray::view_t column_view(gray::view_t const& view, u32 x)
	{
		pixel_range_t range;
		range.x_begin = x;
		range.x_end = x + 1;
		range.y_begin = 0;
		range.y_end = view.height;

		return sub_view(view, range);
	}


	gray::view_t row_view(gray::image_t const& image, u32 x_begin, u32 x_end, u32 y)
	{
		pixel_range_t range;
		range.x_begin = x_begin;
		range.x_end = x_end;
		range.y_begin = y;
		range.y_end = y + 1;

		return sub_view(image, range);
	}


	gray::view_t row_view(gray::view_t const& view, u32 x_begin, u32 x_end, u32 y)
	{
		pixel_range_t range;
		range.x_begin = x_begin;
		range.x_end = x_end;
		range.y_begin = y;
		range.y_end = y + 1;

		return sub_view(view, range);
	}


	gray::view_t column_view(gray::image_t const& image, u32 y_begin, u32 y_end, u32 x)
	{
		pixel_range_t range;
		range.x_begin = x;
		range.x_end = x + 1;
		range.y_begin = y_begin;
		range.y_end = y_end;

		return sub_view(image, range);
	}


	gray::view_t column_view(gray::view_t const& view, u32 y_begin, u32 y_end, u32 x)
	{
		pixel_range_t range;
		range.x_begin = x;
		range.x_end = x + 1;
		range.y_begin = y_begin;
		range.y_end = y_end;

		return sub_view(view, range);
	}


#ifndef LIBIMAGE_NO_WRITE

	void write_image(gray::image_t const& image_src, const char* file_path_dst)
	{
		assert(image_src.width);
		assert(image_src.height);
		assert(image_src.data);

		int width = static_cast<int>(image_src.width);
		int height = static_cast<int>(image_src.height);
		int channels = 1;
		auto const data = image_src.data;

		int result = 0;

		auto ext = fs::path(file_path_dst).extension();

		assert(ext == ".bmp" || ext == ".png");

		if (ext == ".bmp" || ext == ".BMP")
		{
			result = stbi_write_bmp(file_path_dst, width, height, channels, data);
		}
		else if (ext == ".png" || ext == ".PNG")
		{
			int stride_in_bytes = width * channels;

			result = stbi_write_png(file_path_dst, width, height, channels, data, stride_in_bytes);
		}
		else if (ext == ".jpg" || ext == ".jpeg" || ext == ".JPG" || ext == ".JPEG")
		{
			// TODO: quality?
			// stbi_write_jpg(char const *filename, int w, int h, int comp, const void *data, int quality);
		}

		assert(result);
	}


	static void make_image(gray::view_t const& view_src, gray::image_t& image_dst)
	{
		make_image(image_dst, view_src.width, view_src.height);

		std::transform(view_src.begin(), view_src.end(), image_dst.begin(), [](auto p) { return p; });
	}


	void write_view(gray::view_t const& view_src, const char* file_path_dst)
	{
		gray::image_t image;
		make_image(view_src, image);

		write_image(image, file_path_dst);
	}

#endif // !LIBIMAGE_NO_WRITE


#ifndef LIBIMAGE_NO_RESIZE

	void resize_image(gray::image_t const& image_src, gray::image_t& image_dst)
	{
		assert(image_src.width);
		assert(image_src.height);
		assert(image_src.data);
		assert(image_dst.width);
		assert(image_dst.height);

		int channels = 1;

		int width_src = static_cast<int>(image_src.width);
		int height_src = static_cast<int>(image_src.height);
		int stride_bytes_src = width_src * channels;

		int width_dst = static_cast<int>(image_dst.width);
		int height_dst = static_cast<int>(image_dst.height);
		int stride_bytes_dst = width_dst * channels;

		int result = 0;

		image_dst.data = (gray::pixel_t*)malloc(sizeof(gray::pixel_t) * image_dst.width * image_dst.height);

		result = stbir_resize_uint8(
			(u8*)image_src.data, width_src, height_src, stride_bytes_src,
			(u8*)image_dst.data, width_dst, height_dst, stride_bytes_dst,
			channels);

		assert(result);
	}


	gray::view_t make_resized_view(gray::image_t const& image_src, gray::image_t& image_dst)
	{
		resize_image(image_src, image_dst);

		return make_view(image_dst);
	}

#endif // !LIBIMAGE_NO_RESIZE

#endif // !#ifndef LIBIMAGE_NO_GRAYSCALE

}

