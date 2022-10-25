#include "../libimage.hpp"
#include "stb_include.hpp"

#include <algorithm>
#include <cstring>


static bool has_extension(const char* filename, const char* ext)
{
	size_t file_length = std::strlen(filename);
	size_t ext_length = std::strlen(ext);

	return !std::strcmp(&filename[file_length - ext_length], ext);
}


static bool is_bmp(const char* filename)
{
	return has_extension(filename, ".bmp") || has_extension(filename, ".BMP");
}


static bool is_png(const char* filename)
{
	return has_extension(filename, ".png") || has_extension(filename, ".PNG");
}


static bool is_jpg(const char* filename)
{
	return has_extension(filename, ".jpg") || 
		has_extension(filename, ".jpeg") || 
		has_extension(filename, ".JPG") || 
		has_extension(filename, ".JPEG");
}


namespace libimage
{

#ifndef LIBIMAGE_NO_COLOR

	void read_image_from_file(const char* img_path_src, Image& image_dst)
	{
		int width = 0;
		int height = 0;
		int image_channels = 0;
		int desired_channels = 4;

		auto data = (Pixel*)stbi_load(img_path_src, &width, &height, &image_channels, desired_channels);

		assert(data);
		assert(width);
		assert(height);

		image_dst.data = data;
		image_dst.width = width;
		image_dst.height = height;
	}

#ifndef LIBIMAGE_NO_WRITE

	void write_image(Image const& image_src, const char* file_path_dst)
	{
		assert(image_src.width);
		assert(image_src.height);
		assert(image_src.data);

		int width = static_cast<int>(image_src.width);
		int height = static_cast<int>(image_src.height);
		int channels = static_cast<int>(RGBA_CHANNELS);
		auto const data = image_src.data;

		int result = 0;

		if(is_bmp(file_path_dst))
		{
			result = stbi_write_bmp(file_path_dst, width, height, channels, data);
		}
		else if(is_png(file_path_dst))
		{
			int stride_in_bytes = width * channels;

			result = stbi_write_png(file_path_dst, width, height, channels, data, stride_in_bytes);
		}
		else if(is_jpg(file_path_dst))
		{
			// TODO: quality?
			// stbi_write_jpg(char const *filename, int w, int h, int comp, const void *data, int quality);
		}

		assert(result);
	}


	/*static void make_image(View const& view_src, Image& image_dst)
	{
		make_image(image_dst, view_src.width, view_src.height);

		copy(view_src, image_dst);
	}


	void write_view(View const& view_src, const char* file_path_dst)
	{
		Image image;
		make_image(view_src, image);

		write_image(image, file_path_dst);

		destroy_image(image);
	}*/

#endif // !LIBIMAGE_NO_WRITE


#ifndef LIBIMAGE_NO_RESIZE

	void resize_image(Image const& image_src, Image& image_dst)
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

		if (!image_dst.data)
		{
			image_dst.data = (Pixel*)malloc(sizeof(Pixel) * image_dst.width * image_dst.height);
		}		

		result = stbir_resize_uint8(
			(u8*)image_src.data, width_src, height_src, stride_bytes_src,
			(u8*)image_dst.data, width_dst, height_dst, stride_bytes_dst,
			channels);

		assert(result);
	}


	View make_resized_view(Image const& img_src, Image& img_dst)
	{
		resize_image(img_src, img_dst);

		return make_view(img_dst);
	}

#endif // !LIBIMAGE_NO_RESIZE

#endif // !LIBIMAGE_NO_COLOR
	
#ifndef LIBIMAGE_NO_GRAYSCALE

	void read_image_from_file(const char* file_path_src, gray::Image& image_dst)
	{
		int width = 0;
		int height = 0;
		int image_channels = 0;
		int desired_channels = 1;

		auto data = (gray::Pixel*)stbi_load(file_path_src, &width, &height, &image_channels, desired_channels);

		assert(data);
		assert(width);
		assert(height);

		image_dst.data = data;
		image_dst.width = width;
		image_dst.height = height;
	}

#ifndef LIBIMAGE_NO_WRITE

	void write_image(gray::Image const& image_src, const char* file_path_dst)
	{
		assert(image_src.width);
		assert(image_src.height);
		assert(image_src.data);

		int width = static_cast<int>(image_src.width);
		int height = static_cast<int>(image_src.height);
		int channels = 1;
		auto const data = image_src.data;

		int result = 0;
		
		if(is_bmp(file_path_dst))
		{
			result = stbi_write_bmp(file_path_dst, width, height, channels, data);
		}
		else if(is_png(file_path_dst))
		{
			int stride_in_bytes = width * channels;

			result = stbi_write_png(file_path_dst, width, height, channels, data, stride_in_bytes);
		}
		else if(is_jpg(file_path_dst))
		{
			// TODO: quality?
			// stbi_write_jpg(char const *filename, int w, int h, int comp, const void *data, int quality);
		}

		assert(result);
	}


	/*static void make_image(gray::View const& view_src, gray::Image& image_dst)
	{
		make_image(image_dst, view_src.width, view_src.height);

		copy(view_src, image_dst);
	}


	void write_view(gray::View const& view_src, const char* file_path_dst)
	{
		gray::Image image;
		make_image(view_src, image);

		write_image(image, file_path_dst);

		destroy_image(image);
	}*/

#endif // !LIBIMAGE_NO_WRITE


#ifndef LIBIMAGE_NO_RESIZE

	void resize_image(gray::Image const& image_src, gray::Image& image_dst)
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

		if (!image_dst.data)
		{
			image_dst.data = (gray::Pixel*)malloc(sizeof(gray::Pixel) * image_dst.width * image_dst.height);
		}

		result = stbir_resize_uint8(
			(u8*)image_src.data, width_src, height_src, stride_bytes_src,
			(u8*)image_dst.data, width_dst, height_dst, stride_bytes_dst,
			channels);

		assert(result);
	}


	gray::View make_resized_view(gray::Image const& image_src, gray::Image& image_dst)
	{
		resize_image(image_src, image_dst);

		return make_view(image_dst);
	}

#endif // !LIBIMAGE_NO_RESIZE

#endif // !LIBIMAGE_NO_GRAYSCALE

}

