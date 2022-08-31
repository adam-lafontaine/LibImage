#pragma once

#include "defines.hpp"


namespace libimage
{
	constexpr auto RGB_CHANNELS = 3u;
	constexpr auto RGBA_CHANNELS = 4u;


	class PlatformPixel
	{
	public:

		u8 red = 0;
		u8 green = 0;
		u8 blue = 0;
		u8 alpha = 0;
	};


	inline PlatformPixel to_pixel(u8 r, u8 g, u8 b, u8 a)
	{
		PlatformPixel p{};
		p.red = r;
		p.green = g;
		p.blue = b;
		p.alpha = a;

		return p;
	}


	inline PlatformPixel to_pixel(u8 r, u8 g, u8 b)
	{
		return to_pixel(r, g, b, 255);
	}


	class PlatformImage
	{
	public:

		u32 width = 0;
		u32 height = 0;

		PlatformPixel* data = nullptr;
	};


	void make_image(PlatformImage& image, u32 width, u32 height);

	void destroy_image(PlatformImage& image);

	void read_image_from_file(const char* img_path_src, PlatformImage& image_dst);	


	class ImageRGBAr32
	{
	public:

		u32 width = 0;
		u32 height = 0;

		r32* red = nullptr;
		r32* green = nullptr;
		r32* blue = nullptr;
		r32* alpha = nullptr;
	};


	void make_image(ImageRGBAr32& image, u32 width, u32 height);

	void destroy_image(ImageRGBAr32& image);	

	void transform(ImageRGBAr32 const& src, PlatformImage const& dst);

	void transform(PlatformImage const& src, ImageRGBAr32 const& dst);


	class ImageRGBr32
	{
	public:

		u32 width = 0;
		u32 height = 0;

		r32* red = nullptr;
		r32* green = nullptr;
		r32* blue = nullptr;
	};


	void make_image(ImageRGBr32& image, u32 width, u32 height);

	void destroy_image(ImageRGBr32& image);

	void transform(ImageRGBr32 const& src, PlatformImage const& dst);

	void transform(PlatformImage const& src, ImageRGBr32 const& dst);


	class PlatformImageGRAY
	{
	public:

		u32 width;
		u32 height;

		u8* data = nullptr;
	};


	void make_image(PlatformImageGRAY& image, u32 width, u32 height);

	void destroy_image(PlatformImageGRAY& image);

	void read_image_from_file(const char* file_path_src, PlatformImageGRAY& image_dst);


	class ImageGRAYr32
	{
	public:

		u32 width;
		u32 height;

		r32* data = nullptr;
	};


	void make_image(ImageGRAYr32& image, u32 width, u32 height);

	void destroy_image(ImageGRAYr32& image);	

	void transform(ImageGRAYr32 const& src, PlatformImageGRAY const& dst);

	void transform(PlatformImageGRAY const& src, ImageGRAYr32 const& dst);


#ifndef LIBIMAGE_NO_WRITE

	void write_image(PlatformImage const& image_src, const char* file_path_dst);

	void write_image(PlatformImageGRAY const& image_src, const char* file_path_dst);

#endif // !LIBIMAGE_NO_WRITE


#ifndef LIBIMAGE_NO_RESIZE

	void resize_image(PlatformImage const& image_src, PlatformImage& image_dst);

	void resize_image(PlatformImageGRAY const& image_src, PlatformImageGRAY& image_dst);

#endif // !LIBIMAGE_NO_RESIZE
	
}


