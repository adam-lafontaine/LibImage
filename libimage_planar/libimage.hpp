#pragma once

#include "defines.hpp"


namespace libimage
{
	constexpr auto RGB_CHANNELS = 3u;
	constexpr auto RGBA_CHANNELS = 4u;

	enum class RGB : int
	{
		R = 0, G = 1, B = 2
	};


	enum class RGBA : int
	{
		R = 0, G = 1, B = 2, A = 3
	};


	typedef union
	{
		struct
		{
			u8 red;
			u8 green;
			u8 blue;
			u8 alpha;
		};

		u8 channels[4];

		u32 value;

	} PlatformPixel;


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

	PlatformPixel* row_begin(PlatformImage const& image, u32 y);

	PlatformPixel* xy_at(PlatformImage const& image, u32 x, u32 y);


	class PlatformImageView
	{
	public:

		PlatformPixel* image_data = 0;
		u32 image_width = 0;

		u32 x_begin = 0;
		u32 x_end = 0;
		u32 y_begin = 0;
		u32 y_end = 0;

		u32 width = 0;
		u32 height = 0;
	};


	PlatformImageView make_view(PlatformImage const& image);

	PlatformImageView sub_view(PlatformImage const& image, Range2Du32 const& range);

	PlatformImageView sub_view(PlatformImageView const& view, Range2Du32 const& range);

	PlatformPixel* row_begin(PlatformImageView const& view, u32 y);

	PlatformPixel* xy_at(PlatformImageView const& view, u32 x, u32 y);


	class ImageRGBAr32
	{
	public:

		u32 width = 0;
		u32 height = 0;

		union
		{
			struct
			{
				r32* red;
				r32* green;
				r32* blue;
				r32* alpha;
			};

			r32* channel_data[4];
		};
	};


	void make_image(ImageRGBAr32& image, u32 width, u32 height);

	void destroy_image(ImageRGBAr32& image);

	r32* row_begin(ImageRGBAr32 const& image, u32 y, RGBA channel);

	r32* xy_at(ImageRGBAr32 const& image, u32 x, u32 y, RGBA channel);

	void transform(ImageRGBAr32 const& src, PlatformImage const& dst);

	void transform(PlatformImage const& src, ImageRGBAr32 const& dst);


	class ImageViewRGBAr32
	{
	public:

		u32 image_width = 0;

		union
		{
			struct
			{
				r32* image_red;
				r32* image_green;
				r32* image_blue;
				r32* image_alpha;
			};

			r32* image_channel_data[4];
		};

		u32 x_begin = 0;
		u32 x_end = 0;
		u32 y_begin = 0;
		u32 y_end = 0;

		u32 width = 0;
		u32 height = 0;
	};


	ImageViewRGBAr32 make_view(ImageRGBAr32 const& image);

	ImageViewRGBAr32 sub_view(ImageRGBAr32 const& image, Range2Du32 const& range);

	ImageViewRGBAr32 sub_view(ImageViewRGBAr32 const& view, Range2Du32 const& range);

	r32* row_begin(ImageViewRGBAr32 const& view, u32 y);

	r32* xy_at(ImageViewRGBAr32 const& view, u32 x, u32 y);


	class ImageRGBr32
	{
	public:

		u32 width = 0;
		u32 height = 0;

		union
		{
			struct
			{
				r32* red;
				r32* green;
				r32* blue;
			};

			r32* channel_data[3];
		};
	};


	void make_image(ImageRGBr32& image, u32 width, u32 height);

	void destroy_image(ImageRGBr32& image);

	r32* row_begin(ImageRGBr32 const& image, u32 y, RGB channel);

	r32* xy_at(ImageRGBr32 const& image, u32 x, u32 y, RGB channel);

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

	u8* row_begin(PlatformImageGRAY const& image, u32 y);

	u8* xy_at(PlatformImageGRAY const& image, u32 x, u32 y);


	class ImageGRAYr32
	{
	public:

		u32 width;
		u32 height;

		r32* data = nullptr;
	};


	void make_image(ImageGRAYr32& image, u32 width, u32 height);

	void destroy_image(ImageGRAYr32& image);

	r32* row_begin(ImageGRAYr32 const& image, u32 y);

	r32* xy_at(ImageGRAYr32 const& image, u32 x, u32 y);

	void transform(ImageGRAYr32 const& src, PlatformImageGRAY const& dst);

	void transform(PlatformImageGRAY const& src, ImageGRAYr32 const& dst);


	void read_image_from_file(const char* img_path_src, PlatformImage& image_dst);

	void read_image_from_file(const char* file_path_src, PlatformImageGRAY& image_dst);


#ifndef LIBIMAGE_NO_WRITE

	void write_image(PlatformImage const& image_src, const char* file_path_dst);

	void write_image(PlatformImageGRAY const& image_src, const char* file_path_dst);

#endif // !LIBIMAGE_NO_WRITE


#ifndef LIBIMAGE_NO_RESIZE

	void resize_image(PlatformImage const& image_src, PlatformImage& image_dst);

	void resize_image(PlatformImageGRAY const& image_src, PlatformImageGRAY& image_dst);

#endif // !LIBIMAGE_NO_RESIZE
	
}


