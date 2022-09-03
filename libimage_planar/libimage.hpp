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

	} Pixel;


	constexpr inline Pixel to_pixel(u8 r, u8 g, u8 b, u8 a)
	{
		Pixel p{};
		p.red = r;
		p.green = g;
		p.blue = b;
		p.alpha = a;

		return p;
	}


	constexpr inline Pixel to_pixel(u8 r, u8 g, u8 b)
	{
		return to_pixel(r, g, b, 255);
	}


	constexpr inline Pixel to_pixel(u8 value)
	{
		return to_pixel(value, value, value, 255);
	}


	class Image
	{
	public:

		u32 width = 0;
		u32 height = 0;

		Pixel* data = nullptr;
	};


	void make_image(Image& image, u32 width, u32 height);

	void destroy_image(Image& image);

	Pixel* row_begin(Image const& image, u32 y);

	Pixel* xy_at(Image const& image, u32 x, u32 y);


	class View
	{
	public:

		Pixel* image_data = 0;
		u32 image_width = 0;

		union
		{
			Range2Du32 range;

			struct
			{
				u32 x_begin;
				u32 x_end;
				u32 y_begin;
				u32 y_end;
			};
		};

		u32 width = 0;
		u32 height = 0;
	};


	View make_view(Image const& image);

	View sub_view(Image const& image, Range2Du32 const& range);

	View sub_view(View const& view, Range2Du32 const& range);

	Pixel* row_begin(View const& view, u32 y);

	Pixel* xy_at(View const& view, u32 x, u32 y);


	class Image4Cr32
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


	void make_image(Image4Cr32& image, u32 width, u32 height);

	void destroy_image(Image4Cr32& image);

	r32* row_begin(Image4Cr32 const& image, u32 y, RGBA channel);

	r32* xy_at(Image4Cr32 const& image, u32 x, u32 y, RGBA channel);	


	class View4Cr32
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

		union
		{
			Range2Du32 range;

			struct
			{
				u32 x_begin;
				u32 x_end;
				u32 y_begin;
				u32 y_end;
			};
		};

		u32 width = 0;
		u32 height = 0;
	};


	View4Cr32 make_view(Image4Cr32 const& image);

	View4Cr32 sub_view(Image4Cr32 const& image, Range2Du32 const& range);

	View4Cr32 sub_view(View4Cr32 const& view, Range2Du32 const& range);

	r32* row_begin(View4Cr32 const& view, u32 y, RGBA channel);

	r32* xy_at(View4Cr32 const& view, u32 x, u32 y, RGBA channel);


	class Image3Cr32
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


	void make_image(Image3Cr32& image, u32 width, u32 height);

	void destroy_image(Image3Cr32& image);

	r32* row_begin(Image3Cr32 const& image, u32 y, RGB channel);

	r32* xy_at(Image3Cr32 const& image, u32 x, u32 y, RGB channel);


	class View3Cr32
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
			};

			r32* image_channel_data[3];
		};

		union
		{
			Range2Du32 range;

			struct
			{
				u32 x_begin;
				u32 x_end;
				u32 y_begin;
				u32 y_end;
			};
		};

		u32 width = 0;
		u32 height = 0;
	};


	View3Cr32 make_view(Image3Cr32 const& image);

	View3Cr32 sub_view(Image3Cr32 const& image, Range2Du32 const& range);

	View3Cr32 sub_view(View3Cr32 const& view, Range2Du32 const& range);

	r32* row_begin(View3Cr32 const& view, u32 y, RGB channel);

	r32* xy_at(View3Cr32 const& view, u32 x, u32 y, RGB channel);


	namespace gray
	{
		class Image
		{
		public:

			u32 width;
			u32 height;

			u8* data = nullptr;
		};


		class View
		{
		public:

			u8* image_data = 0;
			u32 image_width = 0;

			union
			{
				Range2Du32 range;

				struct
				{
					u32 x_begin;
					u32 x_end;
					u32 y_begin;
					u32 y_end;
				};
			};

			u32 width = 0;
			u32 height = 0;
		};
	}


	void make_image(gray::Image& image, u32 width, u32 height);

	void destroy_image(gray::Image& image);

	u8* row_begin(gray::Image const& image, u32 y);

	u8* xy_at(gray::Image const& image, u32 x, u32 y);

	gray::View make_view(gray::Image const& image);

	gray::View sub_view(gray::Image const& image, Range2Du32 const& range);

	gray::View sub_view(gray::View const& view, Range2Du32 const& range);

	u8* row_begin(gray::View const& view, u32 y);

	u8* xy_at(gray::View const& view, u32 x, u32 y);


	class Image1Cr32
	{
	public:

		u32 width;
		u32 height;

		r32* data = nullptr;
	};


	void make_image(Image1Cr32& image, u32 width, u32 height);

	void destroy_image(Image1Cr32& image);

	r32* row_begin(Image1Cr32 const& image, u32 y, RGB channel);

	r32* xy_at(Image1Cr32 const& image, u32 x, u32 y, RGB channel);


	class View1Cr32
	{
	public:

		r32* image_data = 0;
		u32 image_width = 0;

		union
		{
			Range2Du32 range;

			struct
			{
				u32 x_begin;
				u32 x_end;
				u32 y_begin;
				u32 y_end;
			};
		};

		u32 width = 0;
		u32 height = 0;
	};


	View1Cr32 make_view(Image1Cr32 const& image);

	View1Cr32 sub_view(Image1Cr32 const& image, Range2Du32 const& range);

	View1Cr32 sub_view(View1Cr32 const& view, Range2Du32 const& range);

	r32* row_begin(View1Cr32 const& view, u32 y);

	r32* xy_at(View1Cr32 const& view, u32 x, u32 y);	
}


/* convert */

namespace libimage
{
	void convert(Image4Cr32 const& src, Image const& dst);

	void convert(Image const& src, Image4Cr32 const& dst);


	void convert(Image4Cr32 const& src, View const& dst);

	void convert(View const& src, Image4Cr32 const& dst);


	void convert(View4Cr32 const& src, Image const& dst);

	void convert(Image const& src, View4Cr32 const& dst);


	void convert(View4Cr32 const& src, View const& dst);

	void convert(View const& src, View4Cr32 const& dst);


	void convert(Image3Cr32 const& src, Image const& dst);

	void convert(Image const& src, Image3Cr32 const& dst);


	void convert(Image3Cr32 const& src, View const& dst);

	void convert(View const& src, Image3Cr32 const& dst);


	void convert(View3Cr32 const& src, Image const& dst);

	void convert(Image const& src, View3Cr32 const& dst);


	void convert(View3Cr32 const& src, View const& dst);

	void convert(View const& src, View3Cr32 const& dst);


	void convert(Image1Cr32 const& src, gray::Image const& dst);

	void convert(gray::Image const& src, Image1Cr32 const& dst);


	void convert(Image1Cr32 const& src, gray::View const& dst);

	void convert(gray::View const& src, Image1Cr32 const& dst);


	void convert(View1Cr32 const& src, gray::Image const& dst);

	void convert(gray::Image const& src, View1Cr32 const& dst);


	void convert(View1Cr32 const& src, gray::View const& dst);

	void convert(gray::View const& src, View1Cr32 const& dst);
}


/* fill */

namespace libimage
{
	void fill(Image const& image, Pixel color);

	void fill(View const& view, Pixel color);

	void fill(Image4Cr32 const& image, Pixel color);

	void fill(View4Cr32 const& view, Pixel color);

	void fill(Image3Cr32 const& image, Pixel color);

	void fill(View3Cr32 const& view, Pixel color);

	void fill(gray::Image const& image, u8 gray);

	void fill(gray::View const& view, u8 gray);

	void fill(Image1Cr32 const& image, u8 gray);

	void fill(View1Cr32 const& image, u8 gray);
}


/* copy */

namespace libimage
{
	void copy(gray::Image const& src, gray::Image const& dst);

	void copy(gray::Image const& src, gray::View const& dst);

	void copy(gray::View const& src, gray::Image const& dst);

	void copy(gray::View const& src, gray::View const& dst);


	void copy(Image1Cr32 const& src, Image1Cr32 const& dst);

	void copy(Image1Cr32 const& src, View1Cr32 const& dst);

	void copy(View1Cr32 const& src, Image1Cr32 const& dst);

	void copy(View1Cr32 const& src, View1Cr32 const& dst);
}



/* stb wrappers */

namespace libimage
{
	void read_image_from_file(const char* img_path_src, Image& image_dst);

	void read_image_from_file(const char* file_path_src, gray::Image& image_dst);


#ifndef LIBIMAGE_NO_WRITE

	void write_image(Image const& image_src, const char* file_path_dst);

	void write_image(gray::Image const& image_src, const char* file_path_dst);

#endif // !LIBIMAGE_NO_WRITE


#ifndef LIBIMAGE_NO_RESIZE

	void resize_image(Image const& image_src, Image& image_dst);

	void resize_image(gray::Image const& image_src, gray::Image& image_dst);

#endif // !LIBIMAGE_NO_RESIZE

}

#ifndef LIBIMAGE_NO_FILESYSTEM

#include <filesystem>

namespace fs = std::filesystem;


namespace libimage
{

	inline void read_image_from_file(fs::path const& img_path_src, Image& image_dst)
	{
		return read_image_from_file(img_path_src.string().c_str(), image_dst);
	}


	inline void read_image_from_file(fs::path const& img_path_src, gray::Image& image_dst)
	{
		return read_image_from_file(img_path_src.string().c_str(), image_dst);
	}

#ifndef LIBIMAGE_NO_WRITE

	inline void write_image(Image const& image_src, fs::path const& file_path_dst)
	{
		write_image(image_src, file_path_dst.string().c_str());
	}

	inline void write_image(gray::Image const& image_src, fs::path const& file_path_dst)
	{
		write_image(image_src, file_path_dst.string().c_str());
	}

#endif // !LIBIMAGE_NO_WRITE
	
}

#endif // !LIBIMAGE_NO_FILESYSTEM





