#pragma once

#include "./device/device.hpp"

#include <functional>
#include <array>


/* constants, enums */

namespace libimage
{
	constexpr auto RGB_CHANNELS = 3u;
	constexpr auto RGBA_CHANNELS = 4u;

	// platform dependent, endianness
	class RGBAr32p
	{
	public:
		r32* R;
		r32* G;
		r32* B;
		r32* A;
	};


	class RGBr32p
	{
	public:
		r32* R;
		r32* G;
		r32* B;
	};


	class HSVr32p
	{
	public:
		r32* H;
		r32* S;
		r32* V;
	};


	class RGBAu8
	{
	public:
		u8 red;
		u8 green;
		u8 blue;
		u8 alpha;
	};


	enum class RGB : int
	{
		R = 0, G = 1, B = 2
	};


	enum class RGBA : int
	{
		R = 0, G = 1, B = 2, A = 3
	};


	enum class GA : int
	{
		G = 0, A = 1
	};


	enum class HSV : int
	{
		H = 0, S = 1, V = 2
	};


	enum class XY : int
	{
		X = 0, Y = 1
	};


	template <typename T>
	constexpr inline int id_cast(T channel)
	{
		return static_cast<int>(channel);
	}
}


/* platform image */

namespace libimage
{
	typedef union pixel_t
	{
		u8 channels[4] = {};

		u32 value;

		RGBAu8 rgba;

	} Pixel;


	constexpr inline Pixel to_pixel(u8 r, u8 g, u8 b, u8 a)
	{
		Pixel p{};
		p.channels[id_cast(RGBA::R)] = r;
		p.channels[id_cast(RGBA::G)] = g;
		p.channels[id_cast(RGBA::B)] = b;
		p.channels[id_cast(RGBA::A)] = a;

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
			Range2Du32 range = {};

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
				Range2Du32 range = {};

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
}


/* planar */

namespace libimage
{	
	class View1r32
	{
	public:

		r32* image_data = 0;
		u32 image_width = 0;

		union
		{
			Range2Du32 range = {};

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


	template <size_t N>
	class ViewCHr32
	{
	public:

		u32 image_width = 0;

		r32* image_channel_data[N] = {};

		union
		{
			Range2Du32 range = {};

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


	using View4r32 = ViewCHr32<4>;
	using View3r32 = ViewCHr32<3>;
	using View2r32 = ViewCHr32<2>;

	using ViewRGBAr32 = View4r32;
	using ViewRGBr32 = View3r32;

	using ViewHSVr32 = View3r32;


	ViewRGBr32 make_rgb_view(ViewRGBAr32 const& image);


	template <size_t N>
	class PixelCHr32
	{
	public:

		static constexpr u32 n_channels = N;

		r32* channels[N] = {};
	};


	using Pixel4r32 = PixelCHr32<4>;
	using Pixel3r32 = PixelCHr32<3>;
	using Pixel2r32 = PixelCHr32<2>;


	class PixelRGBAr32
	{
	public:

		static constexpr u32 n_channels = 4;

		union 
		{
			RGBAr32p rgba;

			r32* channels[4] = {};
		};

		// undefined in device code
		/*
		r32& red() { return *rgba.R; }
		r32& green() { return *rgba.G; }
		r32& blue() { return *rgba.B; }
		r32& alpha() { return *rgba.A; }
		*/
	};


	class PixelRGBr32
	{
	public:

		static constexpr u32 n_channels = 3;

		union 
		{
			RGBr32p rgb;

			r32* channels[3] = {};
		};
		
		// undefined in device code
		/*
		r32& red() { return *rgb.R; }
		r32& green() { return *rgb.G; }
		r32& blue() { return *rgb.B; }
		*/
	};


	class PixelHSVr32
	{
	public:

		static constexpr u32 n_channels = 3;

		union
		{
			HSVr32p hsv;

			r32* channels[3] = {};
		};

		// undefined in device code
		/*
		r32& hue() { return *hsv.H; }
		r32& sat() { return *hsv.S; }
		r32& val() { return *hsv.V; }
		*/
	};
}


/* row_begin */

//namespace libimage
//{
//	r32* row_begin(View1r32 const& view, u32 y);
//
//	Pixel4r32 row_begin(View4r32 const& view, u32 y);
//
//	Pixel3r32 row_begin(View3r32 const& view, u32 y);
//
//	Pixel2r32 row_begin(View2r32 const& view, u32 y);
//
//	PixelRGBAr32 rgba_row_begin(View4r32 const& view, u32 y);
//
//	PixelRGBr32 rgb_row_begin(View3r32 const& view, u32 y);
//}


/* xy_at */

namespace libimage
{
	r32* xy_at(View1r32 const& view, u32 x, u32 y);

	Pixel4r32 xy_at(View4r32 const& view, u32 x, u32 y);

	Pixel3r32 xy_at(View3r32 const& view, u32 x, u32 y);

	Pixel2r32 xy_at(View2r32 const& view, u32 x, u32 y);

	PixelRGBAr32 rgba_xy_at(ViewRGBAr32 const& view, u32 x, u32 y);

	PixelRGBr32 rgb_xy_at(ViewRGBr32 const& view, u32 x, u32 y);

	PixelHSVr32 hsv_xy_at(ViewHSVr32 const& view, u32 x, u32 y);
}


/* make_view */

namespace libimage
{	
	using Buffer32 = cuda::MemoryBuffer<r32>;


	void make_view(View4r32& view, u32 width, u32 height, Buffer32& buffer);

	void make_view(View3r32& view, u32 width, u32 height, Buffer32& buffer);

	void make_view(View2r32& view, u32 width, u32 height, Buffer32& buffer);

	void make_view(View1r32& view, u32 width, u32 height, Buffer32& buffer);	
}


/* sub_view */

namespace libimage
{
	View4r32 sub_view(View4r32 const& view, Range2Du32 const& range);

	View3r32 sub_view(View3r32 const& view, Range2Du32 const& range);

	View2r32 sub_view(View2r32 const& view, Range2Du32 const& range);

	View1r32 sub_view(View1r32 const& view, Range2Du32 const& range);
}


/* map */

namespace libimage
{
	void map(View1r32 const& device_src, gray::View const& host_dst, Buffer32& host_buffer);

	void map(gray::View const& host_src, View1r32 const& device_dst, Buffer32& host_buffer);

	void map(View1r32 const& device_src, gray::View const& host_dst, Buffer32& host_buffer, r32 gray_min, r32 gray_max);

	void map(gray::View const& host_src, View1r32 const& device_dst, Buffer32& host_buffer, r32 gray_min, r32 gray_max);
}


/* map_rgb */

namespace libimage
{
	void map_rgb(ViewRGBAr32 const& device_src, View const& host_dst, Buffer32& host_buffer);

	void map_rgb(View const& host_src, ViewRGBAr32 const& device_dst, Buffer32& host_buffer);


	void map_rgb(ViewRGBr32 const& device_src, View const& host_dst, Buffer32& host_buffer);

	void map_rgb(View const& host_src, ViewRGBr32 const& device_dst, Buffer32& host_buffer);
}


/* map_hsv */

namespace libimage
{
	void map_rgb_hsv(ViewRGBr32 const& src, ViewHSVr32 const& dst);

	void map_hsv_rgb(ViewHSVr32 const& src, ViewRGBr32 const& dst);


	//void map_hsv(View const& host_src, ViewHSVr32 const& device_dst);

	//void map_hsv(ViewHSVr32 const& device_src, View const& host_dst);
}


/* fill */

namespace libimage
{
	void fill(View const& view, Pixel color);

	void fill(gray::View const& view, u8 gray);

	void fill(ViewRGBAr32 const& view, Pixel color);

	void fill(ViewRGBr32 const& view, Pixel color);

	void fill(View1r32 const& view, r32 gray32);

	void fill(View1r32 const& view, u8 gray);
}


/* copy */

namespace libimage
{
	void copy(View const& src, View const& dst);
	
	void copy(gray::View const& src, gray::View const& dst);


	void copy(View4r32 const& src, View4r32 const& dst);

	void copy(View3r32 const& src, View3r32 const& dst);

	void copy(View2r32 const& src, View2r32 const& dst);

	void copy(View1r32 const& src, View1r32 const& dst);
}


/* for_each_pixel */

namespace libimage
{
	using u8_f = std::function<void(u8& p)>;

	using r32_f = std::function<void(r32& p)>;


	void for_each_pixel(gray::Image const& image, u8_f const& func);

	void for_each_pixel(gray::View const& image, u8_f const& func);


	void for_each_pixel(View1r32 const& image, r32_f const& func);

}


/* for_each_xy */

namespace libimage
{
	using xy_f = std::function<void(u32 x, u32 y)>;


	void for_each_xy(Image const& image, xy_f const& func);

	void for_each_xy(View const& view, xy_f const& func);


	void for_each_xy(gray::Image const& image, xy_f const& func);

	void for_each_xy(gray::View const& view, xy_f const& func);


	void for_each_xy(View4r32 const& view, xy_f const& func);

	void for_each_xy(View3r32 const& view, xy_f const& func);

	void for_each_xy(View2r32 const& view, xy_f const& func);

	void for_each_xy(View1r32 const& view, xy_f const& func);
}


/* grayscale */

namespace libimage
{
	void grayscale(Image const& src, gray::Image const& dst);

	void grayscale(Image const& src, gray::View const& dst);

	void grayscale(View const& src, gray::Image const& dst);

	void grayscale(View const& src, gray::View const& dst);


	void grayscale(View3r32 const& src, View1r32 const& dst);
}


/* select_channel */

namespace libimage
{
	View1r32 select_channel(ViewRGBAr32 const& view, RGBA channel);

	View1r32 select_channel(ViewRGBr32 const& view, RGB channel);

	View1r32 select_channel(ViewHSVr32 const& view, HSV channel);

	View1r32 select_channel(View2r32 const& view, GA channel);

	View1r32 select_channel(View2r32 const& view, XY channel);
}


/* alpha_blend */

namespace libimage
{
	void alpha_blend(View4r32 const& src, View3r32 const& cur, View3r32 const& dst);

	void alpha_blend(View2r32 const& src, View1r32 const& cur, View1r32 const& dst);
}


/* transform */

namespace libimage
{
	using lut_t = std::array<u8, 256>;

	using u8_to_u8_f = std::function<u8(u8)>;

	lut_t to_lut(u8_to_u8_f const& f);

	using r32_to_r32_f = std::function<r32(r32)>;


	void transform(gray::Image const& src, gray::Image const& dst, lut_t const& lut);

	void transform(gray::Image const& src, gray::View const& dst, lut_t const& lut);

	void transform(gray::View const& src, gray::Image const& dst, lut_t const& lut);

	void transform(gray::View const& src, gray::View const& dst, lut_t const& lut);


	void transform(View1r32 const& src, View1r32 const& dst, r32_to_r32_f const& func);
}


/* threshold */

namespace libimage
{
	void threshold(gray::Image const& src, gray::Image const& dst, u8 min, u8 max);

	void threshold(gray::Image const& src, gray::View const& dst, u8 min, u8 max);

	void threshold(gray::View const& src, gray::Image const& dst, u8 min, u8 max);

	void threshold(gray::View const& src, gray::View const& dst, u8 min, u8 max);


	void threshold(View1r32 const& src, View1r32 const& dst, r32 min, r32 max);
}


/* contrast */

namespace libimage
{
	void contrast(gray::Image const& src, gray::Image const& dst, u8 min, u8 max);

	void contrast(gray::Image const& src, gray::View const& dst, u8 min, u8 max);

	void contrast(gray::View const& src, gray::Image const& dst, u8 min, u8 max);

	void contrast(gray::View const& src, gray::View const& dst, u8 min, u8 max);


	void contrast(View1r32 const& src, View1r32 const& dst, r32 min, r32 max);
}


/* blur */

namespace libimage
{
	void blur(View1r32 const& src, View1r32 const& dst);

	void blur(View3r32 const& src, View3r32 const& dst);
}


/* gradients */

namespace libimage
{
	void gradients(View1r32 const& src, View1r32 const& dst);

	void gradients_xy(View1r32 const& src, View2r32 const& xy_dst);
}


/* edges */

namespace libimage
{
	inline void edges(View1r32 const& src, View1r32 const& dst) { gradients(src, dst); }

	void edges(View1r32 const& src, View1r32 const& dst, r32 threshold);

	void edges_xy(View1r32 const& src, View2r32 const& xy_dst, r32 threshold);
}


/* corners */

namespace libimage
{
	void corners(View1r32 const& src, View2r32 const& temp, View1r32 const& dst);
}


/* rotate */

namespace libimage
{
	void rotate(View4r32 const& src, View4r32 const& dst, Point2Du32 origin, r32 radians);

	void rotate(View3r32 const& src, View3r32 const& dst, Point2Du32 origin, r32 radians);

	void rotate(View1r32 const& src, View1r32 const& dst, Point2Du32 origin, r32 radians);
}


/* overlay */

namespace libimage
{
	void overlay(View3r32 const& src, View1r32 const& binary, Pixel color, View3r32 const& dst);

	void overlay(View1r32 const& src, View1r32 const& binary, u8 gray, View1r32 const& dst);
}


/* scale_down */

namespace libimage
{
	View3r32 scale_down(View3r32 const& src, Buffer32& buffer);

	View1r32 scale_down(View1r32 const& src, Buffer32& buffer);
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

#else

#include <string>

namespace libimage
{

	inline void read_image_from_file(std::string const& img_path_src, Image& image_dst)
	{
		return read_image_from_file(img_path_src.c_str(), image_dst);
	}


	inline void read_image_from_file(std::string const& img_path_src, gray::Image& image_dst)
	{
		return read_image_from_file(img_path_src.c_str(), image_dst);
	}

#ifndef LIBIMAGE_NO_WRITE

	inline void write_image(Image const& image_src, std::string const& file_path_dst)
	{
		write_image(image_src, file_path_dst.c_str());
	}


	inline void write_image(gray::Image const& image_src, std::string const& file_path_dst)
	{
		write_image(image_src, file_path_dst.c_str());
	}

#endif // !LIBIMAGE_NO_WRITE
	
}

#endif // !LIBIMAGE_NO_FILESYSTEM





