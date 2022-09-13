#pragma once

#include "defines.hpp"

#include <functional>
#include <array>


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


	constexpr inline int id_cast(auto channel)
	{
		return static_cast<int>(channel);
	}


	typedef union pixel_t
	{
		u8 channels[4] = {};

		u32 value;

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

	r32* row_begin(View1r32 const& view, u32 y);

	r32* xy_at(View1r32 const& view, u32 x, u32 y);


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
	

	template <size_t N>
	class ChannelData
	{
	public:
		static constexpr u32 n_channels = N;

		r32* channels[N] = {};
	};


	template <size_t N>
	static ChannelData<N> channel_row_begin(ViewCHr32<N> const& view, u32 y)
	{
		auto offset = (size_t)((view.y_begin + y) * view.image_width + view.x_begin);

		ChannelData<N> data{};
		for (u32 ch = 0; ch < N; ++ch)
		{
			data.channels[ch] = view.image_channel_data[ch] + offset;
		}

		return data;
	}


	using View4r32 = ViewCHr32<4>;
	using View3r32 = ViewCHr32<3>;


	View3r32 make_rgb_view(View4r32 const& image);
}


/* make_view */

namespace libimage
{	
	using Buffer32 = MemoryBuffer<r32>;


	void make_view(View4r32& view, u32 width, u32 height, Buffer32& buffer);

	void make_view(View3r32& view, u32 width, u32 height, Buffer32& buffer);

	void make_view(View1r32& view, u32 width, u32 height, Buffer32& buffer);	
}


/* convert */

namespace libimage
{
	void convert(View4r32 const& src, Image const& dst);

	void convert(Image const& src, View4r32 const& dst);


	void convert(View4r32 const& src, View const& dst);

	void convert(View const& src, View4r32 const& dst);


	void convert(View3r32 const& src, Image const& dst);

	void convert(Image const& src, View3r32 const& dst);


	void convert(View3r32 const& src, View const& dst);

	void convert(View const& src, View3r32 const& dst);


	void convert(View1r32 const& src, gray::Image const& dst);

	void convert(gray::Image const& src, View1r32 const& dst);


	void convert(View1r32 const& src, gray::View const& dst);

	void convert(gray::View const& src, View1r32 const& dst);
}


/* sub_view */

namespace libimage
{
	View4r32 sub_view(View4r32 const& view, Range2Du32 const& range);

	View3r32 sub_view(View3r32 const& view, Range2Du32 const& range);

	View1r32 sub_view(View1r32 const& view, Range2Du32 const& range);
}


/* fill */

namespace libimage
{
	void fill(Image const& image, Pixel color);

	void fill(View const& view, Pixel color);

	void fill(gray::Image const& image, u8 gray);

	void fill(gray::View const& view, u8 gray);

	void fill(View4r32 const& view, Pixel color);

	void fill(View3r32 const& view, Pixel color);

	void fill(View1r32 const& image, u8 gray);
}


/* copy */

namespace libimage
{
	void copy(Image const& src, Image const& dst);

	void copy(Image const& src, View const& dst);

	void copy(View const& src, Image const& dst);

	void copy(View const& src, View const& dst);


	void copy(gray::Image const& src, gray::Image const& dst);

	void copy(gray::Image const& src, gray::View const& dst);

	void copy(gray::View const& src, gray::Image const& dst);

	void copy(gray::View const& src, gray::View const& dst);


	void copy(View4r32 const& src, View4r32 const& dst);

	void copy(View3r32 const& src, View3r32 const& dst);

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

	void for_each_xy(View3r32 const& image, xy_f const& func);

	void for_each_xy(View1r32 const& view, xy_f const& func);
}


/* grayscale */

namespace libimage
{
	void grayscale(Image const& src, gray::Image const& dst);

	void grayscale(Image const& src, gray::View const& dst);

	void grayscale(View const& src, gray::Image const& dst);

	void grayscale(View const& src, gray::View const& dst);


	void grayscale(View4r32 const& src, View1r32 const& dst);

	void grayscale(View3r32 const& src, View1r32 const& dst);
}


/* select_channel */

namespace libimage
{
	View1r32 select_channel(View4r32 const& image, RGBA channel);

	View1r32 select_channel(View3r32 const& image, RGB channel);
}


/* alpha_blend */

namespace libimage
{
	void alpha_blend(View4r32 const& src, View3r32 const& cur, View3r32 const& dst);

	void alpha_blend(View4r32 const& src, View3r32 const& cur_dst);
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





