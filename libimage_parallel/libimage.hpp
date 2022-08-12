#pragma once

#include <cstdint>

#include <array>
#include <iterator>
#include <cassert>
#include <functional>

#include "defines.hpp"


class Point2Du32
{
public:
	u32 x;
	u32 y;
};


class Point2Dr32
{
public:
	r32 x;
	r32 y;
};


// region of interest in an image
class Range2Du32
{
public:
	u32 x_begin;
	u32 x_end;   // one past last x
	u32 y_begin;
	u32 y_end;   // one past last y
};

#ifndef LIBIMAGE_NO_COLOR

/* rgba.hpp  */

namespace libimage
{
	constexpr auto RGB_CHANNELS = 3u;
	constexpr auto RGBA_CHANNELS = 4u;

	enum class Channel : u32
	{
		Red = 0,
		Green,
		Blue,
		Alpha
	};


	inline u32 to_channel_index(Channel ch)
	{
		return (u32)(ch);
	}


	inline void for_each_channel_rgb(std::function<void(u32 ch_id)> const& func)
	{
		for (u32 id = 0; id < RGB_CHANNELS; ++id)
		{
			func(id);
		}
	}


	inline void for_each_channel_rgba(std::function<void(u32 ch_id)> const& func)
	{
		for (u32 id = 0; id < RGBA_CHANNELS; ++id)
		{
			func(id);
		}
	}


	// color pixel
	typedef union rgba_pixel_t
	{
		struct
		{
			u8 red;
			u8 green;
			u8 blue;
			u8 alpha;
		};

		u8 channels[RGBA_CHANNELS];

		u32 value;

	}Pixel;


	constexpr Pixel to_pixel(u8 red, u8 green, u8 blue, u8 alpha)
	{
		Pixel pixel{};
		pixel.red = red;
		pixel.green = green;
		pixel.blue = blue;
		pixel.alpha = alpha;

		return pixel;
	}


	constexpr Pixel to_pixel(u8 red, u8 green, u8 blue)
	{
		return to_pixel(red, green, blue, 255);
	}


	constexpr Pixel to_pixel(u8 value)
	{
		return to_pixel(value, value, value, 255);
	}


	class RGBAImage
	{
	public:
		u32 width = 0;
		u32 height = 0;

		Pixel* data = nullptr;
	};



	// subset of existing image data
	class RGBAImageView
	{
	public:

		Pixel* image_data = 0;
		u32 image_width = 0;

		u32 x_begin = 0;
		u32 x_end = 0;
		u32 y_begin = 0;
		u32 y_end = 0;

		u32 width = 0;
		u32 height = 0;
	};


	using Image = RGBAImage;
	using View = RGBAImageView;

}

#endif // !LIBIMAGE_NO_COLOR


#ifndef LIBIMAGE_NO_GRAYSCALE

/*  gray.hpp  */

namespace libimage
{
	namespace gray
	{
		// grayscale value as an unsigned 8-bit integer
		using Pixel = u8;


		// grayscale image
		class Image
		{
		public:
			u32 width = 0;
			u32 height = 0;

			Pixel* data = nullptr;
		};


		// subset of grayscale image data
		class ImageView
		{
		public:

			Pixel* image_data = 0;
			u32 image_width = 0;

			u32 x_begin = 0;
			u32 x_end = 0;
			u32 y_begin = 0;
			u32 y_end = 0;

			u32 width = 0;
			u32 height = 0;
		};


		using View = ImageView;
	}
}


#endif // !LIBIMAGE_NO_GRAYSCALE


/*  libimage.hpp  */

namespace libimage
{

#ifndef LIBIMAGE_NO_COLOR

	void read_image_from_file(const char* img_path_src, Image& image_dst);

	void make_image(Image& image_dst, u32 width, u32 height);

	void destroy_image(Image& image);

	Pixel* row_begin(Image const& image, u32 y);

	Pixel* xy_at(Image const& image, u32 x, u32 y);

	View make_view(Image const& image);

	View make_view(Image& image, u32 width, u32 height);

	View sub_view(Image const& image, Range2Du32 const& range);

	View sub_view(View const& view, Range2Du32 const& range);

	Pixel* row_begin(View const& view, u32 y);

	Pixel* xy_at(View const& view, u32 x, u32 y);
	


#ifndef LIBIMAGE_NO_WRITE

	void write_image(Image const& image_src, const char* file_path_dst);

	void write_view(View const& view_src, const char* file_path_dst);



#endif // !LIBIMAGE_NO_WRITE


#ifndef LIBIMAGE_NO_RESIZE

	void resize_image(Image const& image_src, Image& image_dst);

	View make_resized_view(Image const& image_src, Image& image_dst);

#endif // !LIBIMAGE_NO_RESIZE

#endif // !LIBIMAGE_NO_COLOR


#ifndef LIBIMAGE_NO_GRAYSCALE

	void read_image_from_file(const char* file_path_src, gray::Image& image_dst);

	void make_image(gray::Image& image_dst, u32 width, u32 height);

	void destroy_image(gray::Image& image);

	gray::Pixel* row_begin(gray::Image const& image, u32 y);

	gray::Pixel* xy_at(gray::Image const& image, u32 x, u32 y);

	gray::View make_view(gray::Image const& image);

	gray::View make_view(gray::Image& image, u32 width, u32 height);

	gray::View sub_view(gray::Image const& image, Range2Du32 const& range);

	gray::View sub_view(gray::View const& view, Range2Du32 const& range);

	gray::Pixel* row_begin(gray::View const& view, u32 y);

	gray::Pixel* xy_at(gray::View const& view, u32 x, u32 y);	


#ifndef LIBIMAGE_NO_WRITE

	void write_image(gray::Image const& image_src, const char* file_path_dst);

	void write_view(gray::View const& view_src, const char* file_path_dst);


#endif // !LIBIMAGE_NO_WRITE

#ifndef LIBIMAGE_NO_RESIZE

	void resize_image(gray::Image const& img_src, gray::Image& img_dst);

	gray::View make_resized_view(gray::Image const& image_src, gray::Image& image_dst);

#endif // !LIBIMAGE_NO_RESIZE

#endif // !LIBIMAGE_NO_GRAYSCALE

}


namespace libimage
{
#ifndef LIBIMAGE_NO_COLOR

	inline void read_image_from_file(std::string const& img_path_src, Image& image_dst)
	{
		read_image_from_file(img_path_src.c_str(), image_dst);
	}

#ifndef LIBIMAGE_NO_WRITE

	inline void write_image(Image const& image_src, std::string const& file_path)
	{
		write_image(image_src, file_path.c_str());
	}

	inline void write_view(View const& view_src, std::string const& file_path)
	{
		write_view(view_src, file_path.c_str());
	}

#endif // !LIBIMAGE_NO_WRITE

#endif // !LIBIMAGE_NO_COLOR

#ifndef LIBIMAGE_NO_GRAYSCALE

	inline void read_image_from_file(std::string const& img_path_src, gray::Image& image_dst)
	{
		return read_image_from_file(img_path_src.c_str(), image_dst);
	}

#ifndef LIBIMAGE_NO_WRITE

	inline void write_image(gray::Image const& image_src, std::string const& file_path_dst)
	{
		write_image(image_src, file_path_dst.c_str());
	}

	inline void write_view(gray::View const& view_src, std::string const& file_path_dst)
	{
		write_view(view_src, file_path_dst.c_str());
	}

#endif // !LIBIMAGE_NO_WRITE

#endif // !LIBIMAGE_NO_GRAYSCALE
}


#ifndef LIBIMAGE_NO_FILESYSTEM

#include <filesystem>
namespace fs = std::filesystem;


namespace libimage
{
#ifndef LIBIMAGE_NO_COLOR

	inline void read_image_from_file(fs::path const& img_path_src, Image& image_dst)
	{
		read_image_from_file(img_path_src.string().c_str(), image_dst);
	}

#ifndef LIBIMAGE_NO_WRITE

	inline void write_image(Image const& image_src, fs::path const& file_path)
	{
		write_image(image_src, file_path.string().c_str());
	}

	inline void write_view(View const& view_src, fs::path const& file_path)
	{
		write_view(view_src, file_path.string().c_str());
	}

#endif // !LIBIMAGE_NO_WRITE

#endif // !LIBIMAGE_NO_COLOR

#ifndef LIBIMAGE_NO_GRAYSCALE

	inline void read_image_from_file(fs::path const& img_path_src, gray::Image& image_dst)
	{
		return read_image_from_file(img_path_src.string().c_str(), image_dst);
	}

#ifndef LIBIMAGE_NO_WRITE

	inline void write_image(gray::Image const& image_src, fs::path const& file_path_dst)
	{
		write_image(image_src, file_path_dst.string().c_str());
	}

	inline void write_view(gray::View const& view_src, fs::path const& file_path_dst)
	{
		write_view(view_src, file_path_dst.string().c_str());
	}

#endif // !LIBIMAGE_NO_WRITE

#endif // !LIBIMAGE_NO_GRAYSCALE
}

#endif // !LIBIMAGE_NO_FILESYSTEM


/*  process.hpp  */

namespace libimage
{
	constexpr u32 VIEW_MIN_DIM = 5;

}


/*  proc_def.hpp  */

namespace libimage
{
	using pixel_f = std::function<void(Pixel& p)>;

	using u8_f = std::function<void(u8& p)>;

	using xy_f = std::function<void(u32 x, u32 y)>;

	using u8_to_bool_f = std::function<bool(u8)>;

	using pixel_to_bool_f = std::function<bool(Pixel)>;

	using pixel_to_pixel_f = std::function<Pixel(Pixel)>;

	using pixel_to_u8_f = std::function<u8(Pixel)>;

	using u8_to_u8_f = std::function<u8(u8)>;

	using lookup_table_t = std::array<u8, 256>;
}


namespace libimage
{
#ifndef LIBIMAGE_NO_COLOR	

	void for_each_pixel(Image const& image, pixel_f const& func);

	void for_each_pixel(View const& view, pixel_f const& func);

	void for_each_xy(Image const& image, xy_f const& func);

	void for_each_xy(View const& view, xy_f const& func);

#endif // !LIBIMAGE_NO_COLOR

#ifndef LIBIMAGE_NO_GRAYSCALE

	void for_each_pixel(gray::Image const& image, u8_f const& func);

	void for_each_pixel(gray::View const& view, u8_f const& func);

	void for_each_xy(gray::Image const& image, xy_f const& func);

	void for_each_xy(gray::View const& view, xy_f const& func);

#endif // !LIBIMAGE_NO_GRAYSCALE
}


namespace libimage
{
	void fill(Image const& image, Pixel color);

	void fill(View const& view, Pixel color);



#ifndef LIBIMAGE_NO_GRAYSCALE

	void fill(gray::Image const& image, u8 gray);

	void fill(gray::View const& view, u8 gray);

#endif // !LIBIMAGE_NO_GRAYSCALE
}


/*  transform.hpp  */

namespace libimage
{


	/*** transform parallel ***/



#ifndef LIBIMAGE_NO_COLOR	

	void transform(Image const& src, Image const& dst, pixel_to_pixel_f const& func);

	void transform(Image const& src, View const& dst, pixel_to_pixel_f const& func);

	void transform(View const& src, Image const& dst, pixel_to_pixel_f const& func);

	void transform(View const& src, View const& dst, pixel_to_pixel_f const& func);


	void transform_in_place(Image const& src_dst, pixel_to_pixel_f const& func);

	void transform_in_place(View const& src_dst, pixel_to_pixel_f const& func);


	void transform_alpha(Image const& src_dst, pixel_to_u8_f const& func);

	void transform_alpha(View const& src_dst, pixel_to_u8_f const& func);




#endif // !LIBIMAGE_NO_COLOR

#ifndef LIBIMAGE_NO_GRAYSCALE

	lookup_table_t to_lookup_table(u8_to_u8_f const& func);

	void transform(gray::Image const& src, gray::Image const& dst, lookup_table_t const& lut);

	void transform(gray::Image const& src, gray::View const& dst, lookup_table_t const& lut);

	void transform(gray::View const& src, gray::Image const& dst, lookup_table_t const& lut);

	void transform(gray::View const& src, gray::View const& dst, lookup_table_t const& lut);


	void transform(gray::Image const& src, gray::Image const& dst, u8_to_u8_f const& func);

	void transform(gray::Image const& src, gray::View const& dst, u8_to_u8_f const& func);

	void transform(gray::View const& src, gray::Image const& dst, u8_to_u8_f const& func);

	void transform(gray::View const& src, gray::View const& dst, u8_to_u8_f const& func);


	void transform_in_place(gray::Image const& src_dst, lookup_table_t const& lut);

	void transform_in_place(gray::View const& src_dst, lookup_table_t const& lut);


	void transform_in_place(gray::Image const& src_dst, u8_to_u8_f const& func);

	void transform_in_place(gray::View const& src_dst, u8_to_u8_f const& func);

#endif // !LIBIMAGE_NO_GRAYSCALE

#ifndef LIBIMAGE_NO_COLOR
#ifndef LIBIMAGE_NO_GRAYSCALE

	void transform(Image const& src, gray::Image const& dst, pixel_to_u8_f const& func);

	void transform(Image const& src, gray::View const& dst, pixel_to_u8_f const& func);

	void transform(View const& src, gray::Image const& dst, pixel_to_u8_f const& func);

	void transform(View const& src, gray::View const& dst, pixel_to_u8_f const& func);

#endif // !LIBIMAGE_NO_GRAYSCALE
#endif // !LIBIMAGE_NO_COLOR


}


/*  copy.hpp  */

namespace libimage
{

#ifndef LIBIMAGE_NO_COLOR	

	void copy(Image const& src, Image const& dst);

	void copy(Image const& src, View const& dst);

	void copy(View const& src, Image const& dst);

	void copy(View const& src, View const& dst);

#endif // !LIBIMAGE_NO_COLOR

#ifndef LIBIMAGE_NO_GRAYSCALE

	lookup_table_t to_lookup_table(u8_to_u8_f const& func);


	void copy(gray::Image const& src, gray::Image const& dst);

	void copy(gray::Image const& src, gray::View const& dst);

	void copy(gray::View const& src, gray::Image const& dst);

	void copy(gray::View const& src, gray::View const& dst);


#endif // !LIBIMAGE_NO_GRAYSCALE


	
}


/*  alpha_blend.hpp  */

#ifndef LIBIMAGE_NO_COLOR

namespace libimage
{

	void alpha_blend(Image const& src, Image const& current, Image const& dst);

	void alpha_blend(Image const& src, Image const& current, View const& dst);

	void alpha_blend(Image const& src, View const& current, Image const& dst);

	void alpha_blend(Image const& src, View const& current, View const& dst);

	void alpha_blend(View const& src, Image const& current, Image const& dst);

	void alpha_blend(View const& src, Image const& current, View const& dst);

	void alpha_blend(View const& src, View const& current, Image const& dst);

	void alpha_blend(View const& src, View const& current, View const& dst);


	void alpha_blend(Image const& src, Image const& current_dst);

	void alpha_blend(Image const& src, View const& current_dst);

	void alpha_blend(View const& src, Image const& current_dst);

	void alpha_blend(View const& src, View const& current_dst);



}

#endif // !LIBIMAGE_NO_COLOR


/*  grayscale.hpp  */

#ifndef LIBIMAGE_NO_GRAYSCALE
#ifndef LIBIMAGE_NO_COLOR

namespace libimage
{

	void grayscale(Image const& src, gray::Image const& dst);

	void grayscale(Image const& src, gray::View const& dst);

	void grayscale(View const& src, gray::Image const& dst);

	void grayscale(View const& src, gray::View const& dst);


	void alpha_grayscale(Image const& src);

	void alpha_grayscale(View const& src);


}

#endif // !LIBIMAGE_NO_COLOR
#endif // !LIBIMAGE_NO_GRAYSCALE


/*  binary.hpp  */

#ifndef LIBIMAGE_NO_GRAYSCALE

namespace libimage
{
	void binarize(gray::Image const& src, gray::Image const& dst, u8_to_bool_f const& cond);

	void binarize(gray::Image const& src, gray::View const& dst, u8_to_bool_f const& cond);

	void binarize(gray::View const& src, gray::Image const& dst, u8_to_bool_f const& cond);

	void binarize(gray::View const& src, gray::View const& dst, u8_to_bool_f const& cond);


	void binarize_in_place(gray::Image const& src_dst, u8_to_bool_f const& func);

	void binarize_in_place(gray::View const& src_dst, u8_to_bool_f const& func);


#ifndef LIBIMAGE_NO_COLOR

	void binarize(Image const& src, gray::Image const& dst, pixel_to_bool_f const& cond);

	void binarize(Image const& src, gray::View const& dst, pixel_to_bool_f const& cond);

	void binarize(View const& src, gray::Image const& dst, pixel_to_bool_f const& cond);

	void binarize(View const& src, gray::View const& dst, pixel_to_bool_f const& cond);

#endif // !LIBIMAGE_NO_COLOR


	Point2Du32 centroid(gray::Image const& src);

	Point2Du32 centroid(gray::Image const& src, u8_to_bool_f const& func);


	Point2Du32 centroid(gray::View const& src);

	Point2Du32 centroid(gray::View const& src, u8_to_bool_f const& func);

	
	void skeleton(gray::Image const& src, gray::Image const& dst);

	void skeleton(gray::Image const& src, gray::View const& dst);

	void skeleton(gray::View const& src, gray::Image const& dst);

	void skeleton(gray::View const& src, gray::View const& dst);

}


// threshold overloads
namespace libimage
{
	inline void binarize_th(gray::Image const& src, gray::Image const& dst, u8 th) { binarize(src, dst, [&th](u8 p) { return p >= th; }); }

	inline void binarize_th(gray::Image const& src, gray::View const& dst, u8 th) { binarize(src, dst, [&th](u8 p) { return p >= th; }); }

	inline void binarize_th(gray::View const& src, gray::Image const& dst, u8 th) { binarize(src, dst, [&th](u8 p) { return p >= th; }); }

	inline void binarize_th(gray::View const& src, gray::View const& dst, u8 th) { binarize(src, dst, [&th](u8 p) { return p >= th; }); }


	inline void binarize_th(gray::Image const& src, gray::Image const& dst, u8 min_th, u8 max_th) { binarize(src, dst, [&](u8 p) { return min_th <= p && p <= max_th; }); }

	inline void binarize_th(gray::Image const& src, gray::View const& dst, u8 min_th, u8 max_th) { binarize(src, dst, [&](u8 p) { return min_th <= p && p <= max_th; }); }

	inline void binarize_th(gray::View const& src, gray::Image const& dst, u8 min_th, u8 max_th) { binarize(src, dst, [&](u8 p) { return min_th <= p && p <= max_th; }); }

	inline void binarize_th(gray::View const& src, gray::View const& dst, u8 min_th, u8 max_th) { binarize(src, dst, [&](u8 p) { return min_th <= p && p <= max_th; }); }


}

#endif // !LIBIMAGE_NO_GRAYSCALE


/*  contrast.hpp  */

namespace libimage
{

#ifndef LIBIMAGE_NO_GRAYSCALE

	void contrast(gray::Image const& src, gray::Image const& dst, u8 src_low, u8 src_high);

	void contrast(gray::Image const& src, gray::View const& dst, u8 src_low, u8 src_high);

	void contrast(gray::View const& src, gray::Image const& dst, u8 src_low, u8 src_high);

	void contrast(gray::View const& src, gray::View const& dst, u8 src_low, u8 src_high);


	void contrast_in_place(gray::Image const& src_dst, u8 src_low, u8 src_high);

	void contrast_in_place(gray::View const& src_dst, u8 src_low, u8 src_high);

#endif // !LIBIMAGE_NO_GRAYSCALE


}



/*  blur.hpp  */

#ifndef LIBIMAGE_NO_GRAYSCALE

namespace libimage
{


#ifndef LIBIMAGE_NO_GRAYSCALE


	void blur(gray::Image const& src, gray::Image const& dst);

	void blur(gray::Image const& src, gray::View const& dst);

	void blur(gray::View const& src, gray::Image const& dst);

	void blur(gray::View const& src, gray::View const& dst);


#endif // !LIBIMAGE_NO_GRAYSCALE


}

#endif // !LIBIMAGE_NO_GRAYSCALE


/*  edges.hpp  */

#ifndef LIBIMAGE_NO_GRAYSCALE

namespace libimage
{


#ifndef LIBIMAGE_NO_GRAYSCALE

	void edges(gray::Image const& src, gray::Image const& dst, u8_to_bool_f const& cond);

	void edges(gray::Image const& src, gray::View const& dst, u8_to_bool_f const& cond);

	void edges(gray::View const& src, gray::Image const& dst, u8_to_bool_f const& cond);

	void edges(gray::View const& src, gray::View const& dst, u8_to_bool_f const& cond);


#endif // !LIBIMAGE_NO_GRAYSCALE


}

#endif // !LIBIMAGE_NO_GRAYSCALE


/*  gradients.hpp  */

#ifndef LIBIMAGE_NO_GRAYSCALE

namespace libimage
{


#ifndef LIBIMAGE_NO_GRAYSCALE


	void gradients(gray::Image const& src, gray::Image const& dst);

	void gradients(gray::Image const& src, gray::View const& dst);

	void gradients(gray::View const& src, gray::Image const& dst);

	void gradients(gray::View const& src, gray::View const& dst);

#endif // !LIBIMAGE_NO_GRAYSCALE


}

#endif // !LIBIMAGE_NO_GRAYSCALE


/*  rotate.hpp  */

namespace libimage
{


#ifndef LIBIMAGE_NO_COLOR

	void rotate(Image const& src, Image const& dst, u32 origin_x, u32 origin_y, r32 theta);

	void rotate(Image const& src, View const& dst, u32 origin_x, u32 origin_y, r32 theta);

	void rotate(View const& src, Image const& dst, u32 origin_x, u32 origin_y, r32 theta);

	void rotate(View const& src, View const& dst, u32 origin_x, u32 origin_y, r32 theta);


	void rotate(Image const& src, Image const& dst, Point2Du32 origin, r32 theta);

	void rotate(Image const& src, View const& dst, Point2Du32 origin, r32 theta);

	void rotate(View const& src, Image const& dst, Point2Du32 origin, r32 theta);

	void rotate(View const& src, View const& dst, Point2Du32 origin, r32 theta);

#endif // !LIBIMAGE_NO_COLOR

#ifndef LIBIMAGE_NO_GRAYSCALE

	void rotate(gray::Image const& src, gray::Image const& dst, u32 origin_x, u32 origin_y, r32 theta);

	void rotate(gray::Image const& src, gray::View const& dst, u32 origin_x, u32 origin_y, r32 theta);

	void rotate(gray::View const& src, gray::Image const& dst, u32 origin_x, u32 origin_y, r32 theta);

	void rotate(gray::View const& src, gray::View const& dst, u32 origin_x, u32 origin_y, r32 theta);


	void rotate(gray::Image const& src, gray::Image const& dst, Point2Du32 origin, r32 theta);

	void rotate(gray::Image const& src, gray::View const& dst, Point2Du32 origin, r32 theta);

	void rotate(gray::View const& src, gray::Image const& dst, Point2Du32 origin, r32 theta);

	void rotate(gray::View const& src, gray::View const& dst, Point2Du32 origin, r32 theta);


#endif // !LIBIMAGE_NO_GRAYSCALE


}

