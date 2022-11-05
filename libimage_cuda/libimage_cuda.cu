#include "libimage.hpp"
#include "./device/cuda_def.cuh"
#include "./device/device.hpp"

#include <algorithm>

using RGB = libimage::RGB;
using RGBA = libimage::RGBA;
using GA = libimage::GA;
using HSV = libimage::HSV;
using XY = libimage::XY;

using View1r32 = libimage::View1r32;

template <size_t N>
using PixelCHr32 = libimage::PixelCHr32<N>;

template <size_t N>
using ViewCHr32 = libimage::ViewCHr32<N>;

using View4r32 = ViewCHr32<4>;
using View3r32 = ViewCHr32<3>;
using View2r32 = ViewCHr32<2>;

using Pixel4r32 = PixelCHr32<4>;
using Pixel3r32 = PixelCHr32<3>;
using Pixel2r32 = PixelCHr32<2>;

using ViewRGBAr32 = View4r32;
using ViewRGBr32 = View3r32;
using ViewHSVr32 = View3r32;

using PixelRGBAr32 = libimage::PixelRGBAr32;
using PixelRGBr32 = libimage::PixelRGBr32;
using PixelHSVr32 = libimage::PixelHSVr32;


namespace gpuf
{
	template <typename T>
	GPU_CONSTEXPR_FUNCTION
	inline int id_cast(T channel)
	{
		return static_cast<int>(channel);
	}
}


/* row_begin */

namespace gpuf
{
	GPU_FUNCTION
	static r32* row_begin(View1r32 const& view, u32 y)
	{
		assert(y < view.height);

		auto offset = (view.y_begin + y) * view.image_width + view.x_begin;

		auto ptr = view.image_data + (u64)(offset);
		assert(ptr);

		return ptr;
	}


	template <size_t N>
	GPU_FUNCTION
	static PixelCHr32<N> row_begin(ViewCHr32<N> const& view, u32 y)
	{
		assert(y < view.height);

		auto offset = (view.y_begin + y) * view.image_width + view.x_begin;

		PixelCHr32<N> p{};

		for (u32 ch = 0; ch < N; ++ch)
		{
			p.channels[ch] = view.image_channel_data[ch] + offset;
		}

		return p;
	}


	GPU_FUNCTION
	static PixelRGBr32 rgb_row_begin(ViewRGBr32 const& view, u32 y)
	{
		constexpr auto R = gpuf::id_cast(RGB::R);
		constexpr auto G = gpuf::id_cast(RGB::G);
		constexpr auto B = gpuf::id_cast(RGB::B);

		assert(y < view.height);

		auto offset = (view.y_begin + y) * view.image_width + view.x_begin;

		PixelRGBr32 p{};

		p.rgb.R = view.image_channel_data[R] + offset;
		p.rgb.G = view.image_channel_data[G] + offset;
		p.rgb.B = view.image_channel_data[B] + offset;

		return p;
	}


	GPU_FUNCTION
	static PixelHSVr32 hsv_row_begin(ViewHSVr32 const& view, u32 y)
	{
		constexpr auto H = gpuf::id_cast(HSV::H);
		constexpr auto S = gpuf::id_cast(HSV::S);
		constexpr auto V = gpuf::id_cast(HSV::V);

		assert(y < view.height);

		auto offset = (view.y_begin + y) * view.image_width + view.x_begin;

		PixelHSVr32 p{};

		p.hsv.H = view.image_channel_data[H] + offset;
		p.hsv.S = view.image_channel_data[S] + offset;
		p.hsv.V = view.image_channel_data[V] + offset;

		return p;
	}


	GPU_FUNCTION
	static r32* row_offset_begin(View1r32 const& view, u32 y, int y_offset)
	{
		int y_eff = y + y_offset;

		auto offset = (view.y_begin + y_eff) * view.image_width + view.x_begin;

		auto ptr = view.image_data + (u64)(offset);
		assert(ptr);

		return ptr;
	}


	template <size_t N>
	GPU_FUNCTION
	static r32* channel_row_begin(ViewCHr32<N> const& view, u32 y, u32 ch)
	{
		assert(y < view.height);

		auto offset = (size_t)((view.y_begin + y) * view.image_width + view.x_begin);

		return view.image_channel_data[ch] + offset;
	}


	template <size_t N>
	GPU_FUNCTION
	static r32* channel_row_offset_begin(ViewCHr32<N> const& view, u32 y, int y_offset, u32 ch)
	{
		int y_eff = y + y_offset;

		auto offset = (size_t)((view.y_begin + y_eff) * view.image_width + view.x_begin);

		return view.image_channel_data[ch] + offset;
	}
}



/* xy_at */

namespace gpuf
{
	GPU_FUNCTION
	static r32* xy_at(View1r32 const& view, u32 x, u32 y)
	{
		assert(y < view.height);
		assert(x < view.width);

		return gpuf::row_begin(view, y) + x;
	}	


	template <size_t N>
	GPU_FUNCTION	
	static PixelCHr32<N> xy_at_n(ViewCHr32<N> const& view, u32 x, u32 y)
	{
		assert(y < view.height);
		assert(x < view.width);

		auto offset = (size_t)((view.y_begin + y) * view.image_width + view.x_begin) + x;

		PixelCHr32<N> p{};

		for (u32 ch = 0; ch < N; ++ch)
		{
			p.channels[ch] = view.image_channel_data[ch] + offset;
		}

		return p;
	}


	GPU_FUNCTION
	static Pixel4r32 xy_at(View4r32 const& view, u32 x, u32 y)
	{
		return gpuf::xy_at_n(view, x, y);
	}


	GPU_FUNCTION
	static Pixel3r32 xy_at(View3r32 const& view, u32 x, u32 y)
	{
		return gpuf::xy_at_n(view, x, y);
	}


	GPU_FUNCTION
	static Pixel2r32 xy_at(View2r32 const& view, u32 x, u32 y)
	{
		return gpuf::xy_at_n(view, x, y);
	}


	template <size_t N>
	GPU_FUNCTION	
	static r32* channel_xy_at(ViewCHr32<N> const& view, u32 x, u32 y, u32 ch)
	{
		assert(y < view.height);
		assert(x < view.width);

		auto offset = (size_t)((view.y_begin + y) * view.image_width + view.x_begin) + x;

		return view.image_channel_data[ch] + offset;
	}


	GPU_FUNCTION
	static PixelRGBAr32 rgba_xy_at(ViewRGBAr32 const& view, u32 x, u32 y)
	{
		constexpr auto R = gpuf::id_cast(RGBA::R);
		constexpr auto G = gpuf::id_cast(RGBA::G);
		constexpr auto B = gpuf::id_cast(RGBA::B);
		constexpr auto A = gpuf::id_cast(RGBA::A);

		assert(y < view.height);
		assert(x < view.width);

		auto offset = (size_t)((view.y_begin + y) * view.image_width + view.x_begin) + x;

		PixelRGBAr32 p{};

		p.rgba.R = view.image_channel_data[R] + offset;
		p.rgba.G = view.image_channel_data[G] + offset;
		p.rgba.B = view.image_channel_data[B] + offset;
		p.rgba.A = view.image_channel_data[A] + offset;

		return p;
	}


	GPU_FUNCTION
	PixelRGBr32 rgb_xy_at(ViewRGBr32 const& view, u32 x, u32 y)
	{
		constexpr auto R = gpuf::id_cast(RGB::R);
		constexpr auto G = gpuf::id_cast(RGB::G);
		constexpr auto B = gpuf::id_cast(RGB::B);

		assert(y < view.height);
		assert(x < view.width);

		auto offset = (size_t)((view.y_begin + y) * view.image_width + view.x_begin) + x;

		PixelRGBr32 p{};

		p.rgb.R = view.image_channel_data[R] + offset;
		p.rgb.G = view.image_channel_data[G] + offset;
		p.rgb.B = view.image_channel_data[B] + offset;

		return p;
	}


	GPU_FUNCTION
	PixelHSVr32 hsv_xy_at(ViewHSVr32 const& view, u32 x, u32 y)
	{
		constexpr auto H = gpuf::id_cast(HSV::H);
		constexpr auto S = gpuf::id_cast(HSV::S);
		constexpr auto V = gpuf::id_cast(HSV::V);

		assert(y < view.height);
		assert(x < view.width);

		auto offset = (size_t)((view.y_begin + y) * view.image_width + view.x_begin) + x;

		PixelHSVr32 p{};

		p.hsv.H = view.image_channel_data[H] + offset;
		p.hsv.S = view.image_channel_data[S] + offset;
		p.hsv.V = view.image_channel_data[V] + offset;

		return p;
	}
}


class HSVr32
{
public:
    r32 hue;
    r32 sat;
    r32 val;
};


class RGBr32
{
public:
    r32 red;
    r32 green;
    r32 blue;
};


namespace gpuf
{    

    GPU_FUNCTION
    static HSVr32 rgb_hsv(r32 r, r32 g, r32 b)
	{
		auto max = fmaxf(r, fmaxf(g, b));
		auto min = fminf(r, fmaxf(g, b));

		auto c = max - min;

		r32 value = max;

		r32 sat = max == 0 ? 0.0f : (c / value);

		r32 hue = 60.0f;

		if (max == min)
		{
			hue = 0.0f;
		}
		else if (max == r)
		{
			hue *= ((g - b) / c);
		}
		else if (max == g)
		{
			hue *= ((b - r) / c + 2);
		}
		else // max == b
		{
			hue *= ((r - g) / c + 4);
		}

		hue /= 360.0f;

		return { hue, sat, value };
	}


    GPU_FUNCTION
    static RGBr32 hsv_rgb(r32 h, r32 s, r32 v)
	{
		auto c = s * v;
		auto m = v - c;

		auto d = h * 360.0f / 60.0f;

		auto x = c * (1.0f - fabsf(fmodf(d, 2.0f) - 1.0f));

		auto r = m;
		auto g = m;
		auto b = m;

		switch (int(d))
		{
		case 0:
			r += c;
			g += x;
			break;
		case 1:
			r += x;
			g += c;
			break;
		case 2:
			g += c;
			b += x;
			break;
		case 3:
			g += x;
			b += c;
			break;
		case 4:
			r += x;
			b += c;
			break;
		default:
			r += c;
			b += x;
			break;
		}

		return { r, g, b };
	}
}


using ViewRGBr32 = libimage::ViewRGBr32;
using ViewHSVr32 = libimage::ViewHSVr32;


GPU_KERNAL
static void gpu_map_hsv(ViewRGBr32 src, ViewHSVr32 dst, u32 n_threads)
{
    int t = blockDim.x * blockIdx.x + threadIdx.x;
    if (t >= n_threads)
    {
        return;
    }

    
}


/* map_hsv */

namespace libimage
{
	void map_hsv(ViewHSVr32 const& device_src, Image const& host_dst)
	{

	}


	void map_hsv(Image const& host_src, ViewHSVr32 const& device_dst)
	{

	}


	void map_hsv(ViewHSVr32 const& device_src, View const& host_dst)
	{

	}


	void map_hsv(View const& host_src, ViewHSVr32 const& device_dst)
	{

	}


	void map_hsv(ViewRGBr32 const& src, ViewHSVr32 const& dst)
	{

	}
}