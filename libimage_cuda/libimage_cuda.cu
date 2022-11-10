#include "include.hpp"
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

using ViewRGBAr32 = libimage::ViewRGBAr32;
using ViewRGBr32 = libimage::ViewRGBr32;
using ViewHSVr32 = libimage::ViewHSVr32;
using ViewGAr32 = libimage::ViewGAr32;

using PixelRGBAr32 = libimage::PixelRGBAr32;
using PixelRGBr32 = libimage::PixelRGBr32;
using PixelHSVr32 = libimage::PixelHSVr32;
using PixelGAr32 = libimage::PixelGAr32;

using Pixel = libimage::Pixel;


constexpr int THREADS_PER_BLOCK = 512;

constexpr int calc_thread_blocks(u32 n_threads)
{
    return (n_threads + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
}


GPU_CONSTEXPR_FUNCTION r32 div16(int i) { return i / 16.0f; }

GPU_GLOBAL_CONSTANT r32 GAUSS_3X3[]
{
    div16(1), div16(2), div16(1),
    div16(2), div16(4), div16(2),
    div16(1), div16(2), div16(1),
};


GPU_CONSTEXPR_FUNCTION r32 div256(int i) { return i / 256.0f; }

GPU_GLOBAL_CONSTANT r32 GAUSS_5X5[]
{
    div256(1), div256(4),  div256(6),  div256(4),  div256(1),
    div256(4), div256(16), div256(24), div256(16), div256(4),
    div256(6), div256(24), div256(36), div256(24), div256(6),
    div256(4), div256(16), div256(24), div256(16), div256(4),
    div256(1), div256(4),  div256(6),  div256(4),  div256(1),
};


GPU_GLOBAL_CONSTANT r32 GRAD_X_3X3[]
{
    -0.2f,  0.0f,  0.2f,
	-0.6f,  0.0f,  0.6f,
	-0.2f,  0.0f,  0.2f,
};


GPU_GLOBAL_CONSTANT r32 GRAD_Y_3X3[]
{
    -0.2f, -0.6f, -0.2f,
	 0.0f,  0.0f,  0.0f,
	 0.2f,  0.6f,  0.2f,
};


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


class RGBAr32
{
public:
    r32 red;
    r32 green;
    r32 blue;
	r32 alpha;
};


class ChannelXY
{
public:
	u32 ch;
	u32 x;
	u32 y;
};


namespace libimage
{
	static RGBr32 to_RGBr32(RGBAu8 const& rgba8)
	{
		RGBr32 rgb32 {};

		rgb32.red = to_channel_r32(rgba8.red);
		rgb32.green = to_channel_r32(rgba8.green);
		rgb32.blue = to_channel_r32(rgba8.blue);

		return rgb32;
	}


	static RGBAr32 to_RGBAr32(RGBAu8 const& rgba8)
	{
		RGBAr32 rgba32{};

		rgba32.red = to_channel_r32(rgba8.red);
		rgba32.green = to_channel_r32(rgba8.green);
		rgba32.blue = to_channel_r32(rgba8.blue);
		rgba32.alpha = to_channel_r32(rgba8.alpha);

		return rgba32;
	}
}


namespace gpuf
{
	template <typename T>
	GPU_CONSTEXPR_FUNCTION
	inline int id_cast(T channel)
	{
		return static_cast<int>(channel);
	}
	

	template <class VIEW>
	GPU_FUNCTION
	static Point2Du32 get_thread_xy(VIEW const& view, u32 thread_id)
	{
		// n_threads = width * height
		Point2Du32 p{};

		p.y = thread_id / view.width;
		p.x = thread_id - p.y * view.width;

		return p;
	}


	template <size_t N>
	GPU_FUNCTION
	static ChannelXY get_thread_channel_xy(ViewCHr32<N> const& view, u32 thread_id)
	{
		// n_threads = N * width * height
		auto width = view.width;
		auto height = view.height;

		ChannelXY cxy{};

		cxy.ch = thread_id / (width * height);
		cxy.y = (thread_id - width * height * cxy.ch) / width;
		cxy.x = (thread_id - width * height * cxy.ch) - cxy.y * width;

		return cxy;
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
	static PixelRGBr32 rgb_xy_at(ViewRGBr32 const& view, u32 x, u32 y)
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
	static PixelHSVr32 hsv_xy_at(ViewHSVr32 const& view, u32 x, u32 y)
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


	GPU_FUNCTION
	static PixelGAr32 ga_xy_at(ViewGAr32 const& view, u32 x, u32 y)
	{
		constexpr auto G = gpuf::id_cast(GA::G);
		constexpr auto A = gpuf::id_cast(GA::A);

		assert(y < view.height);
		assert(x < view.width);

		auto offset = (size_t)((view.y_begin + y) * view.image_width + view.x_begin) + x;

		PixelGAr32 p{};

		p.ga.G = view.image_channel_data[G] + offset;
		p.ga.A = view.image_channel_data[A] + offset;

		return p;
	}
}


/* select channel */

namespace gpuf
{
	template <size_t N>
	GPU_FUNCTION
	View1r32 select_channel(ViewCHr32<N> const& view, u32 ch)
	{
		View1r32 view1{};

		view1.image_width = view.image_width;
		view1.range = view.range;
		view1.width = view.width;
		view1.height = view.height;

		view1.image_data = view.image_channel_data[ch];

		return view1;
	}
}

/* map_hsv */

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


namespace gpu
{
	GPU_KERNAL
	static void map_rgb_hsv(ViewRGBr32 src, ViewHSVr32 dst, u32 n_threads)
	{
		int t = blockDim.x * blockIdx.x + threadIdx.x;
		if (t >= n_threads)
		{
			return;
		}

		assert(n_threads == src.width * src.height);

		auto xy = gpuf::get_thread_xy(src, t);

		auto s = gpuf::rgb_xy_at(src, xy.x, xy.y).rgb;

		auto hsv = gpuf::rgb_hsv(*s.R, *s.G, *s.B);
		
		auto d = gpuf::hsv_xy_at(dst, xy.x, xy.y).hsv;
		*d.H = hsv.hue;
		*d.S = hsv.sat;
		*d.V = hsv.val;
	}


	GPU_KERNAL
	static void map_hsv_rgb(ViewHSVr32 src, ViewRGBr32 dst, u32 n_threads)
	{
		auto t = blockDim.x * blockIdx.x + threadIdx.x;
		if (t >= n_threads)
		{
			return;
		}

		assert(n_threads == src.width * src.height);

		auto xy = gpuf::get_thread_xy(src, t);

		auto s = gpuf::hsv_xy_at(src, xy.x, xy.y).hsv;

		auto rgb = gpuf::hsv_rgb(*s.H, *s.S, *s.V);

		auto d = gpuf::rgb_xy_at(dst, xy.x, xy.y).rgb;
		*d.R = rgb.red;
		*d.G = rgb.green;
		*d.B = rgb.blue;
	}
}


namespace libimage
{
	void map_rgb_hsv(ViewRGBr32 const& src, ViewHSVr32 const& dst)
	{
		assert(verify(src, dst));

		auto const width = src.width;
		auto const height = src.height;

		auto const n_threads = width * height;
		auto const n_blocks = calc_thread_blocks(n_threads);
		constexpr auto block_size = THREADS_PER_BLOCK;

		cuda_launch_kernel(gpu::map_rgb_hsv, n_blocks, block_size, src, dst, n_threads);

		auto result = cuda::launch_success("gpu::map_rgb_hsv");
		assert(result);
	}
	

	void map_hsv_rgb(ViewHSVr32 const& src, ViewRGBr32 const& dst)
	{
		assert(src.image_channel_data[0]);
		assert(src.width);
		assert(src.height);
		assert(dst.image_channel_data[0]);
		assert(dst.width);
		assert(dst.height);
		assert(src.width == dst.width);
		assert(src.height == dst.height);

		assert(verify(src, dst));


		auto const width = src.width;
		auto const height = src.height;

		auto const n_threads = width * height;
		auto const n_blocks = calc_thread_blocks(n_threads);
		constexpr auto block_size = THREADS_PER_BLOCK;

		cuda_launch_kernel(gpu::map_hsv_rgb, n_blocks, block_size, src, dst, n_threads);

		auto result = cuda::launch_success("gpu::map_hsv_rgb");
		assert(result);
	}


	void map_rgb_hsv(View3r32 const& view)
	{
		assert(verify(view));

		auto const width = view.width;
		auto const height = view.height;

		auto const n_threads = width * height;
		auto const n_blocks = calc_thread_blocks(n_threads);
		constexpr auto block_size = THREADS_PER_BLOCK;

		cuda_launch_kernel(gpu::map_rgb_hsv, n_blocks, block_size, view, view, n_threads);

		auto result = cuda::launch_success("gpu::map_rgb_hsv in place");
		assert(result);
	}


	void map_hsv_rgb(View3r32 const& view)
	{
		assert(verify(view));

		auto const width = view.width;
		auto const height = view.height;

		auto const n_threads = width * height;
		auto const n_blocks = calc_thread_blocks(n_threads);
		constexpr auto block_size = THREADS_PER_BLOCK;

		cuda_launch_kernel(gpu::map_hsv_rgb, n_blocks, block_size, view, view, n_threads);

		auto result = cuda::launch_success("gpu::map_hsv_rgb in place");
		assert(result);
	}
}


/* fill */

namespace gpu
{
	GPU_KERNAL
	static void fill_rgba(ViewRGBAr32 view, RGBAr32 color, u32 n_threads)
	{
		auto t = blockDim.x * blockIdx.x + threadIdx.x;
		if (t >= n_threads)
		{
			return;
		}

		assert(n_threads == view.width * view.height);

		auto xy = gpuf::get_thread_xy(view, t);

		auto p = gpuf::rgba_xy_at(view, xy.x, xy.y).rgba;
		*p.R = color.red;
		*p.G = color.green;
		*p.B = color.blue;
		*p.A = color.alpha;
	}


	GPU_KERNAL
	static void fill_rgb(ViewRGBr32 view, RGBr32 color, u32 n_threads)
	{
		auto t = blockDim.x * blockIdx.x + threadIdx.x;
		if (t >= n_threads)
		{
			return;
		}

		assert(n_threads == view.width * view.height);

		auto xy = gpuf::get_thread_xy(view, t);

		auto p = gpuf::rgb_xy_at(view, xy.x, xy.y).rgb;
		*p.R = color.red;
		*p.G = color.green;
		*p.B = color.blue;
	}


	GPU_KERNAL
	static void fill_1(View1r32 view, r32 gray, u32 n_threads)
	{
		auto t = blockDim.x * blockIdx.x + threadIdx.x;
		if (t >= n_threads)
		{
			return;
		}

		assert(n_threads = view.width * view.height);

		auto xy = gpuf::get_thread_xy(view, t);

		auto& p = *gpuf::xy_at(view, xy.x, xy.y);

		p = gray;
	}
}


namespace libimage
{
	void fill(ViewRGBAr32 const& view, Pixel color)
	{
		assert(verify(view));

		auto const width = view.width;
		auto const height = view.height;

		auto const n_threads = width * height;
		auto const n_blocks = calc_thread_blocks(n_threads);
		constexpr auto block_size = THREADS_PER_BLOCK;

		cuda_launch_kernel(gpu::fill_rgba, n_blocks, block_size, view, to_RGBAr32(color.rgba), n_threads);

		auto result = cuda::launch_success("gpu::fill_rgba");
		assert(result);
	}


	void fill(ViewRGBr32 const& view, Pixel color)
	{
		assert(verify(view));

		auto const width = view.width;
		auto const height = view.height;

		auto const n_threads = width * height;
		auto const n_blocks = calc_thread_blocks(n_threads);
		constexpr auto block_size = THREADS_PER_BLOCK;

		cuda_launch_kernel(gpu::fill_rgb, n_blocks, block_size, view, to_RGBr32(color.rgba), n_threads);

		auto result = cuda::launch_success("gpu::fill_rgb");
		assert(result);
	}


	void fill(View1r32 const& view, r32 gray32)
	{
		assert(verify(view));

		auto const width = view.width;
		auto const height = view.height;

		auto const n_threads = width * height;
		auto const n_blocks = calc_thread_blocks(n_threads);
		constexpr auto block_size = THREADS_PER_BLOCK;

		cuda_launch_kernel(gpu::fill_1, n_blocks, block_size, view, gray32, n_threads);

		auto result = cuda::launch_success("gpu::fill_1");
		assert(result);
	}


	void fill(View1r32 const& view, u8 gray)
	{
		assert(verify(view));

		auto const width = view.width;
		auto const height = view.height;

		auto const n_threads = width * height;
		auto const n_blocks = calc_thread_blocks(n_threads);
		constexpr auto block_size = THREADS_PER_BLOCK;

		cuda_launch_kernel(gpu::fill_1, n_blocks, block_size, view, to_channel_r32(gray), n_threads);

		auto result = cuda::launch_success("gpu::fill_1 - u8");
		assert(result);
	}
}


/* copy */

namespace gpuf
{
	GPU_FUNCTION
	static void copy(View1r32 const& src, View1r32 const& dst, u32 x, u32 y)
	{
		auto& s = *gpuf::xy_at(src, x, y);
		auto& d = *gpuf::xy_at(dst, x, y);

		d = s;
	}
}


namespace gpu
{
	GPU_KERNAL
	static void copy_1(View1r32 src, View1r32 dst, u32 n_threads)
	{
		auto t = blockDim.x * blockIdx.x + threadIdx.x;
		if (t >= n_threads)
		{
			return;
		}

		assert(n_threads == src.width * src.height);

		auto xy = gpuf::get_thread_xy(src, t);

		gpuf::copy(src, dst, xy.x, xy.y);
	}


	template <size_t N>
	GPU_KERNAL
	static void copy_n(ViewCHr32<N> src, ViewCHr32<N> dst, u32 n_threads)
	{
		auto t = blockDim.x * blockIdx.x + threadIdx.x;
		if (t >= n_threads)
		{
			return;
		}

		assert(n_threads == N * src.width * src.height);

		auto cxy = gpuf::get_thread_channel_xy(src, t);

		auto src_ch = gpuf::select_channel(src, cxy.ch);
		auto dst_ch = gpuf::select_channel(dst, cxy.ch);

		gpuf::copy(src_ch, dst_ch, cxy.x, cxy.y);
	}
}


namespace libimage
{
	void copy(View4r32 const& src, View4r32 const& dst)
	{
		assert(verify(src, dst));

		auto const width = src.width;
		auto const height = src.height;

		auto const n_threads = 4 * width * height;
		auto const n_blocks = calc_thread_blocks(n_threads);
		constexpr auto block_size = THREADS_PER_BLOCK;

		cuda_launch_kernel(gpu::copy_n, n_blocks, block_size, src, dst, n_threads);

		auto result = cuda::launch_success("gpu::copy_4");
		assert(result);
	}


	void copy(View3r32 const& src, View3r32 const& dst)
	{
		assert(verify(src, dst));

		auto const width = src.width;
		auto const height = src.height;

		auto const n_threads = 3 * width * height;
		auto const n_blocks = calc_thread_blocks(n_threads);
		constexpr auto block_size = THREADS_PER_BLOCK;

		cuda_launch_kernel(gpu::copy_n, n_blocks, block_size, src, dst, n_threads);

		auto result = cuda::launch_success("gpu::copy_3");
		assert(result);
	}


	void copy(View2r32 const& src, View2r32 const& dst)
	{
		assert(verify(src, dst));

		auto const width = src.width;
		auto const height = src.height;

		auto const n_threads = 2 * width * height;
		auto const n_blocks = calc_thread_blocks(n_threads);
		constexpr auto block_size = THREADS_PER_BLOCK;

		cuda_launch_kernel(gpu::copy_n, n_blocks, block_size, src, dst, n_threads);

		auto result = cuda::launch_success("gpu::copy_2");
		assert(result);
	}


	void copy(View1r32 const& src, View1r32 const& dst)
	{
		assert(verify(src, dst));

		auto const width = src.width;
		auto const height = src.height;

		auto const n_threads = width * height;
		auto const n_blocks = calc_thread_blocks(n_threads);
		constexpr auto block_size = THREADS_PER_BLOCK;

		cuda_launch_kernel(gpu::copy_1, n_blocks, block_size, src, dst, n_threads);

		auto result = cuda::launch_success("gpu::copy_1");
		assert(result);
	}
}


/* multiply */

namespace gpu
{
	GPU_KERNAL
	static void multiply(View1r32 view, r32 factor, u32 n_threads)
	{
		auto t = blockDim.x * blockIdx.x + threadIdx.x;
		if (t >= n_threads)
		{
			return;
		}

		assert(n_threads == view.width * view.height);

		auto xy = gpuf::get_thread_xy(view, t);

		auto& s = *gpuf::xy_at(view, xy.x, xy.y);

		s *= factor;
	}
}


namespace libimage
{
	void multiply(View1r32 const& view, r32 factor)
	{
		assert(verify(view));

		if (factor < 0.0f)
		{
			factor = 0.0f;
		}

		auto const width = view.width;
		auto const height = view.height;

		auto const n_threads = width * height;
		auto const n_blocks = calc_thread_blocks(n_threads);
		constexpr auto block_size = THREADS_PER_BLOCK;

		cuda_launch_kernel(gpu::multiply, n_blocks, block_size, view, factor, n_threads);

		auto result = cuda::launch_success("gpu::multiply");
		assert(result);
	}
}


/* grayscale */

namespace gpuf
{
	constexpr r32 COEFF_RED = 0.299f;
	constexpr r32 COEFF_GREEN = 0.587f;
	constexpr r32 COEFF_BLUE = 0.114f;


	GPU_CONSTEXPR_FUNCTION
	r32 rgb_grayscale_standard(r32 red, r32 green, r32 blue)
	{
		return COEFF_RED * red + COEFF_GREEN * green + COEFF_BLUE * blue;
	}
}


namespace gpu
{
	GPU_KERNAL
	static void grayscale(ViewRGBr32 src, View1r32 dst, u32 n_threads)
	{
		auto t = blockDim.x * blockIdx.x + threadIdx.x;
		if (t >= n_threads)
		{
			return;
		}

		assert(n_threads == src.width * src.height);

		auto xy = gpuf::get_thread_xy(src, t);

		auto s = gpuf::rgb_xy_at(src, xy.x, xy.y).rgb;
		auto d = gpuf::xy_at(dst, xy.x, xy.y);

		*d = gpuf::rgb_grayscale_standard(*s.R, *s.G, *s.B);
	}
}


namespace libimage
{
	void grayscale(ViewRGBr32 const& src, View1r32 const& dst)
	{
		assert(verify(src, dst));

		auto const width = src.width;
		auto const height = src.height;

		auto const n_threads = width * height;
		auto const n_blocks = calc_thread_blocks(n_threads);
		constexpr auto block_size = THREADS_PER_BLOCK;

		cuda_launch_kernel(gpu::grayscale, n_blocks, block_size, src, dst, n_threads);

		auto result = cuda::launch_success("gpu::grayscale");
		assert(result);
	}
}


/* alpha_blend */

namespace gpuf
{
	GPU_FUNCTION
	static r32 blend_linear(r32 lhs, r32 rhs, r32 alpha)
	{
		assert(alpha >= 0.0f);
		assert(alpha <= 1.0f);

		return alpha * lhs + (1.0f - alpha) * rhs;
	}
}


namespace gpu
{
	GPU_KERNAL
	static void alpha_blend_rgba(ViewRGBAr32 src, ViewRGBr32 cur, ViewRGBr32 dst, u32 n_threads)
	{
		auto t = blockDim.x * blockIdx.x + threadIdx.x;
		if (t >= n_threads)
		{
			return;
		}

		assert(n_threads == src.width * src.height);

		auto xy = gpuf::get_thread_xy(src, t);

		auto s = gpuf::rgba_xy_at(src, xy.x, xy.y).rgba;
		auto c = gpuf::rgb_xy_at(cur, xy.x, xy.y).rgb;
		auto d = gpuf::rgb_xy_at(dst, xy.x, xy.y).rgb;

		auto a = *s.A;

		*d.R = gpuf::blend_linear(*s.R, *c.R, a);
		*d.G = gpuf::blend_linear(*s.G, *c.G, a);
		*d.B = gpuf::blend_linear(*s.B, *c.B, a);
	}


	GPU_KERNAL
	static void alpha_blend_ga(ViewGAr32 src, View1r32 cur, View1r32 dst, u32 n_threads)
	{
		auto t = blockDim.x * blockIdx.x + threadIdx.x;
		if (t >= n_threads)
		{
			return;
		}

		assert(n_threads == src.width * src.height);

		auto xy = gpuf::get_thread_xy(src, t);

		auto s = gpuf::ga_xy_at(src, xy.x, xy.y).ga;
		auto c = gpuf::xy_at(cur, xy.x, xy.y);
		auto d = gpuf::xy_at(dst, xy.x, xy.y);

		*d = gpuf::blend_linear(*s.G, *c, *s.A);
	}
}


namespace libimage
{
	void alpha_blend(ViewRGBAr32 const& src, ViewRGBr32 const& cur, ViewRGBr32 const& dst)
	{
		assert(verify(src, dst));
		assert(verify(src, cur));

		auto const width = src.width;
		auto const height = src.height;

		auto const n_threads = width * height;
		auto const n_blocks = calc_thread_blocks(n_threads);
		constexpr auto block_size = THREADS_PER_BLOCK;

		cuda_launch_kernel(gpu::alpha_blend_rgba, n_blocks, block_size, src, cur, dst, n_threads);

		auto result = cuda::launch_success("gpu::alpha_blend_rgba");
		assert(result);
	}


	void alpha_blend(ViewGAr32 const& src, View1r32 const& cur, View1r32 const& dst)
	{
		assert(verify(src, dst));
		assert(verify(src, cur));

		auto const width = src.width;
		auto const height = src.height;

		auto const n_threads = width * height;
		auto const n_blocks = calc_thread_blocks(n_threads);
		constexpr auto block_size = THREADS_PER_BLOCK;

		cuda_launch_kernel(gpu::alpha_blend_ga, n_blocks, block_size, src, cur, dst, n_threads);

		auto result = cuda::launch_success("gpu::alpha_blend_ga");
		assert(result);
	}
}


/* threshold */

namespace gpu
{
	GPU_KERNAL
	static void threshold(View1r32 src, View1r32 dst, r32 min, r32 max, u32 n_threads)
	{
		auto t = blockDim.x * blockIdx.x + threadIdx.x;
		if (t >= n_threads)
		{
			return;
		}

		assert(n_threads == src.width * src.height);

		auto xy = gpuf::get_thread_xy(src, t);

		auto& s = *gpuf::xy_at(src, xy.x, xy.y);
		auto& d = *gpuf::xy_at(dst, xy.x, xy.y);

		d = s >= min && s <= max ? 1.0f : 0.0f;
	}
}


namespace libimage
{
	void threshold(View1r32 const& src, View1r32 const& dst, r32 min, r32 max)
	{
		assert(verify(src, dst));

		auto const width = src.width;
		auto const height = src.height;

		auto const n_threads = width * height;
		auto const n_blocks = calc_thread_blocks(n_threads);
		constexpr auto block_size = THREADS_PER_BLOCK;

		cuda_launch_kernel(gpu::threshold, n_blocks, block_size, src, dst, min, max, n_threads);

		auto result = cuda::launch_success("gpu::threshold");
		assert(result);
	}


	void threshold(View1r32 const& src_dst, r32 min, r32 max)
	{
		assert(verify(src_dst));

		auto const width = src_dst.width;
		auto const height = src_dst.height;

		auto const n_threads = width * height;
		auto const n_blocks = calc_thread_blocks(n_threads);
		constexpr auto block_size = THREADS_PER_BLOCK;

		cuda_launch_kernel(gpu::threshold, n_blocks, block_size, src_dst, src_dst, min, max, n_threads);

		auto result = cuda::launch_success("gpu::threshold in place");
		assert(result);
	}
}


/* contrast */

namespace gpuf
{
	GPU_FUNCTION
	static r32 lerp_clamp(r32 src_low, r32 src_high, r32 dst_low, r32 dst_high, r32 val)
	{
		if (val < src_low)
		{
			return dst_low;
		}
		else if (val > src_high)
		{
			return dst_high;
		}

		auto const ratio = (val - src_low) / (src_high - src_low);

		assert(ratio >= 0.0f);
		assert(ratio <= 1.0f);

		auto const diff = ratio * (dst_high - dst_low);

		return dst_low + diff;
	}
}


namespace gpu
{
	GPU_KERNAL
	static void contrast(View1r32 src, View1r32 dst, r32 min, r32 max, u32 n_threads)
	{
		auto t = blockDim.x * blockIdx.x + threadIdx.x;
		if (t >= n_threads)
		{
			return;
		}

		assert(n_threads == src.width * src.height);

		auto xy = gpuf::get_thread_xy(src, t);

		auto& s = *gpuf::xy_at(src, xy.x, xy.y);
		auto& d = *gpuf::xy_at(dst, xy.x, xy.y);

		d = gpuf::lerp_clamp(min, max, 0.0f, 1.0f, s);
	}
}


namespace libimage
{
	void contrast(View1r32 const& src, View1r32 const& dst, r32 min, r32 max)
	{
		assert(verify(src, dst));

		auto const width = src.width;
		auto const height = src.height;

		auto const n_threads = width * height;
		auto const n_blocks = calc_thread_blocks(n_threads);
		constexpr auto block_size = THREADS_PER_BLOCK;

		cuda_launch_kernel(gpu::contrast, n_blocks, block_size, src, dst, min, max, n_threads);

		auto result = cuda::launch_success("gpu::contrast");
		assert(result);
	}


	void contrast(View1r32 const& src_dst, r32 min, r32 max)
	{
		assert(verify(src_dst));

		auto const width = src_dst.width;
		auto const height = src_dst.height;

		auto const n_threads = width * height;
		auto const n_blocks = calc_thread_blocks(n_threads);
		constexpr auto block_size = THREADS_PER_BLOCK;

		cuda_launch_kernel(gpu::contrast, n_blocks, block_size, src_dst, src_dst, min, max, n_threads);

		auto result = cuda::launch_success("gpu::contrast in place");
		assert(result);
	}
}


/* blur */

namespace gpuf
{
	GPU_FUNCTION
	inline bool is_outer_edge(u32 width, u32 height, u32 x, u32 y)
	{
		return 
			y == 0 || 
			y == height - 1 || 
			x == 0 || 
			x == width - 1;
	}


	GPU_FUNCTION
	inline bool is_inner_edge(u32 width, u32 height, u32 x, u32 y)
	{
		return 
			y == 1 || 
			y == height - 2 || 
			x == 1 || 
			x == width - 2;
	}


	GPU_FUNCTION
	static r32 convolve_gauss_3x3(View1r32 const& view, u32 x, u32 y)
	{
		int const ry_begin = -1;
		int const ry_end = 2;
		int const rx_begin = -1;
		int const rx_end = 2;

		u32 w = 0;
		r32 acc = 0.0f;

		for (int ry = ry_begin; ry < ry_end; ++ry)
		{
			auto p = gpuf::row_offset_begin(view, y, ry);
			for (int rx = rx_begin; rx < rx_end; ++rx)
			{
				acc += (p + rx)[x] * GAUSS_3X3[w];
				++w;
			}
		}

		return acc;
	}


	GPU_FUNCTION
	static r32 convolve_gauss_5x5(View1r32 const& view, u32 x, u32 y)
	{
		int const ry_begin = -2;
		int const ry_end = 3;
		int const rx_begin = -2;
		int const rx_end = 3;

		u32 w = 0;
		r32 acc = 0.0f;

		for (int ry = ry_begin; ry < ry_end; ++ry)
		{
			auto p = gpuf::row_offset_begin(view, y, ry);
			for (int rx = rx_begin; rx < rx_end; ++rx)
			{
				acc += (p + rx)[x] * GAUSS_5X5[w];
				++w;
			}
		}

		return acc;
	}


	GPU_FUNCTION
	static void blur(View1r32 const& src, View1r32 const& dst, u32 x, u32 y)
	{
		auto& d = *gpuf::xy_at(dst, x, y);

		auto width = src.width;
		auto height = src.height;

		if (gpuf::is_outer_edge(width, height, x, y))
		{
			d = *gpuf::xy_at(src, x, y);
		}
		else if (gpuf::is_inner_edge(width, height, x, y))
		{
			d = gpuf::convolve_gauss_3x3(src, x, y);
		}
		else
		{
			d = gpuf::convolve_gauss_5x5(src, x, y);
		}
	}
}


namespace gpu
{
	GPU_KERNAL
	static void blur_1(View1r32 src, View1r32 dst, u32 n_threads)
	{
		auto t = blockDim.x * blockIdx.x + threadIdx.x;
		if (t >= n_threads)
		{
			return;
		}

		assert(n_threads == src.width * src.height);

		auto xy = gpuf::get_thread_xy(src, t);

		gpuf::blur(src, dst, xy.x, xy.y);
	}


	template <size_t N>
	GPU_KERNAL
	static void blur_n(ViewCHr32<N> src, ViewCHr32<N> dst, u32 n_threads)
	{
		auto t = blockDim.x * blockIdx.x + threadIdx.x;
		if (t >= n_threads)
		{
			return;
		}

		assert(n_threads == N * src.width * src.height);

		auto cxy = gpuf::get_thread_channel_xy(src, t);

		auto src_ch = gpuf::select_channel(src, cxy.ch);
		auto dst_ch = gpuf::select_channel(dst, cxy.ch);

		gpuf::blur(src_ch, dst_ch, cxy.x, cxy.y);
	}
}


namespace libimage
{
	void blur(View1r32 const& src, View1r32 const& dst)
	{
		assert(verify(src, dst));

		auto const width = src.width;
		auto const height = src.height;

		auto const n_threads = width * height;
		auto const n_blocks = calc_thread_blocks(n_threads);
		constexpr auto block_size = THREADS_PER_BLOCK;

		cuda_launch_kernel(gpu::blur_1, n_blocks, block_size, src, dst, n_threads);

		auto result = cuda::launch_success("gpu::blur_1");
		assert(result);
	}


	void blur(View3r32 const& src, View3r32 const& dst)
	{
		assert(verify(src, dst));

		auto const width = src.width;
		auto const height = src.height;

		auto const n_threads = 3 * width * height;
		auto const n_blocks = calc_thread_blocks(n_threads);
		constexpr auto block_size = THREADS_PER_BLOCK;

		cuda_launch_kernel(gpu::blur_n, n_blocks, block_size, src, dst, n_threads);

		auto result = cuda::launch_success("gpu::blur_3");
		assert(result);
	}
}


/* gradients */

namespace gpuf
{
	GPU_FUNCTION
	static Point2Dr32 gradient_xy_3x3(View1r32 const& view, u32 x, u32 y)
	{
		int const ry_begin = -1;
		int const ry_end = 2;
		int const rx_begin = -1;
		int const rx_end = 2;

		u32 w = 0;
		r32 gx = 0.0f;
		r32 gy = 0.0f;

		for (int ry = ry_begin; ry < ry_end; ++ry)
		{
			auto p = gpuf::row_offset_begin(view, y, ry);

			for (int rx = rx_begin; rx < rx_end; ++rx)
			{
				auto val = (p + rx)[x];
				gx += val * GRAD_X_3X3[w];
				gy += val * GRAD_Y_3X3[w];
				++w;
			}
		}

		Point2Dr32 p{};
		p.x = gx;
		p.y = gy;

		return p;
	}


	GPU_FUNCTION
	static void gradients_hypot(View1r32 const& src, View1r32 const& dst, u32 x, u32 y)
	{
		auto& d = *gpuf::xy_at(dst, x, y);

		auto width = src.width;
		auto height = src.height;

		if (gpuf::is_outer_edge(width, height, x, y))
		{
			d = 0.0f;
		}
		else
		{
			auto grad = gpuf::gradient_xy_3x3(src, x, y);
			d = hypotf(grad.x, grad.y);
		}
	}


	GPU_FUNCTION
	static void gradients_xy(View1r32 const& src, View2r32 const& xy_dst, u32 x, u32 y)
	{
		constexpr auto x_ch = gpuf::id_cast(XY::X);
		constexpr auto y_ch = gpuf::id_cast(XY::Y);

		auto& dx = *channel_xy_at(xy_dst, x, y, x_ch);
		auto& dy = *channel_xy_at(xy_dst, x, y, y_ch);

		auto width = src.width;
		auto height = src.height;

		if (gpuf::is_outer_edge(width, height, x, y))
		{
			dx = 0.0f;
			dy = 0.0f;
		}
		else
		{
			auto grad = gpuf::gradient_xy_3x3(src, x, y);
			dx = grad.x;
			dy = grad.y;
		}
	}

}


namespace gpu
{
	GPU_KERNAL
	static void gradients(View1r32 src, View1r32 dst, u32 n_threads)
	{
		auto t = blockDim.x * blockIdx.x + threadIdx.x;
		if (t >= n_threads)
		{
			return;
		}

		assert(n_threads == src.width * src.height);

		auto xy = gpuf::get_thread_xy(src, t);

		gpuf::gradients_hypot(src, dst, xy.x, xy.y);
	}


	GPU_KERNAL
	static void gradients_xy(View1r32 src, View2r32 dst, u32 n_threads)
	{
		auto t = blockDim.x * blockIdx.x + threadIdx.x;
		if (t >= n_threads)
		{
			return;
		}

		assert(n_threads == src.width * src.height);

		auto xy = gpuf::get_thread_xy(src, t);

		gpuf::gradients_xy(src, dst, xy.x, xy.y);
	}
}


namespace libimage
{
	void gradients(View1r32 const& src, View1r32 const& dst)
	{
		assert(verify(src, dst));

		auto const width = src.width;
		auto const height = src.height;

		auto const n_threads = width * height;
		auto const n_blocks = calc_thread_blocks(n_threads);
		constexpr auto block_size = THREADS_PER_BLOCK;

		cuda_launch_kernel(gpu::gradients, n_blocks, block_size, src, dst, n_threads);

		auto result = cuda::launch_success("gpu::gradients");
		assert(result);
	}


	void gradients_xy(View1r32 const& src, View2r32 const& xy_dst)
	{
		assert(verify(src, xy_dst));		

		auto const width = src.width;
		auto const height = src.height;

		auto const n_threads = width * height;
		auto const n_blocks = calc_thread_blocks(n_threads);
		constexpr auto block_size = THREADS_PER_BLOCK;

		cuda_launch_kernel(gpu::gradients_xy, n_blocks, block_size, src, xy_dst, n_threads);

		auto result = cuda::launch_success("gpu::gradients_xy");
		assert(result);
	}
}