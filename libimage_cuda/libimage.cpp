#include "libimage.hpp"

#include <cstdlib>

#ifndef LIBIMAGE_NO_PARALLEL

#include <execution>
// -ltbb


template <class LIST_T, class FUNC_T>
static void do_for_each(LIST_T const& list, FUNC_T const& func)
{
	std::for_each(std::execution::par, list.begin(), list.end(), func);
}

#else

template <class LIST_T, class FUNC_T>
static void do_for_each(LIST_T const& list, FUNC_T const& func)
{
	std::for_each(list.begin(), list.end(), func);
}

#endif // !LIBIMAGE_NO_PARALLEL


using id_func_t = std::function<void(u32)>;


class ThreadProcess
{
public:
	u32 thread_id = 0;
	id_func_t process;
};


using ProcList = std::array<ThreadProcess, N_THREADS>;


static ProcList make_proc_list(id_func_t const& id_func)
{
	ProcList list = { 0 };

	for (u32 i = 0; i < N_THREADS; ++i)
	{
		list[i] = { i, id_func };
	}

	return list;
}


static void execute_procs(ProcList const& list)
{
	auto const func = [](ThreadProcess const& t) { t.process(t.thread_id); };

	do_for_each(list, func);
}


static void process_rows(u32 height, id_func_t const& row_func)
{
	auto const rows_per_thread = height / N_THREADS;

	auto const thread_proc = [&](u32 id)
	{
		auto y_begin = id * rows_per_thread;
		auto y_end = (id == N_THREADS - 1 ? height : (id + 1) * rows_per_thread);

		for (u32 y = y_begin; y < y_end; ++y)
		{
			row_func(y);
		}
	};

	execute_procs(make_proc_list(thread_proc));
}

static constexpr std::array<r32, 256> channel_r32_lut()
{
	std::array<r32, 256> lut = {};

	for (u32 i = 0; i < 256; ++i)
	{
		lut[i] = i / 255.0f;
	}

	return lut;
}


static constexpr r32 to_channel_r32(u8 value)
{
	constexpr auto lut = channel_r32_lut();

	return lut[value];
}


static constexpr u8 to_channel_u8(r32 value)
{
	if (value < 0.0f)
	{
		value = 0.0f;
	}
	else if (value > 1.0f)
	{
		value = 1.0f;
	}

	return (u8)(u32)(value * 255 + 0.5f);
}


static r32 lerp_to_r32(u8 value, r32 min, r32 max)
{
	assert(min < max);

	return min + (value / 255.0f) * (max - min);
}


static u8 lerp_to_u8(r32 value, r32 min, r32 max)
{
	assert(min < max);
	assert(value >= min);
	assert(value <= max);

	if (value < min)
	{
		value = min;
	}
	else if (value > max)
	{
		value = max;
	}

	auto ratio = (value - min) / (max - min);

	return (u8)(u32)(ratio * 255 + 0.5f);
}



/* verify */

#ifndef NDEBUG

namespace libimage
{
	static bool verify(Image const& image)
	{
		return image.width && image.height && image.data;
	}


	static bool verify(View const& view)
	{
		return view.image_width && view.width && view.height && view.image_data;
	}


	template <size_t N>
	static bool verify(ViewCHr32<N> const& view)
	{
		return view.image_width && view.width && view.height && view.image_channel_data[0];
	}


	static bool verify(gray::Image const& image)
	{
		return image.width && image.height && image.data;
	}


	static bool verify(gray::View const& view)
	{
		return view.image_width && view.width && view.height && view.image_data;
	}


	static bool verify(View1r32 const& view)
	{
		return view.image_width && view.width && view.height && view.image_data;
	}


	template <class IMG_A, class IMG_B>
	static bool verify(IMG_A const& lhs, IMG_B const& rhs)
	{
		return
			verify(lhs) && verify(rhs) &&
			lhs.width == rhs.width &&
			lhs.height == rhs.height;
	}


	template <class IMG>
	static bool verify(IMG const& image, Range2Du32 const& range)
	{
		return
			verify(image) &&
			range.x_begin < range.x_end &&
			range.y_begin < range.y_end &&
			range.x_begin < image.width &&
			range.x_end <= image.width &&
			range.y_begin < image.height &&
			range.y_end <= image.height;
	}
}

#endif // !NDEBUG


/* platform image */

namespace libimage
{
	static constexpr Pixel to_pixel(r32 r, r32 g, r32 b, r32 a)
	{
		auto red = to_channel_u8(r);
		auto green = to_channel_u8(g);
		auto blue = to_channel_u8(b);
		auto alpha = to_channel_u8(a);

		return to_pixel(red, green, blue, alpha);
	}


	static constexpr Pixel to_pixel(r32 r, r32 g, r32 b)
	{
		auto red = to_channel_u8(r);
		auto green = to_channel_u8(g);
		auto blue = to_channel_u8(b);
		u8 alpha = 255;

		return to_pixel(red, green, blue, alpha);
	}


	void make_image(Image& image, u32 width, u32 height)
	{
		assert(width);
		assert(height);

		image.data = (Pixel*)malloc(sizeof(Pixel) * width * height);
		assert(image.data);

		image.width = width;
		image.height = height;
	}


	void destroy_image(Image& image)
	{
		if (image.data != nullptr)
		{
			free(image.data);
			image.data = nullptr;
		}
	}


	Pixel* row_begin(Image const& image, u32 y)
	{
		assert(image.width);
		assert(image.height);
		assert(image.data);
		assert(y < image.height);

		auto offset = y * image.width;

		auto ptr = image.data + (u64)(offset);
		assert(ptr);

		return ptr;
	}


	Pixel* xy_at(Image const& image, u32 x, u32 y)
	{
		assert(y < image.height);
		assert(x < image.width);

		return row_begin(image, y) + x;
	}


	View make_view(Image const& image)
	{
		assert(image.width);
		assert(image.height);
		assert(image.data);

		View view;

		view.image_data = image.data;
		view.image_width = image.width;
		view.x_begin = 0;
		view.y_begin = 0;
		view.x_end = image.width;
		view.y_end = image.height;
		view.width = image.width;
		view.height = image.height;

		return view;
	}


	View sub_view(Image const& image, Range2Du32 const& range)
	{
		assert(image.width);
		assert(image.height);
		assert(image.data);

		View view;

		view.image_data = image.data;
		view.image_width = image.width;
		view.range = range;
		view.width = range.x_end - range.x_begin;
		view.height = range.y_end - range.y_begin;

		assert(view.width);
		assert(view.height);

		return view;
	}


	View sub_view(View const& view, Range2Du32 const& range)
	{
		assert(view.width);
		assert(view.height);
		assert(view.image_data);

		assert(range.x_begin >= view.x_begin);
		assert(range.x_end <= view.x_end);
		assert(range.y_begin >= view.y_begin);
		assert(range.y_end <= view.y_end);

		View sub_view;

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


	Pixel* row_begin(View const& view, u32 y)
	{
		assert(y < view.height);

		auto offset = (view.y_begin + y) * view.image_width + view.x_begin;

		auto ptr = view.image_data + (u64)(offset);
		assert(ptr);

		return ptr;
	}


	Pixel* xy_at(View const& view, u32 x, u32 y)
	{
		assert(y < view.height);
		assert(x < view.width);

		return row_begin(view, y) + x;
	}


	void make_image(gray::Image& image, u32 width, u32 height)
	{
		assert(width);
		assert(height);

		image.data = (u8*)malloc(sizeof(u8) * width * height);
		assert(image.data);

		image.width = width;
		image.height = height;
	}


	void destroy_image(gray::Image& image)
	{
		if (image.data != nullptr)
		{
			free(image.data);
			image.data = nullptr;
		}
	}


	u8* row_begin(gray::Image const& image, u32 y)
	{
		assert(image.width);
		assert(image.height);
		assert(image.data);
		assert(y < image.height);

		auto offset = y * image.width;

		auto ptr = image.data + (u64)(offset);
		assert(ptr);

		return ptr;
	}


	u8* xy_at(gray::Image const& image, u32 x, u32 y)
	{
		assert(y < image.height);
		assert(x < image.width);

		return row_begin(image, y) + x;
	}


	gray::View make_view(gray::Image const& image)
	{
		assert(verify(image));

		gray::View view;

		view.image_data = image.data;
		view.image_width = image.width;
		view.x_begin = 0;
		view.y_begin = 0;
		view.x_end = image.width;
		view.y_end = image.height;
		view.width = image.width;
		view.height = image.height;

		assert(verify(view));

		return view;
	}


	gray::View sub_view(gray::Image const& image, Range2Du32 const& range)
	{
		assert(verify(image, range));

		gray::View view;

		view.image_data = image.data;
		view.image_width = image.width;
		view.range = range;
		view.width = range.x_end - range.x_begin;
		view.height = range.y_end - range.y_begin;

		assert(verify(view));

		return view;
	}


	gray::View sub_view(gray::View const& view, Range2Du32 const& range)
	{
		assert(verify(view, range));

		gray::View sub_view;

		sub_view.image_data = view.image_data;
		sub_view.image_width = view.image_width;
		sub_view.x_begin = view.x_begin + range.x_begin;
		sub_view.y_begin = view.y_begin + range.y_begin;
		sub_view.x_end = view.x_begin + range.x_end;
		sub_view.y_end = view.y_begin + range.y_end;
		sub_view.width = range.x_end - range.x_begin;
		sub_view.height = range.y_end - range.y_begin;

		assert(verify(sub_view));

		return sub_view;
	}


	u8* row_begin(gray::View const& view, u32 y)
	{
		assert(y < view.height);

		auto offset = (view.y_begin + y) * view.image_width + view.x_begin;

		auto ptr = view.image_data + (u64)(offset);
		assert(ptr);

		return ptr;
	}


	u8* xy_at(gray::View const& view, u32 x, u32 y)
	{
		assert(y < view.height);
		assert(x < view.width);

		return row_begin(view, y) + x;
	}
}


/* planar */

namespace libimage
{
	ViewRGBr32 make_rgb_view(ViewRGBAr32 const& view)
	{
		assert(verify(view));

		View3r32 view3;

		view3.image_width = view.image_width;
		view3.range = view.range;
		view3.width = view.width;
		view3.height = view.height;

		view3.image_channel_data[id_cast(RGB::R)] = view.image_channel_data[id_cast(RGBA::R)];
		view3.image_channel_data[id_cast(RGB::G)] = view.image_channel_data[id_cast(RGBA::G)];
		view3.image_channel_data[id_cast(RGB::B)] = view.image_channel_data[id_cast(RGBA::B)];

		assert(verify(view3));

		return view3;
	}
}


/* row_begin */

namespace libimage
{
	static r32* row_begin(View1r32 const& view, u32 y)
	{
		assert(y < view.height);

		auto offset = (view.y_begin + y) * view.image_width + view.x_begin;

		auto ptr = view.image_data + (u64)(offset);
		assert(ptr);

		return ptr;
	}


	template <size_t N>
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

/*
	static Pixel4r32 row_begin(View4r32 const& view, u32 y)
	{
		return row_begin_n(view, y);
	}


	static Pixel3r32 row_begin(View3r32 const& view, u32 y)
	{
		return row_begin_n(view, y);
	}


	static Pixel2r32 row_begin(View2r32 const& view, u32 y)
	{
		return row_begin_n(view, y);
	}*/

/*
	static PixelRGBAr32 rgba_row_begin(ViewRGBAr32 const& view, u32 y)
	{
		assert(y < view.height);

		auto offset = (view.y_begin + y) * view.image_width + view.x_begin;

		PixelRGBAr32 p{};

		p.rgba.R = view.image_channel_data[id_cast(RGBA::R)] + offset;
		p.rgba.G = view.image_channel_data[id_cast(RGBA::G)] + offset;
		p.rgba.B = view.image_channel_data[id_cast(RGBA::B)] + offset;
		p.rgba.A = view.image_channel_data[id_cast(RGBA::A)] + offset;

		return p;
	}*/


	static PixelRGBr32 rgb_row_begin(ViewRGBr32 const& view, u32 y)
	{
		assert(y < view.height);

		auto offset = (view.y_begin + y) * view.image_width + view.x_begin;

		PixelRGBr32 p{};

		p.rgb.R = view.image_channel_data[id_cast(RGB::R)] + offset;
		p.rgb.G = view.image_channel_data[id_cast(RGB::G)] + offset;
		p.rgb.B = view.image_channel_data[id_cast(RGB::B)] + offset;

		return p;
	}


	static PixelHSVr32 hsv_row_begin(ViewHSVr32 const& view, u32 y)
	{
		assert(y < view.height);

		auto offset = (view.y_begin + y) * view.image_width + view.x_begin;

		PixelHSVr32 p{};

		p.hsv.H = view.image_channel_data[id_cast(HSV::H)] + offset;
		p.hsv.S = view.image_channel_data[id_cast(HSV::S)] + offset;
		p.hsv.V = view.image_channel_data[id_cast(HSV::V)] + offset;

		return p;
	}


	static r32* row_offset_begin(View1r32 const& view, u32 y, int y_offset)
	{
		int y_eff = y + y_offset;

		auto offset = (view.y_begin + y_eff) * view.image_width + view.x_begin;

		auto ptr = view.image_data + (u64)(offset);
		assert(ptr);

		return ptr;
	}


	template <size_t N>
	static r32* channel_row_begin(ViewCHr32<N> const& view, u32 y, u32 ch)
	{
		assert(y < view.height);

		auto offset = (size_t)((view.y_begin + y) * view.image_width + view.x_begin);

		return view.image_channel_data[ch] + offset;
	}


	template <size_t N>
	static r32* channel_row_offset_begin(ViewCHr32<N> const& view, u32 y, int y_offset, u32 ch)
	{
		int y_eff = y + y_offset;

		auto offset = (size_t)((view.y_begin + y_eff) * view.image_width + view.x_begin);

		return view.image_channel_data[ch] + offset;
	}
}


/* xy_at */

namespace libimage
{
	r32* xy_at(View1r32 const& view, u32 x, u32 y)
	{
		assert(y < view.height);
		assert(x < view.width);

		return row_begin(view, y) + x;
	}	


	template <size_t N>
	PixelCHr32<N> xy_at_n(ViewCHr32<N> const& view, u32 x, u32 y)
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


	Pixel4r32 xy_at(View4r32 const& view, u32 x, u32 y)
	{
		return xy_at_n(view, x, y);
	}


	Pixel3r32 xy_at(View3r32 const& view, u32 x, u32 y)
	{
		return xy_at_n(view, x, y);
	}


	Pixel2r32 xy_at(View2r32 const& view, u32 x, u32 y)
	{
		return xy_at_n(view, x, y);
	}


	template <size_t N>
	static r32* channel_xy_at(ViewCHr32<N> const& view, u32 x, u32 y, u32 ch)
	{
		assert(y < view.height);
		assert(x < view.width);

		auto offset = (size_t)((view.y_begin + y) * view.image_width + view.x_begin) + x;

		return view.image_channel_data[ch] + offset;
	}


	PixelRGBAr32 rgba_xy_at(ViewRGBAr32 const& view, u32 x, u32 y)
	{
		assert(y < view.height);
		assert(x < view.width);

		auto offset = (size_t)((view.y_begin + y) * view.image_width + view.x_begin) + x;

		PixelRGBAr32 p{};

		p.rgba.R = view.image_channel_data[id_cast(RGBA::R)] + offset;
		p.rgba.G = view.image_channel_data[id_cast(RGBA::G)] + offset;
		p.rgba.B = view.image_channel_data[id_cast(RGBA::B)] + offset;
		p.rgba.A = view.image_channel_data[id_cast(RGBA::A)] + offset;

		return p;
	}


	PixelRGBr32 rgb_xy_at(ViewRGBr32 const& view, u32 x, u32 y)
	{
		assert(y < view.height);
		assert(x < view.width);

		auto offset = (size_t)((view.y_begin + y) * view.image_width + view.x_begin) + x;

		PixelRGBr32 p{};

		p.rgb.R = view.image_channel_data[id_cast(RGB::R)] + offset;
		p.rgb.G = view.image_channel_data[id_cast(RGB::G)] + offset;
		p.rgb.B = view.image_channel_data[id_cast(RGB::B)] + offset;

		return p;
	}


	PixelHSVr32 hsv_xy_at(ViewHSVr32 const& view, u32 x, u32 y)
	{
		assert(y < view.height);
		assert(x < view.width);

		auto offset = (size_t)((view.y_begin + y) * view.image_width + view.x_begin) + x;

		PixelHSVr32 p{};

		p.hsv.H = view.image_channel_data[id_cast(HSV::H)] + offset;
		p.hsv.S = view.image_channel_data[id_cast(HSV::S)] + offset;
		p.hsv.V = view.image_channel_data[id_cast(HSV::V)] + offset;

		return p;
	}
}


/* make_view */

namespace libimage
{
	template <size_t N>
	static void do_make_view(ViewCHr32<N>& view, u32 width, u32 height, Buffer32& buffer)
	{
		view.image_width = width;
		view.x_begin = 0;
		view.y_begin = 0;
		view.x_end = width;
		view.y_end = height;
		view.width = width;
		view.height = height;

		for (u32 ch = 0; ch < N; ++ch)
		{
			view.image_channel_data[ch] = buffer.push(width * height);
		}
	}


	void make_view(View4r32& view, u32 width, u32 height, Buffer32& buffer)
	{
		do_make_view(view, width, height, buffer);
	}


	void make_view(View3r32& view, u32 width, u32 height, Buffer32& buffer)
	{
		do_make_view(view, width, height, buffer);
	}


	void make_view(View2r32& view, u32 width, u32 height, Buffer32& buffer)
	{
		do_make_view(view, width, height, buffer);
	}


	void make_view(View1r32& view, u32 width, u32 height, Buffer32& buffer)
	{
		view.image_data = buffer.push(width * height);
		view.image_width = width;
		view.x_begin = 0;
		view.y_begin = 0;
		view.x_end = width;
		view.y_end = height;
		view.width = width;
		view.height = height;
	}


    void make_host_view(View1r32& view, u32 width, u32 height)
	{		
		view.image_width = width;
		view.x_begin = 0;
		view.y_begin = 0;
		view.x_end = width;
		view.y_end = height;
		view.width = width;
		view.height = height;

        view.image_data = (r32*)std::malloc(sizeof(r32) * width * height);
	}


    template <size_t N>
    static void make_host_view(ViewCHr32<N>& view, u32 width, u32 height)
    {
        view.image_width = width;
		view.x_begin = 0;
		view.y_begin = 0;
		view.x_end = width;
		view.y_end = height;
		view.width = width;
		view.height = height;

        auto data = (r32*)std::malloc(N * sizeof(r32) * width * height)

        assert(data);

		for (u32 ch = 0; ch < N; ++ch)
        {
            view.image_channel_data[ch] = data + ch * width * height;
        }
    }


    static void destroy_host_view(View1r32& view)
    {
        std::free(view.image_data);
    }


    template <size_t N>
    static void destroy_host_view(ViewCHr32<N>& view)
    {
        std::free(view.image_channel_data[0]);
    }
}


/* map */

namespace libimage
{
    using u8_to_r32_f = std::function<r32(u8)>;
	using r32_to_u8_f = std::function<u8(r32)>;


    template <class IMG_U8>
    static void map_device_to_host(View1r32 const& device_src, IMG_U8 const& host_dst, r32_to_u8_f const& func)
    {
        auto const width = device_src.width;
        auto const height = device_src.height;
        auto const bytes_per_row = sizeof(r32) * width;

        View1r32 host_v;
        make_host_view(host_v, width, height);

        auto const row_func = [&](u32 y)
		{
			auto s = row_begin(device_src, y);
			auto h = row_begin(host_v, y);

            if(!cuda::memcpy_to_host(s, h, bytes_per_row))
            {
                return;
            }

            auto d = row_begin(host_dst, y);

			for (u32 x = 0; x < width; ++x)
			{
				d[x] = func(h[x]);
			}
		};

		process_rows(height, row_func);

        destroy_host_view(host_v);
    }


    template <class IMG_U8>
    static void map_host_to_device(IMG_U8 const& host_src, View1r32 const& device_dst, u8_to_r32_f const& func)
    {
        auto const width = device_dst.width;
        auto const height = device_dst.height;

        auto const bytes_per_row = sizeof(r32) * width;

        View1r32 host_v;
        make_host_view(host_v, width, height);

        auto const row_func = [&](u32 y)
		{
			auto s = row_begin(host_src, y);
			auto h = row_begin(host_v, y);

            for (u32 x = 0; x < width; ++x)
			{
				h[x] = func(s[x]);
			}

            auto d = row_begin(device_dst, y);

            if(!cuda::memcpy_to_device(h, d, bytes_per_row))
            {
                assert(false);
            }
		};

		process_rows(height, row_func);

        destroy_host_view(host_v);
    }


	void map(View1r32 const& device_src, gray::Image const& host_dst)
    {
        assert(verify(device_src, host_dst));

        map_device_to_host(device_src, host_dst, to_channel_u8);
    }


	void map(gray::Image const& host_src, View1r32 const& device_dst)
    {
        assert(verify(host_src, device_dst));

        map_host_to_device(host_src, device_dst, to_channel_r32);
    }


	void map(View1r32 const& device_src, gray::View const& host_dst)
    {
        assert(verify(device_src, host_dst));

        map_device_to_host(device_src, host_dst, to_channel_u8);
    }


	void map(gray::View const& host_src, View1r32 const& device_dst)
    {
        assert(verify(host_src, device_dst));

        map_host_to_device(host_src, device_dst, to_channel_r32);
    }


	void map(View1r32 const& device_src, gray::Image const& host_dst, r32 gray_min, r32 gray_max)
    {
        assert(verify(device_src, host_dst));

		auto const func = [&](r32 p) { return lerp_to_u8(p, gray_min, gray_max); };

        map_device_to_host(device_src, host_dst, func);
    }


	void map(gray::Image const& host_src, View1r32 const& device_dst, r32 gray_min, r32 gray_max)
    {
        assert(verify(host_src, device_dst));

		auto const func = [&](u8 p) { return lerp_to_r32(p, gray_min, gray_max); };

        map_host_to_device(host_src, device_dst, func);
    }


	void map(View1r32 const& device_src, gray::View const& host_dst, r32 gray_min, r32 gray_max)
    {
        assert(verify(device_src, host_dst));

		auto const func = [&](r32 p) { return lerp_to_u8(p, gray_min, gray_max); };

        map_device_to_host(device_src, host_dst, func);
    }


	void map(gray::View const& host_src, View1r32 const& device_dst, r32 gray_min, r32 gray_max)
    {
        assert(verify(host_src, device_dst));

		auto const func = [&](u8 p) { return lerp_to_r32(p, gray_min, gray_max); };

        map_host_to_device(host_src, device_dst, func);
    }
}