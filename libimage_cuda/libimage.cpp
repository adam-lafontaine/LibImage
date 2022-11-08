#include "include.hpp"

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
	template <size_t N, class BUFFER>
	static void do_make_view(ViewCHr32<N>& view, u32 width, u32 height, BUFFER& buffer)
	{
		assert(buffer.avail() >= N * width * height);

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


	template <class BUFFER>
	void do_make_view(View1r32& view, u32 width, u32 height, BUFFER& buffer)
	{
		assert(buffer.avail() >= width * height);

		view.image_data = buffer.push(width * height);
		view.image_width = width;
		view.x_begin = 0;
		view.y_begin = 0;
		view.x_end = width;
		view.y_end = height;
		view.width = width;
		view.height = height;
	}


	void make_view(View4r32& view, u32 width, u32 height, HostBuffer32& buffer)
	{
		do_make_view(view, width, height, buffer);
	}


	void make_view(View3r32& view, u32 width, u32 height, HostBuffer32& buffer)
	{
		do_make_view(view, width, height, buffer);
	}


	void make_view(View2r32& view, u32 width, u32 height, HostBuffer32& buffer)
	{
		do_make_view(view, width, height, buffer);
	}


	void make_view(View1r32& view, u32 width, u32 height, HostBuffer32& buffer)
	{
		do_make_view(view, width, height, buffer);
	}
	

	void make_view(View4r32& view, u32 width, u32 height, DeviceBuffer32& buffer)
	{
		do_make_view(view, width, height, buffer);
	}


	void make_view(View3r32& view, u32 width, u32 height, DeviceBuffer32& buffer)
	{
		do_make_view(view, width, height, buffer);
	}


	void make_view(View2r32& view, u32 width, u32 height, DeviceBuffer32& buffer)
	{
		do_make_view(view, width, height, buffer);
	}


	void make_view(View1r32& view, u32 width, u32 height, DeviceBuffer32& buffer)
	{
		do_make_view(view, width, height, buffer);
	}


	template <size_t N>
	static void pop_view(HostBuffer32& buffer, ViewCHr32<N> const& view)
	{
		buffer.pop(N * view.width * view.height);
	}


	static void pop_view(HostBuffer32& buffer, View1r32 const& view)
	{
		buffer.pop(view.width * view.height);
	}

}


/* sub_view */

namespace libimage
{
	template <size_t N>
	static ViewCHr32<N> do_sub_view(ViewCHr32<N> const& view, Range2Du32 const& range)
	{
		ViewCHr32<N> sub_view;

		sub_view.image_width = view.image_width;
		sub_view.x_begin = view.x_begin + range.x_begin;
		sub_view.y_begin = view.y_begin + range.y_begin;
		sub_view.x_end = view.x_begin + range.x_end;
		sub_view.y_end = view.y_begin + range.y_end;
		sub_view.width = range.x_end - range.x_begin;
		sub_view.height = range.y_end - range.y_begin;

		for (u32 ch = 0; ch < N; ++ch)
		{
			sub_view.image_channel_data[ch] = view.image_channel_data[ch];
		}

		return sub_view;
	}


	View4r32 sub_view(View4r32 const& view, Range2Du32 const& range)
	{
		assert(verify(view));

		auto sub_view = do_sub_view(view, range);

		assert(verify(sub_view));

		return sub_view;
	}


	View3r32 sub_view(View3r32 const& view, Range2Du32 const& range)
	{
		assert(verify(view, range));

		auto sub_view = do_sub_view(view, range);

		assert(verify(sub_view));

		return sub_view;
	}


	View2r32 sub_view(View2r32 const& view, Range2Du32 const& range)
	{
		assert(verify(view, range));

		auto sub_view = do_sub_view(view, range);

		assert(verify(sub_view));

		return sub_view;
	}


	View1r32 sub_view(View1r32 const& view, Range2Du32 const& range)
	{
		assert(verify(view, range));

		View1r32 sub_view;

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
}


/* select_channel */

namespace libimage
{
	template <size_t N>
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


	View1r32 select_channel(ViewRGBAr32 const& view, RGBA channel)
	{
		assert(verify(view));

		auto ch = id_cast(channel);

		auto view1 = select_channel(view, ch);

		assert(verify(view1));

		return view1;
	}


	View1r32 select_channel(ViewRGBr32 const& view, RGB channel)
	{
		assert(verify(view));

		auto ch = id_cast(channel);

		auto view1 = select_channel(view, ch);

		assert(verify(view1));

		return view1;
	}


	View1r32 select_channel(ViewHSVr32 const& view, HSV channel)
	{
		assert(verify(view));

		auto ch = id_cast(channel);

		auto view1 = select_channel(view, ch);

		assert(verify(view1));

		return view1;
	}


	View1r32 select_channel(View2r32 const& view, GA channel)
	{
		assert(verify(view));

		auto ch = id_cast(channel);
		assert(ch >= 0);
		assert(ch < 2);

		auto view1 = select_channel(view, ch);

		assert(verify(view1));

		return view1;
	}


	View1r32 select_channel(View2r32 const& view, XY channel)
	{
		assert(verify(view));

		auto ch = id_cast(channel);
		assert(ch >= 0);
		assert(ch < 2);

		auto view1 = select_channel(view, ch);

		assert(verify(view1));

		return view1;
	}	
}


/* map */

namespace libimage
{
    using u8_to_r32_f = std::function<r32(u8)>;
	using r32_to_u8_f = std::function<u8(r32)>;


	template <class IMG_U8>
	static void map_device_to_host(View1r32 const& device_src, IMG_U8 const& host_dst, View1r32 const& host_v, r32_to_u8_f const& func)
	{
		assert(verify(device_src, host_v));

		auto const width = device_src.width;
        auto const height = device_src.height;
        auto const bytes_per_row = sizeof(r32) * width;

        auto const row_func = [&](u32 y)
		{
			auto s = row_begin(device_src, y);
			auto h = row_begin(host_v, y);

            if(!cuda::memcpy_to_host(s, h, bytes_per_row)) { assert(false); }

            auto d = row_begin(host_dst, y);

			for (u32 x = 0; x < width; ++x)
			{
				d[x] = func(h[x]);
			}
		};

		process_rows(height, row_func);
	}


	template <class IMG_U8>
	static void map_host_to_device(IMG_U8 const& host_src, View1r32 const& device_dst, View1r32 const& host_v, u8_to_r32_f const& func)
	{
		assert(verify(device_dst, host_v));

		auto const width = device_dst.width;
        auto const height = device_dst.height;

        auto const bytes_per_row = sizeof(r32) * width;

        auto const row_func = [&](u32 y)
		{
			auto s = row_begin(host_src, y);
			auto h = row_begin(host_v, y);

            for (u32 x = 0; x < width; ++x)
			{
				h[x] = func(s[x]);
			}

            auto d = row_begin(device_dst, y);

            if(!cuda::memcpy_to_device(h, d, bytes_per_row)) { assert(false); }
		};

		process_rows(height, row_func);
	}


	void map(View1r32 const& device_src, gray::View const& host_dst, HostBuffer32& host_buffer)
    {
        assert(verify(device_src, host_dst));

		View1r32 host_v;
		make_view(host_v, device_src.width, device_src.height, host_buffer);

        map_device_to_host(device_src, host_dst, host_v, to_channel_u8);

		pop_view(host_buffer, host_v);
    }


	void map(gray::View const& host_src, View1r32 const& device_dst, HostBuffer32& host_buffer)
    {
        assert(verify(host_src, device_dst));

		View1r32 host_v;
		make_view(host_v, device_dst.width, device_dst.height, host_buffer);

        map_host_to_device(host_src, device_dst, host_v, to_channel_r32);

		pop_view(host_buffer, host_v);
    }


	void map(View1r32 const& device_src, gray::View const& host_dst, HostBuffer32& host_buffer, r32 gray_min, r32 gray_max)
    {
        assert(verify(device_src, host_dst));

		auto const func = [&](r32 p) { return lerp_to_u8(p, gray_min, gray_max); };

		View1r32 host_v;
		make_view(host_v, device_src.width, device_src.height, host_buffer);

        map_device_to_host(device_src, host_dst, host_v, func);

		pop_view(host_buffer, host_v);
    }


	void map(gray::View const& host_src, View1r32 const& device_dst, HostBuffer32& host_buffer, r32 gray_min, r32 gray_max)
    {
        assert(verify(host_src, device_dst));

		auto const func = [&](u8 p) { return lerp_to_r32(p, gray_min, gray_max); };

        View1r32 host_v;
		make_view(host_v, device_dst.width, device_dst.height, host_buffer);

        map_host_to_device(host_src, device_dst, host_v, func);

		pop_view(host_buffer, host_v);
    }
}


/* map_rgb */

namespace libimage
{
	static void map_rgba_device_to_host(ViewRGBAr32 const& device_src, View const& host_dst, ViewRGBAr32 const& host_v)
	{
		assert(verify(device_src, host_v));

		constexpr auto R = id_cast(RGBA::R);
		constexpr auto G = id_cast(RGBA::G);
		constexpr auto B = id_cast(RGBA::B);
		constexpr auto A = id_cast(RGBA::A);

        auto const width = device_src.width;
        auto const height = device_src.height;

        auto const bytes_per_row = sizeof(r32) * width;       

		auto const row_func = [&](u32 y)
		{			
			auto sr = channel_row_begin(device_src, y, R);
			auto sg = channel_row_begin(device_src, y, G);
			auto sb = channel_row_begin(device_src, y, B);
			auto sa = channel_row_begin(device_src, y, A);

            auto hr = channel_row_begin(host_v, y, R);
			auto hg = channel_row_begin(host_v, y, G);
			auto hb = channel_row_begin(host_v, y, B);
			auto ha = channel_row_begin(host_v, y, A);

            if(!cuda::memcpy_to_host(sr, hr, bytes_per_row)) { assert(false); }
            if(!cuda::memcpy_to_host(sg, hg, bytes_per_row)) { assert(false); }
            if(!cuda::memcpy_to_host(sb, hb, bytes_per_row)) { assert(false); }
            if(!cuda::memcpy_to_host(sa, ha, bytes_per_row)) { assert(false); }

            auto d = row_begin(host_dst, y);

			for (u32 x = 0; x < width; ++x)
			{
				d[x].channels[R] = to_channel_u8(hr[x]);
				d[x].channels[G] = to_channel_u8(hg[x]);
				d[x].channels[B] = to_channel_u8(hb[x]);
				d[x].channels[A] = to_channel_u8(ha[x]);
			}            
		};

		process_rows(height, row_func);
	}


	static void map_rgba_host_to_device(View const& host_src, ViewRGBAr32 const& device_dst, ViewRGBAr32 const& host_v)
	{
		assert(verify(device_dst, host_v));

		constexpr auto R = id_cast(RGBA::R);
		constexpr auto G = id_cast(RGBA::G);
		constexpr auto B = id_cast(RGBA::B);
		constexpr auto A = id_cast(RGBA::A);

        auto const width = device_dst.width;
        auto const height = device_dst.height;

        auto const bytes_per_row = sizeof(r32) * width;

        auto const row_func = [&](u32 y)
        {
            auto s = row_begin(host_src, y);
            auto hr = channel_row_begin(host_v, y, R);
            auto hg = channel_row_begin(host_v, y, G);
            auto hb = channel_row_begin(host_v, y, B);
            auto ha = channel_row_begin(host_v, y, A);

            for (u32 x = 0; x < width; ++x)
            {
                hr[x] = to_channel_r32(s[x].channels[R]);
                hg[x] = to_channel_r32(s[x].channels[G]);
                hb[x] = to_channel_r32(s[x].channels[B]);
                ha[x] = to_channel_r32(s[x].channels[A]);
            }

            auto dr = channel_row_begin(device_dst, y, R);
            auto dg = channel_row_begin(device_dst, y, G);
            auto db = channel_row_begin(device_dst, y, B);
            auto da = channel_row_begin(device_dst, y, A);

            if(!cuda::memcpy_to_device(hr, dr, bytes_per_row)) { assert(false); }
            if(!cuda::memcpy_to_device(hg, dg, bytes_per_row)) { assert(false); }
            if(!cuda::memcpy_to_device(hb, db, bytes_per_row)) { assert(false); }
            if(!cuda::memcpy_to_device(ha, da, bytes_per_row)) { assert(false); }
        };

        process_rows(height, row_func);
	}


	static void map_rgb_device_to_host(ViewRGBr32 const& device_src, View const& host_dst, ViewRGBr32 const& host_v)
	{
		assert(verify(device_src, host_v));

		constexpr auto R = id_cast(RGBA::R);
		constexpr auto G = id_cast(RGBA::G);
		constexpr auto B = id_cast(RGBA::B);
		constexpr auto A = id_cast(RGBA::A);

		constexpr auto ch_max = to_channel_u8(1.0f);

        auto const width = device_src.width;
        auto const height = device_src.height;

        auto const bytes_per_row = sizeof(r32) * width;

		auto const row_func = [&](u32 y)
        {
            auto sr = channel_row_begin(device_src, y, R);
			auto sg = channel_row_begin(device_src, y, G);
			auto sb = channel_row_begin(device_src, y, B);

            auto hr = channel_row_begin(host_v, y, R);
			auto hg = channel_row_begin(host_v, y, G);
			auto hb = channel_row_begin(host_v, y, B);

            if(!cuda::memcpy_to_host(sr, hr, bytes_per_row)) { assert(false); }
            if(!cuda::memcpy_to_host(sg, hg, bytes_per_row)) { assert(false); }
            if(!cuda::memcpy_to_host(sb, hb, bytes_per_row)) { assert(false); }

            auto d = row_begin(host_dst, y);

            for (u32 x = 0; x < width; ++x)
            {
                d[x].channels[R] = to_channel_u8(hr[x]);
				d[x].channels[G] = to_channel_u8(hg[x]);
				d[x].channels[B] = to_channel_u8(hb[x]);
				d[x].channels[A] = ch_max;
            }
        };

        process_rows(height, row_func);
	}


	static void map_rgb_host_to_device(View const& host_src, ViewRGBr32 const& device_dst, ViewRGBr32 const& host_v)
	{
		assert(verify(device_dst, host_v));

		constexpr auto R = id_cast(RGBA::R);
		constexpr auto G = id_cast(RGBA::G);
		constexpr auto B = id_cast(RGBA::B);
		constexpr auto A = id_cast(RGBA::A);

        auto const width = device_dst.width;
        auto const height = device_dst.height;

        auto const bytes_per_row = sizeof(r32) * width;

        auto const row_func = [&](u32 y)
        {
            auto s = row_begin(host_src, y);
            auto hr = channel_row_begin(host_v, y, R);
            auto hg = channel_row_begin(host_v, y, G);
            auto hb = channel_row_begin(host_v, y, B);

            for (u32 x = 0; x < width; ++x)
            {
                hr[x] = to_channel_r32(s[x].channels[R]);
                hg[x] = to_channel_r32(s[x].channels[G]);
                hb[x] = to_channel_r32(s[x].channels[B]);
            }

            auto dr = channel_row_begin(device_dst, y, R);
            auto dg = channel_row_begin(device_dst, y, G);
            auto db = channel_row_begin(device_dst, y, B);

            if(!cuda::memcpy_to_device(hr, dr, bytes_per_row)) { assert(false); }
            if(!cuda::memcpy_to_device(hg, dg, bytes_per_row)) { assert(false); }
            if(!cuda::memcpy_to_device(hb, db, bytes_per_row)) { assert(false); }
        };

        process_rows(height, row_func);
	}


	void map_rgb(ViewRGBAr32 const& device_src, View const& host_dst, HostBuffer32& host_buffer)
    {
        assert(verify(device_src, host_dst));

		ViewRGBAr32 host_v;
		make_view(host_v, device_src.width, device_src.height, host_buffer);

		map_rgba_device_to_host(device_src, host_dst, host_v);

		pop_view(host_buffer, host_v);
    }


	void map_rgb(View const& host_src, ViewRGBAr32 const& device_dst, HostBuffer32& host_buffer)
    {
        assert(verify(host_src, device_dst));

        ViewRGBAr32 host_v;
		make_view(host_v, device_dst.width, device_dst.height, host_buffer);

		map_rgba_host_to_device(host_src, device_dst, host_v);

		pop_view(host_buffer, host_v);
    }


	void map_rgb(ViewRGBr32 const& device_src, View const& host_dst, HostBuffer32& host_buffer)
    {
        assert(verify(device_src, host_dst));

        ViewRGBr32 host_v;
		make_view(host_v, device_src.width, device_src.height, host_buffer);

        map_rgb_device_to_host(device_src, host_dst, host_v);

		pop_view(host_buffer, host_v);
    }


	void map_rgb(View const& host_src, ViewRGBr32 const& device_dst, HostBuffer32& host_buffer)
    {
        assert(verify(host_src, device_dst));

        ViewRGBr32 host_v;
		make_view(host_v, device_dst.width, device_dst.height, host_buffer);

        map_rgb_host_to_device(host_src, device_dst, host_v);

		pop_view(host_buffer, host_v);
    }
}


/* map_hsv */

namespace libimage
{
	void map_rgb_hsv(View const& host_src, ViewHSVr32 const& device_dst, HostBuffer32& host_buffer)
	{
		assert(verify(host_src, device_dst));

		ViewRGBr32 device_rgb = device_dst;
		map_rgb(host_src, device_rgb, host_buffer);

		map_rgb_hsv(device_rgb, device_dst);

		assert(verify(host_src, device_dst));
	}


	void map_hsv_rgb(ViewHSVr32 const& device_src, View const& host_dst, HostBuffer32& host_buffer)
	{
		assert(verify(device_src, host_dst));

		ViewRGBr32 device_rgb = device_src;
		map_hsv_rgb(device_src, device_rgb);

		map_rgb(device_rgb, host_dst, host_buffer);
	}
	
}


/* for_each_pixel */

namespace libimage
{
	void for_each_pixel(gray::View const& view, u8_f const& func)
	{
		assert(verify(view));

		auto const row_func = [&](u32 y)
		{
			auto row = row_begin(view, y);
			for (u32 x = 0; x < view.width; ++x)
			{
				func(row[x]);
			}
		};

		process_rows(view.height, row_func);
	}
}


/* for_each_xy */

namespace libimage
{
	template <class IMG>
	static void do_for_each_xy(IMG const& image, xy_f const& func)
	{
		auto const row_func = [&](u32 y)
		{
			for (u32 x = 0; x < image.width; ++x)
			{
				func(x, y);
			}
		};

		process_rows(image.height, row_func);
	}


	void for_each_xy(View const& view, xy_f const& func)
	{
		assert(verify(view));

		do_for_each_xy(view, func);
	}


	void for_each_xy(gray::View const& view, xy_f const& func)
	{
		assert(verify(view));

		do_for_each_xy(view, func);
	}
}


/* fill */

namespace libimage
{
	template <class IMG, typename PIXEL>
	static void fill_1_channel(IMG const& image, PIXEL color)
	{
		auto const row_func = [&](u32 y)
		{
			auto d = row_begin(image, y);
			for (u32 x = 0; x < image.width; ++x)
			{
				d[x] = color;
			}
		};

		process_rows(image.height, row_func);
	}


	void fill(View const& view, Pixel color)
	{
		assert(verify(view));

		fill_1_channel(view, color);
	}


	void fill(gray::View const& view, u8 gray)
	{
		assert(verify(view));

		fill_1_channel(view, gray);
	}
}


/* copy */

namespace libimage
{
	template <class IMG_SRC, class IMG_DST>
	static void copy_1_channel(IMG_SRC const& src, IMG_DST const& dst)
	{
		auto const row_func = [&](u32 y)
		{
			auto s = row_begin(src, y);
			auto d = row_begin(dst, y);
			for (u32 x = 0; x < src.width; ++x)
			{
				d[x] = s[x];
			}
		};

		process_rows(src.height, row_func);
	}


	void copy(View const& src, View const& dst)
	{
		assert(verify(src, dst));

		copy_1_channel(src, dst);
	}

	
	void copy(gray::View const& src, gray::View const& dst)
	{
		assert(verify(src, dst));

		copy_1_channel(src, dst);
	}
}


/* grayscale */

namespace libimage
{
	constexpr r32 COEFF_RED = 0.299f;
	constexpr r32 COEFF_GREEN = 0.587f;
	constexpr r32 COEFF_BLUE = 0.114f;


	static constexpr r32 rgb_grayscale_standard(r32 red, r32 green, r32 blue)
	{
		return COEFF_RED * red + COEFF_GREEN * green + COEFF_BLUE * blue;
	}


	void grayscale(View const& src, gray::View const& dst)
	{
		assert(verify(src, dst));

		static constexpr auto red = id_cast(RGB::R);
		static constexpr auto green = id_cast(RGB::G);
		static constexpr auto blue = id_cast(RGB::B);

		auto const row_func = [&](u32 y)
		{
			auto s = row_begin(src, y);
			auto d = row_begin(dst, y);
			for (u32 x = 0; x < src.width; ++x)
			{
				auto r = (r32)(s[x].channels[red]);
				auto g = (r32)(s[x].channels[green]);
				auto b = (r32)(s[x].channels[blue]);
				d[x] = (u8)rgb_grayscale_standard(r, g, b);
			}
		};

		process_rows(src.height, row_func);
	}
	
}