#include "libimage.hpp"

#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <cstring>


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


/* Images */

namespace libimage
{
	View3r32 make_rgb_view(View4r32 const& view)
	{
		assert(verify(view));

		View3r32 view3;

		view3.image_width = view.image_width;
		view3.range = view.range;
		view3.width = view.width;
		view3.height = view.height;

		for (u32 ch = 0; ch < 3; ++ch)
		{
			view3.image_channel_data[ch] = view.image_channel_data[ch];
		}

		assert(verify(view3));

		return view3;
	}


	r32* row_begin(View1r32 const& view, u32 y)
	{
		assert(y < view.height);

		auto offset = (view.y_begin + y) * view.image_width + view.x_begin;

		auto ptr = view.image_data + (u64)(offset);
		assert(ptr);

		return ptr;
	}


	r32* xy_at(View1r32 const& view, u32 x, u32 y)
	{
		assert(y < view.height);
		assert(x < view.width);

		return row_begin(view, y) + x;
	}


	static r32* row_offset_begin(View1r32 const& view, int y_offset)
	{
		auto offset = (view.y_begin + y_offset) * view.image_width + view.x_begin;

		auto ptr = view.image_data + (u64)(offset);
		assert(ptr);

		return ptr;
	}


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


	template <size_t N>
	static ChannelData<N> channel_xy_at(ViewCHr32<N> const& view, u32 x, u32 y)
	{
		assert(y < view.height);
		assert(x < view.width);

		auto offset = (size_t)((view.y_begin + y) * view.image_width + view.x_begin) + x;

		ChannelData<N> data{};
		for (u32 ch = 0; ch < N; ++ch)
		{
			data.channels[ch] = view.image_channel_data[ch] + offset;
		}

		return data;
	}


	template <size_t N>
	static ChannelData<N> channel_row_offset_begin(ViewCHr32<N> const& view, int y)
	{
		auto offset = (size_t)((view.y_begin + y) * view.image_width + view.x_begin);

		ChannelData<N> data{};
		for (u32 ch = 0; ch < N; ++ch)
		{
			data.channels[ch] = view.image_channel_data[ch] + offset;
		}

		return data;
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
}


/* convert */

namespace libimage
{
	template <class IMG_INT, size_t N>
	static void interleaved_to_planar(IMG_INT const& src, ViewCHr32<N> const& dst)
	{
		auto const row_func = [&](u32 y)
		{
			auto d = channel_row_begin(dst, y).channels;
			auto s = row_begin(src, y);

			for (u32 ch = 0; ch < N; ++ch)
			{
				for (u32 x = 0; x < src.width; ++x)
				{
					d[ch][x] = to_channel_r32(s[x].channels[ch]);
				}
			}
		};

		process_rows(src.height, row_func);
	}


	template <class IMG_INT, size_t N>
	static void planar_to_interleaved(ViewCHr32<N> const& src, IMG_INT const& dst)
	{
		auto const row_func = [&](u32 y)
		{
			auto s = channel_row_begin(src, y).channels;
			auto d = row_begin(dst, y);

			for (u32 ch = 0; ch < N; ++ch)
			{
				for (u32 x = 0; x < src.width; ++x)
				{
					d[x].channels[ch] = to_channel_u8(s[ch][x]);
				}
			}
		};

		process_rows(src.height, row_func);
	}


	template <class IMG_U8>
	static void channel_r32_to_u8(View1r32 const& src, IMG_U8 const& dst)
	{
		auto const row_func = [&](u32 y)
		{
			auto s = row_begin(src, y);
			auto d = row_begin(dst, y);
			for (u32 x = 0; x < src.width; ++x)
			{
				d[x] = to_channel_u8(s[x]);
			}
		};

		process_rows(src.height, row_func);
	}


	template <class IMG_U8>
	static void channel_u8_to_r32(IMG_U8 const& src, View1r32 const& dst)
	{
		auto const row_func = [&](u32 y)
		{
			auto s = row_begin(src, y);
			auto d = row_begin(dst, y);
			for (u32 x = 0; x < src.width; ++x)
			{
				d[x] = to_channel_r32(s[x]);
			}
		};

		process_rows(src.height, row_func);
	}


	void convert(View4r32 const& src, Image const& dst)
	{
		assert(verify(src, dst));

		planar_to_interleaved(src, dst);
	}


	void convert(Image const& src, View4r32 const& dst)
	{
		assert(verify(src, dst));

		interleaved_to_planar(src, dst);
	}


	void convert(View4r32 const& src, View const& dst)
	{
		assert(verify(src, dst));

		planar_to_interleaved(src, dst);
	}


	void convert(View const& src, View4r32 const& dst)
	{
		assert(verify(src, dst));

		interleaved_to_planar(src, dst);
	}


	void convert(View3r32 const& src, Image const& dst)
	{
		assert(verify(src, dst));

		planar_to_interleaved(src, dst);
	}


	void convert(Image const& src, View3r32 const& dst)
	{
		assert(verify(src, dst));

		interleaved_to_planar(src, dst);
	}


	void convert(View3r32 const& src, View const& dst)
	{
		assert(verify(src, dst));

		planar_to_interleaved(src, dst);
	}


	void convert(View const& src, View3r32 const& dst)
	{
		assert(verify(src, dst));

		interleaved_to_planar(src, dst);
	}


	void convert(View1r32 const& src, gray::Image const& dst)
	{
		assert(verify(src, dst));

		channel_r32_to_u8(src, dst);
	}


	void convert(gray::Image const& src, View1r32 const& dst)
	{
		assert(verify(src, dst));

		channel_u8_to_r32(src, dst);
	}


	void convert(View1r32 const& src, gray::View const& dst)
	{
		assert(verify(src, dst));

		channel_r32_to_u8(src, dst);
	}


	void convert(gray::View const& src, View1r32 const& dst)
	{
		assert(verify(src, dst));

		channel_u8_to_r32(src, dst);
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


	template <size_t N>
	static void fill_n_channels(ViewCHr32<N> const& image, Pixel color)
	{
		r32 channels[N] = {};
		for (u32 ch = 0; ch < N; ++ch)
		{
			channels[ch] = to_channel_r32(color.channels[ch]);
		}

		auto const row_func = [&](u32 y)
		{
			auto d = channel_row_begin(image, y).channels;

			for (u32 ch = 0; ch < N; ++ch)
			{
				for (u32 x = 0; x < image.width; ++x)
				{
					d[ch][x] = channels[ch];
				}
			}
		};

		process_rows(image.height, row_func);
	}


	void fill(Image const& image, Pixel color)
	{
		assert(verify(image));

		fill_1_channel(image, color);
	}


	void fill(View const& view, Pixel color)
	{
		assert(verify(view));

		fill_1_channel(view, color);
	}


	void fill(gray::Image const& image, u8 gray)
	{
		assert(verify(image));

		fill_1_channel(image, gray);
	}


	void fill(gray::View const& view, u8 gray)
	{
		assert(verify(view));

		fill_1_channel(view, gray);
	}


	void fill(View4r32 const& view, Pixel color)
	{
		assert(verify(view));

		fill_n_channels(view, color);
	}


	void fill(View3r32 const& view, Pixel color)
	{
		assert(verify(view));

		fill_n_channels(view, color);
	}


	void fill(View1r32 const& view, u8 gray)
	{
		assert(verify(view));

		auto const gray32 = to_channel_r32(gray);

		auto const row_func = [&](u32 y)
		{
			auto d = row_begin(view, y);
			for (u32 x = 0; x < view.width; ++x)
			{
				d[x] = gray32;
			}
		};

		process_rows(view.height, row_func);
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


	template <size_t NS, size_t ND>
	static void copy_n_channels(ViewCHr32<NS> const& src, ViewCHr32<ND> const& dst)
	{
		static_assert(NS >= ND);

		auto const row_func = [&](u32 y)
		{
			auto s = channel_row_begin(src, y).channels;
			auto d = channel_row_begin(dst, y).channels;

			for (u32 ch = 0; ch < ND; ++ch)
			{
				for (u32 x = 0; x < src.width; ++x)
				{
					d[ch][x] = s[ch][x];
				}
			}
		};

		process_rows(src.height, row_func);
	}


	void copy(Image const& src, Image const& dst)
	{
		assert(verify(src, dst));

		copy_1_channel(src, dst);
	}


	void copy(Image const& src, View const& dst)
	{
		assert(verify(src, dst));

		copy_1_channel(src, dst);
	}


	void copy(View const& src, Image const& dst)
	{
		assert(verify(src, dst));

		copy_1_channel(src, dst);
	}


	void copy(View const& src, View const& dst)
	{
		assert(verify(src, dst));

		copy_1_channel(src, dst);
	}


	void copy(gray::Image const& src, gray::Image const& dst)
	{
		assert(verify(src, dst));

		copy_1_channel(src, dst);
	}


	void copy(gray::Image const& src, gray::View const& dst)
	{
		assert(verify(src, dst));

		copy_1_channel(src, dst);
	}


	void copy(gray::View const& src, gray::Image const& dst)
	{
		assert(verify(src, dst));

		copy_1_channel(src, dst);
	}


	void copy(gray::View const& src, gray::View const& dst)
	{
		assert(verify(src, dst));

		copy_1_channel(src, dst);
	}


	void copy(View4r32 const& src, View4r32 const& dst)
	{
		assert(verify(src, dst));

		copy_n_channels(src, dst);
	}


	void copy(View3r32 const& src, View3r32 const& dst)
	{
		assert(verify(src, dst));

		copy_n_channels(src, dst);
	}


	void copy(View1r32 const& src, View1r32 const& dst)
	{
		assert(verify(src, dst));

		copy_1_channel(src, dst);
	}
}


/* for_each_pixel */

namespace libimage
{
	template <class GRAY_IMG, class GRAY_F>
	void do_for_each_pixel(GRAY_IMG const& image, GRAY_F const& func)
	{
		auto const row_func = [&](u32 y)
		{
			auto row = row_begin(image, y);
			for (u32 x = 0; x < image.width; ++x)
			{
				func(row[x]);
			}
		};

		process_rows(image.height, row_func);
	}


	void for_each_pixel(gray::Image const& image, u8_f const& func)
	{
		assert(verify(image));

		do_for_each_pixel(image, func);
	}


	void for_each_pixel(gray::View const& view, u8_f const& func)
	{
		assert(verify(view));

		do_for_each_pixel(view, func);
	}


	void for_each_pixel(View1r32 const& view, r32_f const& func)
	{
		assert(verify(view));

		do_for_each_pixel(view, func);
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

	void for_each_xy(Image const& image, xy_f const& func)
	{
		assert(verify(image));

		do_for_each_xy(image, func);
	}


	void for_each_xy(View const& view, xy_f const& func)
	{
		assert(verify(view));

		do_for_each_xy(view, func);
	}


	void for_each_xy(gray::Image const& image, xy_f const& func)
	{
		assert(verify(image));

		do_for_each_xy(image, func);
	}


	void for_each_xy(gray::View const& view, xy_f const& func)
	{
		assert(verify(view));

		do_for_each_xy(view, func);
	}


	void for_each_xy(View4r32 const& view, xy_f const& func)
	{
		assert(verify(view));

		do_for_each_xy(view, func);
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


	template <class IMG, class GRAY>
	static void grayscale_platform(IMG const& src, GRAY const& dst)
	{
		constexpr static auto red = id_cast(RGB::R);
		constexpr static auto green = id_cast(RGB::G);
		constexpr static auto blue = id_cast(RGB::B);

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


	template <size_t N>
	static void grayscale_rgb(ViewCHr32<N> const& src, View1r32 const& dst)
	{
		static_assert(N >= 3);

		constexpr static auto red = id_cast(RGB::R);
		constexpr static auto green = id_cast(RGB::G);
		constexpr static auto blue = id_cast(RGB::B);

		auto const row_func = [&](u32 y)
		{
			auto s = channel_row_begin(src, y).channels;
			auto r = s[red];
			auto g = s[green];
			auto b = s[blue];
			auto d = row_begin(dst, y);
			for (u32 x = 0; x < src.width; ++x)
			{
				d[x] = rgb_grayscale_standard(r[x], g[x], b[x]);
			}
		};

		process_rows(src.height, row_func);
	}


	void grayscale(Image const& src, gray::Image const& dst)
	{
		assert(verify(src, dst));

		grayscale_platform(src, dst);
	}

	void grayscale(Image const& src, gray::View const& dst)
	{
		assert(verify(src, dst));

		grayscale_platform(src, dst);
	}


	void grayscale(View const& src, gray::Image const& dst)
	{
		assert(verify(src, dst));

		grayscale_platform(src, dst);
	}


	void grayscale(View const& src, gray::View const& dst)
	{
		assert(verify(src, dst));

		grayscale_platform(src, dst);
	}


	void grayscale(View4r32 const& src, View1r32 const& dst)
	{
		assert(verify(src, dst));

		grayscale_rgb(src, dst);
	}


	void grayscale(View3r32 const& src, View1r32 const& dst)
	{
		assert(verify(src, dst));

		grayscale_rgb(src, dst);
	}


}


/* select_channel */

namespace libimage
{
	View1r32 select_channel(View4r32 const& view, RGBA channel)
	{
		assert(verify(view));

		auto ch = id_cast(channel);

		View1r32 view1{};

		view1.image_width = view.image_width;
		view1.range = view.range;
		view1.width = view.width;
		view1.height = view.height;

		view1.image_data = view.image_channel_data[ch];

		assert(verify(view1));

		return view1;
	}


	View1r32 select_channel(View3r32 const& view, RGB channel)
	{
		assert(verify(view));

		auto ch = id_cast(channel);

		View1r32 view1{};

		view1.image_width = view.image_width;
		view1.range = view.range;
		view1.width = view.width;
		view1.height = view.height;

		view1.image_data = view.image_channel_data[ch];

		assert(verify(view1));

		return view1;
	}
}


/* alpha blend */

namespace libimage
{
	static r32 blend_linear(r32 lhs, r32 rhs, r32 alpha)
	{
		assert(alpha >= 0.0f);
		assert(alpha <= 1.0f);

		return alpha * lhs + (1.0f - alpha) * rhs;
	}


	static void do_alpha_blend(View4r32 const& src, View3r32 const& cur, View3r32 const& dst)
	{
		constexpr static auto red = id_cast(RGBA::R);
		constexpr static auto green = id_cast(RGBA::G);
		constexpr static auto blue = id_cast(RGBA::B);
		constexpr static auto alpha = id_cast(RGBA::A);

		auto const row_func = [&](u32 y)
		{
			auto s = channel_row_begin(src, y).channels;
			auto c = channel_row_begin(cur, y).channels;
			auto d = channel_row_begin(dst, y).channels;

			auto sr = s[red];
			auto sg = s[green];
			auto sb = s[blue];
			auto sa = s[alpha];

			auto cr = c[red];
			auto cg = c[green];
			auto cb = c[blue];

			auto dr = d[red];
			auto dg = d[green];
			auto db = d[blue];

			for (u32 x = 0; x < src.width; ++x)
			{
				dr[x] = blend_linear(sr[x], cr[x], sa[x]);
				dg[x] = blend_linear(sg[x], cg[x], sa[x]);
				db[x] = blend_linear(sb[x], cb[x], sa[x]);
			}
		};

		process_rows(src.height, row_func);
	}


	void alpha_blend(View4r32 const& src, View3r32 const& cur, View3r32 const& dst)
	{
		assert(verify(src, cur));
		assert(verify(src, dst));

		do_alpha_blend(src, cur, dst);
	}


	void alpha_blend(View4r32 const& src, View3r32 const& cur_dst)
	{
		assert(verify(src, cur_dst));

		do_alpha_blend(src, cur_dst, cur_dst);
	}
}


/* transform */

namespace libimage
{
	lut_t to_lut(u8_to_u8_f const& f)
	{
		lut_t lut = { 0 };

		for (u32 i = 0; i < 256; ++i)
		{
			lut[i] = f(i);
		}

		//process_rows(256, [&](u32 i) { lut[i] = f(i); });

		return lut;
	}


	template <class IMG_S, class IMG_D>
	static void do_transform_lut(IMG_S const& src, IMG_D const& dst, lut_t const& lut)
	{
		auto const row_func = [&](u32 y)
		{
			auto s = row_begin(src, y);
			auto d = row_begin(dst, y);
			for (u32 x = 0; x < src.width; ++x)
			{
				d[x] = lut[s[x]];
			}
		};

		process_rows(src.height, row_func);
	}


	void transform(gray::Image const& src, gray::Image const& dst, lut_t const& lut)
	{
		assert(verify(src, dst));

		do_transform_lut(src, dst, lut);
	}


	void transform(gray::Image const& src, gray::View const& dst, lut_t const& lut)
	{
		assert(verify(src, dst));

		do_transform_lut(src, dst, lut);
	}


	void transform(gray::View const& src, gray::Image const& dst, lut_t const& lut)
	{
		assert(verify(src, dst));

		do_transform_lut(src, dst, lut);
	}


	void transform(gray::View const& src, gray::View const& dst, lut_t const& lut)
	{
		assert(verify(src, dst));

		do_transform_lut(src, dst, lut);
	}


	void transform(View1r32 const& src, View1r32 const& dst, r32_to_r32_f const& func)
	{
		assert(verify(src, dst));

		auto const row_func = [&](u32 y)
		{
			auto s = row_begin(src, y);
			auto d = row_begin(dst, y);
			for (u32 x = 0; x < src.width; ++x)
			{
				d[x] = func(s[x]);
			}
		};

		process_rows(src.height, row_func);
	}

}


/* threshold */

namespace libimage
{
	void threshold(gray::Image const& src, gray::Image const& dst, u8 min, u8 max)
	{
		assert(verify(src, dst));
		assert(min < max);

		auto const lut = to_lut([&](u8 p) { return p >= min && p <= max ? 255 : 0; });		

		do_transform_lut(src, dst, lut);
	}


	void threshold(gray::Image const& src, gray::View const& dst, u8 min, u8 max)
	{
		assert(verify(src, dst));
		assert(min < max);

		auto const lut = to_lut([&](u8 p) { return p >= min && p <= max ? 255 : 0; });

		do_transform_lut(src, dst, lut);
	}


	void threshold(gray::View const& src, gray::Image const& dst, u8 min, u8 max)
	{
		assert(verify(src, dst));
		assert(min < max);

		auto const lut = to_lut([&](u8 p) { return p >= min && p <= max ? 255 : 0; });

		do_transform_lut(src, dst, lut);
	}


	void threshold(gray::View const& src, gray::View const& dst, u8 min, u8 max)
	{
		assert(verify(src, dst));
		assert(min < max);

		auto const lut = to_lut([&](u8 p) { return p >= min && p <= max ? 255 : 0; });

		do_transform_lut(src, dst, lut);
	}


	void threshold(View1r32 const& src, View1r32 const& dst, r32 min, r32 max)
	{
		assert(verify(src, dst));
		assert(min >= 0.0f && min <= 1.0f);
		assert(max >= 0.0f && max <= 1.0f);
		assert(min < max);

		auto const row_func = [&](u32 y)
		{
			auto s = row_begin(src, y);
			auto d = row_begin(dst, y);
			for (u32 x = 0; x < src.width; ++x)
			{
				d[x] = s[x] >= min && s[x] <= max ? 1.0f : 0.0f;
			}
		};

		process_rows(src.height, row_func);
	}
}


/* contrast */

namespace libimage
{
	static constexpr u8 lerp_clamp(u8 src_low, u8 src_high, u8 dst_low, u8 dst_high, u8 val)
	{
		if (val < src_low)
		{
			return dst_low;
		}
		else if (val > src_high)
		{
			return dst_high;
		}

		auto const ratio = (r32)(val - src_low) / (src_high - src_low);

		assert(ratio >= 0.0f);
		assert(ratio <= 1.0f);

		auto const diff = ratio * (dst_high - dst_low);

		return dst_low + (u8)diff;
	}


	static constexpr r32 lerp_clamp(r32 src_low, r32 src_high, r32 dst_low, r32 dst_high, r32 val)
	{
		if (val < src_low)
		{
			return dst_low;
		}
		else if (val > src_high)
		{
			return dst_high;
		}

		auto const ratio = (r32)(val - src_low) / (src_high - src_low);

		assert(ratio >= 0.0f);
		assert(ratio <= 1.0f);

		auto const diff = ratio * (dst_high - dst_low);

		return dst_low + (r32)diff;
	}


	void contrast(gray::Image const& src, gray::Image const& dst, u8 min, u8 max)
	{
		assert(verify(src, dst));
		assert(min < max);

		auto const lut = to_lut([&](u8 p) { return lerp_clamp(min, max, 0, 255, p); });

		do_transform_lut(src, dst, lut);
	}


	void contrast(gray::Image const& src, gray::View const& dst, u8 min, u8 max)
	{
		assert(verify(src, dst));
		assert(min < max);

		auto const lut = to_lut([&](u8 p) { return lerp_clamp(min, max, 0, 255, p); });

		do_transform_lut(src, dst, lut);
	}


	void contrast(gray::View const& src, gray::Image const& dst, u8 min, u8 max)
	{
		assert(verify(src, dst));
		assert(min < max);

		auto const lut = to_lut([&](u8 p) { return lerp_clamp(min, max, 0, 255, p); });

		do_transform_lut(src, dst, lut);
	}


	void contrast(gray::View const& src, gray::View const& dst, u8 min, u8 max)
	{
		assert(verify(src, dst));
		assert(min < max);

		auto const lut = to_lut([&](u8 p) { return lerp_clamp(min, max, 0, 255, p); });

		do_transform_lut(src, dst, lut);
	}


	void contrast(View1r32 const& src, View1r32 const& dst, r32 min, r32 max)
	{
		assert(verify(src, dst));
		assert(min >= 0.0f && min <= 1.0f);
		assert(max >= 0.0f && max <= 1.0f);
		assert(min < max);

		auto const row_func = [&](u32 y)
		{
			auto s = row_begin(src, y);
			auto d = row_begin(dst, y);
			for (u32 x = 0; x < src.width; ++x)
			{
				d[x] = lerp_clamp(min, max, 0.0f, 1.0f, s[x]);
			}
		};

		process_rows(src.height, row_func);
	}
}


/* gradients edges */

namespace libimage
{
	static constexpr r32 GRAD_X_3X3[9]
	{
		-0.25f,  0.0f,  0.25f,
		-0.50f,  0.0f,  0.50f,
		-0.25f,  0.0f,  0.25f,
	};


	static constexpr r32 GRAD_Y_3X3[9]
	{
		-0.25f, -0.50f, -0.25f,
		 0.0f,   0.0f,   0.0f,
		 0.25f,  0.50f,  0.25f,
	};


	static void zero_outer(View1r32 const& view)
	{
		auto const top_bottom = [&]() 
		{
			auto top = row_begin(view, 0);
			auto bottom = row_begin(view, view.height - 1);
			for (u32 x = 0; x < view.width; ++x)
			{
				top[x] = bottom[x] = 0.0f;
			}
		};
		
		auto const left_right = [&]() 
		{
			for (u32 y = 1; y < view.height - 1; ++y)
			{
				auto row = row_begin(view, y);
				row[0] = row[view.width - 1] = 0.0f;
			}
		};

		std::array<std::function<void()>, 2> f_list
		{
			top_bottom, left_right
		};

		do_for_each(f_list, [](auto const& f) { f(); });
	}


	static void convolve_gradients_3x3(View1r32 const& src, View1r32 const& dst)
	{
		// TODO: simd

		int const ry_begin = -1;
		int const ry_end = 2;
		int const rx_begin = -1;
		int const rx_end = 2;

		auto const row_func = [&](u32 y) 
		{
			u32 w = 0;
			r32 gx = 0.0f;
			r32 gy = 0.0f;
			
			auto d = row_begin(dst, y);
			for (u32 x = 0; x < src.width; ++x)
			{
				w = 0;
				gx = 0.0f;
				gy = 0.0f;

				for (int ry = ry_begin; ry < ry_end; ++ry)
				{
					auto s = row_offset_begin(src, y + ry);
					for (int rx = rx_begin; rx < rx_end; ++rx)
					{
						gx += (s + rx)[x] * GRAD_X_3X3[w];
						gy += (s + rx)[x] * GRAD_Y_3X3[w];
						++w;
					}
				}

				gx = fabs(gx);
				gy = fabs(gy);

				d[x] = gx > gy ? gx : gy;

				// TODO: f(gx, gy)
			}
		};

		process_rows(src.height, row_func);
	}


	static void convolve_edges_3x3(View1r32 const& src, View1r32 const& dst, r32 threshold)
	{
		// TODO: simd

		int const ry_begin = -1;
		int const ry_end = 2;
		int const rx_begin = -1;
		int const rx_end = 2;

		auto const row_func = [&](u32 y)
		{
			u32 w = 0;
			r32 gx = 0.0f;
			r32 gy = 0.0f;

			auto d = row_begin(dst, y);
			for (u32 x = 0; x < src.width; ++x)
			{
				w = 0;
				gx = 0.0f;
				gy = 0.0f;

				for (int ry = ry_begin; ry < ry_end; ++ry)
				{
					auto s = row_offset_begin(src, y + ry);
					for (int rx = rx_begin; rx < rx_end; ++rx)
					{
						gx += (s + rx)[x] * GRAD_X_3X3[w];
						gy += (s + rx)[x] * GRAD_Y_3X3[w];
						++w;
					}
				}

				gx = fabs(gx);
				gy = fabs(gy);

				d[x] = (gx > gy ? gx : gy) >= threshold ? 1.0f : 0.0f;

				// TODO: f(gx, gy)
			}
		};

		process_rows(src.height, row_func);
	}


	void gradients(View1r32 const& src, View1r32 const& dst)
	{
		assert(verify(src, dst));

		zero_outer(dst);

		Range2Du32 inner{};
		inner.x_begin = 1;
		inner.x_end = src.width - 1;
		inner.y_begin = 1;
		inner.y_end = src.height - 1;

		convolve_gradients_3x3(sub_view(src, inner), sub_view(dst, inner));
	}


	void edges(View1r32 const& src, View1r32 const& dst, r32 threshold)
	{
		assert(verify(src, dst));
		assert(threshold >= 0.0f && threshold <= 1.0f);

		zero_outer(dst);

		Range2Du32 inner{};
		inner.x_begin = 1;
		inner.x_end = src.width - 1;
		inner.y_begin = 1;
		inner.y_end = src.height - 1;

		convolve_edges_3x3(sub_view(src, inner), sub_view(dst, inner), threshold);
	}
}


/* blur */

namespace libimage
{
	constexpr r32 D3 = 16.0f;
	constexpr r32 GAUSS_3X3[9]
	{
		(1 / D3), (2 / D3), (1 / D3),
		(2 / D3), (4 / D3), (2 / D3),
		(1 / D3), (2 / D3), (1 / D3),
	};

	constexpr r32 D5 = 256.0f;
	constexpr r32 GAUSS_5X5[25]
	{
		(1 / D5), (4 / D5),  (6 / D5),  (4 / D5),  (1 / D5),
		(4 / D5), (16 / D5), (24 / D5), (16 / D5), (4 / D5),
		(6 / D5), (24 / D5), (36 / D5), (24 / D5), (6 / D5),
		(4 / D5), (16 / D5), (24 / D5), (16 / D5), (4 / D5),
		(1 / D5), (4 / D5),  (6 / D5),  (4 / D5),  (1 / D5),
	};


	template <size_t N>
	static void copy_outer(ViewCHr32<N> const& src, ViewCHr32<N> const& dst)
	{
		auto const width = src.width;
		auto const height = src.height;

		auto const top_bottom = [&]()
		{
			auto s_top = channel_row_begin(src, 0).channels;
			auto s_bottom = channel_row_begin(src, height - 1).channels;
			auto d_top = channel_row_begin(dst, 0).channels;
			auto d_bottom = channel_row_begin(dst, height - 1).channels;

			for (u32 ch = 0; ch < N; ++ch)
			{
				for (u32 x = 0; x < width; ++x)
				{
					d_top[ch][x] = s_top[ch][x]; // TODO: simd
					d_bottom[ch][x] = s_bottom[ch][x];
				}
			}
		};

		auto const left_right = [&]()
		{
			for (u32 y = 1; y < height - 1; ++y)
			{
				auto s_row = channel_row_begin(src, y).channels;
				auto d_row = channel_row_begin(dst, y).channels;

				for (u32 ch = 0; ch < N; ++ch)
				{
					d_row[ch][0] = s_row[ch][0];
					d_row[ch][width - 1] = s_row[ch][width - 1];
				}				
			}
		};

		std::array<std::function<void()>, 2> f_list
		{
			top_bottom, left_right
		};

		do_for_each(f_list, [](auto const& f) { f(); });
	}


	template <size_t N>
	static void convolve_gauss_3x3_outer(ViewCHr32<N> const& src, ViewCHr32<N> const& dst)
	{
		auto const width = src.width;
		auto const height = src.height;

		int const ry_begin = -1;
		int const ry_end = 2;
		int const rx_begin = -1;
		int const rx_end = 2;

		auto const top_bottom = [&]()
		{
			u32 w = 0;

			auto d_top_row = channel_row_begin(dst, 0).channels;
			auto d_bottom_row = channel_row_begin(dst, height - 1).channels;

			for (u32 ch = 0; ch < N; ++ch)
			{
				for (u32 x = 0; x < width; ++x)
				{
					w = 0;
					auto& d_top = d_top_row[ch][x];
					auto& d_bottom = d_bottom_row[ch][x];
					d_top = d_bottom = 0.0f;

					for (int ry = ry_begin; ry < ry_end; ++ry)
					{
						auto s_top = channel_row_offset_begin(src, 0 + ry).channels;
						auto s_bottom = channel_row_offset_begin(src, height - 1 + ry).channels;

						for (int rx = rx_begin; rx < rx_end; ++rx)
						{
							d_top += (s_top[ch] + rx)[x] * GAUSS_3X3[w];
							d_bottom += (s_bottom[ch] + rx)[x] * GAUSS_3X3[w];

							++w;
						}						
					}
				}
			}
		};

		auto const left_right = [&]() 
		{
			u32 w = 0;

			for (u32 y = 1; y < height - 1; ++y)
			{				
				auto d_row = channel_row_begin(dst, y).channels;

				for (u32 ch = 0; ch < N; ++ch)
				{
					w = 0;
					auto& d_left = d_row[ch][0];
					auto& d_right = d_row[ch][width - 1];

					d_left = d_right = 0.0f;

					for (int ry = ry_begin; ry < ry_end; ++ry)
					{
						auto s_row = channel_row_begin(src, y + ry).channels;

						for (int rx = rx_begin; rx < rx_end; ++rx)
						{
							auto s_left = s_row[ch];
							auto s_right = s_left + width - 1;

							d_left += *(s_left + rx) * GAUSS_3X3[w];
							d_right += *(s_right + rx) * GAUSS_3X3[w];

							++w;
						}
					}
				}
			}
		};

		std::array<std::function<void()>, 2> f_list
		{
			top_bottom, left_right
		};

		do_for_each(f_list, [](auto const& f) { f(); });
	}


	template <size_t N>
	static void convolve_gauss_5x5(ViewCHr32<N> const& src, ViewCHr32<N> const& dst)
	{
		assert(verify(src, dst));

		auto const width = src.width;
		auto const height = src.height;

		int const ry_begin = -2;
		int const ry_end = 3;
		int const rx_begin = -2;
		int const rx_end = 3;

		auto const row_func = [&](u32 y) 
		{
			u32 w = 0;

			auto d_row = channel_row_begin(dst, y).channels;

			for (u32 ch = 0; ch < N; ++ch)
			{
				for (u32 x = 0; x < src.width; ++x)
				{
					w = 0;
					auto& d = d_row[ch][x];

					d = 0.0f;

					for (int ry = ry_begin; ry < ry_end; ++ry)
					{
						auto s = channel_row_offset_begin(src, y + ry).channels;

						for (int rx = rx_begin; rx < rx_end; ++rx)
						{
							d += (s[ch] + rx)[x] * GAUSS_5X5[w];

							++w;
						}
					}
				}
			}			
		};

		process_rows(src.height, row_func);
	}


	static void copy_outer(View1r32 const& src, View1r32 const& dst)
	{
		auto const width = src.width;
		auto const height = src.height;

		auto const top_bottom = [&]()
		{
			auto s_top = row_begin(src, 0);
			auto s_bottom = row_begin(src, height - 1);
			auto d_top = row_begin(dst, 0);
			auto d_bottom = row_begin(dst, height - 1);
			for (u32 x = 0; x < width; ++x)
			{
				d_top[x] = s_top[x];
				d_bottom[x] = s_bottom[x];
			}
		};

		auto const left_right = [&]()
		{
			for (u32 y = 1; y < height - 1; ++y)
			{
				auto s_row = row_begin(src, y);
				auto d_row = row_begin(dst, y);

				d_row[0] = s_row[0];
				d_row[width - 1] = s_row[width - 1];
			}
		};

		std::array<std::function<void()>, 2> f_list
		{
			top_bottom, left_right
		};

		do_for_each(f_list, [](auto const& f) { f(); });
	}


	static void convolve_gauss_3x3_outer(View1r32 const& src, View1r32 const& dst)
	{
		auto const width = src.width;
		auto const height = src.height;

		int const ry_begin = -1;
		int const ry_end = 2;
		int const rx_begin = -1;
		int const rx_end = 2;

		auto const top_bottom = [&]()
		{
			u32 w = 0;
			r32 b_top = 0.0f;
			r32 b_bottom = 0.0f;

			auto d_top = row_begin(dst, 0);
			auto d_bottom = row_begin(dst, height - 1);
			for (u32 x = 0; x < width; ++x)
			{
				w = 0;
				b_top = 0.0f;
				b_bottom = 0.0f;

				for (int ry = ry_begin; ry < ry_end; ++ry)
				{
					auto s_top = row_offset_begin(src, 0 + ry);
					auto s_bottom = row_offset_begin(src, height - 1 + ry);
					for (int rx = rx_begin; rx < rx_end; ++rx)
					{
						b_top += (s_top + rx)[x] * GAUSS_3X3[w];
						b_bottom += (s_bottom + rx)[x] * GAUSS_3X3[w];
						++w;
					}
				}

				d_top[x] = b_top;
				d_bottom[x] = b_bottom;
			}
		};

		auto const left_right = [&]()
		{
			u32 w = 0;
			r32 b_left = 0.0f;
			r32 b_right = 0.0f;

			for (u32 y = 1; y < height - 1; ++y)
			{
				w = 0;
				b_left = 0.0f;
				b_right = 0.0f;

				auto d_left = row_begin(dst, y);
				auto d_right = d_left + width - 1;

				for (int ry = ry_begin; ry < ry_end; ++ry)
				{
					auto s_left = row_begin(src, y + ry);
					auto s_right = s_left + width - 1;

					for (int rx = rx_begin; rx < rx_end; ++rx)
					{
						b_left += *(s_left + rx) * GAUSS_3X3[w];
						b_right += *(s_right + rx) * GAUSS_3X3[w];
						++w;
					}
				}

				*d_left = b_left;
				*d_right = b_right;
			}
		};

		std::array<std::function<void()>, 2> f_list
		{
			top_bottom, left_right
		};

		do_for_each(f_list, [](auto const& f) { f(); });
	}


	static void convolve_gauss_5x5(View1r32 const& src, View1r32 const& dst)
	{
		assert(verify(src, dst));

		auto const width = src.width;
		auto const height = src.height;

		int const ry_begin = -2;
		int const ry_end = 3;
		int const rx_begin = -2;
		int const rx_end = 3;

		auto const row_func = [&](u32 y)
		{
			u32 w = 0;
			r32 g = 0.0f;

			auto d = row_begin(dst, y);
			for (u32 x = 0; x < src.width; ++x)
			{
				w = 0;
				g = 0.0f;

				for (int ry = ry_begin; ry < ry_end; ++ry)
				{
					auto s = row_offset_begin(src, y + ry);
					for (int rx = rx_begin; rx < rx_end; ++rx)
					{
						g += (s + rx)[x] * GAUSS_5X5[w];
						++w;
					}
				}

				d[x] = g;
			}
		};

		process_rows(src.height, row_func);
	}


	void blur(View1r32 const& src, View1r32 const& dst)
	{
		assert(verify(src, dst));

		copy_outer(src, dst);

		Range2Du32 inner{};
		inner.x_begin = 1;
		inner.x_end = src.width - 1;
		inner.y_begin = 1;
		inner.y_end = src.height - 1;

		convolve_gauss_3x3_outer(sub_view(src, inner), sub_view(dst, inner));

		inner.x_begin = 2;
		inner.x_end = src.width - 2;
		inner.y_begin = 2;
		inner.y_end = src.height - 2;

		convolve_gauss_5x5(sub_view(src, inner), sub_view(dst, inner));
	}


	void blur(View3r32 const& src, View3r32 const& dst)
	{
		assert(verify(src, dst));

		copy_outer(src, dst);

		Range2Du32 inner{};
		inner.x_begin = 1;
		inner.x_end = src.width - 1;
		inner.y_begin = 1;
		inner.y_end = src.height - 1;

		convolve_gauss_3x3_outer(sub_view(src, inner), sub_view(dst, inner));

		inner.x_begin = 2;
		inner.x_end = src.width - 2;
		inner.y_begin = 2;
		inner.y_end = src.height - 2;

		convolve_gauss_5x5(sub_view(src, inner), sub_view(dst, inner));
	}
}


/* rotate */

namespace libimage
{
	static Point2Dr32 find_rotation_src(Point2Du32 const& pt, Point2Du32 const& origin, r32 radians)
	{
		auto dx_dst = (r32)pt.x - (r32)origin.x;
		auto dy_dst = (r32)pt.y - (r32)origin.y;

		auto radius = std::hypotf(dx_dst, dy_dst);

		auto theta_dst = atan2f(dy_dst, dx_dst);
		auto theta_src = theta_dst - radians;

		auto dx_src = radius * cosf(theta_src);
		auto dy_src = radius * sinf(theta_src);

		Point2Dr32 pt_src{};
		pt_src.x = (r32)origin.x + dx_src;
		pt_src.y = (r32)origin.y + dy_src;

		return pt_src;
	}


	template <size_t N>
	void do_rotate(ViewCHr32<N> const& src, ViewCHr32<N> const& dst, Point2Du32 origin, r32 radians)
	{
		auto const zero = 0.0f;
		auto const width = (r32)src.width;
		auto const height = (r32)src.height;

		auto const row_func = [&](u32 y)
		{
			auto d = channel_row_begin(dst, y).channels;
			for (u32 x = 0; x < src.width; ++x)
			{
				auto src_xy = find_rotation_src({ x, y }, origin, radians);
				if (src_xy.x < zero || src_xy.x >= width || src_xy.y < zero || src_xy.y >= height)
				{
					for (u32 ch = 0; ch < N; ++ch)
					{
						d[ch][x] = 0.0f; // alpha?
					}
				}
				else
				{
					auto s = channel_xy_at(src, (u32)floorf(src_xy.x), (u32)floorf(src_xy.y)).channels;
					for (u32 ch = 0; ch < N; ++ch)
					{
						d[ch][x] = *s[ch];
					}
				}
			}
		};

		process_rows(src.height, row_func);
	}


	void rotate(View4r32 const& src, View4r32 const& dst, Point2Du32 origin, r32 radians)
	{
		assert(verify(src, dst));

		do_rotate(src, dst, origin, radians);
	}


	void rotate(View3r32 const& src, View3r32 const& dst, Point2Du32 origin, r32 radians)
	{
		assert(verify(src, dst));

		do_rotate(src, dst, origin, radians);		
	}


	void rotate(View1r32 const& src, View1r32 const& dst, Point2Du32 origin, r32 radians)
	{
		assert(verify(src, dst));

		auto const zero = 0.0f;
		auto const width = (r32)src.width;
		auto const height = (r32)src.height;

		auto const row_func = [&](u32 y) 
		{
			auto d = row_begin(dst, y);
			for(u32 x = 0; x < src.width; ++x)
			{
				auto src_xy = find_rotation_src({ x, y }, origin, radians);

				if (src_xy.x < zero || src_xy.x >= width || src_xy.y < zero || src_xy.y >= height)
				{
					d[x] = 0.0f;
				}
				else
				{
					d[x] = *xy_at(src, (u32)floorf(src_xy.x), (u32)floorf(src_xy.y));
				}
			}
		};

		process_rows(src.height, row_func);
	}
}