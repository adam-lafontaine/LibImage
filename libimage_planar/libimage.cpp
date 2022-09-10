#include "libimage.hpp"

#include <cstdlib>
#include <algorithm>
#include <cmath>


template <class LIST_T, class FUNC_T>
static void do_for_each_seq(LIST_T const& list, FUNC_T const& func)
{
	std::for_each(list.begin(), list.end(), func);
}


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
	do_for_each_seq(list, func);
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


static void process_rows(u32 row_begin, u32 row_end, id_func_t const& row_func)
{
	assert(row_end > row_begin);

	auto const height = row_end - row_begin;
	auto const rows_per_thread = height / N_THREADS;

	auto const thread_proc = [&](u32 id)
	{
		auto y_begin = row_begin + id * rows_per_thread;
		auto y_end = row_begin + (id == N_THREADS - 1 ? height : (id + 1) * rows_per_thread);

		for (u32 y = y_begin; y < y_end; ++y)
		{
			row_func(y);
		}
	};

	execute_procs(make_proc_list(thread_proc));
}


static constexpr r32 to_channel_r32(u8 value)
{
	return value / 255.0f;
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
	static bool verify(ImageCHr32<N> const& image)
	{
		return image.width && image.height && image.channel_data[0];
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


	static bool verify(Image1Cr32 const& image)
	{
		return image.width && image.height && image.data;
	}


	static bool verify(View1Cr32 const& view)
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
}


/* image templates */

namespace libimage
{

	template <size_t N>
	void do_make_image(ImageCHr32<N>& image, u32 width, u32 height)
	{
		auto n_pixels = width * height;

		auto data = (r32*)malloc(sizeof(r32) * N * n_pixels);
		assert(data);

		image.width = width;
		image.height = height;

		for (u32 ch = 0; ch < N; ++ch)
		{
			image.channel_data[ch] = data + ch * n_pixels;
		}
	}


	template <size_t N>
	static void do_destroy_image(ImageCHr32<N>& image)
	{
		if (image.channel_data[0])
		{
			free(image.channel_data[0]);
			for (u32 ch = 0; ch < N; ++ch)
			{
				image.channel_data[ch] = nullptr;
			}
		}

		image.width = 0;
		image.height = 0;
	}


	template <size_t N>
	static ViewCHr32<N> do_make_view(ImageCHr32<N> const& image)
	{
		ViewCHr32<N> view;

		view.image_width = image.width;
		view.x_begin = 0;
		view.y_begin = 0;
		view.x_end = image.width;
		view.y_end = image.height;
		view.width = image.width;
		view.height = image.height;

		for (u32 ch = 0; ch < N; ++ch)
		{
			view.image_channel_data[ch] = image.channel_data[ch];
		}

		return view;
	}


	template <size_t N>
	static ViewCHr32<N> do_sub_view(ImageCHr32<N> const& image, Range2Du32 const& range)
	{
		ViewCHr32<N> view;

		view.image_width = image.width;
		view.range = range;
		view.width = range.x_end - range.x_begin;
		view.height = range.y_end - range.y_begin;

		for (u32 ch = 0; ch < N; ++ch)
		{
			view.image_channel_data[ch] = image.channel_data[ch];
		}

		return view;
	}


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


	template <size_t N>
	static std::array<r32*, N> channel_row_begin(ImageCHr32<N> const& image, u32 y)
	{
		auto offset = (size_t)(y * image.width);

		std::array<r32*, N> data = {};
		for (u32 ch = 0; ch < N; ++ch)
		{
			data[ch] = image.channel_data[ch] + offset;
		}

		return data;
	}


	template <size_t N>
	static std::array<r32*, N> channel_row_begin(ViewCHr32<N> const& view, u32 y)
	{
		auto offset = (size_t)((view.y_begin + y) * view.image_width + view.x_begin);

		std::array<r32*, N> data = {};
		for (u32 ch = 0; ch < N; ++ch)
		{
			data[ch] = view.image_channel_data[ch] + offset;
		}

		return data;
	}
}


/* Images */

namespace libimage
{
	void make_image(Image4Cr32& image, u32 width, u32 height)
	{
		assert(width);
		assert(height);

		do_make_image(image, width, height);
	}


	void destroy_image(Image4Cr32& image)
	{
		do_destroy_image(image);
	}


	View4Cr32 make_view(Image4Cr32 const& image)
	{
		assert(verify(image));

		auto view = do_make_view(image);

		assert(verify(view));

		return view;
	}


	View4Cr32 sub_view(Image4Cr32 const& image, Range2Du32 const& range)
	{
		assert(verify(image, range));

		auto view = do_sub_view(image, range);

		assert(verify(view));

		return view;
	}


	View4Cr32 sub_view(View4Cr32 const& view, Range2Du32 const& range)
	{
		assert(verify(view));

		auto sub_view = do_sub_view(view, range);

		assert(verify(sub_view));

		return sub_view;
	}
	

	void make_image(Image3Cr32& image, u32 width, u32 height)
	{
		assert(width);
		assert(height);

		do_make_image(image, width, height);

		assert(verify(image));
	}


	void destroy_image(Image3Cr32& image)
	{
		do_destroy_image(image);
	}
	

	View3Cr32 make_view(Image3Cr32 const& image)
	{
		assert(verify(image));

		auto view = do_make_view(image);

		assert(verify(view));

		return view;
	}


	View3Cr32 sub_view(Image3Cr32 const& image, Range2Du32 const& range)
	{
		assert(verify(image, range));

		auto view = do_sub_view(image, range);

		assert(verify(view));

		return view;
	}


	View3Cr32 sub_view(View3Cr32 const& view, Range2Du32 const& range)
	{
		assert(verify(view, range));

		auto sub_view = do_sub_view(view, range);

		assert(verify(sub_view));

		return sub_view;
	}


	View3Cr32 make_rgb_view(Image4Cr32 const& image)
	{
		assert(verify(image));

		View3Cr32 view;

		view.image_width = image.width;
		view.x_begin = 0;
		view.y_begin = 0;
		view.x_end = image.width;
		view.y_end = image.height;
		view.width = image.width;
		view.height = image.height;

		for (u32 ch = 0; ch < 3; ++ch)
		{
			view.image_channel_data[ch] = image.channel_data[ch];
		}

		assert(verify(view));

		return view;
	}


	View3Cr32 make_rgb_view(View4Cr32 const& view)
	{
		assert(verify(view));

		View3Cr32 view3;

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


	/*r32* row_begin(View3Cr32 const& view, u32 y, RGB channel)
	{
		auto ch = id_cast(channel);

		assert(y < view.height);
		assert(view.image_channel_data[ch]);

		auto offset = (view.y_begin + y) * view.image_width + view.x_begin;

		auto ptr = view.image_channel_data[ch] + (u64)(offset);
		assert(ptr);

		return ptr;
	}*/


	/*r32* xy_at(View3Cr32 const& view, u32 x, u32 y, RGB channel)
	{
		assert(y < view.height);
		assert(x < view.width);

		return row_begin(view, y, channel) + x;
	}*/
	

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


	void make_image(Image1Cr32& image, u32 width, u32 height)
	{
		assert(width);
		assert(height);

		image.data = (r32*)malloc(sizeof(r32) * width * height);
		assert(image.data);

		image.width = width;
		image.height = height;
	}


	void destroy_image(Image1Cr32& image)
	{
		if (image.data != nullptr)
		{
			free(image.data);
			image.data = nullptr;
		}
	}


	r32* row_begin(Image1Cr32 const& image, u32 y)
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


	r32* xy_at(Image1Cr32 const& image, u32 x, u32 y)
	{
		assert(y < image.height);
		assert(x < image.width);

		return row_begin(image, y) + x;
	}
		

	View1Cr32 make_view(Image1Cr32 const& image)
	{
		assert(verify(image));

		View1Cr32 view;

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
	
	
	View1Cr32 sub_view(Image1Cr32 const& image, Range2Du32 const& range)
	{
		assert(verify(image, range));

		View1Cr32 view;

		view.image_data = image.data;
		view.image_width = image.width;
		view.range = range;
		view.width = range.x_end - range.x_begin;
		view.height = range.y_end - range.y_begin;

		assert(view.width);
		assert(view.height);

		return view;
	}


	View1Cr32 sub_view(View1Cr32 const& view, Range2Du32 const& range)
	{
		assert(verify(view, range));

		View1Cr32 sub_view;

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


	r32* row_begin(View1Cr32 const& view, u32 y)
	{
		assert(y < view.height);

		auto offset = (view.y_begin + y) * view.image_width + view.x_begin;

		auto ptr = view.image_data + (u64)(offset);
		assert(ptr);

		return ptr;
	}


	r32* xy_at(View1Cr32 const& view, u32 x, u32 y)
	{
		assert(y < view.height);
		assert(x < view.width);

		return row_begin(view, y) + x;
	}
		
}


/* convert */

namespace libimage
{
	template <class IMG_INT, class IMG_PLA>
	static void interleaved_to_planar(IMG_INT const& src, IMG_PLA const& dst)
	{
		auto const row_func = [&](u32 y)
		{
			auto d = channel_row_begin(dst, y);
			auto s = row_begin(src, y);
			for (u32 x = 0; x < src.width; ++x)
			{
				for (u32 ch = 0; ch < d.size(); ++ch)
				{
					d[ch][x] = to_channel_r32(s[x].channels[ch]);
				}
			}
		};

		process_rows(src.height, row_func);
	}


	template <class IMG_INT, class IMG_PLA>
	static void planar_to_interleaved(IMG_PLA const& src, IMG_INT const& dst)
	{
		auto const row_func = [&](u32 y)
		{
			auto s = channel_row_begin(src, y);
			auto d = row_begin(dst, y);

			for (u32 x = 0; x < src.width; ++x)
			{
				for (u32 ch = 0; ch < s.size(); ++ch)
				{
					d[x].channels[ch] = to_channel_u8(s[ch][x]);
				}
			}
		};

		process_rows(src.height, row_func);
	}


	void convert(Image4Cr32 const& src, Image const& dst)
	{
		assert(verify(src, dst));

		planar_to_interleaved(src, dst);
	}


	void convert(Image const& src, Image4Cr32 const& dst)
	{
		assert(verify(src, dst));

		interleaved_to_planar(src, dst);
	}


	void convert(Image4Cr32 const& src, View const& dst)
	{
		assert(verify(src, dst));

		planar_to_interleaved(src, dst);
	}


	void convert(View const& src, Image4Cr32 const& dst)
	{
		assert(verify(src, dst));

		interleaved_to_planar(src, dst);
	}


	void convert(View4Cr32 const& src, Image const& dst)
	{
		assert(verify(src, dst));

		planar_to_interleaved(src, dst);
	}


	void convert(Image const& src, View4Cr32 const& dst)
	{
		assert(verify(src, dst));

		interleaved_to_planar(src, dst);
	}


	void convert(View4Cr32 const& src, View const& dst)
	{
		assert(verify(src, dst));

		planar_to_interleaved(src, dst);
	}


	void convert(View const& src, View4Cr32 const& dst)
	{
		assert(verify(src, dst));

		interleaved_to_planar(src, dst);
	}


	void convert(Image3Cr32 const& src, Image const& dst)
	{
		assert(verify(src, dst));

		planar_to_interleaved(src, dst);
	}


	void convert(Image const& src, Image3Cr32 const& dst)
	{
		assert(verify(src, dst));

		interleaved_to_planar(src, dst);
	}


	void convert(Image3Cr32 const& src, View const& dst)
	{
		assert(verify(src, dst));

		planar_to_interleaved(src, dst);
	}


	void convert(View const& src, Image3Cr32 const& dst)
	{
		assert(verify(src, dst));

		interleaved_to_planar(src, dst);
	}


	void convert(View3Cr32 const& src, Image const& dst)
	{
		assert(verify(src, dst));

		planar_to_interleaved(src, dst);
	}


	void convert(Image const& src, View3Cr32 const& dst)
	{
		assert(verify(src, dst));

		interleaved_to_planar(src, dst);
	}


	void convert(View3Cr32 const& src, View const& dst)
	{
		assert(verify(src, dst));

		planar_to_interleaved(src, dst);
	}


	void convert(View const& src, View3Cr32 const& dst)
	{
		assert(verify(src, dst));

		interleaved_to_planar(src, dst);
	}


	void convert(Image1Cr32 const& src, gray::Image const& dst)
	{
		assert(verify(src, dst));

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


	void convert(gray::Image const& src, Image1Cr32 const& dst)
	{
		assert(verify(src, dst));

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


	void convert(Image1Cr32 const& src, gray::View const& dst)
	{
		assert(verify(src, dst));

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


	void convert(gray::View const& src, Image1Cr32 const& dst)
	{
		assert(verify(src, dst));

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


	void convert(View1Cr32 const& src, gray::Image const& dst)
	{
		assert(verify(src, dst));

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


	void convert(gray::Image const& src, View1Cr32 const& dst)
	{
		assert(verify(src, dst));

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


	void convert(View1Cr32 const& src, gray::View const& dst)
	{
		assert(verify(src, dst));

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


	void convert(gray::View const& src, View1Cr32 const& dst)
	{
		assert(verify(src, dst));

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


	template <class IMG, typename PIXEL>
	static void fill_n_channels(IMG const& image, PIXEL color)
	{
		auto const row_func = [&](u32 y)
		{
			auto d = channel_row_begin(image, y);
			for (u32 x = 0; x < image.width; ++x)
			{
				for (u32 ch = 0; ch < d.size(); ++ch)
				{
					d[ch][x] = to_channel_r32(color.channels[ch]);
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


	void fill(View4Cr32 const& view, Pixel color)
	{
		assert(verify(view));

		fill_n_channels(view, color);
	}


	void fill(View3Cr32 const& view, Pixel color)
	{
		assert(verify(view));

		fill_n_channels(view, color);
	}


	void fill(View1Cr32 const& view, u8 gray)
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


	template <class IMG_SRC, class IMG_DST>
	static void copy_n_channels(IMG_SRC const& src, IMG_DST const& dst)
	{
		auto const row_func = [&](u32 y)
		{
			auto s = channel_row_begin(src, y);
			auto d = channel_row_begin(dst, y);
			for (u32 x = 0; x < src.width; ++x)
			{
				for (u32 ch = 0; ch < d.size(); ++ch)
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


	void copy(View4Cr32 const& src, View4Cr32 const& dst)
	{
		assert(verify(src, dst));

		copy_n_channels(src, dst);
	}


	void copy(View3Cr32 const& src, View3Cr32 const& dst)
	{
		assert(verify(src, dst));

		copy_n_channels(src, dst);
	}


	void copy(View1Cr32 const& src, View1Cr32 const& dst)
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
		verify(image);

		do_for_each_pixel(image, func);
	}


	void for_each_pixel(gray::View const& view, u8_f const& func)
	{
		verify(view);

		do_for_each_pixel(view, func);
	}


	void for_each_pixel(View1Cr32 const& view, r32_f const& func)
	{
		verify(view);

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


	void for_each_xy(View4Cr32 const& view, xy_f const& func)
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


	template <class IMG, class GRAY>
	static void grayscale_rgb(IMG const& src, GRAY const& dst)
	{
		constexpr static auto red = id_cast(RGB::R);
		constexpr static auto green = id_cast(RGB::G);
		constexpr static auto blue = id_cast(RGB::B);

		auto const row_func = [&](u32 y)
		{
			auto s = channel_row_begin(src, y);
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


	void grayscale(View4Cr32 const& src, View1Cr32 const& dst)
	{
		assert(verify(src, dst));

		grayscale_rgb(src, dst);
	}


	void grayscale(View3Cr32 const& src, View1Cr32 const& dst)
	{
		assert(verify(src, dst));

		grayscale_rgb(src, dst);
	}


}


/* select_channel */

namespace libimage
{
	/*View1Cr32 select_channel(Image4Cr32 const& image, RGBA channel)
	{
		assert(verify(image));

		auto ch = id_cast(channel);

		View1Cr32 view1{};

		view1.image_width = image.width;
		view1.x_begin = 0;
		view1.y_begin = 0;
		view1.x_end = image.width;
		view1.y_end = image.height;
		view1.width = image.width;
		view1.height = image.height;

		view1.image_data = image.channel_data[ch];

		assert(verify(view1));

		return view1;
	}*/


	View1Cr32 select_channel(View4Cr32 const& view, RGBA channel)
	{
		assert(verify(view));

		auto ch = id_cast(channel);

		View1Cr32 view1{};

		view1.image_width = view.image_width;
		view1.range = view.range;
		view1.width = view.width;
		view1.height = view.height;

		view1.image_data = view.image_channel_data[ch];

		assert(verify(view1));

		return view1;
	}


	/*View1Cr32 select_channel(Image3Cr32 const& image, RGB channel)
	{
		assert(verify(image));

		auto ch = id_cast(channel);

		View1Cr32 view1{};

		view1.image_width = image.width;
		view1.x_begin = 0;
		view1.y_begin = 0;
		view1.x_end = image.width;
		view1.y_end = image.height;
		view1.width = image.width;
		view1.height = image.height;

		view1.image_data = image.channel_data[ch];

		assert(verify(view1));

		return view1;
	}*/


	View1Cr32 select_channel(View3Cr32 const& view, RGB channel)
	{
		assert(verify(view));

		auto ch = id_cast(channel);

		View1Cr32 view1{};

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


	template <class IMG_4_SRC, class IMG_3_CUR, class IMG_3_DST>
	static void do_alpha_blend(IMG_4_SRC const& src, IMG_3_CUR const& cur, IMG_3_DST const& dst)
	{
		constexpr static auto red = id_cast(RGBA::R);
		constexpr static auto green = id_cast(RGBA::G);
		constexpr static auto blue = id_cast(RGBA::B);
		constexpr static auto alpha = id_cast(RGBA::A);

		auto const row_func = [&](u32 y)
		{
			auto s = channel_row_begin(src, y);
			auto c = channel_row_begin(cur, y);
			auto d = channel_row_begin(dst, y);

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


	void alpha_blend(View4Cr32 const& src, View3Cr32 const& cur, View3Cr32 const& dst)
	{
		assert(verify(src, cur));
		assert(verify(src, dst));

		do_alpha_blend(src, cur, dst);
	}


	void alpha_blend(View4Cr32 const& src, View3Cr32 const& cur_dst)
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


	template <class IMG_1C_SRC, class IMG_1C_DST>
	static void do_transform_r32(IMG_1C_SRC const& src, IMG_1C_DST const& dst, r32_to_r32_f const& func)
	{
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


	void transform(View1Cr32 const& src, View1Cr32 const& dst, r32_to_r32_f const& func)
	{
		assert(verify(src, dst));

		do_transform_r32(src, dst, func);
	}

}


