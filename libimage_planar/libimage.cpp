#include "libimage.hpp"

#include <cstdlib>
#include <algorithm>
#include <functional>
#include <array>
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


static inline r32 to_channel_r32(u8 value)
{
	return value / 255.0f;
}


static inline u8 to_channel_u8(r32 value)
{
	if (value < 0.0f)
	{
		value = 0.0f;
	}
	else if (value > 1.0f)
	{
		value = 1.0f;
	}

	return (u8)std::round(value * 255);
}


namespace libimage
{
	static Pixel to_pixel(r32 r, r32 g, r32 b, r32 a)
	{
		Pixel p{};
		p.red = to_channel_u8(r);
		p.green = to_channel_u8(g);
		p.blue = to_channel_u8(b);
		p.alpha = to_channel_u8(a);

		return p;
	}


	static Pixel to_pixel(r32 r, r32 g, r32 b)
	{
		Pixel p{};
		p.red = to_channel_u8(r);
		p.green = to_channel_u8(g);
		p.blue = to_channel_u8(b);
		p.alpha = 255;

		return p;
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


	void make_image(Image4Cr32& image, u32 width, u32 height)
	{
		assert(width);
		assert(height);

		auto n_pixels = width * height;

		auto data = (r32*)malloc(sizeof(r32) * 4 * n_pixels);
		assert(data);

		image.width = width;
		image.height = height;

		image.red = data;
		image.green = image.red + n_pixels;
		image.blue = image.green + n_pixels;
		image.alpha = image.blue + n_pixels;
	}


	void destroy_image(Image4Cr32& image)
	{
		if (image.red != nullptr)
		{
			free(image.red);
			image.red = nullptr;
			image.green = nullptr;
			image.blue = nullptr;
			image.alpha = nullptr;
		}
	}


	r32* row_begin(Image4Cr32 const& image, u32 y, RGBA channel)
	{
		auto ch = static_cast<int>(channel);

		assert(image.width);
		assert(image.height);
		assert(image.channel_data[ch]);
		assert(y < image.height);

		auto offset = y * image.width;

		auto ptr = image.channel_data[ch] + (u64)(offset);
		assert(ptr);

		return ptr;
	}


	r32* xy_at(Image4Cr32 const& image, u32 x, u32 y, RGBA channel)
	{
		assert(y < image.height);
		assert(x < image.width);

		return row_begin(image, y, channel) + x;
	}
	

	View4Cr32 make_view(Image4Cr32 const& image)
	{
		assert(image.width);
		assert(image.height);
		assert(image.red);

		View4Cr32 view;

		view.image_red = image.red;
		view.image_green = image.green;
		view.image_blue = image.blue;
		view.image_alpha = image.alpha;
		view.image_width = image.width;
		view.x_begin = 0;
		view.y_begin = 0;
		view.x_end = image.width;
		view.y_end = image.height;
		view.width = image.width;
		view.height = image.height;

		return view;
	}


	View4Cr32 sub_view(Image4Cr32 const& image, Range2Du32 const& range)
	{
		assert(image.width);
		assert(image.height);
		assert(image.red);

		View4Cr32 view;

		view.image_red = image.red;
		view.image_green = image.green;
		view.image_blue = image.blue;
		view.image_alpha = image.alpha;
		view.image_width = image.width;
		view.range = range;
		view.width = range.x_end - range.x_begin;
		view.height = range.y_end - range.y_begin;

		assert(view.width);
		assert(view.height);

		return view;
	}


	View4Cr32 sub_view(View4Cr32 const& view, Range2Du32 const& range)
	{
		assert(view.width);
		assert(view.height);
		assert(view.image_red);

		assert(range.x_begin >= view.x_begin);
		assert(range.x_end <= view.x_end);
		assert(range.y_begin >= view.y_begin);
		assert(range.y_end <= view.y_end);

		View4Cr32 sub_view;

		sub_view.image_red = view.image_red;
		sub_view.image_green = view.image_green;
		sub_view.image_blue = view.image_blue;
		sub_view.image_alpha = view.image_alpha;
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


	r32* row_begin(View4Cr32 const& view, u32 y, RGBA channel)
	{
		auto ch = static_cast<int>(channel);

		assert(y < view.height);
		assert(view.image_channel_data[ch]);

		auto offset = (view.y_begin + y) * view.image_width + view.x_begin;

		auto ptr = view.image_channel_data[ch] + (u64)(offset);
		assert(ptr);

		return ptr;
	}


	r32* xy_at(View4Cr32 const& view, u32 x, u32 y, RGBA channel)
	{
		assert(y < view.height);
		assert(x < view.width);

		return row_begin(view, y, channel) + x;
	}
	

	void make_image(Image3Cr32& image, u32 width, u32 height)
	{
		assert(width);
		assert(height);

		auto n_pixels = width * height;

		auto data = (r32*)malloc(sizeof(r32) * 3 * n_pixels);
		assert(data);

		image.width = width;
		image.height = height;

		image.red = data;
		image.green = image.red + n_pixels;
		image.blue = image.green + n_pixels;
	}


	void destroy_image(Image3Cr32& image)
	{
		if (image.red != nullptr)
		{
			free(image.red);
			image.red = nullptr;
			image.green = nullptr;
			image.blue = nullptr;
		}
	}


	r32* row_begin(Image3Cr32 const& image, u32 y, RGB channel)
	{
		auto ch = static_cast<int>(channel);

		assert(image.width);
		assert(image.height);
		assert(image.channel_data[ch]);
		assert(y < image.height);

		auto offset = y * image.width;

		auto ptr = image.channel_data[ch] + (u64)(offset);
		assert(ptr);

		return ptr;
	}


	r32* xy_at(Image3Cr32 const& image, u32 x, u32 y, RGB channel)
	{
		assert(y < image.height);
		assert(x < image.width);

		return row_begin(image, y, channel) + x;
	}
	

	View3Cr32 make_view(Image3Cr32 const& image)
	{
		assert(image.width);
		assert(image.height);
		assert(image.red);

		View3Cr32 view;

		view.image_red = image.red;
		view.image_green = image.green;
		view.image_blue = image.blue;
		view.image_width = image.width;
		view.x_begin = 0;
		view.y_begin = 0;
		view.x_end = image.width;
		view.y_end = image.height;
		view.width = image.width;
		view.height = image.height;

		return view;
	}


	View3Cr32 sub_view(Image3Cr32 const& image, Range2Du32 const& range)
	{
		assert(image.width);
		assert(image.height);
		assert(image.red);

		View3Cr32 view;

		view.image_red = image.red;
		view.image_green = image.green;
		view.image_blue = image.blue;
		view.image_width = image.width;
		view.range = range;
		view.width = range.x_end - range.x_begin;
		view.height = range.y_end - range.y_begin;

		assert(view.width);
		assert(view.height);

		return view;
	}


	View3Cr32 sub_view(View3Cr32 const& view, Range2Du32 const& range)
	{
		assert(view.width);
		assert(view.height);
		assert(view.image_red);

		assert(range.x_begin >= view.x_begin);
		assert(range.x_end <= view.x_end);
		assert(range.y_begin >= view.y_begin);
		assert(range.y_end <= view.y_end);

		View3Cr32 sub_view;

		sub_view.image_red = view.image_red;
		sub_view.image_green = view.image_green;
		sub_view.image_blue = view.image_blue;
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


	r32* row_begin(View3Cr32 const& view, u32 y, RGB channel)
	{
		auto ch = static_cast<int>(channel);

		assert(y < view.height);
		assert(view.image_channel_data[ch]);

		auto offset = (view.y_begin + y) * view.image_width + view.x_begin;

		auto ptr = view.image_channel_data[ch] + (u64)(offset);
		assert(ptr);

		return ptr;
	}


	r32* xy_at(View3Cr32 const& view, u32 x, u32 y, RGB channel)
	{
		assert(y < view.height);
		assert(x < view.width);

		return row_begin(view, y, channel) + x;
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
		assert(image.width);
		assert(image.height);
		assert(image.data);

		gray::View view;

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


	gray::View sub_view(gray::Image const& image, Range2Du32 const& range)
	{
		assert(image.width);
		assert(image.height);
		assert(image.data);

		gray::View view;

		view.image_data = image.data;
		view.image_width = image.width;
		view.range = range;
		view.width = range.x_end - range.x_begin;
		view.height = range.y_end - range.y_begin;

		assert(view.width);
		assert(view.height);

		return view;
	}


	gray::View sub_view(gray::View const& view, Range2Du32 const& range)
	{
		assert(view.width);
		assert(view.height);
		assert(view.image_data);

		assert(range.x_begin >= view.x_begin);
		assert(range.x_end <= view.x_end);
		assert(range.y_begin >= view.y_begin);
		assert(range.y_end <= view.y_end);

		gray::View sub_view;

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
		assert(image.width);
		assert(image.height);
		assert(image.data);

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
		assert(image.width);
		assert(image.height);
		assert(image.data);

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
		assert(view.width);
		assert(view.height);
		assert(view.image_data);

		assert(range.x_begin >= view.x_begin);
		assert(range.x_end <= view.x_end);
		assert(range.y_begin >= view.y_begin);
		assert(range.y_end <= view.y_end);

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
	void convert(Image4Cr32 const& src, Image const& dst)
	{
		assert(src.width == dst.width);
		assert(src.height == dst.height);
		assert(src.red);
		assert(dst.data);

		auto const row_func = [&](u32 y)
		{
			auto r = row_begin(src, y, RGBA::R);
			auto g = row_begin(src, y, RGBA::G);
			auto b = row_begin(src, y, RGBA::B);
			auto a = row_begin(src, y, RGBA::A);
			auto d = row_begin(dst, y);
			for (u32 x = 0; x < src.width; ++x)
			{
				d[x] = to_pixel(r[x], g[x], b[x], a[x]);
			}
		};

		process_rows(src.height, row_func);
	}


	void convert(Image const& src, Image4Cr32 const& dst)
	{
		assert(src.width == dst.width);
		assert(src.height == dst.height);
		assert(src.data);
		assert(dst.red);

		auto const row_func = [&](u32 y)
		{
			auto r = row_begin(dst, y, RGBA::R);
			auto g = row_begin(dst, y, RGBA::G);
			auto b = row_begin(dst, y, RGBA::B);
			auto a = row_begin(dst, y, RGBA::A);
			auto s = row_begin(src, y);
			for (u32 x = 0; x < src.width; ++x)
			{
				r[x] = to_channel_r32(s[x].red);
				g[x] = to_channel_r32(s[x].green);
				b[x] = to_channel_r32(s[x].blue);
				a[x] = to_channel_r32(s[x].alpha);
			}
		};

		process_rows(src.height, row_func);
	}


	void convert(Image4Cr32 const& src, View const& dst)
	{
		assert(src.width == dst.width);
		assert(src.height == dst.height);
		assert(src.red);
		assert(dst.image_data);

		auto const row_func = [&](u32 y)
		{
			auto r = row_begin(src, y, RGBA::R);
			auto g = row_begin(src, y, RGBA::G);
			auto b = row_begin(src, y, RGBA::B);
			auto a = row_begin(src, y, RGBA::A);
			auto d = row_begin(dst, y);
			for (u32 x = 0; x < src.width; ++x)
			{
				d[x] = to_pixel(r[x], g[x], b[x], a[x]);
			}
		};

		process_rows(src.height, row_func);
	}


	void convert(View const& src, Image4Cr32 const& dst)
	{
		assert(src.width == dst.width);
		assert(src.height == dst.height);
		assert(src.image_data);
		assert(dst.red);

		auto const row_func = [&](u32 y)
		{
			auto r = row_begin(dst, y, RGBA::R);
			auto g = row_begin(dst, y, RGBA::G);
			auto b = row_begin(dst, y, RGBA::B);
			auto a = row_begin(dst, y, RGBA::A);
			auto s = row_begin(src, y);
			for (u32 x = 0; x < src.width; ++x)
			{
				r[x] = to_channel_r32(s[x].red);
				g[x] = to_channel_r32(s[x].green);
				b[x] = to_channel_r32(s[x].blue);
				a[x] = to_channel_r32(s[x].alpha);
			}
		};

		process_rows(src.height, row_func);
	}


	void convert(View4Cr32 const& src, Image const& dst)
	{
		assert(src.width == dst.width);
		assert(src.height == dst.height);
		assert(src.image_red);
		assert(dst.data);

		auto const row_func = [&](u32 y)
		{
			auto r = row_begin(src, y, RGBA::R);
			auto g = row_begin(src, y, RGBA::G);
			auto b = row_begin(src, y, RGBA::B);
			auto a = row_begin(src, y, RGBA::A);
			auto d = row_begin(dst, y);
			for (u32 x = 0; x < src.width; ++x)
			{
				d[x] = to_pixel(r[x], g[x], b[x], a[x]);
			}
		};

		process_rows(src.height, row_func);
	}


	void convert(Image const& src, View4Cr32 const& dst)
	{
		assert(src.width == dst.width);
		assert(src.height == dst.height);
		assert(src.data);
		assert(dst.image_red);

		auto const row_func = [&](u32 y)
		{
			auto r = row_begin(dst, y, RGBA::R);
			auto g = row_begin(dst, y, RGBA::G);
			auto b = row_begin(dst, y, RGBA::B);
			auto a = row_begin(dst, y, RGBA::A);
			auto s = row_begin(src, y);
			for (u32 x = 0; x < src.width; ++x)
			{
				r[x] = to_channel_r32(s[x].red);
				g[x] = to_channel_r32(s[x].green);
				b[x] = to_channel_r32(s[x].blue);
				a[x] = to_channel_r32(s[x].alpha);
			}
		};

		process_rows(src.height, row_func);
	}


	void convert(View4Cr32 const& src, View const& dst)
	{
		assert(src.width == dst.width);
		assert(src.height == dst.height);
		assert(src.image_red);
		assert(dst.image_data);

		auto const row_func = [&](u32 y)
		{
			auto r = row_begin(src, y, RGBA::R);
			auto g = row_begin(src, y, RGBA::G);
			auto b = row_begin(src, y, RGBA::B);
			auto a = row_begin(src, y, RGBA::A);
			auto d = row_begin(dst, y);
			for (u32 x = 0; x < src.width; ++x)
			{
				d[x] = to_pixel(r[x], g[x], b[x], a[x]);
			}
		};

		process_rows(src.height, row_func);
	}


	void convert(View const& src, View4Cr32 const& dst)
	{
		assert(src.width == dst.width);
		assert(src.height == dst.height);
		assert(src.image_data);
		assert(dst.image_red);

		auto const row_func = [&](u32 y)
		{
			auto r = row_begin(dst, y, RGBA::R);
			auto g = row_begin(dst, y, RGBA::G);
			auto b = row_begin(dst, y, RGBA::B);
			auto a = row_begin(dst, y, RGBA::A);
			auto s = row_begin(src, y);
			for (u32 x = 0; x < src.width; ++x)
			{
				r[x] = to_channel_r32(s[x].red);
				g[x] = to_channel_r32(s[x].green);
				b[x] = to_channel_r32(s[x].blue);
				a[x] = to_channel_r32(s[x].alpha);
			}
		};

		process_rows(src.height, row_func);
	}


	void convert(Image3Cr32 const& src, Image const& dst)
	{
		assert(src.width == dst.width);
		assert(src.height == dst.height);
		assert(src.red);
		assert(dst.data);

		auto const row_func = [&](u32 y)
		{
			auto r = row_begin(src, y, RGB::R);
			auto g = row_begin(src, y, RGB::G);
			auto b = row_begin(src, y, RGB::B);
			auto d = row_begin(dst, y);
			for (u32 x = 0; x < src.width; ++x)
			{
				d[x] = to_pixel(r[x], g[x], b[x]);
			}
		};

		process_rows(src.height, row_func);
	}


	void convert(Image const& src, Image3Cr32 const& dst)
	{
		assert(src.width == dst.width);
		assert(src.height == dst.height);
		assert(src.data);
		assert(dst.red);

		auto const row_func = [&](u32 y)
		{
			auto r = row_begin(dst, y, RGB::R);
			auto g = row_begin(dst, y, RGB::G);
			auto b = row_begin(dst, y, RGB::B);
			auto s = row_begin(src, y);
			for (u32 x = 0; x < src.width; ++x)
			{
				r[x] = to_channel_r32(s[x].red);
				g[x] = to_channel_r32(s[x].green);
				b[x] = to_channel_r32(s[x].blue);
			}
		};

		process_rows(src.height, row_func);
	}


	void convert(Image3Cr32 const& src, View const& dst)
	{
		assert(src.width == dst.width);
		assert(src.height == dst.height);
		assert(src.red);
		assert(dst.image_data);

		auto const row_func = [&](u32 y)
		{
			auto r = row_begin(src, y, RGB::R);
			auto g = row_begin(src, y, RGB::G);
			auto b = row_begin(src, y, RGB::B);
			auto d = row_begin(dst, y);
			for (u32 x = 0; x < src.width; ++x)
			{
				d[x] = to_pixel(r[x], g[x], b[x]);
			}
		};

		process_rows(src.height, row_func);
	}


	void convert(View const& src, Image3Cr32 const& dst)
	{
		assert(src.width == dst.width);
		assert(src.height == dst.height);
		assert(src.image_data);
		assert(dst.red);

		auto const row_func = [&](u32 y)
		{
			auto r = row_begin(dst, y, RGB::R);
			auto g = row_begin(dst, y, RGB::G);
			auto b = row_begin(dst, y, RGB::B);
			auto s = row_begin(src, y);
			for (u32 x = 0; x < src.width; ++x)
			{
				r[x] = to_channel_r32(s[x].red);
				g[x] = to_channel_r32(s[x].green);
				b[x] = to_channel_r32(s[x].blue);
			}
		};

		process_rows(src.height, row_func);
	}


	void convert(View3Cr32 const& src, Image const& dst)
	{
		assert(src.width == dst.width);
		assert(src.height == dst.height);
		assert(src.image_red);
		assert(dst.data);

		auto const row_func = [&](u32 y)
		{
			auto r = row_begin(src, y, RGB::R);
			auto g = row_begin(src, y, RGB::G);
			auto b = row_begin(src, y, RGB::B);
			auto d = row_begin(dst, y);
			for (u32 x = 0; x < src.width; ++x)
			{
				d[x] = to_pixel(r[x], g[x], b[x]);
			}
		};

		process_rows(src.height, row_func);
	}


	void convert(Image const& src, View3Cr32 const& dst)
	{
		assert(src.width == dst.width);
		assert(src.height == dst.height);
		assert(src.data);
		assert(dst.image_red);

		auto const row_func = [&](u32 y)
		{
			auto r = row_begin(dst, y, RGB::R);
			auto g = row_begin(dst, y, RGB::G);
			auto b = row_begin(dst, y, RGB::B);
			auto s = row_begin(src, y);
			for (u32 x = 0; x < src.width; ++x)
			{
				r[x] = to_channel_r32(s[x].red);
				g[x] = to_channel_r32(s[x].green);
				b[x] = to_channel_r32(s[x].blue);
			}
		};

		process_rows(src.height, row_func);
	}


	void convert(View3Cr32 const& src, View const& dst)
	{
		assert(src.width == dst.width);
		assert(src.height == dst.height);
		assert(src.image_red);
		assert(dst.image_data);

		auto const row_func = [&](u32 y)
		{
			auto r = row_begin(src, y, RGB::R);
			auto g = row_begin(src, y, RGB::G);
			auto b = row_begin(src, y, RGB::B);
			auto d = row_begin(dst, y);
			for (u32 x = 0; x < src.width; ++x)
			{
				d[x] = to_pixel(r[x], g[x], b[x]);
			}
		};

		process_rows(src.height, row_func);
	}


	void convert(View const& src, View3Cr32 const& dst)
	{
		assert(src.width == dst.width);
		assert(src.height == dst.height);
		assert(src.image_data);
		assert(dst.image_red);

		auto const row_func = [&](u32 y)
		{
			auto r = row_begin(dst, y, RGB::R);
			auto g = row_begin(dst, y, RGB::G);
			auto b = row_begin(dst, y, RGB::B);
			auto s = row_begin(src, y);
			for (u32 x = 0; x < src.width; ++x)
			{
				r[x] = to_channel_r32(s[x].red);
				g[x] = to_channel_r32(s[x].green);
				b[x] = to_channel_r32(s[x].blue);
			}
		};

		process_rows(src.height, row_func);
	}


	void convert(Image1Cr32 const& src, gray::Image const& dst)
	{
		assert(src.width == dst.width);
		assert(src.height == dst.height);
		assert(src.data);
		assert(dst.data);

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
		assert(src.width == dst.width);
		assert(src.height == dst.height);
		assert(src.data);
		assert(dst.data);

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
		assert(src.width == dst.width);
		assert(src.height == dst.height);
		assert(src.data);
		assert(dst.image_data);

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
		assert(src.width == dst.width);
		assert(src.height == dst.height);
		assert(src.image_data);
		assert(dst.data);

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
		assert(src.width == dst.width);
		assert(src.height == dst.height);
		assert(src.image_data);
		assert(dst.data);

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
		assert(src.width == dst.width);
		assert(src.height == dst.height);
		assert(src.data);
		assert(dst.image_data);

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
		assert(src.width == dst.width);
		assert(src.height == dst.height);
		assert(src.image_data);
		assert(dst.image_data);

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
		assert(src.width == dst.width);
		assert(src.height == dst.height);
		assert(src.image_data);
		assert(dst.image_data);

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