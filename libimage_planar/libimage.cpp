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


static inline int to_channel_index(auto channel)
{
	return static_cast<int>(channel);
}


/* verify */

#ifndef NDEBUG

namespace libimage
{
	static bool verify(Image const& image)
	{
		return image.width && image.height && image.data;
	}


	static bool verify(View const& image)
	{
		return image.width && image.height && image.image_data;
	}


	static bool verify(Image4Cr32 const& image)
	{
		return image.width && image.height && image.red;
	}


	static bool verify(View4Cr32 const& image)
	{
		return image.width && image.height && image.image_red;
	}


	static bool verify(Image3Cr32 const& image)
	{
		return image.width && image.height && image.red;
	}


	static bool verify(View3Cr32 const& image)
	{
		return image.width && image.height && image.image_red;
	}


	static bool verify(gray::Image const& image)
	{
		return image.width && image.height && image.data;
	}


	static bool verify(gray::View const& image)
	{
		return image.width && image.height && image.image_data;
	}


	static bool verify(Image1Cr32 const& image)
	{
		return image.width && image.height && image.data;
	}


	static bool verify(View1Cr32 const& image)
	{
		return image.width && image.height && image.image_data;
	}


	template <class IMG_A, class IMG_B>
	static bool verify(IMG_A const& lhs, IMG_B const& rhs)
	{
		return
			verify(lhs) && verify(rhs) &&
			lhs.width == rhs.width &&
			lhs.height == rhs.height;
	}
}

#endif // !NDEBUG


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
		auto ch = to_channel_index(channel);

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
		auto ch = to_channel_index(channel);

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
		auto ch = to_channel_index(channel);

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


	View3Cr32 make_rgb_view(Image4Cr32 const& image)
	{
		assert(verify(image));

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


	View3Cr32 make_rgb_view(View4Cr32 const& view)
	{
		assert(verify(view));

		View3Cr32 view3;

		view3.image_red = view.image_red;
		view3.image_green = view.image_green;
		view3.image_blue = view.image_blue;
		view3.image_width = view.image_width;
		view3.range = view.range;
		view3.width = view.width;
		view3.height = view.height;

		return view3;
	}


	r32* row_begin(View3Cr32 const& view, u32 y, RGB channel)
	{
		auto ch = to_channel_index(channel);

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
		assert(verify(src, dst));

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
		assert(verify(src, dst));

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
		assert(verify(src, dst));

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
		assert(verify(src, dst));

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
		assert(verify(src, dst));

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
		assert(verify(src, dst));

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
		assert(verify(src, dst));

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
		assert(verify(src, dst));

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
		assert(verify(src, dst));

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
		assert(verify(src, dst));

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
		assert(verify(src, dst));

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
		assert(verify(src, dst));

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
		assert(verify(src, dst));

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
		assert(verify(src, dst));

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
		assert(verify(src, dst));

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
		assert(verify(src, dst));

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


	void fill(Image4Cr32 const& image, Pixel color)
	{
		assert(verify(image));

		auto const row_func = [&](u32 y)
		{
			auto r = row_begin(image, y, RGBA::R);
			auto g = row_begin(image, y, RGBA::G);
			auto b = row_begin(image, y, RGBA::B);
			auto a = row_begin(image, y, RGBA::A);
			for (u32 x = 0; x < image.width; ++x)
			{
				r[x] = to_channel_r32(color.red);
				g[x] = to_channel_r32(color.green);
				b[x] = to_channel_r32(color.blue);
				a[x] = to_channel_r32(color.alpha);
			}
		};

		process_rows(image.height, row_func);
	}


	void fill(View4Cr32 const& view, Pixel color)
	{
		assert(verify(view));

		auto const row_func = [&](u32 y)
		{
			auto r = row_begin(view, y, RGBA::R);
			auto g = row_begin(view, y, RGBA::G);
			auto b = row_begin(view, y, RGBA::B);
			auto a = row_begin(view, y, RGBA::A);
			for (u32 x = 0; x < view.width; ++x)
			{
				r[x] = to_channel_r32(color.red);
				g[x] = to_channel_r32(color.green);
				b[x] = to_channel_r32(color.blue);
				a[x] = to_channel_r32(color.alpha);
			}
		};

		process_rows(view.height, row_func);
	}


	void fill(Image3Cr32 const& image, Pixel color)
	{
		assert(verify(image));

		auto const row_func = [&](u32 y)
		{
			auto r = row_begin(image, y, RGB::R);
			auto g = row_begin(image, y, RGB::G);
			auto b = row_begin(image, y, RGB::B);
			for (u32 x = 0; x < image.width; ++x)
			{
				r[x] = to_channel_r32(color.red);
				g[x] = to_channel_r32(color.green);
				b[x] = to_channel_r32(color.blue);
			}
		};

		process_rows(image.height, row_func);
	}


	void fill(View3Cr32 const& view, Pixel color)
	{
		assert(verify(view));

		auto const row_func = [&](u32 y)
		{
			auto r = row_begin(view, y, RGB::R);
			auto g = row_begin(view, y, RGB::G);
			auto b = row_begin(view, y, RGB::B);
			for (u32 x = 0; x < view.width; ++x)
			{
				r[x] = to_channel_r32(color.red);
				g[x] = to_channel_r32(color.green);
				b[x] = to_channel_r32(color.blue);
			}
		};

		process_rows(view.height, row_func);
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


	void fill(Image1Cr32 const& image, u8 gray)
	{
		assert(verify(image));

		fill_1_channel(image, gray);
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
	static void copy_4_channels(IMG_SRC const& src, IMG_DST const& dst)
	{
		auto const row_func = [&](u32 y)
		{
			auto sr = row_begin(src, y, RGBA::R);
			auto sg = row_begin(src, y, RGBA::G);
			auto sb = row_begin(src, y, RGBA::B);
			auto sa = row_begin(src, y, RGBA::A);

			auto dr = row_begin(dst, y, RGBA::R);
			auto dg = row_begin(dst, y, RGBA::G);
			auto db = row_begin(dst, y, RGBA::B);
			auto da = row_begin(dst, y, RGBA::A);
			for (u32 x = 0; x < src.width; ++x)
			{
				dr[x] = sr[x];
				dg[x] = sg[x];
				db[x] = sb[x];
				da[x] = sa[x];
			}
		};

		process_rows(src.height, row_func);
	}


	template <class IMG_SRC, class IMG_DST>
	static void copy_3_channels(IMG_SRC const& src, IMG_DST const& dst)
	{
		auto const row_func = [&](u32 y)
		{
			auto sr = row_begin(src, y, RGB::R);
			auto sg = row_begin(src, y, RGB::G);
			auto sb = row_begin(src, y, RGB::B);

			auto dr = row_begin(dst, y, RGB::R);
			auto dg = row_begin(dst, y, RGB::G);
			auto db = row_begin(dst, y, RGB::B);
			for (u32 x = 0; x < src.width; ++x)
			{
				dr[x] = sr[x];
				dg[x] = sg[x];
				db[x] = sb[x];
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


	void copy(Image4Cr32 const& src, Image4Cr32 const& dst)
	{
		assert(verify(src, dst));

		copy_4_channels(src, dst);
	}


	void copy(Image4Cr32 const& src, View4Cr32 const& dst)
	{
		assert(verify(src, dst));

		copy_4_channels(src, dst);
	}


	void copy(View4Cr32 const& src, Image4Cr32 const& dst)
	{
		assert(verify(src, dst));

		copy_4_channels(src, dst);
	}


	void copy(View4Cr32 const& src, View4Cr32 const& dst)
	{
		assert(verify(src, dst));

		copy_4_channels(src, dst);
	}


	void copy(Image3Cr32 const& src, Image3Cr32 const& dst)
	{
		assert(verify(src, dst));

		copy_3_channels(src, dst);
	}


	void copy(Image3Cr32 const& src, View3Cr32 const& dst)
	{
		assert(verify(src, dst));

		copy_3_channels(src, dst);
	}


	void copy(View3Cr32 const& src, Image3Cr32 const& dst)
	{
		assert(verify(src, dst));

		copy_3_channels(src, dst);
	}


	void copy(View3Cr32 const& src, View3Cr32 const& dst)
	{
		assert(verify(src, dst));

		copy_3_channels(src, dst);
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


	void copy(Image1Cr32 const& src, Image1Cr32 const& dst)
	{
		assert(verify(src, dst));

		copy_1_channel(src, dst);
	}


	void copy(Image1Cr32 const& src, View1Cr32 const& dst)
	{
		assert(verify(src, dst));

		copy_1_channel(src, dst);
	}


	void copy(View1Cr32 const& src, Image1Cr32 const& dst)
	{
		assert(verify(src, dst));

		copy_1_channel(src, dst);
	}


	void copy(View1Cr32 const& src, View1Cr32 const& dst)
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


	template <class IMG, class GRAY>
	static void grayscale_platform(IMG const& src, GRAY const& dst)
	{
		auto const row_func = [&](u32 y)
		{
			auto s = row_begin(src, y);
			auto d = row_begin(dst, y);
			for (u32 x = 0; x < src.width; ++x)
			{
				auto r = (r32)(s[x].red);
				auto g = (r32)(s[x].green);
				auto b = (r32)(s[x].blue);
				d[x] = (u8)rgb_grayscale_standard(r, g, b);
			}
		};

		process_rows(src.height, row_func);
	}


	template <class IMG, class GRAY>
	static void grayscale_rgba(IMG const& src, GRAY const& dst)
	{
		auto const row_func = [&](u32 y)
		{
			auto r = row_begin(src, y, RGBA::R);
			auto g = row_begin(src, y, RGBA::G);
			auto b = row_begin(src, y, RGBA::B);
			auto d = row_begin(dst, y);
			for (u32 x = 0; x < src.width; ++x)
			{
				d[x] = rgb_grayscale_standard(r[x], g[x], b[x]);
			}
		};

		process_rows(src.height, row_func);
	}


	template <class IMG, class GRAY>
	static void grayscale_rgb(IMG const& src, GRAY const& dst)
	{
		auto const row_func = [&](u32 y)
		{
			auto r = row_begin(src, y, RGB::R);
			auto g = row_begin(src, y, RGB::G);
			auto b = row_begin(src, y, RGB::B);
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


	void grayscale(Image4Cr32 const& src, Image1Cr32 const& dst)
	{
		assert(verify(src, dst));

		grayscale_rgba(src, dst);
	}


	void grayscale(Image4Cr32 const& src, View1Cr32 const& dst)
	{
		assert(verify(src, dst));

		grayscale_rgba(src, dst);
	}


	void grayscale(View4Cr32 const& src, Image1Cr32 const& dst)
	{
		assert(verify(src, dst));

		grayscale_rgba(src, dst);
	}


	void grayscale(View4Cr32 const& src, View1Cr32 const& dst)
	{
		assert(verify(src, dst));

		grayscale_rgba(src, dst);
	}



	void grayscale(Image3Cr32 const& src, Image1Cr32 const& dst)
	{
		assert(verify(src, dst));

		grayscale_rgb(src, dst);
	}


	void grayscale(Image3Cr32 const& src, View1Cr32 const& dst)
	{
		assert(verify(src, dst));

		grayscale_rgb(src, dst);
	}


	void grayscale(View3Cr32 const& src, Image1Cr32 const& dst)
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
	View1Cr32 select_channel(Image4Cr32 const& image, RGBA channel)
	{
		assert(verify(image));

		auto ch = to_channel_index(channel);

		View1Cr32 view1{};

		view1.image_width = image.width;
		view1.x_begin = 0;
		view1.y_begin = 0;
		view1.x_end = image.width;
		view1.y_end = image.height;
		view1.width = image.width;
		view1.height = image.height;

		view1.image_data = image.channel_data[ch];

		return view1;
	}


	View1Cr32 select_channel(View4Cr32 const& view, RGBA channel)
	{
		assert(verify(view));

		auto ch = to_channel_index(channel);

		View1Cr32 view1{};

		view1.image_width = view.image_width;
		view1.range = view.range;
		view1.width = view.width;
		view1.height = view.height;

		view1.image_data = view.image_channel_data[ch];

		return view1;
	}


	View1Cr32 select_channel(Image3Cr32 const& image, RGB channel)
	{
		assert(verify(image));

		auto ch = to_channel_index(channel);

		View1Cr32 view1{};

		view1.image_width = image.width;
		view1.x_begin = 0;
		view1.y_begin = 0;
		view1.x_end = image.width;
		view1.y_end = image.height;
		view1.width = image.width;
		view1.height = image.height;

		view1.image_data = image.channel_data[ch];

		return view1;
	}


	View1Cr32 select_channel(View3Cr32 const& view, RGB channel)
	{
		assert(verify(view));

		auto ch = to_channel_index(channel);

		View1Cr32 view1{};

		view1.image_width = view.image_width;
		view1.range = view.range;
		view1.width = view.width;
		view1.height = view.height;

		view1.image_data = view.image_channel_data[ch];

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
		auto const row_func = [&](u32 y)
		{
			auto sr = row_begin(src, y, RGBA::R);
			auto sg = row_begin(src, y, RGBA::G);
			auto sb = row_begin(src, y, RGBA::B);
			auto sa = row_begin(src, y, RGBA::A);

			auto cr = row_begin(cur, y, RGB::R);
			auto cg = row_begin(cur, y, RGB::G);
			auto cb = row_begin(cur, y, RGB::B);

			auto dr = row_begin(dst, y, RGB::R);
			auto dg = row_begin(dst, y, RGB::G);
			auto db = row_begin(dst, y, RGB::B);

			for (u32 x = 0; x < src.width; ++x)
			{
				dr[x] = blend_linear(sr[x], cr[x], sa[x]);
				dg[x] = blend_linear(sg[x], cg[x], sa[x]);
				db[x] = blend_linear(sb[x], cb[x], sa[x]);
			}
		};

		process_rows(src.height, row_func);
	}


	void alpha_blend(Image4Cr32 const& src, Image3Cr32 const& cur, Image3Cr32 const& dst)
	{		
		assert(verify(src, cur));
		assert(verify(src, dst));

		do_alpha_blend(src, cur, dst);
	}


	void alpha_blend(Image4Cr32 const& src, Image3Cr32 const& cur, View3Cr32 const& dst)
	{
		assert(verify(src, cur));
		assert(verify(src, dst));

		do_alpha_blend(src, cur, dst);
	}


	void alpha_blend(Image4Cr32 const& src, View3Cr32 const& cur, Image3Cr32 const& dst)
	{
		assert(verify(src, cur));
		assert(verify(src, dst));

		do_alpha_blend(src, cur, dst);
	}


	void alpha_blend(Image4Cr32 const& src, View3Cr32 const& cur, View3Cr32 const& dst)
	{
		assert(verify(src, cur));
		assert(verify(src, dst));

		do_alpha_blend(src, cur, dst);
	}


	void alpha_blend(View4Cr32 const& src, Image3Cr32 const& cur, Image3Cr32 const& dst)
	{
		assert(verify(src, cur));
		assert(verify(src, dst));

		do_alpha_blend(src, cur, dst);
	}


	void alpha_blend(View4Cr32 const& src, Image3Cr32 const& cur, View3Cr32 const& dst)
	{
		assert(verify(src, cur));
		assert(verify(src, dst));

		do_alpha_blend(src, cur, dst);
	}


	void alpha_blend(View4Cr32 const& src, View3Cr32 const& cur, Image3Cr32 const& dst)
	{
		assert(verify(src, cur));
		assert(verify(src, dst));

		do_alpha_blend(src, cur, dst);
	}


	void alpha_blend(View4Cr32 const& src, View3Cr32 const& cur, View3Cr32 const& dst)
	{
		assert(verify(src, cur));
		assert(verify(src, dst));

		do_alpha_blend(src, cur, dst);
	}


	void alpha_blend(Image4Cr32 const& src, Image3Cr32 const& cur_dst)
	{
		assert(verify(src, cur_dst));

		do_alpha_blend(src, cur_dst, cur_dst);
	}


	void alpha_blend(Image4Cr32 const& src, View3Cr32 const& cur_dst)
	{
		assert(verify(src, cur_dst));

		do_alpha_blend(src, cur_dst, cur_dst);
	}


	void alpha_blend(View4Cr32 const& src, Image3Cr32 const& cur_dst)
	{
		assert(verify(src, cur_dst));

		do_alpha_blend(src, cur_dst, cur_dst);
	}


	void alpha_blend(View4Cr32 const& src, View3Cr32 const& cur_dst)
	{
		assert(verify(src, cur_dst));

		do_alpha_blend(src, cur_dst, cur_dst);
	}
}