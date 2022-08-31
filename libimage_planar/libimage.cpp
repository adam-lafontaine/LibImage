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


class ThreadProcess
{
public:
	u32 thread_id = 0;
	std::function<void(u32)> process;
};


using ProcList = std::array<ThreadProcess, N_THREADS>;


static ProcList make_proc_list(std::function<void(u32)> const& id_func)
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
	static PlatformPixel to_pixel(r32 r, r32 g, r32 b, r32 a)
	{
		PlatformPixel p{};
		p.red = to_channel_u8(r);
		p.green = to_channel_u8(g);
		p.blue = to_channel_u8(b);
		p.alpha = to_channel_u8(a);

		return p;
	}


	void make_image(PlatformImage& image, u32 width, u32 height)
	{
		assert(width);
		assert(height);
		
		image.data = (PlatformPixel*)malloc(sizeof(PlatformPixel) * width * height);
		assert(image.data);

		image.width = width;
		image.height = height;
	}


	void destroy_image(PlatformImage& image)
	{
		if (image.data != nullptr)
		{
			free(image.data);
			image.data = nullptr;
		}
	}


	void make_image(ImageRGBAr32& image, u32 width, u32 height)
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


	void destroy_image(ImageRGBAr32& image)
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


	void transform(ImageRGBAr32 const& src, PlatformImage const& dst)
	{
		assert(src.width == dst.width);
		assert(src.height == dst.height);
		assert(src.red);
		assert(dst.data);

		auto const width = src.width;
		auto const height = src.height;
		auto const rows_per_thread = height / N_THREADS;

		auto const thread_proc = [&](u32 id)
		{
			auto y_begin = id * rows_per_thread;
			auto y_end = (id == N_THREADS - 1 ? height : (id + 1) * rows_per_thread);

			for (u32 y = y_begin; y < y_end; ++y)
			{
				auto r = src.red + y * width;
				auto g = src.green + y * width;
				auto b = src.blue + y * width;
				auto a = src.alpha + y * width;
				auto dst_begin = dst.data + y * width;
				for (u32 x = 0; x < width; ++x)
				{
					dst_begin[x] = to_pixel(r[x], g[x], b[x], a[x]);
				}
			}
		};

		execute_procs(make_proc_list(thread_proc));
	}


	void transform(PlatformImage const& src, ImageRGBAr32 const& dst)
	{
		assert(src.width == dst.width);
		assert(src.height == dst.height);
		assert(src.data);
		assert(dst.red);

		auto const width = src.width;
		auto const height = src.height;
		auto const rows_per_thread = height / N_THREADS;

		auto const thread_proc = [&](u32 id)
		{
			auto y_begin = id * rows_per_thread;
			auto y_end = (id == N_THREADS - 1 ? height : (id + 1) * rows_per_thread);

			for (u32 y = y_begin; y < y_end; ++y)
			{
				auto r = dst.red + y * width;
				auto g = dst.green + y * width;
				auto b = dst.blue + y * width;
				auto a = dst.alpha + y * width;
				auto src_begin = src.data + y * width;
				for (u32 x = 0; x < width; ++x)
				{
					auto s = src_begin[x];
					r[x] = to_channel_r32(s.red);
					g[x] = to_channel_r32(s.green);
					b[x] = to_channel_r32(s.blue);
					a[x] = to_channel_r32(s.alpha);
				}
			}
		};

		execute_procs(make_proc_list(thread_proc));
	}


	void make_image(ImageRGBr32& image, u32 width, u32 height)
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


	void destroy_image(ImageRGBr32& image)
	{
		if (image.red != nullptr)
		{
			free(image.red);
			image.red = nullptr;
			image.green = nullptr;
			image.blue = nullptr;
		}
	}


	void transform(ImageRGBr32 const& src, PlatformImage const& dst)
	{
		assert(src.width == dst.width);
		assert(src.height == dst.height);
		assert(src.red);
		assert(dst.data);

		auto const width = src.width;
		auto const height = src.height;
		auto const rows_per_thread = height / N_THREADS;

		auto const thread_proc = [&](u32 id)
		{
			auto y_begin = id * rows_per_thread;
			auto y_end = (id == N_THREADS - 1 ? height : (id + 1) * rows_per_thread);

			for (u32 y = y_begin; y < y_end; ++y)
			{
				auto r = src.red + y * width;
				auto g = src.green + y * width;
				auto b = src.blue + y * width;
				auto dst_begin = dst.data + y * width;
				for (u32 x = 0; x < width; ++x)
				{
					dst_begin[x] = to_pixel(r[x], g[x], b[x]);
				}
			}
		};

		execute_procs(make_proc_list(thread_proc));
	}


	void transform(PlatformImage const& src, ImageRGBr32 const& dst)
	{
		assert(src.width == dst.width);
		assert(src.height == dst.height);
		assert(src.data);
		assert(dst.red);

		auto const width = src.width;
		auto const height = src.height;
		auto const rows_per_thread = height / N_THREADS;

		auto const thread_proc = [&](u32 id)
		{
			auto y_begin = id * rows_per_thread;
			auto y_end = (id == N_THREADS - 1 ? height : (id + 1) * rows_per_thread);

			for (u32 y = y_begin; y < y_end; ++y)
			{
				auto r = dst.red + y * width;
				auto g = dst.green + y * width;
				auto b = dst.blue + y * width;
				auto src_begin = src.data + y * width;
				for (u32 x = 0; x < width; ++x)
				{
					auto s = src_begin[x];
					r[x] = to_channel_r32(s.red);
					g[x] = to_channel_r32(s.green);
					b[x] = to_channel_r32(s.blue);
				}
			}
		};

		execute_procs(make_proc_list(thread_proc));
	}


	void make_image(PlatformImageGRAY& image, u32 width, u32 height)
	{
		assert(width);
		assert(height);

		image.data = (u8*)malloc(sizeof(u8) * width * height);
		assert(image.data);

		image.width = width;
		image.height = height;
	}


	void destroy_image(PlatformImageGRAY& image)
	{
		if (image.data != nullptr)
		{
			free(image.data);
			image.data = nullptr;
		}
	}


	void make_image(ImageGRAYr32& image, u32 width, u32 height)
	{
		assert(width);
		assert(height);

		image.data = (r32*)malloc(sizeof(r32) * width * height);
		assert(image.data);

		image.width = width;
		image.height = height;
	}


	void destroy_image(ImageGRAYr32& image)
	{
		if (image.data != nullptr)
		{
			free(image.data);
			image.data = nullptr;
		}
	}


	void transform(ImageGRAYr32 const& src, PlatformImageGRAY const& dst)
	{
		assert(src.width == dst.width);
		assert(src.height == dst.height);
		assert(src.data);
		assert(dst.data);

		auto const width = src.width;
		auto const height = src.height;
		auto const rows_per_thread = height / N_THREADS;

		auto const thread_proc = [&](u32 id)
		{
			auto y_begin = id * rows_per_thread;
			auto y_end = (id == N_THREADS - 1 ? height : (id + 1) * rows_per_thread);

			for (u32 y = y_begin; y < y_end; ++y)
			{
				auto s = src.data + y * width;
				auto d = dst.data + y * width;
				for (u32 x = 0; x < width; ++x)
				{
					d[x] = to_channel_u8(s[x]);
				}
			}
		};

		execute_procs(make_proc_list(thread_proc));
	}


	void transform(PlatformImageGRAY const& src, ImageGRAYr32 const& dst)
	{
		assert(src.width == dst.width);
		assert(src.height == dst.height);
		assert(src.data);
		assert(dst.data);

		auto const width = src.width;
		auto const height = src.height;
		auto const rows_per_thread = height / N_THREADS;

		auto const thread_proc = [&](u32 id)
		{
			auto y_begin = id * rows_per_thread;
			auto y_end = (id == N_THREADS - 1 ? height : (id + 1) * rows_per_thread);

			for (u32 y = y_begin; y < y_end; ++y)
			{
				auto s = src.data + y * width;
				auto d = dst.data + y * width;
				for (u32 x = 0; x < width; ++x)
				{
					d[x] = to_channel_r32(s[x]);
				}
			}
		};

		execute_procs(make_proc_list(thread_proc));
	}

}


