#include "libimage.hpp"
#include "simd_def.hpp"

#include <algorithm>
#include <cmath>


static void do_for_each_seq(auto& list, auto const& func)
{
	std::for_each(list.begin(), list.end(), func);
}


#ifndef LIBIMAGE_NO_PARALLEL

#include <execution>

constexpr u32 N_THREADS = 8;

static void do_for_each(auto& list, auto const& func)
{
	std::for_each(std::execution::par, list.begin(), list.end(), func);
}

#else

constexpr u32 N_THREADS = 1;

static void do_for_each(auto const& list, auto const& func)
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


/*  libimage.cpp  */

namespace libimage
{

#ifndef LIBIMAGE_NO_COLOR

	void make_image(Image& image_dst, u32 width, u32 height)
	{
		assert(width);
		assert(height);

		image_dst.width = width;
		image_dst.height = height;
		image_dst.data = (Pixel*)malloc(sizeof(Pixel) * width * height);

		assert(image_dst.data);
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


	View make_view(Image const& img)
	{
		assert(img.width);
		assert(img.height);
		assert(img.data);

		View view;

		view.image_data = img.data;
		view.image_width = img.width;
		view.x_begin = 0;
		view.y_begin = 0;
		view.x_end = img.width;
		view.y_end = img.height;
		view.width = img.width;
		view.height = img.height;

		return view;
	}


	View make_view(Image& image, u32 width, u32 height)
	{
		make_image(image, width, height);
		return make_view(image);
	}


	View sub_view(Image const& image, Range2Du32 const& range)
	{
		assert(image.width);
		assert(image.height);
		assert(image.data);

		View sub_view;

		sub_view.image_data = image.data;
		sub_view.image_width = image.width;
		sub_view.x_begin = range.x_begin;
		sub_view.y_begin = range.y_begin;
		sub_view.x_end = range.x_end;
		sub_view.y_end = range.y_end;
		sub_view.width = range.x_end - range.x_begin;
		sub_view.height = range.y_end - range.y_begin;

		assert(sub_view.width);
		assert(sub_view.height);

		return sub_view;
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


	


#endif // !LIBIMAGE_NO_COLOR

#ifndef LIBIMAGE_NO_GRAYSCALE

	void make_image(gray::Image& image_dst, u32 width, u32 height)
	{
		assert(width);
		assert(height);

		image_dst.width = width;
		image_dst.height = height;
		image_dst.data = (gray::Pixel*)malloc(sizeof(gray::Pixel) * width * height);

		assert(image_dst.data);
	}


	void destroy_image(gray::Image& image)
	{
		if (image.data != nullptr)
		{
			free(image.data);
			image.data = nullptr;
		}
	}


	gray::Pixel* row_begin(gray::Image const& image, u32 y)
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


	gray::Pixel* xy_at(gray::Image const& image, u32 x, u32 y)
	{
		assert(image.width);
		assert(image.height);
		assert(image.data);
		assert(y < image.height);
		assert(x < image.width);

		return row_begin(image, y) + x;
	}


	gray::View make_view(gray::Image const& img)
	{
		assert(img.width);
		assert(img.height);
		assert(img.data);

		gray::View view;

		view.image_data = img.data;
		view.image_width = img.width;
		view.x_begin = 0;
		view.y_begin = 0;
		view.x_end = img.width;
		view.y_end = img.height;
		view.width = img.width;
		view.height = img.height;

		return view;
	}


	gray::View make_view(gray::Image& image, u32 width, u32 height)
	{
		make_image(image, width, height);
		return make_view(image);
	}


	gray::View sub_view(gray::Image const& image, Range2Du32 const& range)
	{
		assert(image.width);
		assert(image.height);
		assert(image.data);

		gray::View sub_view;

		sub_view.image_data = image.data;
		sub_view.image_width = image.width;
		sub_view.x_begin = range.x_begin;
		sub_view.y_begin = range.y_begin;
		sub_view.x_end = range.x_end;
		sub_view.y_end = range.y_end;
		sub_view.width = range.x_end - range.x_begin;
		sub_view.height = range.y_end - range.y_begin;

		assert(sub_view.width);
		assert(sub_view.height);

		return sub_view;
	}


	gray::View sub_view(gray::View const& view, Range2Du32 const& range)
	{
		assert(view.width);
		assert(view.height);
		assert(view.image_data);

		assert(range.x_begin < view.x_end);
		assert(range.x_end <= view.x_end);
		assert(range.y_begin < view.y_end);
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


	gray::Pixel* row_begin(gray::View const& view, u32 y)
	{
		assert(y < view.height);

		auto offset = (view.y_begin + y) * view.image_width + view.x_begin;

		auto ptr = view.image_data + (u64)(offset);
		assert(ptr);

		return ptr;
	}


	gray::Pixel* xy_at(gray::View const& view, u32 x, u32 y)
	{
		assert(y < view.height);
		assert(x < view.width);

		return row_begin(view, y) + x;
	}
		

#endif // !LIBIMAGE_NO_GRAYSCALE

}


/*  verify  */

namespace libimage
{
#ifndef LIBIMAGE_NO_COLOR


	inline bool verify(View const& view)
	{
		return view.image_data && view.width && view.height;
	}


	inline bool verify(Image const& img)
	{
		return img.data && img.width && img.height;
	}


	inline bool verify(Image const& src, Image const& dst)
	{
		return verify(src) && verify(dst) && dst.width == src.width && dst.height == src.height;
	}


	inline bool verify(Image const& src, View const& dst)
	{
		return verify(src) && verify(dst) && dst.width == src.width && dst.height == src.height;
	}


	inline bool verify(View const& src, Image const& dst)
	{
		return verify(src) && verify(dst) && dst.width == src.width && dst.height == src.height;
	}


	inline bool verify(View const& src, View const& dst)
	{
		return verify(src) && verify(dst) && dst.width == src.width && dst.height == src.height;
	}


#endif // !LIBIMAGE_NO_COLOR


#ifndef LIBIMAGE_NO_GRAYSCALE

	inline bool verify(gray::Image const& img)
	{
		return img.data && img.width && img.height;
	}


	inline bool verify(gray::View const& view)
	{
		return view.image_data && view.width && view.height;
	}


	inline bool verify(gray::Image const& src, gray::Image const& dst)
	{
		return verify(src) && verify(dst) && dst.width == src.width && dst.height == src.height;
	}


	inline bool verify(gray::Image const& src, gray::View const& dst)
	{
		return verify(src) && verify(dst) && dst.width == src.width && dst.height == src.height;
	}


	inline bool verify(gray::View const& src, gray::Image const& dst)
	{
		return verify(src) && verify(dst) && dst.width == src.width && dst.height == src.height;
	}


	inline bool verify(gray::View const& src, gray::View const& dst)
	{
		return verify(src) && verify(dst) && dst.width == src.width && dst.height == src.height;
	}


#endif // !LIBIMAGE_NO_GRAYSCALE


#ifndef LIBIMAGE_NO_COLOR
#ifndef LIBIMAGE_NO_GRAYSCALE

	inline bool verify(Image const& src, gray::Image const& dst)
	{
		return verify(src) && verify(dst) && dst.width == src.width && dst.height == src.height;
	}


	inline bool verify(Image const& src, gray::View const& dst)
	{
		return verify(src) && verify(dst) && dst.width == src.width && dst.height == src.height;
	}


	inline bool verify(View const& src, gray::Image const& dst)
	{
		return verify(src) && verify(dst) && dst.width == src.width && dst.height == src.height;
	}


	inline bool verify(View const& src, gray::View const& dst)
	{
		return verify(src) && verify(dst) && dst.width == src.width && dst.height == src.height;
	}

#endif // !LIBIMAGE_NO_GRAYSCALE
#endif // !LIBIMAGE_NO_COLOR
}


template <typename PIXEL_T, class PIXEL_F>
static void do_for_each_pixel_in_span(PIXEL_T* row_begin, u32 length, PIXEL_F const& func)
{
	for (u32 i = 0; i < length; ++i)
	{
		func(row_begin[i]);
	}
}


template <class XY_F>
static void do_for_each_xy_in_span(u32 y, u32 x_begin, u32 length, XY_F const& func)
{
	for (u32 x = x_begin; x < x_begin + length; ++x)
	{
		func(x, y);
	}
}


template <class XY_F>
static void do_for_each_xy_in_row(u32 y, u32 length, XY_F const& func)
{
	auto x_begin = 0;

	do_for_each_xy_in_span(y, x_begin, length, func);
}


template <typename SRC_PIXEL_T, typename DST_PIXEL_T, class SRC_TO_DST_F>
static void do_transform_row(SRC_PIXEL_T* src_begin, DST_PIXEL_T* dst_begin, u32 length, SRC_TO_DST_F const& func)
{
	for (u32 i = 0; i < length; ++i)
	{
		dst_begin[i] = func(src_begin[i]);
	}
}


template <typename SRC_A_PIXEL_T, typename SRC_B_PIXEL_T, typename DST_PIXEL_T, class SRC_TO_DST_F>
static void do_transform_row2(SRC_A_PIXEL_T* src_a_begin, SRC_B_PIXEL_T* src_b_begin, DST_PIXEL_T* dst_begin, u32 length, SRC_TO_DST_F const& func)
{
	for (u32 i = 0; i < length; ++i)
	{
		dst_begin[i] = func(src_a_begin[i], src_b_begin[i]);
	}
}


template <class IMG_T>
static Range2Du32 make_range(IMG_T const& img)
{
	Range2Du32 r{};

	r.x_begin = 0;
	r.x_end = img.width;
	r.y_begin = 0;
	r.y_end = img.height;

	return r;
}


template <class IMG_T>
static void do_for_each_xy_in_range_by_row(IMG_T const& image, Range2Du32 const& range, std::function<void(u32 x, u32 y)> const& func)
{
	auto const height = range.y_end - range.y_begin;
	auto const width = range.x_end - range.x_begin;
	auto const rows_per_thread = height / N_THREADS;

	auto const thread_proc = [&](u32 id)
	{
		auto y_begin = range.y_begin + id * rows_per_thread;
		auto y_end = range.y_begin + (id == N_THREADS - 1 ? height : (id + 1) * rows_per_thread);

		for (u32 y = y_begin; y < y_end; ++y)
		{
			do_for_each_xy_in_span(y, range.x_begin, width, func);
		}
	};

	execute_procs(make_proc_list(thread_proc));
}


template <class IMG_T>
static void do_for_each_xy_by_row(IMG_T const& image, std::function<void(u32 x, u32 y)> const& func)
{
	auto const range = make_range(image);
	do_for_each_xy_in_range_by_row(image, range, func);
}


namespace libimage
{
	template <class IMG_T, class PIXEL_F>
	static void do_for_each_pixel_in_range_by_row(IMG_T const& image, Range2Du32 const& range, PIXEL_F const& func)
	{
		auto const height = range.y_end - range.y_begin;
		auto const width = range.x_end - range.x_begin;
		auto const rows_per_thread = height / N_THREADS;

		auto const thread_proc = [&](u32 id)
		{
			auto y_begin = range.y_begin + id * rows_per_thread;
			auto y_end = range.y_begin + (id == N_THREADS - 1 ? height : (id + 1) * rows_per_thread);

			for (u32 y = y_begin; y < y_end; ++y)
			{
				auto img_begin = row_begin(image, y) + range.x_begin;
				do_for_each_pixel_in_span(img_begin, width, func);
			}
		};

		execute_procs(make_proc_list(thread_proc));
	}


	template <class IMG_T, class PIXEL_F>
	static void do_for_each_pixel_by_row(IMG_T const& image, PIXEL_F const& func)
	{
		auto range = make_range(image);
		do_for_each_pixel_in_range_by_row(image, range, func);
	}


	template <class SRC_IMG_T, class DST_IMG_T, class SRC_TO_DST_F>
	static void do_transform_by_row(SRC_IMG_T const& src, DST_IMG_T const& dst, SRC_TO_DST_F const& func)
	{
		auto const height = src.height;
		auto const width = src.width;
		auto const rows_per_thread = height / N_THREADS;

		auto const thread_proc = [&](u32 id)
		{
			auto y_begin = id * rows_per_thread;
			auto y_end = id == N_THREADS - 1 ? height : (id + 1) * rows_per_thread;

			for (u32 y = y_begin; y < y_end; ++y)
			{
				do_transform_row(row_begin(src, y), row_begin(dst, y), width, func);
			}
		};

		execute_procs(make_proc_list(thread_proc));
	}


	template <class SRC_A_IMG_T, class SRC_B_IMG_T, class DST_IMG_T, class SRC_TO_DST_F>
	static void do_transform_by_row2(SRC_A_IMG_T const& src_a, SRC_B_IMG_T const& src_b, DST_IMG_T const& dst, SRC_TO_DST_F const& func)
	{
		auto const height = src_a.height;
		auto const width = src_a.width;
		auto const rows_per_thread = height / N_THREADS;

		auto const thread_proc = [&](u32 id)
		{
			auto y_begin = id * rows_per_thread;
			auto y_end = id == N_THREADS - 1 ? height : (id + 1) * rows_per_thread;

			for (u32 y = y_begin; y < y_end; ++y)
			{
				do_transform_row2(row_begin(src_a, y), row_begin(src_b, y), row_begin(dst, y), width, func);
			}
		};

		execute_procs(make_proc_list(thread_proc));
	}


}



/* for each */

namespace libimage
{

#ifndef LIBIMAGE_NO_COLOR

	void for_each_pixel(Image const& image, pixel_f const& func)
	{
		do_for_each_pixel_by_row(image, func);
	}


	void for_each_pixel(View const& view, pixel_f const& func)
	{
		do_for_each_pixel_by_row(view, func);
	}


	void for_each_xy(Image const& image, xy_f const& func)
	{
		do_for_each_xy_by_row(image, func);
	}


	void for_each_xy(View const& view, xy_f const& func)
	{
		do_for_each_xy_by_row(view, func);
	}

#endif // !LIBIMAGE_NO_COLOR

#ifndef LIBIMAGE_NO_GRAYSCALE

	void for_each_pixel(gray::Image const& image, u8_f const& func)
	{
		do_for_each_pixel_by_row(image, func);
	}


	void for_each_pixel(gray::View const& view, u8_f const& func)
	{
		do_for_each_pixel_by_row(view, func);
	}


	void for_each_xy(gray::Image const& image, xy_f const& func)
	{
		do_for_each_xy_by_row(image, func);
	}


	void for_each_xy(gray::View const& view, xy_f const& func)
	{
		do_for_each_xy_by_row(view, func);
	}

#endif // !LIBIMAGE_NO_GRAYSCALE
}


/* simd helpers */

#ifndef LIBIMAGE_NO_SIMD

namespace libimage
{
	static inline void copy_vec_len(Pixel* src, Pixelr32Planar& dst)
	{
		for (u32 i = 0; i < simd::VEC_LEN; ++i)
		{
			dst.red[i] = (r32)src[i].red;
			dst.green[i] = (r32)src[i].green;
			dst.blue[i] = (r32)src[i].blue;
			dst.alpha[i] = (r32)src[i].alpha;
		}
	}


	static inline void copy_vec_len(u8* red, u8* green, u8* blue, Pixelr32Planar& dst)
	{
		for (u32 i = 0; i < simd::VEC_LEN; ++i)
		{
			dst.red[i] = (r32)red[i];
			dst.green[i] = (r32)green[i];
			dst.blue[i] = (r32)blue[i];
			dst.alpha[i] = 255.0f;
		}
	}


	template <class IMG_T, class SIMD_F>
	static void do_simd_for_each_in_range_by_row(IMG_T const& image, Range2Du32 const& range, SIMD_F const& func)
	{
		auto const height = range.y_end - range.y_begin;
		auto const width = range.x_end - range.x_begin;
		auto const rows_per_thread = height / N_THREADS;

		auto const thread_proc = [&](u32 id)
		{
			auto y_begin = range.y_begin + id * rows_per_thread;
			auto y_end = range.y_begin + (id == N_THREADS - 1 ? height : (id + 1) * rows_per_thread);

			for (u32 y = y_begin; y < y_end; ++y)
			{
				auto img_begin = row_begin(image, y) + range.x_begin;
				func(img_begin, width);
			}
		};

		execute_procs(make_proc_list(thread_proc));
	}


	template <class IMG_T, class SIMD_F>
	static void do_simd_for_each_by_row(IMG_T const& image, SIMD_F const& func)
	{
		auto range = make_range(image);
		do_simd_for_each_in_range_by_row(image, range, func);
	}


	template <class SRC_IMG_T, class DST_IMG_T, class SIMD_F>
	static void do_simd_transform_by_row(SRC_IMG_T const& src, DST_IMG_T const& dst, SIMD_F const& func)
	{
		auto const height = src.height;
		auto const width = src.width;
		auto const rows_per_thread = height / N_THREADS;

		auto const thread_proc = [&](u32 id)
		{
			auto y_begin = id * rows_per_thread;
			auto y_end = id == N_THREADS - 1 ? height : (id + 1) * rows_per_thread;

			for (u32 y = y_begin; y < y_end; ++y)
			{
				func(row_begin(src, y), row_begin(dst, y), width);
			}
		};

		execute_procs(make_proc_list(thread_proc));
	}


	template <class SRC_A_IMG_T, class SRC_B_IMG_T, class DST_IMG_T, class SIMD_F>
	static void do_simd_transform_by_row2(SRC_A_IMG_T const& src_a, SRC_B_IMG_T const& src_b, DST_IMG_T const& dst, SIMD_F const& func)
	{
		auto const height = src_a.height;
		auto const width = src_a.width;
		auto const rows_per_thread = height / N_THREADS;

		auto const thread_proc = [&](u32 id)
		{
			auto y_begin = id * rows_per_thread;
			auto y_end = id == N_THREADS - 1 ? height : (id + 1) * rows_per_thread;

			for (u32 y = y_begin; y < y_end; ++y)
			{
				func(row_begin(src_a, y), row_begin(src_b, y), row_begin(dst, y), width);
			}
		};

		execute_procs(make_proc_list(thread_proc));
	}

}

#endif // !LIBIMAGE_NO_SIMD


/*  transform  */

namespace libimage
{

#ifndef LIBIMAGE_NO_COLOR


	void transform(Image const& src, Image const& dst, pixel_to_pixel_f const& func)
	{
		assert(verify(src, dst));
		do_transform_by_row(src, dst, func);
	}


	void transform(Image const& src, View const& dst, pixel_to_pixel_f const& func)
	{
		assert(verify(src, dst));
		do_transform_by_row(src, dst, func);
	}


	void transform(View const& src, Image const& dst, pixel_to_pixel_f const& func)
	{
		assert(verify(src, dst));
		do_transform_by_row(src, dst, func);
	}


	void transform(View const& src, View const& dst, pixel_to_pixel_f const& func)
	{
		assert(verify(src, dst));
		do_transform_by_row(src, dst, func);
	}


	void transform_in_place(Image const& src_dst, pixel_to_pixel_f const& func)
	{
		assert(verify(src_dst));
		auto const update = [&](Pixel& p) { p = func(p); };		
		do_for_each_pixel_by_row(src_dst, update);
	}


	void transform_in_place(View const& src_dst, pixel_to_pixel_f const& func)
	{
		assert(verify(src_dst));
		auto const update = [&](Pixel& p) { p = func(p); };
		do_for_each_pixel_by_row(src_dst, update);
	}


	void transform_alpha(Image const& src_dst, pixel_to_u8_f const& func)
	{
		assert(verify(src_dst));
		auto const update = [&](Pixel& p) { p.alpha = func(p); };
		do_for_each_pixel_by_row(src_dst, update);
	}


	void transform_alpha(View const& src_dst, pixel_to_u8_f const& func)
	{
		assert(verify(src_dst));
		auto const update = [&](Pixel& p) { p.alpha = func(p); };
		do_for_each_pixel_by_row(src_dst, update);
	}


#endif // !LIBIMAGE_NO_COLOR


#ifndef LIBIMAGE_NO_GRAYSCALE

	static lookup_table_t to_lookup_table(u8_to_u8_f const& func)
	{
		lookup_table_t lut = { 0 };

		/*u32_range_t ids(0u, 256u);

		std::for_each(ids.begin(), ids.end(), [&](u32 id) { lut[id] = func(id); });*/

		for (u32 i = 0; i < 256; ++i)
		{
			lut[i] = func(i);
		}

		return lut;
	}


	void transform(gray::Image const& src, gray::Image const& dst, lookup_table_t const& lut)
	{
		assert(verify(src, dst));
		auto const conv = [&lut](u8 p) { return lut[p]; };
		do_transform_by_row(src, dst, conv);
	}


	void transform(gray::Image const& src, gray::View const& dst, lookup_table_t const& lut)
	{
		assert(verify(src, dst));
		auto const conv = [&lut](u8 p) { return lut[p]; };
		do_transform_by_row(src, dst, conv);
	}


	void transform(gray::View const& src, gray::Image const& dst, lookup_table_t const& lut)
	{
		assert(verify(src, dst));
		auto const conv = [&lut](u8 p) { return lut[p]; };
		do_transform_by_row(src, dst, conv);
	}


	void transform(gray::View const& src, gray::View const& dst, lookup_table_t const& lut)
	{
		assert(verify(src, dst));
		auto const conv = [&lut](u8 p) { return lut[p]; };
		do_transform_by_row(src, dst, conv);
	}


	void transform(gray::Image const& src, gray::Image const& dst, u8_to_u8_f const& func)
	{
		assert(verify(src, dst));
		auto const lut = to_lookup_table(func);
		transform(src, dst, lut);
	}


	void transform(gray::Image const& src, gray::View const& dst, u8_to_u8_f const& func)
	{
		assert(verify(src, dst));
		auto const lut = to_lookup_table(func);
		transform(src, dst, lut);
	}


	void transform(gray::View const& src, gray::Image const& dst, u8_to_u8_f const& func)
	{
		assert(verify(src, dst));
		auto const lut = to_lookup_table(func);
		transform(src, dst, lut);
	}


	void transform(gray::View const& src, gray::View const& dst, u8_to_u8_f const& func)
	{
		assert(verify(src, dst));
		auto const lut = to_lookup_table(func);
		transform(src, dst, lut);
	}


	void transform_in_place(gray::Image const& src_dst, lookup_table_t const& lut)
	{
		assert(verify(src_dst));
		auto const conv = [&lut](u8& p) { p = lut[p]; };
		do_for_each_pixel_by_row(src_dst, conv);
	}

	void transform_in_place(gray::View const& src_dst, lookup_table_t const& lut)
	{
		assert(verify(src_dst));
		auto const conv = [&lut](u8& p) { p = lut[p]; };
		do_for_each_pixel_by_row(src_dst, conv);
	}


	void transform_in_place(gray::Image const& src_dst, u8_to_u8_f const& func)
	{
		assert(verify(src_dst));
		auto const lut = to_lookup_table(func);
		transform_in_place(src_dst, lut);
	}


	void transform_in_place(gray::View const& src_dst, u8_to_u8_f const& func)
	{
		assert(verify(src_dst));
		auto const lut = to_lookup_table(func);
		transform_in_place(src_dst, lut);
	}


#endif // !LIBIMAGE_NO_GRAYSCALE


#ifndef LIBIMAGE_NO_COLOR
#ifndef LIBIMAGE_NO_GRAYSCALE

	void transform(Image const& src, gray::Image const& dst, pixel_to_u8_f const& func)
	{
		assert(verify(src, dst));
		do_transform_by_row(src, dst, func);
	}


	void transform(Image const& src, gray::View const& dst, pixel_to_u8_f const& func)
	{
		assert(verify(src, dst));
		do_transform_by_row(src, dst, func);
	}


	void transform(View const& src, gray::Image const& dst, pixel_to_u8_f const& func)
	{
		assert(verify(src, dst));
		do_transform_by_row(src, dst, func);
	}


	void transform(View const& src, gray::View const& dst, pixel_to_u8_f const& func)
	{
		assert(verify(src, dst));
		do_transform_by_row(src, dst, func);
	}


#endif // !LIBIMAGE_NO_GRAYSCALE
#endif // !LIBIMAGE_NO_COLOR




}


/* fill */

namespace libimage
{
#ifndef LIBIMAGE_NO_COLOR

	void fill(Image const& image, Pixel color)
	{
		auto const func = [&](Pixel& p) { p = color; };
		for_each_pixel(image, func);
	}


	void fill(View const& view, Pixel color)
	{
		auto const func = [&](Pixel& p) { p = color; };
		for_each_pixel(view, func);
	}

#endif // !LIBIMAGE_NO_COLOR

#ifndef LIBIMAGE_NO_GRAYSCALE

	void fill(gray::Image const& image, u8 gray)
	{
		auto const func = [&](u8& p) { p = gray; };
		for_each_pixel(image, func);
	}


	void fill(gray::View const& view, u8 gray)
	{
		auto const func = [&](u8& p) { p = gray; };
		for_each_pixel(view, func);
	}

#endif // !LIBIMAGE_NO_GRAYSCALE
}


/*  copy  */

namespace libimage
{

#ifndef LIBIMAGE_NO_COLOR

#ifndef LIBIMAGE_NO_SIMD

	static void simd_copy_row(Pixel* src_begin, Pixel* dst_begin, u32 length)
	{
		assert(sizeof(Pixel) == sizeof(r32));

		constexpr u32 STEP = simd::VEC_LEN;

		r32* src = 0;
		r32* dst = 0;
		simd::vec_t vec{};

		auto const do_simd = [&](u32 i) 
		{
			src = (r32*)(src_begin + i);
			dst = (r32*)(dst_begin + i);

			vec = simd::load(src);
			simd::store(dst, vec);
		};

		for (u32 i = 0; i < length - STEP; i += STEP)
		{
			do_simd(i);
		}

		do_simd(length - STEP);
	}



	template <class SRC_IMG_T, class DST_IMG_T>
	static void do_copy(SRC_IMG_T const& src, DST_IMG_T const& dst)
	{
		do_simd_transform_by_row(src, dst, simd_copy_row);
	}

#else

	template <class SRC_IMG_T, class DST_IMG_T>
	static void do_copy(SRC_IMG_T const& src, DST_IMG_T const& dst)
	{
		auto const func = [](auto p) { return p; };
		do_transform_by_row(src, dst, func);
	}

#endif // !LIBIMAGE_NO_SIMD


	void copy(Image const& src, Image const& dst)
	{
		assert(verify(src, dst));
		
		do_copy(src, dst);
	}


	void copy(Image const& src, View const& dst)
	{
		assert(verify(src, dst));

		do_copy(src, dst);
	}


	void copy(View const& src, Image const& dst)
	{
		assert(verify(src, dst));

		do_copy(src, dst);
	}


	void copy(View const& src, View const& dst)
	{
		assert(verify(src, dst));

		do_copy(src, dst);
	}


#endif // !LIBIMAGE_NO_COLOR

#ifndef LIBIMAGE_NO_GRAYSCALE

#ifndef LIBIMAGE_NO_SIMD

	static void simd_copy_gray_row(u8* src_begin, u8* dst_begin, u32 length)
	{
		constexpr u32 STEP = simd::VEC_LEN * sizeof(r32) / sizeof(u8);

		r32* src = 0;
		r32* dst = 0;
		simd::vec_t vec{};

		auto const do_simd = [&](u32 i) 
		{
			src = (r32*)(src_begin + i);
			dst = (r32*)(dst_begin + i);

			vec = simd::load(src);
			simd::store(dst, vec);
		};

		for (u32 i = 0; i < length - STEP; i += STEP)
		{
			do_simd(i);
		}

		do_simd(length - STEP);
	}


	template <class SRC_IMG_T, class DST_IMG_T>
	static void do_copy_gray(SRC_IMG_T const& src, DST_IMG_T const& dst)
	{
		do_simd_transform_by_row(src, dst, simd_copy_gray_row);
	}

#else

	template <class SRC_IMG_T, class DST_IMG_T>
	static void do_copy_gray(SRC_IMG_T const& src, DST_IMG_T const& dst)
	{
		auto const func = [](auto p) { return p; };
		do_transform_by_row(src, dst, func);
	}

#endif // !LIBIMAGE_NO_SIMD


	void copy(gray::Image const& src, gray::Image const& dst)
	{
		assert(verify(src, dst));

		do_copy_gray(src, dst);
	}


	void copy(gray::Image const& src, gray::View const& dst)
	{
		assert(verify(src, dst));

		do_copy_gray(src, dst);
	}


	void copy(gray::View const& src, gray::Image const& dst)
	{
		assert(verify(src, dst));

		do_copy_gray(src, dst);
	}


	void copy(gray::View const& src, gray::View const& dst)
	{
		assert(verify(src, dst));

		do_copy_gray(src, dst);
	}

#endif // !LIBIMAGE_NO_GRAYSCALE



}


/*  alpha_blend  */

#ifndef LIBIMAGE_NO_COLOR

namespace libimage
{
	static Pixel alpha_blend_linear(Pixel src, Pixel current)
	{
		auto const a = (r32)(src.alpha) / 255.0f;

		auto const blend = [&](u8 s, u8 c)
		{
			auto sf = (r32)(s);
			auto cf = (r32)(c);

			auto blended = a * sf + (1.0f - a) * cf;

			return (u8)(blended);
		};

		auto red = blend(src.red, current.red);
		auto green = blend(src.green, current.green);
		auto blue = blend(src.blue, current.blue);

		return to_pixel(red, green, blue);
	}


#ifndef LIBIMAGE_NO_SIMD

	static void simd_alpha_blend_row(Pixel* src_begin, Pixel* cur_begin, Pixel* dst_begin, u32 length)
	{
		constexpr u32 N = simd::VEC_LEN;
		constexpr u32 STEP = N;

		r32 one = 1.0f;
		r32 u8max = 255.0f;

		auto const do_simd = [&](u32 i)
		{
			// pixels are interleaved
			// make them planar
			Pixelr32Planar src_mem{};
			Pixelr32Planar cur_mem{};
			Pixelr32Planar dst_mem{};

			auto src = src_begin + i;
			auto cur = cur_begin + i;
			auto dst = dst_begin + i;

			for (u32 j = 0; j < N; ++j)
			{
				src_mem.red[j] = (r32)src[j].red;
				src_mem.green[j] = (r32)src[j].green;
				src_mem.blue[j] = (r32)src[j].blue;
				src_mem.alpha[j] = (r32)src[j].alpha;

				cur_mem.red[j] = (r32)cur[j].red;
				cur_mem.green[j] = (r32)cur[j].green;
				cur_mem.blue[j] = (r32)cur[j].blue;
			}

			auto one_vec = simd::load_broadcast(&one);
			auto u8max_vec = simd::load_broadcast(&u8max);

			auto src_a_vec = simd::divide(simd::load(src_mem.alpha), u8max_vec);
			auto cur_a_vec = simd::subtract(one_vec, src_a_vec);

			auto dst_vec = simd::fmadd(src_a_vec, simd::load(src_mem.red), simd::multiply(cur_a_vec, simd::load(cur_mem.red)));
			simd::store(dst_mem.red, dst_vec);

			dst_vec = simd::fmadd(src_a_vec, simd::load(src_mem.green), simd::multiply(cur_a_vec, simd::load(cur_mem.green)));
			simd::store(dst_mem.green, dst_vec);

			dst_vec = simd::fmadd(src_a_vec, simd::load(src_mem.blue), simd::multiply(cur_a_vec, simd::load(cur_mem.blue)));
			simd::store(dst_mem.blue, dst_vec);

			for (u32 j = 0; j < N; ++j)
			{
				dst[j].red = (u8)dst_mem.red[j];
				dst[j].green = (u8)dst_mem.green[j];
				dst[j].blue = (u8)dst_mem.blue[j];
				dst[j].alpha = 255;
			}
		};

		for (u32 i = 0; i < length - STEP; i += STEP)
		{
			do_simd(i);
		}

		do_simd(length - STEP);
	}


	template <class SRC_A_IMG_T, class SRC_B_IMG_T, class DST_IMG_T>
	static void do_alpha_blend(SRC_A_IMG_T const& src_a, SRC_B_IMG_T const& src_b, DST_IMG_T const& dst)
	{
		do_simd_transform_by_row2(src_a, src_b, dst, simd_alpha_blend_row);
	}


#else

	template <class SRC_A_IMG_T, class SRC_B_IMG_T, class DST_IMG_T>
	static void do_alpha_blend(SRC_A_IMG_T const& src_a, SRC_B_IMG_T const& src_b, DST_IMG_T const& dst)
	{
		do_transform_by_row2(src_a, src_b, dst, alpha_blend_linear);
	}

#endif !LIBIMAGE_NO_SIMD	


	void alpha_blend(Image const& src, Image const& current, Image const& dst)
	{
		assert(verify(src, current));
		assert(verify(src, dst));

		do_alpha_blend(src, current, dst);
	}


	void alpha_blend(Image const& src, Image const& current, View const& dst)
	{
		assert(verify(src, current));
		assert(verify(src, dst));

		do_alpha_blend(src, current, dst);
	}


	void alpha_blend(Image const& src, View const& current, Image const& dst)
	{
		assert(verify(src, current));
		assert(verify(src, dst));

		do_alpha_blend(src, current, dst);
	}


	void alpha_blend(Image const& src, View const& current, View const& dst)
	{
		assert(verify(src, current));
		assert(verify(src, dst));

		do_alpha_blend(src, current, dst);
	}


	void alpha_blend(View const& src, Image const& current, Image const& dst)
	{
		assert(verify(src, current));
		assert(verify(src, dst));

		do_alpha_blend(src, current, dst);
	}


	void alpha_blend(View const& src, Image const& current, View const& dst)
	{
		assert(verify(src, current));
		assert(verify(src, dst));

		do_alpha_blend(src, current, dst);
	}


	void alpha_blend(View const& src, View const& current, Image const& dst)
	{
		assert(verify(src, current));
		assert(verify(src, dst));

		do_alpha_blend(src, current, dst);
	}


	void alpha_blend(View const& src, View const& current, View const& dst)
	{
		assert(verify(src, current));
		assert(verify(src, dst));

		do_alpha_blend(src, current, dst);
	}


	void alpha_blend(Image const& src, Image const& current_dst)
	{
		assert(verify(src, current_dst));

		do_alpha_blend(src, current_dst, current_dst);
	}


	void alpha_blend(Image const& src, View const& current_dst)
	{
		assert(verify(src, current_dst));

		do_alpha_blend(src, current_dst, current_dst);
	}


	void alpha_blend(View const& src, Image const& current_dst)
	{
		assert(verify(src, current_dst));

		do_alpha_blend(src, current_dst, current_dst);
	}


	void alpha_blend(View const& src, View const& current_dst)
	{
		assert(verify(src, current_dst));

		do_alpha_blend(src, current_dst, current_dst);
	}
}

#endif // !LIBIMAGE_NO_COLOR


/*  grayscale  */

#ifndef LIBIMAGE_NO_GRAYSCALE
#ifndef LIBIMAGE_NO_COLOR


constexpr r32 COEFF_RED = 0.299f;
constexpr r32 COEFF_GREEN = 0.587f;
constexpr r32 COEFF_BLUE = 0.114f;


static constexpr u8 rgb_grayscale_standard(u8 red, u8 green, u8 blue)
{
	return (u8)(COEFF_RED * red + COEFF_GREEN * green + COEFF_BLUE * blue);
}


namespace libimage
{	

	static constexpr u8 pixel_grayscale_standard(Pixel const& p)
	{
		return rgb_grayscale_standard(p.red, p.green, p.blue);
	}

#ifndef LIBIMAGE_NO_SIMD

	constexpr std::array<r32, 3> STANDARD_GRAYSCALE_COEFFS{ COEFF_RED, COEFF_GREEN, COEFF_BLUE };


	static void simd_grayscale_row(Pixel* src_begin, u8* dst_begin, u32 length)
	{
		constexpr u32 STEP = simd::VEC_LEN;

		auto weights = STANDARD_GRAYSCALE_COEFFS.data();

		auto red_w_vec = simd::load_broadcast(weights);
		auto green_w_vec = simd::load_broadcast(weights + 1);
		auto blue_w_vec = simd::load_broadcast(weights + 2);

		simd::vec_t src_vec{};
		simd::vec_t dst_vec{};
		Pixelr32Planar mem{};

		auto const do_simd = [&](u32 i)
		{
			// pixels are interleaved
			// make them planar
			
			copy_vec_len(src_begin + i, mem);

			src_vec = simd::load(mem.red);
			dst_vec = simd::multiply(src_vec, red_w_vec);

			src_vec = simd::load(mem.green);
			dst_vec = simd::fmadd(src_vec, green_w_vec, dst_vec);

			src_vec = simd::load(mem.blue);
			dst_vec = simd::fmadd(src_vec, blue_w_vec, dst_vec);

			simd::store(mem.alpha, dst_vec);

			simd::cast_copy_len(mem.alpha, dst_begin + i);
		};

		for (u32 i = 0; i < length - STEP; i += STEP)
		{
			do_simd(i);
		}

		do_simd(length - STEP);
	}


	template <class SRC_IMG_T, class DST_IMG_T>
	static void do_grayscale(SRC_IMG_T const& src, DST_IMG_T const& dst)
	{
		do_simd_transform_by_row(src, dst, simd_grayscale_row);
	}

#else

	template <class SRC_IMG_T, class DST_IMG_T>
	static void do_grayscale(SRC_IMG_T const& src, DST_IMG_T const& dst)
	{
		do_transform_by_row(src, dst, pixel_grayscale_standard);
	}


#endif // !LIBIMAGE_NO_SIMD	


	void grayscale(Image const& src, gray::Image const& dst)
	{
		assert(verify(src, dst));

		do_grayscale(src, dst);
	}


	void grayscale(Image const& src, gray::View const& dst)
	{
		assert(verify(src, dst));

		do_grayscale(src, dst);
	}


	void grayscale(View const& src, gray::Image const& dst)
	{
		assert(verify(src, dst));

		do_grayscale(src, dst);
	}


	void grayscale(View const& src, gray::View const& dst)
	{
		assert(verify(src, dst));

		do_grayscale(src, dst);
	}

	void alpha_grayscale(Image const& src)
	{
		transform_alpha(src, pixel_grayscale_standard);
	}


	void alpha_grayscale(View const& src)
	{
		transform_alpha(src, pixel_grayscale_standard);
	}


}

#endif // !LIBIMAGE_NO_COLOR
#endif // !LIBIMAGE_NO_GRAYSCALE


/*  binary  */

#ifndef LIBIMAGE_NO_GRAYSCALE

namespace libimage
{
	void binarize(gray::Image const& src, gray::Image const& dst, u8_to_bool_f const& cond)
	{
		auto const conv = [&](u8 p) { return cond(p) ? 255 : 0; };
		transform(src, dst, conv);
	}


	void binarize(gray::Image const& src, gray::View const& dst, u8_to_bool_f const& cond)
	{
		auto const conv = [&](u8 p) { return cond(p) ? 255 : 0; };
		transform(src, dst, conv);
	}


	void binarize(gray::View const& src, gray::Image const& dst, u8_to_bool_f const& cond)
	{
		auto const conv = [&](u8 p) { return cond(p) ? 255 : 0; };
		transform(src, dst, conv);
	}


	void binarize(gray::View const& src, gray::View const& dst, u8_to_bool_f const& cond)
	{
		auto const conv = [&](u8 p) { return cond(p) ? 255 : 0; };
		transform(src, dst, conv);
	}


	void binarize_in_place(gray::Image const& src_dst, u8_to_bool_f const& cond)
	{
		auto const conv = [&](u8 p) { return cond(p) ? 255 : 0; };
		transform_in_place(src_dst, conv);
	}


	void binarize_in_place(gray::View const& src_dst, u8_to_bool_f const& cond)
	{
		auto const conv = [&](u8 p) { return cond(p) ? 255 : 0; };
		transform_in_place(src_dst, conv);
	}


#ifndef LIBIMAGE_NO_COLOR

	void binarize(Image const& src, gray::Image const& dst, pixel_to_bool_f const& cond)
	{
		auto const conv = [&](Pixel p) { return cond(p) ? 255 : 0; };
		transform(src, dst, conv);
	}


	void binarize(Image const& src, gray::View const& dst, pixel_to_bool_f const& cond)

	{
		auto const conv = [&](Pixel p) { return cond(p) ? 255 : 0; };
		transform(src, dst, conv);
	}


	void binarize(View const& src, gray::Image const& dst, pixel_to_bool_f const& cond)
	{
		auto const conv = [&](Pixel p) { return cond(p) ? 255 : 0; };
		transform(src, dst, conv);
	}


	void binarize(View const& src, gray::View const& dst, pixel_to_bool_f const& cond)
	{
		auto const conv = [&](Pixel p) { return cond(p) ? 255 : 0; };
		transform(src, dst, conv);
	}


#endif // !LIBIMAGE_NO_COLOR


	template <class GRAY_IMG_T>
	Point2Du32 do_centroid(GRAY_IMG_T const& src, u8_to_bool_f const& func)
	{
		constexpr u32 n_threads = N_THREADS;
		u32 h = src.height / n_threads;

		std::array<u32, n_threads> thread_totals = { 0 };
		std::array<u32, n_threads> thread_x_totals = { 0 };
		std::array<u32, n_threads> thread_y_totals = { 0 };

		u32 total = 0;
		u32 x_total = 0;
		u32 y_total = 0;

		auto const row_func = [&](u32 y)
		{
			if (y >= src.height)
			{
				return;
			}

			auto thread_id = y - n_threads * (y / n_threads);

			assert(thread_id < n_threads);

			auto row = row_begin(src, y);
			for (u32 x = 0; x < src.width; ++x)
			{
				u32 val = func(row[x]) ? 1 : 0;

				thread_totals[thread_id] += val;
				thread_x_totals[thread_id] += x * val;
				thread_y_totals[thread_id] += y * val;
			}
		};

		for (u32 y_begin = 0; y_begin < src.height; y_begin += n_threads)
		{
			thread_totals = { 0 };
			thread_x_totals = { 0 };
			thread_y_totals = { 0 };

			for (u32 y = y_begin; y < y_begin + n_threads; ++y)
			{
				row_func(y);
			}

			for (u32 i = 0; i < n_threads; ++i)
			{
				total += thread_totals[i];
				x_total += thread_x_totals[i];
				y_total += thread_y_totals[i];
			}
		}

		Point2Du32 pt{};

		if (total == 0)
		{
			pt.x = src.width / 2;
			pt.y = src.height / 2;
		}
		else
		{
			pt.x = x_total / total;
			pt.y = y_total / total;
		}

		return pt;
	}


	Point2Du32 centroid(gray::Image const& src)
	{
		assert(verify(src));

		auto const func = [](u8 p) { return p > 0; };
		return do_centroid(src, func);
	}


	Point2Du32 centroid(gray::Image const& src, u8_to_bool_f const& func)
	{
		assert(verify(src));

		return do_centroid(src, func);
	}


	Point2Du32 centroid(gray::View const& src)
	{
		auto const func = [](u8 p) { return p > 0; };
		return do_centroid(src, func);
	}


	Point2Du32 centroid(gray::View const& src, u8_to_bool_f const& func)
	{
		assert(verify(src));

		return do_centroid(src, func);
	}


	template <class GRAY_IMG_T>
	static bool do_neighbors(GRAY_IMG_T const& img, u32 x, u32 y)
	{
		assert(x >= 1);
		assert(x < img.width);
		assert(y >= 1);
		assert(y < img.height);

		constexpr std::array<int, 8> x_neighbors = { -1,  0,  1,  1,  1,  0, -1, -1 };
		constexpr std::array<int, 8> y_neighbors = { -1, -1, -1,  0,  1,  1,  1,  0 };

		constexpr auto n_neighbors = x_neighbors.size();
		int value_total = 0;
		u32 value_count = 0;
		u32 flip = 0;

		auto xi = (u32)(x + x_neighbors[n_neighbors - 1]);
		auto yi = (u32)(y + y_neighbors[n_neighbors - 1]);
		auto val = *xy_at(img, xi, yi);
		bool is_on = val != 0;

		for (u32 i = 0; i < n_neighbors; ++i)
		{
			xi = (u32)(x + x_neighbors[i]);
			yi = (u32)(y + y_neighbors[i]);

			val = *xy_at(img, xi, yi);
			flip += (val != 0) != is_on;

			is_on = val != 0;
			value_count += is_on;
		}

		return value_count > 1 && value_count < 7 && flip == 2;
	}


	template <class GRAY_IMG_T>
	static u32 do_skeleton_once(GRAY_IMG_T const& img)
	{
		u32 pixel_count = 0;

		auto width = img.width;
		auto height = img.height;

		auto const xy_func = [&](u32 x, u32 y)
		{
			auto& p = *xy_at(img, x, y);
			if (p == 0)
			{
				return;
			}

			if (do_neighbors(img, x, y))
			{
				p = 0;
			}

			pixel_count += p > 0;
		};

		u32 x_begin = 1;
		u32 x_end = width - 1;
		u32 y_begin = 1;
		u32 y_end = height - 2;
		u32 x = 0;
		u32 y = 0;

		auto const done = [&]() { return !(x_begin < x_end&& y_begin < y_end); };

		while (!done())
		{
			// iterate clockwise
			y = y_begin;
			x = x_begin;
			for (; x < x_end; ++x)
			{
				xy_func(x, y);
			}
			--x;

			for (++y; y < y_end; ++y)
			{
				xy_func(x, y);
			}
			--y;

			for (--x; x >= x_begin; --x)
			{
				xy_func(x, y);
			}
			++x;

			for (--y; y > y_begin; --y)
			{
				xy_func(x, y);
			}
			++y;

			++x_begin;
			++y_begin;
			--x_end;
			--y_end;

			if (done())
			{
				break;
			}

			// iterate counter clockwise
			for (++x; y < y_end; ++y)
			{
				xy_func(x, y);
			}
			--y;

			for (++x; x < x_end; ++x)
			{
				xy_func(x, y);
			}
			--x;

			for (--y; y >= y_begin; --y)
			{
				xy_func(x, y);
			}
			++y;

			for (--x; x >= x_begin; --x)
			{
				xy_func(x, y);
			}
			++x;

			++x_begin;
			++y_begin;
			--x_end;
			--y_end;
		}

		return pixel_count;
	}


	template<class GRAY_SRC_IMG_T, class GRAY_DST_IMG_T>
	static void do_skeleton(GRAY_SRC_IMG_T const& src, GRAY_DST_IMG_T const& dst)
	{
		copy(src, dst);

		u32 current_count = 0;
		u32 pixel_count = do_skeleton_once(dst);
		u32 max_iter = 100; // src.width / 2;

		for (u32 i = 1; pixel_count != current_count && i < max_iter; ++i)
		{
			current_count = pixel_count;
			pixel_count = do_skeleton_once(dst);
		}
	}


	void skeleton(gray::Image const& src, gray::Image const& dst)
	{
		assert(verify(src, dst));

		do_skeleton(src, dst);
	}


	void skeleton(gray::Image const& src, gray::View const& dst)
	{
		assert(verify(src, dst));

		do_skeleton(src, dst);
	}


	void skeleton(gray::View const& src, gray::Image const& dst)
	{
		assert(verify(src, dst));

		do_skeleton(src, dst);
	}


	void skeleton(gray::View const& src, gray::View const& dst)
	{
		assert(verify(src, dst));

		do_skeleton(src, dst);
	}
	
}

#endif // !LIBIMAGE_NO_GRAYSCALE


/*  contrast  */

constexpr u8 U8_MIN = 0;
constexpr u8 U8_MAX = 255;


#ifndef LIBIMAGE_NO_GRAYSCALE

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

	auto const ratio = (static_cast<r64>(val) - src_low) / (src_high - src_low);

	assert(ratio >= 0.0);
	assert(ratio <= 1.0);

	auto const diff = ratio * (dst_high - dst_low);

	return dst_low + static_cast<u8>(diff);
}

namespace libimage
{

	void contrast(gray::Image const& src, gray::Image const& dst, u8 src_low, u8 src_high)
	{
		assert(src_low < src_high);
		auto const conv = [&](u8 p) { return lerp_clamp(src_low, src_high, U8_MIN, U8_MAX, p); };
		transform(src, dst, conv);
	}


	void contrast(gray::Image const& src, gray::View const& dst, u8 src_low, u8 src_high)
	{
		assert(src_low < src_high);
		auto const conv = [&](u8 p) { return lerp_clamp(src_low, src_high, U8_MIN, U8_MAX, p); };
		transform(src, dst, conv);
	}


	void contrast(gray::View const& src, gray::Image const& dst, u8 src_low, u8 src_high)
	{
		assert(src_low < src_high);
		auto const conv = [&](u8 p) { return lerp_clamp(src_low, src_high, U8_MIN, U8_MAX, p); };
		transform(src, dst, conv);
	}



	void contrast(gray::View const& src, gray::View const& dst, u8 src_low, u8 src_high)
	{
		assert(src_low < src_high);
		auto const conv = [&](u8 p) { return lerp_clamp(src_low, src_high, U8_MIN, U8_MAX, p); };
		transform(src, dst, conv);
	}


	void contrast_in_place(gray::Image const& src_dst, u8 src_low, u8 src_high)
	{
		assert(src_low < src_high);
		auto const conv = [&](u8 p) { return lerp_clamp(src_low, src_high, U8_MIN, U8_MAX, p); };
		transform_in_place(src_dst, conv);
	}


	void contrast_in_place(gray::View const& src_dst, u8 src_low, u8 src_high)
	{
		assert(src_low < src_high);
		auto const conv = [&](u8 p) { return lerp_clamp(src_low, src_high, U8_MIN, U8_MAX, p); };
		transform_in_place(src_dst, conv);
	}





}

#endif // !LIBIMAGE_NO_GRAYSCALE


/*  convolve  */

#ifndef LIBIMAGE_NO_GRAYSCALE

namespace libimage
{
	constexpr r32 D3 = 16.0f;
	constexpr std::array<r32, 9> GAUSS_3X3
	{
		(1 / D3), (2 / D3), (1 / D3),
		(2 / D3), (4 / D3), (2 / D3),
		(1 / D3), (2 / D3), (1 / D3),
	};

	constexpr r32 D5 = 256.0f;
	constexpr std::array<r32, 25> GAUSS_5X5
	{
		(1 / D5), (4 / D5),  (6 / D5),  (4 / D5),  (1 / D5),
		(4 / D5), (16 / D5), (24 / D5), (16 / D5), (4 / D5),
		(6 / D5), (24 / D5), (36 / D5), (24 / D5), (6 / D5),
		(4 / D5), (16 / D5), (24 / D5), (16 / D5), (4 / D5),
		(1 / D5), (4 / D5),  (6 / D5),  (4 / D5),  (1 / D5),
	};


	constexpr std::array<r32, 9> GRAD_X_3X3
	{
		-0.25f,  0.0f,  0.25f,
		-0.50f,  0.0f,  0.50f,
		-0.25f,  0.0f,  0.25f,
	};


	constexpr std::array<r32, 9> GRAD_Y_3X3
	{
		-0.25f, -0.50f, -0.25f,
		 0.0f,   0.0f,   0.0f,
		 0.25f,  0.50f,  0.25f,
	};


	class Matrix2Dr32
	{
	public:
		u32 width;
		u32 height;

		r32* data;
	};


	template <size_t N>
	static void fill_kernel_data(r32* dst, std::array<r32, N> const& data)
	{
		for (u32 i = 0; i < data.size(); ++i)
		{
			dst[i] = data[i];
		}
	}


	class ConvolveProps
	{
	public:
		u8* src_begin;
		u8* dst_begin;

		u32 length;
		u32 pitch;

		Matrix2Dr32 kernel;
	};


	class Convolve2Props
	{
	public:
		u8* src_begin;
		u8* dst_begin;
		u8* src2_begin;
		u8* dst2_begin;

		u32 length;
		u32 pitch;

		Matrix2Dr32 kernel;
	};	


#ifndef LIBIMAGE_NO_SIMD


	static void simd_convolve_span(ConvolveProps const& props)
	{
		assert(props.kernel.width % 2 == 1);
		assert(props.kernel.height % 2 == 1);

		int ry_begin = 0 - (props.kernel.height / 2);
		int ry_end = props.kernel.height / 2 + 1;
		int rx_begin = 0 - (props.kernel.width / 2);
		int rx_end = props.kernel.width / 2 + 1;

		auto weights = props.kernel.data;
		u32 w = 0;

		constexpr u32 N = simd::VEC_LEN;
		constexpr u32 STEP = N;

		auto const do_simd = [&](int i)
		{
			MemoryVector mem{};
			w = 0;
			auto acc_vec = simd::setzero();
			auto src_vec = simd::setzero();

			for (int ry = ry_begin; ry < ry_end; ++ry)
			{
				for (int rx = rx_begin; rx < rx_end; ++rx, ++w)
				{
					int offset = ry * props.pitch + rx + i;

					auto ptr = props.src_begin + offset;
					simd::cast_copy_len(ptr, mem.data);
					src_vec = simd::load(mem.data);
					auto weight = simd::load_broadcast(weights + w);
					acc_vec = simd::fmadd(weight, src_vec, acc_vec);
				}
			}

			simd::store(mem.data, acc_vec);

			simd::cast_copy_len(mem.data, props.dst_begin + i);
		};

		for (u32 i = 0; i < props.length - STEP; i += STEP)
		{
			do_simd(i);
		}

		do_simd(props.length - STEP);
	}


	static void simd_convolve_span(Convolve2Props const& props)
	{
		assert(props.kernel.width % 2 == 1);
		assert(props.kernel.height % 2 == 1);

		int ry_begin = 0 - (props.kernel.height / 2);
		int ry_end = props.kernel.height / 2 + 1;
		int rx_begin = 0 - (props.kernel.width / 2);
		int rx_end = props.kernel.width / 2 + 1;

		auto weights = props.kernel.data;
		u32 w = 0;

		constexpr u32 N = simd::VEC_LEN;
		constexpr u32 STEP = N;

		auto const do_simd = [&](int i)
		{
			MemoryVector mem{};
			MemoryVector mem2{};
			
			auto acc_vec = simd::setzero();
			auto src_vec = simd::setzero();
			auto acc2_vec = simd::setzero();
			auto src2_vec = simd::setzero();

			w = 0;

			for (int ry = ry_begin; ry < ry_end; ++ry)
			{
				for (int rx = rx_begin; rx < rx_end; ++rx, ++w)
				{
					int offset = ry * props.pitch + rx + i;

					auto ptr = props.src_begin + offset;
					simd::cast_copy_len(ptr, mem.data);
					src_vec = simd::load(mem.data);
					auto weight = simd::load_broadcast(weights + w);
					acc_vec = simd::fmadd(weight, src_vec, acc_vec);

					auto ptr2 = props.src2_begin + offset;
					simd::cast_copy_len(ptr2, mem2.data);
					src2_vec = simd::load(mem2.data);
					weight = simd::load_broadcast(weights + w);
					acc2_vec = simd::fmadd(weight, src2_vec, acc2_vec);
				}
			}

			simd::store(mem.data, acc_vec);
			simd::cast_copy_len(mem.data, props.dst_begin + i);

			simd::store(mem2.data, acc_vec);
			simd::cast_copy_len(mem2.data, props.dst2_begin + i);
		};

		for (u32 i = 0; i < props.length - STEP; i += STEP)
		{
			do_simd(i);
		}

		do_simd(props.length - STEP);
	}


	static void convolve_span(ConvolveProps const& props)
	{
		simd_convolve_span(props);
	}


	static void convolve_span(Convolve2Props const& props)
	{
		simd_convolve_span(props);
	}


#else

	static void convolve_span(ConvolveProps const& props)
	{
		assert(props.kernel.width % 2 == 1);
		assert(props.kernel.height % 2 == 1);

		int ry_begin = 0 - (props.kernel.height / 2);
		int ry_end = props.kernel.height / 2 + 1;
		int rx_begin = 0 - (props.kernel.width / 2);
		int rx_end = props.kernel.width / 2 + 1;

		auto weights = props.kernel.data;
		u32 w = 0;

		for (u32 i = 0; i < props.length; ++i)
		{
			w = 0;
			r32 p = 0.0f;
			for (int ry = ry_begin; ry < ry_end; ++ry)
			{
				for (int rx = rx_begin; rx < rx_end; ++rx, ++w)
				{
					int offset = ry * props.pitch + rx + i;

					p += props.src_begin[offset] * weights[w];
				}
			}

			assert(p >= 0.0f);
			assert(p <= 255.0f);

			props.dst_begin[i] = (u8)p;
		}
	}


	static void convolve_span(Convolve2Props const& props)
	{
		assert(props.kernel.width % 2 == 1);
		assert(props.kernel.height % 2 == 1);

		int ry_begin = 0 - (props.kernel.height / 2);
		int ry_end = props.kernel.height / 2 + 1;
		int rx_begin = 0 - (props.kernel.width / 2);
		int rx_end = props.kernel.width / 2 + 1;

		auto weights = props.kernel.data;
		u32 w = 0;

		for (u32 i = 0; i < props.length; ++i)
		{
			w = 0;
			r32 p = 0.0f;
			r32 p2 = 0.0f;
			for (int ry = ry_begin; ry < ry_end; ++ry)
			{
				for (int rx = rx_begin; rx < rx_end; ++rx, ++w)
				{
					int offset = ry * props.pitch + rx + i;

					p += props.src_begin[offset] * weights[w];
					p2 += props.src2_begin[offset] * weights[w];
				}
			}

			assert(p >= 0.0f);
			assert(p <= 255.0f);
			assert(p2 >= 0.0f);
			assert(p2 <= 255.0f);

			props.dst_begin[i] = (u8)p;
			props.dst2_begin[i] = (u8)p2;
		}
	}


#endif // !LIBIMAGE_NO_SIMD

	template<class GRAY_SRC_IMG_T, class GRAY_DST_IMG_T>
	static void do_convolve_in_range(GRAY_SRC_IMG_T const& src, GRAY_DST_IMG_T const& dst, Range2Du32 const& range, Matrix2Dr32 const& kernel)
	{
		auto const height = range.y_end - range.y_begin;
		auto const width = range.x_end - range.x_begin;
		auto const rows_per_thread = height / N_THREADS;
		auto const pitch = (u32)(row_begin(src, 1) - row_begin(src, 0));

		auto const thread_proc = [&](u32 id)
		{
			auto y_begin = range.y_begin + id * rows_per_thread;
			auto y_end = range.y_begin + (id == N_THREADS - 1 ? height : (id + 1) * rows_per_thread);

			for (u32 y = y_begin; y < y_end; ++y)
			{
				ConvolveProps props{};
				props.src_begin = row_begin(src, y) + range.x_begin;
				props.dst_begin = row_begin(dst, y) + range.x_begin;
				props.kernel = kernel;
				props.length = width;
				props.pitch = pitch;

				convolve_span(props);
			}
		};

		execute_procs(make_proc_list(thread_proc));
	}
}

#endif // !LIBIMAGE_NO_GRAYSCALE


/*  blur  */

#ifndef LIBIMAGE_NO_GRAYSCALE

namespace libimage
{

	template<class GRAY_SRC_IMG_T, class GRAY_DST_IMG_T>
	static void do_copy_top_bottom(GRAY_SRC_IMG_T const& src, GRAY_DST_IMG_T const& dst)
	{
		u32 const x_begin = 0;
		u32 const x_end = src.width;
		u32 const y_top = 0;
		u32 const y_bottom = src.height - 1;

		auto const src_top = row_begin(src, y_top);
		auto const dst_top = row_begin(dst, y_top);
		auto const src_bottom = row_begin(src, y_bottom);
		auto const dst_bottom = row_begin(dst, y_bottom);

		for (u32 x = x_begin; x < x_end; ++x)
		{
			dst_top[x] = src_top[x];
			dst_bottom[x] = src_bottom[x];
		}
	}


	template<class GRAY_SRC_IMG_T, class GRAY_DST_IMG_T>
	static void do_copy_left_right(GRAY_SRC_IMG_T const& src, GRAY_DST_IMG_T const& dst)
	{
		u32 const y_begin = 1;
		u32 const y_end = src.height - 1;
		u32 const x_left = 0;
		u32 const x_right = src.width - 1;

		for (u32 y = y_begin; y < y_end; ++y)
		{
			auto src_row = row_begin(src, y);
			auto dst_row = row_begin(dst, y);

			dst_row[x_left] = src_row[x_left];
			dst_row[x_right] = src_row[x_right];
		}
	}


	template<class GRAY_SRC_IMG_T, class GRAY_DST_IMG_T>
	static void do_gauss_inner_top_bottom(GRAY_SRC_IMG_T const& src, GRAY_DST_IMG_T const& dst)
	{
		u32 const x_begin = 1;
		u32 const x_end = src.width - 1;
		u32 const y_top = 1;
		u32 const y_bottom = src.height - 2;

		r32 kernel_data[GAUSS_3X3.size()];
		fill_kernel_data(kernel_data, GAUSS_3X3);
		Matrix2Dr32 kernel{};
		kernel.width = 3;
		kernel.height = 3;
		kernel.data = kernel_data;

		Convolve2Props props{};
		props.src_begin = row_begin(src, y_top) + x_begin;
		props.dst_begin = row_begin(dst, y_top) + x_begin;
		props.src2_begin = row_begin(src, y_bottom) + x_begin;
		props.dst2_begin = row_begin(dst, y_bottom) + x_begin;
		props.length = x_end - x_begin;
		props.pitch = (u32)(row_begin(src, 1) - row_begin(src, 0));
		props.kernel = kernel;
		
		convolve_span(props);
	}


	template<class GRAY_SRC_IMG_T, class GRAY_DST_IMG_T>
	static void do_gauss_inner_left_right(GRAY_SRC_IMG_T const& src, GRAY_DST_IMG_T const& dst)
	{
		u32 const y_begin = 2;
		u32 const y_end = src.height - 2;
		int const x_left = 1;
		int const x_right = src.width - 2;

		auto const pitch = (u32)(row_begin(src, 1) - row_begin(src, 0));
		auto& weights = GAUSS_3X3;

		for (u32 y = y_begin; y < y_end; ++y)
		{
			auto src_row = row_begin(src, y);
			auto dst_row = row_begin(dst, y);

			u32 w = 0;
			r32 l = 0.0f;
			r32 r = 0.0f;
			for (int ry = -1; ry < 2; ++ry)
			{
				for (int rx = -1; rx < 2; ++rx, ++w)
				{
					int offset = ry * pitch + rx;

					l += src_row[x_left + offset] * weights[w];
					r += src_row[x_right + offset] * weights[w];
				}
			}

			dst_row[x_left] = (u8)l;
			dst_row[x_right] = (u8)r;
		}
	}


	template<class GRAY_SRC_IMG_T, class GRAY_DST_IMG_T>
	static void do_inner_gauss(GRAY_SRC_IMG_T const& src, GRAY_DST_IMG_T const& dst)
	{
		Range2Du32 r{};
		r.x_begin = 2;
		r.x_end = src.width - 2;
		r.y_begin = 2;
		r.y_end = src.height - 2;

		r32 kernel_data[GAUSS_5X5.size()];
		fill_kernel_data(kernel_data, GAUSS_5X5);
		Matrix2Dr32 kernel{};
		kernel.width = 5;
		kernel.height = 5;
		kernel.data = kernel_data;

		do_convolve_in_range(src, dst, r, kernel);
	}


	template<class GRAY_SRC_IMG_T, class GRAY_DST_IMG_T>
	static void do_blur(GRAY_SRC_IMG_T const& src, GRAY_DST_IMG_T const& dst)
	{
		std::array<std::function<void()>, 4> f_list =
		{
			[&]() { do_copy_top_bottom(src, dst); },
			[&]() { do_copy_left_right(src, dst); },
			[&]() { do_gauss_inner_top_bottom(src, dst); },
			[&]() { do_gauss_inner_left_right(src, dst); },
		};

		do_for_each(f_list, [](auto const& f) { f(); });

		do_inner_gauss(src, dst);
	}


	void blur(gray::Image const& src, gray::Image const& dst)
	{
		assert(verify(src, dst));
		auto const width = src.width;
		auto const height = src.height;

		assert(width >= VIEW_MIN_DIM);
		assert(height >= VIEW_MIN_DIM);

		do_blur(src, dst);
	}


	void blur(gray::Image const& src, gray::View const& dst)
	{
		assert(verify(src, dst));
		auto const width = src.width;
		auto const height = src.height;

		assert(width >= VIEW_MIN_DIM);
		assert(height >= VIEW_MIN_DIM);

		do_blur(src, dst);
	}


	void blur(gray::View const& src, gray::Image const& dst)
	{
		assert(verify(src, dst));
		auto const width = src.width;
		auto const height = src.height;

		assert(width >= VIEW_MIN_DIM);
		assert(height >= VIEW_MIN_DIM);

		do_blur(src, dst);
	}


	void blur(gray::View const& src, gray::View const& dst)
	{
		assert(verify(src, dst));
		auto const width = src.width;
		auto const height = src.height;

		assert(width >= VIEW_MIN_DIM);
		assert(height >= VIEW_MIN_DIM);

		do_blur(src, dst);
	}
}

#endif // !LIBIMAGE_NO_GRAYSCALE


/*  edges_gradients  */

#ifndef LIBIMAGE_NO_GRAYSCALE

namespace libimage
{
#ifndef LIBIMAGE_NO_SIMD


	static void simd_gradients_span(u8* src_begin, u8* dst_begin, u32 length, u32 pitch)
	{
		constexpr u32 N = simd::VEC_LEN;
		constexpr u32 STEP = N;

		auto const do_simd = [&](int i)
		{
			MemoryVector mem{};
			u32 w = 0;
			auto vec_x = simd::setzero();
			auto vec_y = simd::setzero();
			auto src_vec = simd::setzero();

			for (int ry = -1; ry < 2; ++ry)
			{
				for (int rx = -1; rx < 2; ++rx, ++w)
				{
					int offset = ry * pitch + rx + i;
					auto ptr = src_begin + offset;
					simd::cast_copy_len(ptr, mem.data);

					src_vec = simd::load(mem.data);

					auto weight_x = simd::load_broadcast(GRAD_X_3X3.data() + w);
					auto weight_y = simd::load_broadcast(GRAD_Y_3X3.data() + w);

					vec_x = simd::fmadd(weight_x, src_vec, vec_x);
					vec_y = simd::fmadd(weight_y, src_vec, vec_y);
				}
			}

			vec_x = simd::multiply(vec_x, vec_x);
			vec_y = simd::multiply(vec_y, vec_y);

			auto grad = simd::sqrt(simd::add(vec_x, vec_y));
			simd::store(mem.data, grad);

			simd::cast_copy_len(mem.data, dst_begin + i);
		};

		for (u32 i = 0; i < length - STEP; i += STEP)
		{
			do_simd(i);
		}

		do_simd(length - STEP);
	}


	static void simd_edges_span(u8* src_begin, u8* dst_begin, u32 length, u32 pitch, u8_to_bool_f const& cond)
	{
		constexpr u32 N = simd::VEC_LEN;
		constexpr u32 STEP = N;

		auto const do_simd = [&](int i)
		{
			MemoryVector mem{};
			u32 w = 0;
			auto vec_x = simd::setzero();
			auto vec_y = simd::setzero();
			auto src_vec = simd::setzero();

			for (int ry = -1; ry < 2; ++ry)
			{
				for (int rx = -1; rx < 2; ++rx, ++w)
				{
					int offset = ry * pitch + rx + i;
					auto ptr = src_begin + offset;
					simd::cast_copy_len(ptr, mem.data);

					src_vec = simd::load(mem.data);

					auto weight_x = simd::load_broadcast(GRAD_X_3X3.data() + w);
					auto weight_y = simd::load_broadcast(GRAD_Y_3X3.data() + w);

					vec_x = simd::fmadd(weight_x, src_vec, vec_x);
					vec_y = simd::fmadd(weight_y, src_vec, vec_y);
				}
			}

			vec_x = simd::multiply(vec_x, vec_x);
			vec_y = simd::multiply(vec_y, vec_y);

			auto grad = simd::sqrt(simd::add(vec_x, vec_y));
			simd::store(mem.data, grad);

			simd::transform_len(mem.data, dst_begin + i, [&](r32 val) { return (u8)(cond((u8)val) ? 255 : 0); });
		};

		for (u32 i = 0; i < length - STEP; i += STEP)
		{
			do_simd(i);
		}

		do_simd(length - STEP);
	}


	static void gradients_span(u8* src_begin, u8* dst_begin, u32 length, u32 pitch)
	{
		simd_gradients_span(src_begin, dst_begin, length, pitch);
	}


	static void edges_span(u8* src_begin, u8* dst_begin, u32 length, u32 pitch, u8_to_bool_f const& cond)
	{
		simd_edges_span(src_begin, dst_begin, length, pitch, cond);
	}


#else

	static void gradients_span(u8* src_begin, u8* dst_begin, u32 length, u32 pitch)
	{		
		u32 w = 0;
		for (u32 i = 0; i < length; ++i)
		{
			r32 grad_x = 0.0f;
			r32 grad_y = 0.0f;
			w = 0;
			for (int ry = -1; ry < 2; ++ry)
			{
				for (int rx = -1; rx < 2; ++rx, ++w)
				{
					int offset = ry * pitch + rx + i;
					auto p = src_begin[offset];

					grad_x += GRAD_X_3X3[w] * p;
					grad_y += GRAD_Y_3X3[w] * p;
				}
			}

			auto g = std::hypot(grad_x, grad_y);

			assert(g >= 0.0f);
			assert(g <= 255.0f);

			dst_begin[i] = (u8)g;
		}
	}


	static void edges_span(u8* src_begin, u8* dst_begin, u32 length, u32 pitch, u8_to_bool_f const& cond)
	{
		u32 w = 0;
		for (u32 i = 0; i < length; ++i)
		{
			r32 grad_x = 0.0f;
			r32 grad_y = 0.0f;
			w = 0;
			for (int ry = -1; ry < 2; ++ry)
			{
				for (int rx = -1; rx < 2; ++rx, ++w)
				{
					int offset = ry * pitch + rx + i;
					auto p = src_begin[offset];

					grad_x += GRAD_X_3X3[w] * p;
					grad_y += GRAD_Y_3X3[w] * p;
				}
			}			

			auto g =std::hypot(grad_x, grad_y);

			assert(g >= 0.0f);
			assert(g <= 255.0f);

			dst_begin[i] = cond((u8)g) ? 255 : 0;
		}
	}

#endif !LIBIMAGE_NO_SIMD


	template<class GRAY_SRC_IMG_T, class GRAY_DST_IMG_T>
	static void do_gradients_in_range(GRAY_SRC_IMG_T const& src, GRAY_DST_IMG_T const& dst, Range2Du32 const& range)
	{
		auto const height = range.y_end - range.y_begin;
		auto const width = range.x_end - range.x_begin;
		auto const rows_per_thread = height / N_THREADS;
		auto const pitch = (u32)(row_begin(src, 1) - row_begin(src, 0));

		auto const thread_proc = [&](u32 id)
		{
			auto y_begin = range.y_begin + id * rows_per_thread;
			auto y_end = range.y_begin + (id == N_THREADS - 1 ? height : (id + 1) * rows_per_thread);

			for (u32 y = y_begin; y < y_end; ++y)
			{
				auto src_begin = row_begin(src, y) + range.x_begin;
				auto dst_begin = row_begin(dst, y) + range.x_begin;

				gradients_span(src_begin, dst_begin, width, pitch);
			}
		};

		execute_procs(make_proc_list(thread_proc));
	}


	template<class GRAY_SRC_IMG_T, class GRAY_DST_IMG_T>
	static void do_edges_in_range(GRAY_SRC_IMG_T const& src, GRAY_DST_IMG_T const& dst, Range2Du32 const& range, u8_to_bool_f const& cond)
	{
		auto const height = range.y_end - range.y_begin;
		auto const width = range.x_end - range.x_begin;
		auto const rows_per_thread = height / N_THREADS;
		auto const pitch = (u32)(row_begin(src, 1) - row_begin(src, 0));

		auto const thread_proc = [&](u32 id)
		{
			auto y_begin = range.y_begin + id * rows_per_thread;
			auto y_end = range.y_begin + (id == N_THREADS - 1 ? height : (id + 1) * rows_per_thread);

			for (u32 y = y_begin; y < y_end; ++y)
			{
				auto src_begin = row_begin(src, y) + range.x_begin;
				auto dst_begin = row_begin(dst, y) + range.x_begin;

				edges_span(src_begin, dst_begin, width, pitch, cond);
			}
		};

		execute_procs(make_proc_list(thread_proc));
	}


	template<class GRAY_IMG_T>
	static void do_zero_top_bottom(GRAY_IMG_T const& dst)
	{
		u32 const x_begin = 0;
		u32 const x_end = dst.width;
		u32 const y_top = 0;
		u32 const y_bottom = dst.height - 1;

		auto const dst_top = row_begin(dst, y_top);
		auto const dst_bottom = row_begin(dst, y_bottom);

		for (u32 x = x_begin; x < x_end; ++x)
		{
			dst_top[x] = 0;
			dst_bottom[x] = 0;
		}
	}


	template<class GRAY_IMG_T>
	static void do_zero_left_right(GRAY_IMG_T const& dst)
	{
		u32 const y_begin = 1;
		u32 const y_end = dst.height - 1;
		u32 const x_left = 0;
		u32 const x_right = dst.width - 1;

		for (u32 y = y_begin; y < y_end; ++y)
		{
			auto dst_row = row_begin(dst, y);

			dst_row[x_left] = 0;
			dst_row[x_right] = 0;
		}
	}


	template<class GRAY_SRC_IMG_T, class GRAY_DST_IMG_T>
	static void do_edges(GRAY_SRC_IMG_T const& src, GRAY_DST_IMG_T const& dst, u8_to_bool_f const& cond)
	{
		std::array<std::function<void()>, 2> f_list
		{
			[&]() { do_zero_top_bottom(dst); },
			[&]() { do_zero_left_right(dst); },
		};

		do_for_each(f_list, [](auto const& f) { f(); });

		Range2Du32 r{};
		r.x_begin = 1;
		r.x_end = src.width - 1;
		r.y_begin = 1;
		r.y_end = src.height - 1;

		do_edges_in_range(src, dst, r, cond);
	}


	template<class GRAY_SRC_IMG_T, class GRAY_DST_IMG_T>
	static void do_gradients(GRAY_SRC_IMG_T const& src, GRAY_DST_IMG_T const& dst)
	{
		std::array<std::function<void()>, 2> f_list
		{
			[&]() { do_zero_top_bottom(dst); },
			[&]() { do_zero_left_right(dst); },
		};

		do_for_each(f_list, [](auto const& f) { f(); });

		Range2Du32 r{};
		r.x_begin = 1;
		r.x_end = src.width - 1;
		r.y_begin = 1;
		r.y_end = src.height - 1;

		do_gradients_in_range(src, dst, r);
	}


	void edges(gray::Image const& src, gray::Image const& dst, u8_to_bool_f const& cond)
	{
		assert(verify(src, dst));

		do_edges(src, dst, cond);
	}


	void edges(gray::Image const& src, gray::View const& dst, u8_to_bool_f const& cond)
	{
		assert(verify(src, dst));

		do_edges(src, dst, cond);
	}


	void edges(gray::View const& src, gray::Image const& dst, u8_to_bool_f const& cond)
	{
		assert(verify(src, dst));

		do_edges(src, dst, cond);
	}


	void edges(gray::View const& src, gray::View const& dst, u8_to_bool_f const& cond)
	{
		assert(verify(src, dst));

		do_edges(src, dst, cond);
	}


	void gradients(gray::Image const& src, gray::Image const& dst)
	{
		assert(verify(src, dst));

		do_gradients(src, dst);
	}


	void gradients(gray::Image const& src, gray::View const& dst)
	{
		assert(verify(src, dst));

		do_gradients(src, dst);
	}


	void gradients(gray::View const& src, gray::Image const& dst)
	{
		assert(verify(src, dst));

		do_gradients(src, dst);
	}


	void gradients(gray::View const& src, gray::View const& dst)
	{
		assert(verify(src, dst));

		do_gradients(src, dst);
	}
}

#endif // !LIBIMAGE_NO_GRAYSCALE


/*  rotate  */

static Point2Dr32 find_rotation_src(Point2Du32 const& pt, Point2Du32 const& origin, r32 theta_rotate)
{
	auto dx_dst = (r32)pt.x - (r32)origin.x;
	auto dy_dst = (r32)pt.y - (r32)origin.y;

	auto radius = std::hypotf(dx_dst, dy_dst);

	auto theta_dst = atan2f(dy_dst, dx_dst);
	auto theta_src = theta_dst - theta_rotate;

	auto dx_src = radius * cosf(theta_src);
	auto dy_src = radius * sinf(theta_src);

	Point2Dr32 pt_src{};
	pt_src.x = (r32)origin.x + dx_src;
	pt_src.y = (r32)origin.y + dy_src;

	return pt_src;
}


namespace libimage
{

#ifndef LIBIMAGE_NO_COLOR

	template <typename IMG_T>
	static Pixel do_get_color(IMG_T const& src_image, Point2Dr32 location)
	{
		auto zero = 0.0f;
		auto width = (r32)src_image.width;
		auto height = (r32)src_image.height;

		auto x = location.x;
		auto y = location.y;

		if (x < zero || x >= width || y < zero || y >= height)
		{
			return to_pixel(0, 0, 0);
		}

		return *xy_at(src_image, (u32)floorf(x), (u32)floorf(y));
	}

#endif // !LIBIMAGE_NO_COLOR

#ifndef LIBIMAGE_NO_GRAYSCALE

	template <typename GR_IMG_T>
	static u8 do_get_gray(GR_IMG_T const& src_image, Point2Dr32 location)
	{
		auto zero = 0.0f;
		auto width = (r32)src_image.width;
		auto height = (r32)src_image.height;

		auto x = location.x;
		auto y = location.y;

		if (x < zero || x >= width || y < zero || y >= height)
		{
			return 0;
		}

		return *xy_at(src_image, (u32)floorf(x), (u32)floorf(y));
	}

#endif // !LIBIMAGE_NO_GRAYSCALE




#ifndef LIBIMAGE_NO_COLOR	


	template <typename IMG_SRC_T, typename IMG_DST_T>
	static void do_rotate(IMG_SRC_T const& src, IMG_DST_T const& dst, u32 origin_x, u32 origin_y, r32 theta)
	{
		Point2Du32 origin = { origin_x, origin_y };

		auto const func = [&](u32 x, u32 y) 
		{
			auto src_pt = find_rotation_src({ x, y }, origin, theta);
			*xy_at(dst, x, y) = do_get_color(src, src_pt);
		};

		for_each_xy(src, func);
	}


	void rotate(Image const& src, Image const& dst, u32 origin_x, u32 origin_y, r32 theta)
	{
		assert(verify(src));
		assert(verify(dst));

		do_rotate(src, dst, origin_x, origin_y, theta);
	}


	void rotate(Image const& src, View const& dst, u32 origin_x, u32 origin_y, r32 theta)
	{
		assert(verify(src));
		assert(verify(dst));

		do_rotate(src, dst, origin_x, origin_y, theta);
	}


	void rotate(View const& src, Image const& dst, u32 origin_x, u32 origin_y, r32 theta)
	{
		assert(verify(src));
		assert(verify(dst));

		do_rotate(src, dst, origin_x, origin_y, theta);
	}


	void rotate(View const& src, View const& dst, u32 origin_x, u32 origin_y, r32 theta)
	{
		assert(verify(src));
		assert(verify(dst));

		do_rotate(src, dst, origin_x, origin_y, theta);
	}


	void rotate(Image const& src, Image const& dst, Point2Du32 origin, r32 theta)
	{
		assert(verify(src));
		assert(verify(dst));

		do_rotate(src, dst, origin.x, origin.y, theta);
	}


	void rotate(Image const& src, View const& dst, Point2Du32 origin, r32 theta)
	{
		assert(verify(src));
		assert(verify(dst));

		do_rotate(src, dst, origin.x, origin.y, theta);
	}


	void rotate(View const& src, Image const& dst, Point2Du32 origin, r32 theta)
	{
		assert(verify(src));
		assert(verify(dst));

		do_rotate(src, dst, origin.x, origin.y, theta);
	}


	void rotate(View const& src, View const& dst, Point2Du32 origin, r32 theta)
	{
		assert(verify(src));
		assert(verify(dst));

		do_rotate(src, dst, origin.x, origin.y, theta);
	}


#endif // !LIBIMAGE_NO_COLOR

#ifndef LIBIMAGE_NO_GRAYSCALE

	template <typename GR_IMG_SRC_T, typename GR_IMG_DST_T>
	static void do_rotate_gray(GR_IMG_SRC_T const& src, GR_IMG_DST_T const& dst, u32 origin_x, u32 origin_y, r32 theta)
	{
		Point2Du32 origin = { origin_x, origin_y };

		auto const func = [&](u32 x, u32 y) 
		{
			auto src_pt = find_rotation_src({ x, y }, origin, theta);
			*xy_at(dst, x, y) = do_get_gray(src, src_pt);
		};

		for_each_xy(src, func);
	}


	void rotate(gray::Image const& src, gray::Image const& dst, u32 origin_x, u32 origin_y, r32 theta)
	{
		assert(verify(src));
		assert(verify(dst));

		do_rotate_gray(src, dst, origin_x, origin_y, theta);
	}


	void rotate(gray::Image const& src, gray::View const& dst, u32 origin_x, u32 origin_y, r32 theta)
	{
		assert(verify(src));
		assert(verify(dst));

		do_rotate_gray(src, dst, origin_x, origin_y, theta);
	}


	void rotate(gray::View const& src, gray::Image const& dst, u32 origin_x, u32 origin_y, r32 theta)
	{
		assert(verify(src));
		assert(verify(dst));

		do_rotate_gray(src, dst, origin_x, origin_y, theta);
	}


	void rotate(gray::View const& src, gray::View const& dst, u32 origin_x, u32 origin_y, r32 theta)
	{
		assert(verify(src));
		assert(verify(dst));

		do_rotate_gray(src, dst, origin_x, origin_y, theta);
	}


	void rotate(gray::Image const& src, gray::Image const& dst, Point2Du32 origin, r32 theta)
	{
		assert(verify(src));
		assert(verify(dst));

		do_rotate_gray(src, dst, origin.x, origin.y, theta);
	}


	void rotate(gray::Image const& src, gray::View const& dst, Point2Du32 origin, r32 theta)
	{
		assert(verify(src));
		assert(verify(dst));

		do_rotate_gray(src, dst, origin.x, origin.y, theta);
	}


	void rotate(gray::View const& src, gray::Image const& dst, Point2Du32 origin, r32 theta)
	{
		assert(verify(src));
		assert(verify(dst));

		do_rotate_gray(src, dst, origin.x, origin.y, theta);
	}


	void rotate(gray::View const& src, gray::View const& dst, Point2Du32 origin, r32 theta)
	{
		assert(verify(src));
		assert(verify(dst));

		do_rotate_gray(src, dst, origin.x, origin.y, theta);
	}


#endif // !LIBIMAGE_NO_GRAYSCALE


}