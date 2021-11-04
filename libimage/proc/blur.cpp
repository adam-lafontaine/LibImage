/*

Copyright (c) 2021 Adam Lafontaine

*/

#ifndef LIBIMAGE_NO_GRAYSCALE

#include "process.hpp"
#include "verify.hpp"
#include "index_range.hpp"
#include "convolve.hpp"

#include <algorithm>

#ifndef LIBIMAGE_NO_PARALLEL
#include <execution>
#endif // !LIBIMAGE_NO_PARALLEL

namespace libimage
{
#ifndef LIBIMAGE_NO_PARALLEL


	template<class GRAY_SRC_IMG_T, class GRAY_DST_IMG_T>
	static void copy_top(GRAY_SRC_IMG_T const& src, GRAY_DST_IMG_T const& dst)
	{
		auto src_top = row_view(src, 0);
		auto dst_top = row_view(dst, 0);

		copy(src_top, dst_top);
	}


	template<class GRAY_SRC_IMG_T, class GRAY_DST_IMG_T>
	static void copy_bottom(GRAY_SRC_IMG_T const& src, GRAY_DST_IMG_T const& dst)
	{
		auto src_bottom = row_view(src, src.height - 1);
		auto dst_bottom = row_view(dst, src.height - 1);

		copy(src_bottom, dst_bottom);
	}


	template<class GRAY_SRC_IMG_T, class GRAY_DST_IMG_T>
	static void copy_left(GRAY_SRC_IMG_T const& src, GRAY_DST_IMG_T const& dst)
	{
		pixel_range_t r = {};
		r.x_begin = 0;
		r.x_end = 1;
		r.y_begin = 1;
		r.y_end = src.height - 1;

		auto src_left = sub_view(src, r);
		auto dst_left = sub_view(dst, r);

		copy(src_left, dst_left);
	}


	template<class GRAY_SRC_IMG_T, class GRAY_DST_IMG_T>
	static void copy_right(GRAY_SRC_IMG_T const& src, GRAY_DST_IMG_T const& dst)
	{
		pixel_range_t r = {};
		r.x_begin = src.width - 1;
		r.x_end = src.width;
		r.y_begin = 1;
		r.y_end = src.height - 1;

		auto src_right = sub_view(src, r);
		auto dst_right = sub_view(dst, r);

		copy(src_right, dst_right);
	}


	template<class GRAY_SRC_IMG_T, class GRAY_DST_IMG_T>
	static void gauss_inner_top(GRAY_SRC_IMG_T const& src, GRAY_DST_IMG_T const& dst)
	{
		u32 const x_begin = 1;
		u32 const x_end = src.width - 1;
		u32 const y = 1;

		auto dst_top = dst.row_begin(y);

		u32_range_t x_ids(x_begin, x_end);

		auto const gauss = [&](u32 x)
		{
			dst_top[x] = gauss3(src, x, y);
		};

		std::for_each(std::execution::par, x_ids.begin(), x_ids.end(), gauss);
	}


	template<class GRAY_SRC_IMG_T, class GRAY_DST_IMG_T>
	static void gauss_inner_bottom(GRAY_SRC_IMG_T const& src, GRAY_DST_IMG_T const& dst)
	{
		u32 const x_begin = 1;
		u32 const x_end = src.width - 1;
		u32 const y = src.height - 2;

		auto dst_bottom = dst.row_begin(y);

		u32_range_t x_ids(x_begin, x_end);

		auto const gauss = [&](u32 x)
		{
			dst_bottom[x] = gauss3(src, x, y);
		};

		std::for_each(std::execution::par, x_ids.begin(), x_ids.end(), gauss);
	}


	template<class GRAY_SRC_IMG_T, class GRAY_DST_IMG_T>
	static void gauss_inner_left(GRAY_SRC_IMG_T const& src, GRAY_DST_IMG_T const& dst)
	{
		u32 const y_begin = 2;
		u32 const y_end = src.height - 2;
		u32 const x = 1;

		u32_range_t y_ids(y_begin, y_end);

		auto const gauss = [&](u32 y)
		{
			auto dst_row = dst.row_begin(y);
			dst_row[x] = gauss3(src, x, y);
		};

		std::for_each(std::execution::par, y_ids.begin(), y_ids.end(), gauss);
	}


	template<class GRAY_SRC_IMG_T, class GRAY_DST_IMG_T>
	static void gauss_inner_right(GRAY_SRC_IMG_T const& src, GRAY_DST_IMG_T const& dst)
	{
		u32 const y_begin = 2;
		u32 const y_end = src.height - 2;
		u32 const x = src.width - 2;

		u32_range_t y_ids(y_begin, y_end);

		auto const gauss = [&](u32 y)
		{
			auto dst_row = dst.row_begin(y);
			dst_row[x] = gauss3(src, x, y);
		};

		std::for_each(std::execution::par, y_ids.begin(), y_ids.end(), gauss);
	}


	template<class GRAY_SRC_IMG_T, class GRAY_DST_IMG_T>
	static void inner_gauss(GRAY_SRC_IMG_T const& src, GRAY_DST_IMG_T const& dst)
	{
		u32 const x_begin = 2;
		u32 const x_end = src.width - 2;
		u32_range_t x_ids(x_begin, x_end);

		auto const gauss_row = [&](u32 y)
		{
			auto dst_row = dst.row_begin(y);

			auto const gauss_x = [&](u32 x)
			{
				dst_row[x] = gauss5(src, x, y);
			};

			std::for_each(std::execution::par, x_ids.begin(), x_ids.end(), gauss_x);
		};

		u32 const y_begin = 2;
		u32 const y_end = src.height - 2;

		u32_range_t y_ids(y_begin, y_end);

		std::for_each(std::execution::par, y_ids.begin(), y_ids.end(), gauss_row);
	}


	template<class GRAY_SRC_IMG_T, class GRAY_DST_IMG_T>
	static void do_blur(GRAY_SRC_IMG_T const& src, GRAY_DST_IMG_T const& dst)
	{
		// lambdas in an array
		std::array<std::function<void()>, 9> f_list =
		{
			[&]() { copy_top(src, dst); },
			[&]() { copy_bottom(src, dst); } ,
			[&]() { copy_left(src, dst); } ,
			[&]() { copy_right(src, dst); },
			[&]() { gauss_inner_top(src, dst); },
			[&]() { gauss_inner_bottom(src, dst); },
			[&]() { gauss_inner_left(src, dst); },
			[&]() { gauss_inner_right(src, dst); },
			[&]() { inner_gauss(src, dst); }
		};

		// execute everything
		std::for_each(std::execution::par, f_list.begin(), f_list.end(), [](auto const& f) { f(); });
	}


	void blur(gray::image_t const& src, gray::image_t const& dst)
	{
		assert(verify(src, dst));
		auto const width = src.width;
		auto const height = src.height;

		assert(width >= VIEW_MIN_DIM);
		assert(height >= VIEW_MIN_DIM);

		do_blur(src, dst);
	}


	void blur(gray::image_t const& src, gray::view_t const& dst)
	{
		assert(verify(src, dst));
		auto const width = src.width;
		auto const height = src.height;

		assert(width >= VIEW_MIN_DIM);
		assert(height >= VIEW_MIN_DIM);

		do_blur(src, dst);
	}


	void blur(gray::view_t const& src, gray::image_t const& dst)
	{
		assert(verify(src, dst));
		auto const width = src.width;
		auto const height = src.height;

		assert(width >= VIEW_MIN_DIM);
		assert(height >= VIEW_MIN_DIM);

		do_blur(src, dst);
	}


	void blur(gray::view_t const& src, gray::view_t const& dst)
	{
		assert(verify(src, dst));
		auto const width = src.width;
		auto const height = src.height;

		assert(width >= VIEW_MIN_DIM);
		assert(height >= VIEW_MIN_DIM);

		do_blur(src, dst);
	}


#endif // !LIBIMAGE_NO_PARALLEL


	namespace seq
	{
		template<class SRC_GRAY_IMG_T, class DST_GRAY_IMG_T>
		static void copy_top_bottom(SRC_GRAY_IMG_T const& src, DST_GRAY_IMG_T const& dst)
		{
			u32 x_first = 0;
			u32 y_first = 0;
			u32 x_last = src.width - 1;
			u32 y_last = src.height - 1;
			auto src_top = src.row_begin(y_first);
			auto src_bottom = src.row_begin(y_last);
			auto dst_top = dst.row_begin(y_first);
			auto dst_bottom = dst.row_begin(y_last);
			for (u32 x = x_first; x <= x_last; ++x)
			{
				dst_top[x] = src_top[x];
				dst_bottom[x] = src_bottom[x];
			}
		}


		template<class SRC_GRAY_IMG_T, class DST_GRAY_IMG_T>
		static void copy_left_right(SRC_GRAY_IMG_T const& src, DST_GRAY_IMG_T const& dst)
		{
			u32 x_first = 0;
			u32 y_first = 1;
			u32 x_last = src.width - 1;
			u32 y_last = src.height - 2;
			for (u32 y = y_first; y <= y_last; ++y)
			{
				auto src_row = src.row_begin(y);
				auto dst_row = dst.row_begin(y);
				dst_row[x_first] = src_row[x_first];
				dst_row[x_last] = src_row[x_last];
			}
		}


		template<class SRC_GRAY_IMG_T, class DST_GRAY_IMG_T>
		static void gauss_inner_top_bottom(SRC_GRAY_IMG_T const& src, DST_GRAY_IMG_T const& dst)
		{
			u32 x_first = 1;
			u32 y_first = 1;
			u32 x_last = src.width - 2;
			u32 y_last = src.height - 2;
			auto dst_top = dst.row_begin(y_first);
			auto dst_bottom = dst.row_begin(y_last);
			for (u32 x = x_first; x <= x_last; ++x)
			{
				dst_top[x] = gauss3(src, x, y_first);
				dst_bottom[x] = gauss3(src, x, y_last);
			}
		}


		template<class SRC_GRAY_IMG_T, class DST_GRAY_IMG_T>
		static void gauss_inner_left_right(SRC_GRAY_IMG_T const& src, DST_GRAY_IMG_T const& dst)
		{
			u32 x_first = 1;
			u32 y_first = 2;
			u32 x_last = src.width - 2;
			u32 y_last = src.height - 3;
			for (u32 y = y_first; y <= y_last; ++y)
			{
				auto dst_row = dst.row_begin(y);
				dst_row[x_first] = gauss3(src, x_first, y);
				dst_row[x_last] = gauss3(src, x_last, y);
			}
		}


		template<class GRAY_SRC_IMG_T, class GRAY_DST_IMG_T>
		static void inner_gauss(GRAY_SRC_IMG_T const& src, GRAY_DST_IMG_T const& dst)
		{
			u32 x_first = 2;
			u32 y_first = 2;
			u32 x_last = src.width - 3;
			u32 y_last = src.height - 3;
			for (u32 y = y_first; y <= y_last; ++y)
			{
				auto dst_row = dst.row_begin(y);

				for (u32 x = x_first; x <= x_last; ++x)
				{
					dst_row[x] = gauss5(src, x, y);
				}
			}
		}


		void blur(gray::image_t const& src, gray::image_t const& dst)
		{
			assert(verify(src, dst));
			assert(src.width >= VIEW_MIN_DIM);
			assert(src.height >= VIEW_MIN_DIM);

			seq::copy_top_bottom(src, dst);
			seq::copy_left_right(src, dst);
			seq::gauss_inner_top_bottom(src, dst);
			seq::gauss_inner_left_right(src, dst);
			seq::inner_gauss(src, dst);
		}


		void blur(gray::image_t const& src, gray::view_t const& dst)
		{
			assert(verify(src, dst));
			assert(src.width >= VIEW_MIN_DIM);
			assert(src.height >= VIEW_MIN_DIM);

			seq::copy_top_bottom(src, dst);
			seq::copy_left_right(src, dst);
			seq::gauss_inner_top_bottom(src, dst);
			seq::gauss_inner_left_right(src, dst);
			seq::inner_gauss(src, dst);
		}


		void blur(gray::view_t const& src, gray::image_t const& dst)
		{
			assert(verify(src, dst));
			assert(src.width >= VIEW_MIN_DIM);
			assert(src.height >= VIEW_MIN_DIM);

			seq::copy_top_bottom(src, dst);
			seq::copy_left_right(src, dst);
			seq::gauss_inner_top_bottom(src, dst);
			seq::gauss_inner_left_right(src, dst);
			seq::inner_gauss(src, dst);
		}


		void blur(gray::view_t const& src, gray::view_t const& dst)
		{
			assert(verify(src, dst));
			assert(src.width >= VIEW_MIN_DIM);
			assert(src.height >= VIEW_MIN_DIM);

			seq::copy_top_bottom(src, dst);
			seq::copy_left_right(src, dst);
			seq::gauss_inner_top_bottom(src, dst);
			seq::gauss_inner_left_right(src, dst);
			seq::inner_gauss(src, dst);
		}
	}

#ifndef LIBIMAGE_NO_SIMD

	namespace simd
	{
		void blur(gray::image_t const& src, gray::image_t const& dst)
		{
			assert(verify(src, dst));
			assert(src.width >= VIEW_MIN_DIM);
			assert(src.height >= VIEW_MIN_DIM);
			
			seq::copy_top_bottom(src, dst);
			seq::copy_left_right(src, dst);
			seq::gauss_inner_top_bottom(src, dst);
			seq::gauss_inner_left_right(src, dst);
			simd::inner_gauss(src, dst);
		}


		void blur(gray::image_t const& src, gray::view_t const& dst)
		{
			assert(verify(src, dst));
			assert(src.width >= VIEW_MIN_DIM);
			assert(src.height >= VIEW_MIN_DIM);

			seq::copy_top_bottom(src, dst);
			seq::copy_left_right(src, dst);
			seq::gauss_inner_top_bottom(src, dst);
			seq::gauss_inner_left_right(src, dst);
			simd::inner_gauss(src, dst);
		}


		void blur(gray::view_t const& src, gray::image_t const& dst)
		{
			assert(verify(src, dst));
			assert(src.width >= VIEW_MIN_DIM);
			assert(src.height >= VIEW_MIN_DIM);

			seq::copy_top_bottom(src, dst);
			seq::copy_left_right(src, dst);
			seq::gauss_inner_top_bottom(src, dst);
			seq::gauss_inner_left_right(src, dst);
			simd::inner_gauss(src, dst);
		}


		void blur(gray::view_t const& src, gray::view_t const& dst)
		{
			assert(verify(src, dst));
			assert(src.width >= VIEW_MIN_DIM);
			assert(src.height >= VIEW_MIN_DIM);

			seq::copy_top_bottom(src, dst);
			seq::copy_left_right(src, dst);
			seq::gauss_inner_top_bottom(src, dst);
			seq::gauss_inner_left_right(src, dst);
			simd::inner_gauss(src, dst);
		}
	}

#endif // !LIBIMAGE_NO_SIMD

}

#endif // !LIBIMAGE_NO_GRAYSCALE