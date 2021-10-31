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
			for (u32 x = x_first; x <= x_last; ++x) // top and bottom rows
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
		static void inner_gauss_old(GRAY_SRC_IMG_T const& src, GRAY_DST_IMG_T const& dst)
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


#include <xmmintrin.h>

		typedef union
		{
			__m128 quad;
			r32 lanes[4];

		} quad_r32_t;


		void set_zero(quad_r32_t& q)
		{
			q.quad = _mm_setzero_ps();
		}


		quad_r32_t make_quad_r32()
		{
			quad_r32_t quad;
			quad.quad = _mm_setzero_ps();

			return quad;
		}


		void load_lanes(quad_r32_t& q, r32* begin)
		{
			q.lanes[0] = begin[0];
			q.lanes[1] = begin[1];
			q.lanes[2] = begin[2];
			q.lanes[3] = begin[3];
		}


		

		constexpr r32 D5 = 256.0f;
		constexpr std::array<r32, 25> GAUSS_5X5
		{
			(1 / D5), (4 / D5),  (6 / D5),  (4 / D5),  (1 / D5),
			(4 / D5), (16 / D5), (24 / D5), (16 / D5), (4 / D5),
			(6 / D5), (24 / D5), (36 / D5), (24 / D5), (6 / D5),
			(4 / D5), (16 / D5), (24 / D5), (16 / D5), (4 / D5),
			(1 / D5), (4 / D5),  (6 / D5),  (4 / D5),  (1 / D5),
		};


		static void gauss5(u8* src_begin, u8* dst_begin, u32 length, u32 pitch)
		{
			r32 vals_r32[4];

			auto const do_simd = [&](int i) 
			{
				u32 w = 0;
				auto acc = _mm_setzero_ps();

				for (int gy = -2; gy < 3; ++gy)
				{
					for (int gx = -2; gx < 3; ++gx)
					{
						int offset = gy * pitch + gx;
						auto weight = _mm_load1_ps(GAUSS_5X5.data() + w);

						vals_r32[0] = src_begin[i];
						vals_r32[1] = src_begin[i + 1];
						vals_r32[2] = src_begin[i + 2];
						vals_r32[3] = src_begin[i + 3];

						auto src_val = _mm_load_ps(vals_r32);

						acc = _mm_add_ps(acc, _mm_mul_ps(weight, src_val));

						_mm_store_ps(vals_r32, acc);

						dst_begin[i] = static_cast<u8>(vals_r32[0]);
						dst_begin[i + 1] = static_cast<u8>(vals_r32[1]);
						dst_begin[i + 2] = static_cast<u8>(vals_r32[2]);
						dst_begin[i + 3] = static_cast<u8>(vals_r32[3]);

						++w;
					}
				}
			};

			for (int i = 0; i < length; i += 4)
			{
				do_simd(i);
			}

			do_simd(length - 4);


		}





		template<class GRAY_SRC_IMG_T, class GRAY_DST_IMG_T>
		static void inner_gauss(GRAY_SRC_IMG_T const& src, GRAY_DST_IMG_T const& dst)
		{
			u32 x_first = 2;
			u32 y_first = 2;
			u32 x_last = src.width - 3;
			u32 y_last = src.height - 3;

			auto length = x_last - x_first + 1;
			auto pitch = static_cast<u32>(src.row_begin(1) - src.row_begin(0));

			for (u32 y = y_first; y <= y_last; ++y)
			{
				gauss5(src.row_begin(y), dst.row_begin(y), length, pitch);

				/*auto dst_row = dst.row_begin(y);

				for (u32 x = x_first; x <= x_last; x += 1)
				{
					u32 w = 0;
					r32 total = 0.0f;

					for (u32 gy = y - 2; gy < y + 3; ++gy)
					{
						for (u32 gx = x - 2; gx < x + 3; ++gx)
						{
							auto p = *src.xy_at(gx, gy);
							total += GAUSS_5X5[w++] * p;
						}
					}

					dst_row[x] = static_cast<u8>(total);
				}*/
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
}

#endif // !LIBIMAGE_NO_GRAYSCALE