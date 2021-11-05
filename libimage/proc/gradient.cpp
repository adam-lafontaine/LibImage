/*

Copyright (c) 2021 Adam Lafontaine

*/

#ifndef LIBIMAGE_NO_GRAYSCALE

#include "process.hpp"
#include "verify.hpp"
#include "index_range.hpp"
#include "convolve.hpp"

#include <algorithm>
#include <cmath>

#ifndef LIBIMAGE_NO_PARALLEL
#include <execution>
#endif // !LIBIMAGE_NO_PARALLEL

namespace libimage
{
#ifndef LIBIMAGE_NO_PARALLEL


	template<class GRAY_IMG_T>
	static void fill_zero(GRAY_IMG_T const& view)
	{
		std::fill(std::execution::par, view.begin(), view.end(), 0);
	}


	template<class GRAY_IMG_T>
	static void zero_top(GRAY_IMG_T const& dst)
	{
		auto dst_top = row_view(dst, 0);

		fill_zero(dst_top);
	}


	template<class GRAY_IMG_T>
	static void zero_bottom(GRAY_IMG_T const& dst)
	{
		auto dst_bottom = row_view(dst, dst.height - 1);

		fill_zero(dst_bottom);
	}


	template<class GRAY_IMG_T>
	static void zero_left(GRAY_IMG_T const& dst)
	{
		pixel_range_t r = {};
		r.x_begin = 0;
		r.x_end = 1;
		r.y_begin = 1;
		r.y_end = dst.height - 1;
		auto dst_left = sub_view(dst, r);

		fill_zero(dst_left);
	}


	template<class GRAY_IMG_T>
	static void zero_right(GRAY_IMG_T const& dst)
	{
		pixel_range_t r = {};
		r.x_begin = dst.width - 1;
		r.x_end = dst.width;
		r.y_begin = 1;
		r.y_end = dst.height - 1;
		auto dst_right = sub_view(dst, r);

		fill_zero(dst_right);
	}


	template<class GRAY_SRC_IMG_T, class GRAY_DST_IMG_T>
	static void edges_inner(GRAY_SRC_IMG_T const& src, GRAY_DST_IMG_T const& dst, u8_to_bool_f const& cond)
	{
		u32 const x_begin = 1;
		u32 const x_end = src.width - 1;
		u32_range_t x_ids(x_begin, x_end);

		u32 const y_begin = 1;
		u32 const y_end = src.height - 1;
		u32_range_t y_ids(y_begin, y_end);

		auto const grad_row = [&](u32 y)
		{
			auto dst_row = dst.row_begin(y);

			auto const grad_x = [&](u32 x)
			{
				auto gx = x_gradient(src, x, y);
				auto gy = y_gradient(src, x, y);
				auto g = static_cast<u8>(std::hypot(gx, gy));
				dst_row[x] = cond(g) ? 255 : 0;
			};

			std::for_each(std::execution::par, x_ids.begin(), x_ids.end(), grad_x);
		};

		std::for_each(std::execution::par, y_ids.begin(), y_ids.end(), grad_row);
	}


	template<class GRAY_SRC_IMG_T, class GRAY_DST_IMG_T>
	static void do_edges(GRAY_SRC_IMG_T const& src, GRAY_DST_IMG_T const& dst, u8_to_bool_f const& cond)
	{
		std::array<std::function<void()>, 5> f_list
		{
			[&]() { zero_top(dst); },
			[&]() { zero_bottom(dst); },
			[&]() { zero_left(dst); },
			[&]() { zero_right(dst); },
			[&]() { edges_inner(src, dst, cond); }
		};

		std::for_each(std::execution::par, f_list.begin(), f_list.end(), [](auto const& f) { f(); });
	}


	template<class GRAY_SRC_IMG_T, class GRAY_DST_IMG_T>
	static void gradients_inner(GRAY_SRC_IMG_T const& src, GRAY_DST_IMG_T const& dst)
	{
		u32 const x_begin = 1;
		u32 const x_end = src.width - 1;
		u32_range_t x_ids(x_begin, x_end);

		u32 const y_begin = 1;
		u32 const y_end = src.height - 1;
		u32_range_t y_ids(y_begin, y_end);

		auto const grad_row = [&](u32 y)
		{
			auto dst_row = dst.row_begin(y);

			auto const grad_x = [&](u32 x)
			{
				auto gx = x_gradient(src, x, y);
				auto gy = y_gradient(src, x, y);
				auto g = std::hypot(gx, gy);
				dst_row[x] = static_cast<u8>(g);
			};

			std::for_each(std::execution::par, x_ids.begin(), x_ids.end(), grad_x);
		};

		std::for_each(std::execution::par, y_ids.begin(), y_ids.end(), grad_row);
	}


	template<class GRAY_SRC_IMG_T, class GRAY_DST_IMG_T>
	static void do_gradients(GRAY_SRC_IMG_T const& src, GRAY_DST_IMG_T const& dst)
	{
		std::array<std::function<void()>, 5> f_list
		{
			[&]() { zero_top(dst); },
			[&]() { zero_bottom(dst); },
			[&]() { zero_left(dst); },
			[&]() { zero_right(dst); },
			[&]() { gradients_inner(src, dst); }
		};

		std::for_each(std::execution::par, f_list.begin(), f_list.end(), [](auto const& f) { f(); });
	}


	void edges(gray::image_t const& src, gray::image_t const& dst, gray::image_t const& temp, u8_to_bool_f const& cond)
	{
		assert(verify(src, dst));
		assert(verify(src, temp));

		blur(src, temp);

		do_edges(temp, dst, cond);
	}


	void edges(gray::image_t const& src, gray::view_t const& dst, gray::image_t const& temp, u8_to_bool_f const& cond)
	{
		assert(verify(src, dst));
		assert(verify(src, temp));

		blur(src, temp);

		do_edges(temp, dst, cond);
	}


	void edges(gray::view_t const& src, gray::image_t const& dst, gray::image_t const& temp, u8_to_bool_f const& cond)
	{
		assert(verify(src, dst));
		assert(verify(src, temp));

		blur(src, temp);

		do_edges(temp, dst, cond);
	}


	void edges(gray::view_t const& src, gray::view_t const& dst, gray::image_t const& temp, u8_to_bool_f const& cond)
	{
		assert(verify(src, dst));
		assert(verify(src, temp));

		blur(src, temp);

		do_edges(temp, dst, cond);
	}


	void edges(gray::image_t const& src, gray::image_t const& dst, u8_to_bool_f const& cond)
	{
		assert(verify(src, dst));

		// blur the image first
		gray::image_t temp;
		make_image(temp, src.width, src.height);
		blur(src, temp);

		do_edges(temp, dst, cond);
	}


	void edges(gray::image_t const& src, gray::view_t const& dst, u8_to_bool_f const& cond)
	{
		assert(verify(src, dst));

		// blur the image first
		gray::image_t temp;
		make_image(temp, src.width, src.height);
		blur(src, temp);

		do_edges(temp, dst, cond);
	}


	void edges(gray::view_t const& src, gray::image_t const& dst, u8_to_bool_f const& cond)
	{
		assert(verify(src, dst));

		// blur the image first
		gray::image_t temp;
		make_image(temp, src.width, src.height);
		blur(src, temp);

		do_edges(temp, dst, cond);
	}


	void edges(gray::view_t const& src, gray::view_t const& dst, u8_to_bool_f const& cond)
	{
		assert(verify(src, dst));

		// blur the image first
		gray::image_t temp;
		make_image(temp, src.width, src.height);
		blur(src, temp);

		do_edges(temp, dst, cond);
	}


	void gradients(gray::image_t const& src, gray::image_t const& dst, gray::image_t const& temp)
	{
		assert(verify(src, dst));
		assert(verify(src, temp));

		blur(src, temp);

		do_gradients(temp, dst);
	}


	void gradients(gray::image_t const& src, gray::view_t const& dst, gray::image_t const& temp)
	{
		assert(verify(src, dst));
		assert(verify(src, temp));

		blur(src, temp);

		do_gradients(temp, dst);
	}


	void gradients(gray::view_t const& src, gray::image_t const& dst, gray::image_t const& temp)
	{
		assert(verify(src, dst));
		assert(verify(src, temp));

		blur(src, temp);

		do_gradients(temp, dst);
	}


	void gradients(gray::view_t const& src, gray::view_t const& dst, gray::image_t const& temp)
	{
		assert(verify(src, dst));
		assert(verify(src, temp));

		blur(src, temp);

		do_gradients(temp, dst);
	}


	void gradients(gray::image_t const& src, gray::image_t const& dst)
	{
		assert(verify(src, dst));

		// blur the image first
		gray::image_t temp;
		make_image(temp, src.width, src.height);
		blur(src, temp);

		do_gradients(temp, dst);
	}


	void gradients(gray::image_t const& src, gray::view_t const& dst)
	{
		assert(verify(src, dst));

		// blur the image first
		gray::image_t temp;
		make_image(temp, src.width, src.height);
		blur(src, temp);

		do_gradients(temp, dst);
	}


	void gradients(gray::view_t const& src, gray::image_t const& dst)
	{
		assert(verify(src, dst));

		// blur the image first
		gray::image_t temp;
		make_image(temp, src.width, src.height);
		blur(src, temp);

		do_gradients(temp, dst);
	}


	void gradients(gray::view_t const& src, gray::view_t const& dst)
	{
		assert(verify(src, dst));

		// blur the image first
		gray::image_t temp;
		make_image(temp, src.width, src.height);
		blur(src, temp);

		do_gradients(temp, dst);
	}


#endif // !LIBIMAGE_NO_PARALLEL
	namespace seq
	{
		template<class GRAY_IMG_T>
		static void zero_top_bottom(GRAY_IMG_T const& dst)
		{
			u32 x_first = 0;
			u32 y_first = 0;
			u32 x_last = dst.width - 1;
			u32 y_last = dst.height - 1;
			auto dst_top = dst.row_begin(y_first);
			auto dst_bottom = dst.row_begin(y_last);
			for (u32 x = x_first; x <= x_last; ++x)
			{
				dst_top[x] = 0;
				dst_bottom[x] = 0;
			}
		}


		template<class GRAY_IMG_T>
		static void zero_left_right(GRAY_IMG_T const& dst)
		{
			u32 x_first = 0;
			u32 y_first = 1;
			u32 x_last = dst.width - 1;
			u32 y_last = dst.height - 2;
			for (u32 y = y_first; y <= y_last; ++y)
			{
				auto dst_row = dst.row_begin(y);
				dst_row[x_first] = 0;
				dst_row[x_last] = 0;
			}
		}


		template<class GRAY_SRC_IMG_T, class GRAY_DST_IMG_T>
		static void edges_inner(GRAY_SRC_IMG_T const& src, GRAY_DST_IMG_T const& dst, u8_to_bool_f const& cond)
		{
			u32 x_first = 1;
			u32 y_first = 1;
			u32 x_last = dst.width - 2;
			u32 y_last = dst.height - 2;
			for (u32 y = y_first; y <= y_last; ++y)
			{
				auto dst_row = dst.row_begin(y);
				for (u32 x = x_first; x <= x_last; ++x)
				{
					auto gx = x_gradient(src, x, y);
					auto gy = y_gradient(src, x, y);
					auto g = static_cast<u8>(std::hypot(gx, gy));
					dst_row[x] = cond(g) ? 255 : 0;
				}
			}
		}


		template<class GRAY_SRC_IMG_T, class GRAY_DST_IMG_T>
		static void do_edges(GRAY_SRC_IMG_T const& src, GRAY_DST_IMG_T const& dst, u8_to_bool_f const& cond)
		{
			zero_top_bottom(dst);
			zero_left_right(dst);

			seq::edges_inner(src, dst, cond);
		}


		template<class GRAY_SRC_IMG_T, class GRAY_DST_IMG_T>
		static void gradients_inner(GRAY_SRC_IMG_T const& src, GRAY_DST_IMG_T const& dst)
		{
			u32 x_first = 1;
			u32 y_first = 1;
			u32 x_last = dst.width - 2;
			u32 y_last = dst.height - 2;
			for (u32 y = y_first; y <= y_last; ++y)
			{
				auto dst_row = dst.row_begin(y);
				for (u32 x = x_first; x <= x_last; ++x)
				{
					auto gx = x_gradient(src, x, y);
					auto gy = y_gradient(src, x, y);
					auto g = std::hypot(gx, gy);
					dst_row[x] = static_cast<u8>(g);
				}
			}
		}


		template<class GRAY_SRC_IMG_T, class GRAY_DST_IMG_T>
		static void do_gradients(GRAY_SRC_IMG_T const& src, GRAY_DST_IMG_T const& dst)
		{
			seq::zero_top_bottom(dst);
			seq::zero_left_right(dst);

			seq::gradients_inner(src, dst);
		}


		void edges(gray::image_t const& src, gray::image_t const& dst, u8_to_bool_f const& cond)
		{
			assert(verify(src, dst));

			gray::image_t temp;
			make_image(temp, src.width, src.height);
			seq::blur(src, temp);

			seq::do_edges(temp, dst, cond);
		}


		void edges(gray::image_t const& src, gray::view_t const& dst, u8_to_bool_f const& cond)
		{
			assert(verify(src, dst));

			gray::image_t temp;
			make_image(temp, src.width, src.height);
			seq::blur(src, temp);

			seq::do_edges(temp, dst, cond);
		}


		void edges(gray::view_t const& src, gray::image_t const& dst, u8_to_bool_f const& cond)
		{
			assert(verify(src, dst));

			gray::image_t temp;
			make_image(temp, src.width, src.height);
			seq::blur(src, temp);

			seq::do_edges(temp, dst, cond);
		}


		void edges(gray::view_t const& src, gray::view_t const& dst, u8_to_bool_f const& cond)
		{
			assert(verify(src, dst));

			gray::image_t temp;
			make_image(temp, src.width, src.height);
			seq::blur(src, temp);

			seq::do_edges(temp, dst, cond);
		}


		void edges(gray::image_t const& src, gray::image_t const& dst, gray::image_t const& temp, u8_to_bool_f const& cond)
		{
			assert(verify(src, dst));
			assert(verify(src, temp));

			seq::blur(src, temp);

			seq::do_edges(temp, dst, cond);
		}


		void edges(gray::image_t const& src, gray::view_t const& dst, gray::image_t const& temp, u8_to_bool_f const& cond)
		{
			assert(verify(src, dst));
			assert(verify(src, temp));

			seq::blur(src, temp);

			seq::do_edges(temp, dst, cond);
		}


		void edges(gray::view_t const& src, gray::image_t const& dst, gray::image_t const& temp, u8_to_bool_f const& cond)
		{
			assert(verify(src, dst));
			assert(verify(src, temp));

			seq::blur(src, temp);

			seq::do_edges(temp, dst, cond);
		}


		void edges(gray::view_t const& src, gray::view_t const& dst, gray::image_t const& temp, u8_to_bool_f const& cond)
		{
			assert(verify(src, dst));
			assert(verify(src, temp));

			seq::blur(src, temp);

			seq::do_edges(temp, dst, cond);
		}


		void gradients(gray::image_t const& src, gray::image_t const& dst)
		{
			assert(verify(src, dst));

			gray::image_t temp;
			make_image(temp, src.width, src.height);
			seq::blur(src, temp);

			seq::do_gradients(temp, dst);
		}


		void gradients(gray::image_t const& src, gray::view_t const& dst)
		{
			assert(verify(src, dst));

			gray::image_t temp;
			make_image(temp, src.width, src.height);
			seq::blur(src, temp);

			seq::do_gradients(temp, dst);
		}


		void gradients(gray::view_t const& src, gray::image_t const& dst)
		{
			assert(verify(src, dst));

			gray::image_t temp;
			make_image(temp, src.width, src.height);
			seq::blur(src, temp);

			seq::do_gradients(temp, dst);
		}


		void gradients(gray::view_t const& src, gray::view_t const& dst)
		{
			assert(verify(src, dst));

			gray::image_t temp;
			make_image(temp, src.width, src.height);
			seq::blur(src, temp);

			seq::do_gradients(temp, dst);
		}


		void gradients(gray::image_t const& src, gray::image_t const& dst, gray::image_t const& temp)
		{
			assert(verify(src, dst));
			assert(verify(src, temp));

			seq::blur(src, temp);

			seq::do_gradients(temp, dst);
		}


		void gradients(gray::image_t const& src, gray::view_t const& dst, gray::image_t const& temp)
		{
			assert(verify(src, dst));
			assert(verify(src, temp));

			seq::blur(src, temp);

			seq::do_gradients(temp, dst);
		}


		void gradients(gray::view_t const& src, gray::image_t const& dst, gray::image_t const& temp)
		{
			assert(verify(src, dst));
			assert(verify(src, temp));

			seq::blur(src, temp);

			seq::do_gradients(temp, dst);
		}


		void gradients(gray::view_t const& src, gray::view_t const& dst, gray::image_t const& temp)
		{
			assert(verify(src, dst));
			assert(verify(src, temp));

			seq::blur(src, temp);

			seq::do_gradients(temp, dst);
		}
	}


#ifndef LIBIMAGE_NO_SIMD

	namespace simd
	{

		template<class GRAY_SRC_IMG_T, class GRAY_DST_IMG_T>
		static void do_edges(GRAY_SRC_IMG_T const& src, GRAY_DST_IMG_T const& dst, u8_to_bool_f const& cond)
		{
			seq::zero_top_bottom(dst);
			seq::zero_left_right(dst);

			simd::inner_edges(src, dst, cond);
		}


		template<class GRAY_SRC_IMG_T, class GRAY_DST_IMG_T>
		static void do_gradients(GRAY_SRC_IMG_T const& src, GRAY_DST_IMG_T const& dst)
		{
			seq::zero_top_bottom(dst);
			seq::zero_left_right(dst);

			simd::inner_gradients(src, dst);
		}


		void edges(gray::image_t const& src, gray::image_t const& dst, gray::image_t const& temp, u8_to_bool_f const& cond)
		{
			assert(verify(src, dst));
			assert(verify(src, temp));

			simd::blur(src, temp);
			simd::do_edges(temp, dst, cond);
		}


		void edges(gray::image_t const& src, gray::view_t const& dst, gray::image_t const& temp, u8_to_bool_f const& cond)
		{
			assert(verify(src, dst));
			assert(verify(src, temp));

			simd::blur(src, temp);
			simd::do_edges(temp, dst, cond);
		}


		void edges(gray::view_t const& src, gray::image_t const& dst, gray::image_t const& temp, u8_to_bool_f const& cond)
		{
			assert(verify(src, dst));
			assert(verify(src, temp));

			simd::blur(src, temp);
			simd::do_edges(temp, dst, cond);
		}


		void edges(gray::view_t const& src, gray::view_t const& dst, gray::image_t const& temp, u8_to_bool_f const& cond)
		{
			assert(verify(src, dst));
			assert(verify(src, temp));

			simd::blur(src, temp);
			simd::do_edges(temp, dst, cond);
		}


		void edges(gray::image_t const& src, gray::image_t const& dst, u8_to_bool_f const& cond)
		{
			assert(verify(src, dst));

			gray::image_t temp;
			make_image(temp, src.width, src.height);
			simd::blur(src, temp);

			simd::do_edges(temp, dst, cond);
		}


		void edges(gray::image_t const& src, gray::view_t const& dst, u8_to_bool_f const& cond)
		{
			assert(verify(src, dst));

			gray::image_t temp;
			make_image(temp, src.width, src.height);
			simd::blur(src, temp);

			simd::do_edges(temp, dst, cond);
		}


		void edges(gray::view_t const& src, gray::image_t const& dst, u8_to_bool_f const& cond)
		{
			assert(verify(src, dst));

			gray::image_t temp;
			make_image(temp, src.width, src.height);
			simd::blur(src, temp);

			simd::do_edges(temp, dst, cond);
		}


		void edges(gray::view_t const& src, gray::view_t const& dst, u8_to_bool_f const& cond)
		{
			assert(verify(src, dst));

			gray::image_t temp;
			make_image(temp, src.width, src.height);
			simd::blur(src, temp);

			simd::do_edges(temp, dst, cond);
		}


		void gradients(gray::image_t const& src, gray::image_t const& dst, gray::image_t const& temp)
		{
			assert(verify(src, dst));
			assert(verify(src, temp));

			simd::blur(src, temp);
			simd::do_gradients(temp, dst);
		}


		void gradients(gray::image_t const& src, gray::view_t const& dst, gray::image_t const& temp)
		{
			assert(verify(src, dst));
			assert(verify(src, temp));

			simd::blur(src, temp);
			simd::do_gradients(temp, dst);
		}


		void gradients(gray::view_t const& src, gray::image_t const& dst, gray::image_t const& temp)
		{
			assert(verify(src, dst));
			assert(verify(src, temp));

			simd::blur(src, temp);
			simd::do_gradients(temp, dst);
		}



		void gradients(gray::view_t const& src, gray::view_t const& dst, gray::image_t const& temp)
		{
			assert(verify(src, dst));
			assert(verify(src, temp));

			simd::blur(src, temp);
			simd::do_gradients(temp, dst);
		}


		void gradients(gray::image_t const& src, gray::image_t const& dst)
		{
			assert(verify(src, dst));

			gray::image_t temp;
			make_image(temp, src.width, src.height);
			simd::blur(src, temp);

			simd::do_gradients(temp, dst);
		}


		void gradients(gray::image_t const& src, gray::view_t const& dst)
		{
			assert(verify(src, dst));

			gray::image_t temp;
			make_image(temp, src.width, src.height);
			simd::blur(src, temp);

			simd::do_gradients(temp, dst);
		}


		void gradients(gray::view_t const& src, gray::image_t const& dst)
		{
			assert(verify(src, dst));

			gray::image_t temp;
			make_image(temp, src.width, src.height);
			simd::blur(src, temp);

			simd::do_gradients(temp, dst);
		}


		void gradients(gray::view_t const& src, gray::view_t const& dst)
		{
			assert(verify(src, dst));

			gray::image_t temp;
			make_image(temp, src.width, src.height);
			simd::blur(src, temp);

			simd::do_gradients(temp, dst);
		}

	}

#endif // !LIBIMAGE_NO_SIMD

}


#endif // !LIBIMAGE_NO_GRAYSCALE