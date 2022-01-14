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

#ifndef LIBIMAGE_NO_GRAYSCALE


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


#endif // !LIBIMAGE_NO_GRAYSCALE

#endif // !LIBIMAGE_NO_PARALLEL


	namespace seq
	{
#ifndef LIBIMAGE_NO_GRAYSCALE



#endif // !LIBIMAGE_NO_GRAYSCALE
	}



	namespace simd
	{
#ifndef LIBIMAGE_NO_GRAYSCALE




#endif // !LIBIMAGE_NO_GRAYSCALE
	}
}