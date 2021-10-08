/*

Copyright (c) 2021 Adam Lafontaine

*/

#ifndef LIBIMAGE_NO_GRAYSCALE

#include "process.hpp"
#include "verify.hpp"
#include "index_range.hpp"
#include "convolve.hpp"

#include <algorithm>
#include <execution>

namespace libimage
{
	void edges(gray::view_t const& src, gray::view_t const& dst, u8 threshold)
	{
		assert(verify(src, dst));

		auto const width = src.width;
		auto const height = src.height;

		// blur the image first
		gray::image_t temp;
		auto temp_view = make_view(temp, width, height);
		blur(src, temp_view);

		auto const zero = [](u8 p) { return static_cast<u8>(0); };

		auto const zero_top = [&]()
		{
			auto dst_top = row_view(dst, 0);

			transform_self(dst_top, zero);
		};

		auto const zero_bottom = [&]()
		{
			auto dst_bottom = row_view(dst, height - 1);

			transform_self(dst_bottom, zero);
		};

		auto const zero_left = [&]()
		{
			pixel_range_t r = {};
			r.x_begin = 0;
			r.x_end = 1;
			r.y_begin = 1;
			r.y_end = height - 1;
			auto dst_left = sub_view(dst, r);

			transform_self(dst_left, zero);
		};

		auto const zero_right = [&]()
		{
			pixel_range_t r = {};
			r.x_begin = width - 1;
			r.x_end = width;
			r.y_begin = 1;
			r.y_end = height - 1;
			auto dst_right = sub_view(dst, r);

			transform_self(dst_right, zero);
		};

		// get gradient magnitude of inner pixels
		u32 const x_begin = 1;
		u32 const x_end = width - 1;
		u32_range_t x_ids(x_begin, x_end);

		u32 const y_begin = 1;
		u32 const y_end = height - 1;
		u32_range_t y_ids(y_begin, y_end);

		auto const grad_row = [&](u32 y)
		{
			auto dst_row = dst.row_begin(y);

			auto const grad_x = [&](u32 x)
			{
				auto gx = x_gradient(temp_view, x, y);
				auto gy = y_gradient(temp_view, x, y);
				auto g = std::hypot(gx, gy);
				dst_row[x] = g < threshold ? 0 : 255;
			};

			std::for_each(std::execution::par, x_ids.begin(), x_ids.end(), grad_x);
		};

		auto const gradients_inner = [&]()
		{
			std::for_each(std::execution::par, y_ids.begin(), y_ids.end(), grad_row);
		};

		// put the lambdas in an array
		std::array<std::function<void()>, 5> f_list
		{
			zero_top,
			zero_bottom,
			zero_left,
			zero_right,
			gradients_inner
		};

		// finally execute everything
		std::for_each(std::execution::par, f_list.begin(), f_list.end(), [](auto const& f) { f(); });
	}


	void gradient(gray::view_t const& src, gray::view_t const& dst)
	{
		assert(verify(src, dst));

		auto const width = src.width;
		auto const height = src.height;

		// blur the image first
		gray::image_t temp;
		auto temp_view = make_view(temp, width, height);
		blur(src, temp_view);

		auto const zero = [](u8 p) { return static_cast<u8>(0); };

		auto const zero_top = [&]()
		{
			auto dst_top = row_view(dst, 0);

			transform_self(dst_top, zero);
		};

		auto const zero_bottom = [&]()
		{
			auto dst_bottom = row_view(dst, height - 1);

			transform_self(dst_bottom, zero);
		};

		auto const zero_left = [&]()
		{
			pixel_range_t r = {};
			r.x_begin = 0;
			r.x_end = 1;
			r.y_begin = 1;
			r.y_end = height - 1;
			auto dst_left = sub_view(dst, r);

			transform_self(dst_left, zero);
		};

		auto const zero_right = [&]()
		{
			pixel_range_t r = {};
			r.x_begin = width - 1;
			r.x_end = width;
			r.y_begin = 1;
			r.y_end = height - 1;
			auto dst_right = sub_view(dst, r);

			transform_self(dst_right, zero);
		};

		// get gradient magnitude of inner pixels
		u32 const x_begin = 1;
		u32 const x_end = width - 1;
		u32_range_t x_ids(x_begin, x_end);

		u32 const y_begin = 1;
		u32 const y_end = height - 1;
		u32_range_t y_ids(y_begin, y_end);

		auto const grad_row = [&](u32 y)
		{
			auto dst_row = dst.row_begin(y);

			auto const grad_x = [&](u32 x)
			{
				auto gx = x_gradient(temp_view, x, y);
				auto gy = y_gradient(temp_view, x, y);
				auto g = std::hypot(gx, gy);
				dst_row[x] = static_cast<u8>(g);
			};

			std::for_each(std::execution::par, x_ids.begin(), x_ids.end(), grad_x);
		};

		auto const gradients_inner = [&]()
		{
			std::for_each(std::execution::par, y_ids.begin(), y_ids.end(), grad_row);
		};

		// put the lambdas in an array
		std::array<std::function<void()>, 5> f_list
		{
			zero_top,
			zero_bottom,
			zero_left,
			zero_right,
			gradients_inner
		};

		// finally execute everything
		std::for_each(std::execution::par, f_list.begin(), f_list.end(), [](auto const& f) { f(); });
	}


	namespace seq
	{
		void edges(gray::view_t const& src, gray::view_t const& dst, u8 threshold)
		{
			assert(verify(src, dst));

			auto const width = src.width;
			auto const height = src.height;

			gray::image_t temp;
			auto temp_view = make_view(temp, src.width, src.height);
			blur(src, temp_view);

			// top and bottom rows are black
			u32 x_first = 0;
			u32 y_first = 0;
			u32 x_last = width - 1;
			u32 y_last = height - 1;
			auto dst_top = dst.row_begin(y_first);
			auto dst_bottom = dst.row_begin(y_last);
			for (u32 x = x_first; x <= x_last; ++x)
			{
				dst_top[x] = 0;
				dst_bottom[x] = 0;
			}

			// left and right columns are black
			x_first = 0;
			y_first = 1;
			x_last = width - 1;
			y_last = height - 2;
			for (u32 y = y_first; y <= y_last; ++y)
			{
				auto dst_row = dst.row_begin(y);
				dst_row[x_first] = 0;
				dst_row[x_last] = 0;
			}

			// get gradient magnitude of inner pixels
			x_first = 1;
			y_first = 1;
			x_last = width - 2;
			y_last = height - 2;
			for (u32 y = y_first; y <= y_last; ++y)
			{
				auto dst_row = dst.row_begin(y);
				for (u32 x = x_first; x <= x_last; ++x)
				{
					auto gx = x_gradient(temp_view, x, y);
					auto gy = y_gradient(temp_view, x, y);
					auto g = std::hypot(gx, gy);
					dst_row[x] = g < threshold ? 0 : 255;
				}
			}
		}
	}
}


#endif // !LIBIMAGE_NO_GRAYSCALE