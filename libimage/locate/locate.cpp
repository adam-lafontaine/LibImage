#include "locate.hpp"
#include "../proc/process.hpp"
#include "../proc/index_range.hpp"
#include "../proc/convolve.hpp"

#include <cassert>
#include <algorithm>
#include <execution>
#include <functional>
#include <array>

namespace libimage
{
	static void q_edges(gray::view_t const& src, gray::view_t const& dst, u8 threshold)
	{
		u32 const src_w = src.width;
		u32 const src_h = src.height;
		u32 const dst_w = dst.width;
		u32 const dst_h = dst.height;

		// outer edge pixels of src are ignored
		assert(src_w - dst_w == 2u);
		assert(src_h - dst_h == 2u);
		constexpr auto map_src = [](u32 i) { return i + 1; };

		// get gradient magnitude of inner pixels
		u32 const dst_x_begin = 0;
		u32 const dst_x_end = src_w;
		u32_range_t dst_x_ids(dst_x_begin, dst_x_end);

		u32 const dst_y_begin = 0;
		u32 const dst_y_end = dst_h;
		u32_range_t dst_y_ids(dst_y_begin, dst_y_end);		

		auto const grad_row = [&](u32 dst_y)
		{
			auto dst_row = dst.row_begin(dst_y);

			auto const grad_x = [&](u32 dst_x)
			{
				auto src_x = map_src(dst_x);
				auto src_y = map_src(dst_y);
				auto gx = x_gradient(src, src_x, src_y);
				auto gy = y_gradient(src, src_x, src_y);
				auto g = std::hypot(gx, gy);
				dst_row[dst_x] = g < threshold ? 0 : 255;
			};

			std::for_each(std::execution::par, dst_x_ids.begin(), dst_x_ids.end(), grad_x);
		};

		std::for_each(std::execution::par, dst_y_ids.begin(), dst_y_ids.end(), grad_row);
	}



}