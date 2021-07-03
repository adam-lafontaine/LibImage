#include "locate.hpp"
#include "../proc/process.hpp"
#include "../proc/index_range.hpp"
#include "../proc/convolve.hpp"

#include <cassert>
#include <algorithm>
#include <execution>
#include <functional>
#include <array>
#include <vector>


// outer edge pixels of src are ignored
constexpr u32 DST_OFFSET = 1;
constexpr u32 DELTA_DIM = 2 * DST_OFFSET;
constexpr u32 map_to_src(u32 dst_i) { return dst_i + DST_OFFSET; }

namespace libimage
{
	static void q_edges(gray::view_t const& src, gray::view_t const& dst, u8 threshold)
	{
		u32 const src_w = src.width;
		u32 const src_h = src.height;
		u32 const dst_w = dst.width;
		u32 const dst_h = dst.height;

		// outer edge pixels of src are ignored
		assert(src_w - dst_w == DELTA_DIM);
		assert(src_h - dst_h == DELTA_DIM);

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
				auto src_x = map_to_src(dst_x);
				auto src_y = map_to_src(dst_y);
				auto gx = x_gradient(src, src_x, src_y);
				auto gy = y_gradient(src, src_x, src_y);
				auto g = std::hypot(gx, gy);
				dst_row[dst_x] = g < threshold ? 0 : 255;
			};

			std::for_each(std::execution::par, dst_x_ids.begin(), dst_x_ids.end(), grad_x);
		};

		std::for_each(std::execution::par, dst_y_ids.begin(), dst_y_ids.end(), grad_row);
	}


	static u32 edge_delta(gray::view_t const& lhs, gray::view_t const& rhs)
	{
		assert(lhs.width == rhs.width);
		assert(lhs.height == rhs.height);

		u32 const width = lhs.width;
		u32 const height = rhs.height;

		u32 total = 0;

		for (u32 y = 0; y < height; ++y)
		{
			auto lhs_row = lhs.row_begin(y);
			auto rhs_row = rhs.row_begin(y);
			for (u32 x = 0; x < width; ++x)
			{
				total += (lhs_row[x] != rhs_row[x]);
			}
		}

		return total;
	}


	static gray::view_t trim(gray::view_t const& view)
	{
		u32 const width = view.width;
		u32 const height = view.height;

		pixel_range_t r = { 0, 0, 0, 0 };		

		// search from top
		bool found = false;
		for (u32 y = 0; y < height && !found; ++y)
		{
			auto row = view.row_begin(y);
			for (u32 x = 0; x < width && !found; ++x)
			{
				if (row[x])
				{
					r.y_begin = y;
					found = true;
				}				
			}
		}

		// search from bottom
		found = false;
		for (u32 y = height; y > 0 && !found; --y)
		{
			auto row = view.row_begin(y - 1);
			for (u32 x = width; x > 0 && !found; --x)
			{
				if (row[x - 1])
				{
					r.y_end = y;
					found = true;
				}
			}
		}

		// search from left
		found = false;
		for (u32 x = 0; x < width && !found; ++x)
		{
			for (u32 y = 0; y < height && !found; ++y)
			{
				if (view.row_begin(y)[x])
				{
					r.x_begin = x;
					found = true;
				}
			}
		}

		// search from right
		found = false;
		for (u32 x = width; x > 0 && !found; --x)
		{
			for (u32 y = height; y > 0 && !found; --y)
			{
				if (view.row_begin(y - 1)[x - 1])
				{
					r.x_end = y;
				}
			}
		}

		assert(r.x_end > r.x_begin);
		assert(r.y_end > r.y_begin);

		return sub_view(view, r);
	}


	void locate(gray::view_t const& view, gray::view_t const& pattern)
	{
		u32 const v_width = view.width;
		u32 const v_height = view.height;
		u32 const p_width = pattern.width;
		u32 const p_height = pattern.height;

		assert(p_width < v_width);
		assert(p_height < v_height);

		u32 const edges_width = v_width - 1;
		u32 const edges_height = v_height - 1;

		// TODO: where do these come from?
		u32 const contrast_low = 10;
		u32 const contrast_high = 150;

		gray::image_t contrast;
		gray::view_t contrast_view;
		gray::image_t blur;
		gray::view_t blur_view;
		gray::image_t edges;
		gray::view_t edges_view;

		using func_t = std::function<void()>;
		using func_list_t = std::array<func_t, 3>;

		auto const execute = [](func_t const& f) { f(); };

		// allocate memory on separate threads
		func_list_t allocate
		{
			[&]() { contrast_view = make_view(contrast, v_width, v_height); },
			[&]() { blur_view = make_view(blur, v_width, v_height); },
			[&]() { edges_view = make_view(edges, edges_width, edges_height); }
		};
		std::for_each(std::execution::par, allocate.begin(), allocate.end(), execute);
		
		par::transform_contrast(view, contrast_view, contrast_low, contrast_high);

		par::blur(contrast_view, blur_view);

		// TODO: where does this come from?
		u8 const edge_gradient_threshold = 128;

		q_edges(contrast_view, edges_view, edge_gradient_threshold);

		auto search_view = trim(edges_view);

		// no match found?
		assert(search_view.width > p_width);
		assert(search_view.height > p_height);

		u32 const search_x_begin = 0;
		u32 const search_x_end = search_view.width - p_width;
		u32 const search_y_begin = 0;
		u32 const search_y_end = search_view.height - p_height;

		std::vector<u32> delta_totals((search_y_end - search_x_begin), 0u);
		u32_range_t x_ids(search_x_begin, search_x_end);
		u32_range_t y_ids(search_y_begin, search_y_end);


		auto const row_min_delta = [](u32 y) 
		{
			
		};

		std::for_each(y_ids.begin(), y_ids.end(), row_min_delta);

		// free memory on separe threads
		func_list_t destroy
		{
			[&]() { contrast.clear(); },
			[&]() { blur.clear(); },
			[&]() { edges.clear(); }
		};
		std::for_each(std::execution::par, destroy.begin(), destroy.end(), execute);
	}
}