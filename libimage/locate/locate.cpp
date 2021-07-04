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


// outer pixels of edges src are ignored
constexpr u32 EDGES_OFFSET = 1;
constexpr u32 DELTA_DIM = 2 * EDGES_OFFSET;
constexpr u32 map_to_src(u32 dst_i) { return dst_i + EDGES_OFFSET; }

namespace libimage
{
	using view_list_t = std::vector<gray::view_t>;

	static gray::view_t make_edges_view(gray::image_t& image, u32 src_width, u32 src_height)
	{
		return make_view(image, src_width - EDGES_OFFSET, src_height - EDGES_OFFSET);
	}


	static void q_edges(gray::view_t const& src, gray::view_t const& dst, u8 threshold)
	{
		u32 const src_w = src.width;
		u32 const src_h = src.height;
		u32 const dst_w = dst.width;
		u32 const dst_h = dst.height;

		// outer edge pixels of src are ignored
		assert(src_w - dst_w == DELTA_DIM);
		assert(src_h - dst_h == DELTA_DIM);

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
				if (*view.xy_at(x, y))
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
				if (*view.xy_at(x - 1, y - 1))
				{
					r.x_end = y;
				}
			}
		}

		assert(r.x_end > r.x_begin);
		assert(r.y_end > r.y_begin);

		return sub_view(view, r);
	}


	typedef struct
	{
		u32 contrast_low;
		u32 contrast_high;
		u8 edge_gradient_threshold;

		gray::view_t contrast;
		gray::view_t blur;
		gray::view_t edges;

	} locate_props_t;


	gray::view_t locate_one(gray::view_t const& view, gray::view_t const& pattern, locate_props_t const& props)
	{
		u32 const v_width = view.width;
		u32 const v_height = view.height;
		u32 const p_width = pattern.width;
		u32 const p_height = pattern.height;

		assert(p_width < v_width);
		assert(p_height < v_height);

		auto& contrast_view = props.contrast;
		auto& blur_view = props.blur;
		auto& edges_view = props.edges;
		u32 const contrast_low = props.contrast_low;
		u32 const contrast_high = props.contrast_high;
		u32 const edge_gradient_threshold = props.edge_gradient_threshold;
		
		par::transform_contrast(view, contrast_view, contrast_low, contrast_high);

		par::blur(contrast_view, blur_view);

		q_edges(contrast_view, edges_view, edge_gradient_threshold);

		auto search_view = trim(edges_view);

		// no match found?
		assert(search_view.width > p_width);
		assert(search_view.height > p_height);

		u32 const search_x_begin = 0;
		u32 const search_x_end = search_view.width - p_width;
		u32 const search_y_begin = 0;
		u32 const search_y_end = search_view.height - p_height;

		u32 min_delta = p_width * p_height;
		gray::view_t search_result;
		pixel_range_t r = {};

		auto const check_min_delta = [&](u32 x, u32 y) 
		{			
			r.x_begin = x;
			r.x_end = x + p_width;
			r.y_begin = y;
			r.y_end = y + p_height;

			auto v = sub_view(search_view, r);

			auto delta = edge_delta(v, pattern);

			if (delta < min_delta)
			{
				min_delta = delta;
				search_result = v;
			}
		};

		for_each_xy(search_view, check_min_delta);

		return search_result;
	}


	static bool verify(view_list_t const& views)
	{
		auto w = std::all_of(std::execution::par, views.begin(), views.end(), [&](auto const& v) { return v.width == views[0].width; });
		auto h = std::all_of(std::execution::par, views.begin(), views.end(), [&](auto const& v) { return v.height == views[0].height; });

		return w && h;
	}


	view_list_t locate_many(view_list_t const& views, gray::view_t const& pattern)
	{
		assert(views.size());

		auto& v_first = views[0];
		assert(verify(views));

		u32 const v_width = v_first.width;
		u32 const v_height = v_first.height;
		u32 const v_size = static_cast<u32>(views.size());

		gray::image_t contrast;
		gray::image_t blur;
		gray::image_t edges;

		locate_props_t props = {};
		props.edge_gradient_threshold = 128;
		props.contrast_low = 10;
		props.contrast_high = 150;

		using func_t = std::function<void()>;
		using func_list_t = std::array<func_t, 3>;

		auto const execute = [](func_t const& f) { f(); };

		// allocate memory on separate threads
		func_list_t allocate
		{
			[&]() { props.contrast = make_view(contrast, v_width, v_height); },
			[&]() { props.blur = make_view(blur, v_width, v_height); },
			[&]() { props.edges = make_edges_view(edges, v_width, v_height); }
		};
		std::for_each(std::execution::par, allocate.begin(), allocate.end(), execute);

		u32_range_t view_ids(0u, v_size);
		view_list_t results(v_size);
		auto const locate = [&](u32 i) { results[i] = locate_one(views[i], pattern, props); };
		std::for_each(std::execution::par, view_ids.begin(), view_ids.end(), locate);

		// free memory on separate threads
		func_list_t destroy
		{
			[&]() { contrast.clear(); },
			[&]() { blur.clear(); },
			[&]() { edges.clear(); }
		};
		std::for_each(std::execution::par, destroy.begin(), destroy.end(), execute);

		return results;
	}
}