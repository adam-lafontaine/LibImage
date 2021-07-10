
// Experimental.
// To be used in a separate machine learning libray

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
#include <limits>

constexpr u32 U32_NOT_SET = UINT32_MAX;

namespace libimage
{
	using view_list_t = std::vector<gray::view_t>;
	using image_list_t = std::vector<gray::image_t>;

	typedef struct LocateResult
	{
		u32 x = U32_NOT_SET;
		u32 y = U32_NOT_SET;
		u32 delta = U32_NOT_SET;

	} locate_result_t;

	using result_list_t = std::vector<locate_result_t>;

	typedef struct
	{
		u32 contrast_low;
		u32 contrast_high;

		u8 edge_gradient_min;

	} locate_props_t;


	static bool verify(view_list_t const& views)
	{
		auto w = std::all_of(std::execution::par, views.begin(), views.end(), [&](auto const& v) { return v.width == views[0].width; });
		auto h = std::all_of(std::execution::par, views.begin(), views.end(), [&](auto const& v) { return v.height == views[0].height; });

		return w && h;
	}


	static bool verify(gray::view_t const& view)
	{
		return view.image_data && view.width && view.height;
	}


	static bool verify(gray::view_t const& src, gray::view_t const& dst)
	{
		return verify(src) && verify(dst) && dst.width == src.width && dst.height == src.height;
	}


	static void q_gradient(gray::view_t const& src, gray::view_t const& dst)
	{
		assert(verify(src, dst));

		auto const width = src.width;
		auto const height = src.height;

		auto const zero = [](u8 p) { return static_cast<u8>(0); };

		auto const zero_top = [&]()
		{
			auto dst_top = row_view(dst, 0);

			transform(dst_top, zero);
		};

		auto const zero_bottom = [&]()
		{
			auto dst_bottom = row_view(dst, height - 1);

			transform(dst_bottom, zero);
		};

		auto const zero_left = [&]()
		{
			pixel_range_t r = {};
			r.x_begin = 0;
			r.x_end = 1;
			r.y_begin = 1;
			r.y_end = height - 1;
			auto dst_left = sub_view(dst, r);

			transform(dst_left, zero);
		};

		auto const zero_right = [&]()
		{
			pixel_range_t r = {};
			r.x_begin = width - 1;
			r.x_end = width;
			r.y_begin = 1;
			r.y_end = height - 1;
			auto dst_right = sub_view(dst, r);

			transform(dst_right, zero);
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
				auto gx = x_gradient(src, x, y);
				auto gy = y_gradient(src, x, y);
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
				total += (lhs_row[x] && rhs_row[x]) || (!lhs_row[x] && !rhs_row[x]);
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

	
	locate_result_t search_edges(gray::view_t const& search_view, gray::view_t const& pattern)
	{
		u32 const v_width = search_view.width;
		u32 const v_height = search_view.height;
		u32 const p_width = pattern.width;
		u32 const p_height = pattern.height;

		locate_result_t search_result;

		pixel_range_t r = {};

		auto const x_end = v_width - p_width + 1;
		auto const y_end = v_height - p_height + 1;

		for (u32 y = 0; y < y_end; ++y)
		{
			r.y_begin = y;
			r.y_end = y + p_height;
			for (u32 x = 0; x < x_end; ++x)
			{
				r.x_begin = x;
				r.x_end = x + p_width;				

				auto delta = edge_delta(sub_view(search_view, r), pattern);

				if (delta < search_result.delta)
				{
					search_result = { x, y, delta };
				}
			}
		}

		return search_result;
	}
	
	
	locate_result_t locate_one(gray::view_t const& view, gray::view_t const& pattern, locate_props_t& props, gray::view_t& v1, gray::view_t& v2)
	{
		verify(view, v1);
		verify(view, v2);

		u32 const v_width = view.width;
		u32 const v_height = view.height;
		u32 const p_width = pattern.width;
		u32 const p_height = pattern.height;

		assert(p_width < v_width);
		assert(p_height < v_height);
		
		transform_contrast(view, v1, props.contrast_low, props.contrast_high);
		blur(v1, v2);
		q_gradient(v2, v1);

		auto& grad = v1;
		auto& edges = v2;

		auto const g_max = std::max_element(std::execution::par, grad.begin(), grad.end());

		u32 th_begin = static_cast<u32>(props.edge_gradient_min);
		u32 th_end = static_cast<u32>(*g_max + 1);

		locate_result_t res_min;

		if (th_begin > th_end)
		{
			th_begin = 0;
		}

		for (u32 th = th_begin; th < th_end; ++th)
		{
			binarize(grad, edges, th);

			auto res = search_edges(edges, pattern);
			if (res.delta < res_min.delta)
			{
				res_min = res;
			}
		}

		return res_min;
	}


	static view_list_t make_views(image_list_t& images, u32 width, u32 height)
	{
		assert(images.size());

		view_list_t views(images.size());

		u32_range_t ids(0u, static_cast<u32>(images.size()));

		auto const make = [&](u32 id) { views[id] = make_view(images[id], width, height); };
		std::for_each(std::execution::par, ids.begin(), ids.end(), make);

		return views;
	}


	static void destroy_images(image_list_t& images)
	{
		assert(images.size());

		auto const destroy = [](auto& image) { image.dispose(); };
		std::for_each(std::execution::par, images.begin(), images.end(), destroy);
	}


	result_list_t locate_many(view_list_t const& views, gray::view_t const& pattern)
	{
		assert(views.size());
		assert(verify(views));

		u32 const v_width = views[0].width;
		u32 const v_height = views[0].height;
		u32 const v_size = static_cast<u32>(views.size());

		// scratch memory for parallel image calculations
		image_list_t img_a(views.size());
		image_list_t img_b(views.size());
		view_list_t view_a;
		view_list_t view_b;

		locate_props_t props = {};
		props.edge_gradient_min = 10;
		props.contrast_low = 10;
		props.contrast_high = 150;

		using func_t = std::function<void()>;
		using func_list_t = std::array<func_t, 2>;

		auto const execute = [](func_t const& f) { f(); };
		
		func_list_t create // TODO: together
		{
			[&]() { view_a = make_views(img_a, v_width, v_height); },
			[&]() { view_b = make_views(img_b, v_width, v_height); },
		};

		func_list_t destroy
		{
			[&]() { destroy_images(img_a); },
			[&]() { destroy_images(img_b); },
		};

		u32_range_t view_ids(0u, v_size);
		result_list_t results(v_size);
		auto const locate = [&](u32 i) { results[i] = locate_one(views[i], pattern, props, view_a[i], view_b[i]); };

		// allocate memory on separate threads
		std::for_each(std::execution::par, create.begin(), create.end(), execute);

		// get best matches		
		std::for_each(std::execution::par, view_ids.begin(), view_ids.end(), locate);

		// free memory on separate threads		
		std::for_each(std::execution::par, destroy.begin(), destroy.end(), execute);

		return results;
	}





	void set_template(view_list_t const& views, gray::image_t& dst)
	{
		// edge gradient where all gradient views are most alike

		assert(views.size());
		assert(verify(views));

		u32 const width = views[0].width;
		u32 const height = views[0].height;

		auto pattern = make_view(dst, width, height);

		// scratch memory for parallel image calculations
		image_list_t img_a(views.size());
		image_list_t img_b(views.size());
		view_list_t view_a;
		view_list_t view_b;

		using func_t = std::function<void()>;
		using func_list_t = std::array<func_t, 2>;

		auto const execute = [](func_t const& f) { f(); };

		func_list_t create // TODO: together
		{
			[&]() { view_a = make_views(img_a, width, height); },
			[&]() { view_b = make_views(img_b, width, height); },
		};

		func_list_t destroy
		{
			[&]() { destroy_images(img_a); },
			[&]() { destroy_images(img_b); },
		};


		// allocate memory on separate threads
		std::for_each(std::execution::par, create.begin(), create.end(), execute);




		// free memory on separate threads		
		std::for_each(std::execution::par, destroy.begin(), destroy.end(), execute);
	}
}