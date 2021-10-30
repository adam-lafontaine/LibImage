/*

Copyright (c) 2021 Adam Lafontaine

*/

#include "charts.hpp"

#include <numeric>
#include <algorithm>
#include <cmath>

#ifndef LIBIMAGE_NO_PARALLEL
#include <execution>
#endif // !LIBIMAGE_NO_PARALLEL


namespace libimage
{
#ifndef LIBIMAGE_NO_PARALLEL

#ifndef LIBIMAGE_NO_COLOR

	static void img_fill(image_t const& img, pixel_t const& p)
	{
		std::fill(std::execution::par, img.begin(), img.end(), p);
	}


	static void img_fill(view_t const& img, pixel_t const& p)
	{
		std::fill(std::execution::par, img.begin(), img.end(), p);
	}

#endif // !LIBIMAGE_NO_COLOR
#ifndef LIBIMAGE_NO_GRAYSCALE

	static void img_fill(gray::image_t const& img, gray::pixel_t const& p)
	{
		std::fill(std::execution::par, img.begin(), img.end(), p);
	}


	static void img_fill(gray::view_t const& img, gray::pixel_t const& p)
	{
		std::fill(std::execution::par, img.begin(), img.end(), p);
	}

#endif // !LIBIMAGE_NO_GRAYSCALE

#else

#ifndef LIBIMAGE_NO_COLOR

	static void img_fill(image_t const& img, pixel_t const& p)
	{
		std::fill(img.begin(), img.end(), p);
	}


	static void img_fill(view_t const& img, pixel_t const& p)
	{
		std::fill(img.begin(), img.end(), p);
	}

#endif // !LIBIMAGE_NO_COLOR
#ifndef LIBIMAGE_NO_GRAYSCALE

	static void img_fill(gray::image_t const& img, gray::pixel_t const& p)
	{
		std::fill(img.begin(), img.end(), p);
	}


	static void img_fill(gray::view_t const& img, gray::pixel_t const& p)
	{
		std::fill(img.begin(), img.end(), p);
	}

#endif // !LIBIMAGE_NO_GRAYSCALE
#endif // !LIBIMAGE_NO_PARALLEL
}


namespace libimage
{
#ifndef LIBIMAGE_NO_COLOR	

	constexpr auto ugly_yellow = to_pixel(255, 249, 79);
	constexpr auto ugly_green = to_pixel(153, 255, 51);


	void draw_histogram(rgb_stats_t const& rgb_stats, image_t& image_dst)
	{
		assert(!image_dst.width);
		assert(!image_dst.height);
		assert(!image_dst.data);
		assert(N_HIST_BUCKETS <= CHANNEL_SIZE);

		u32 const max_relative_qty = 200;
		u32 const channel_spacing = 1;
		u32 const channel_height = max_relative_qty + channel_spacing;
		u32 const image_height = channel_height * RGB_CHANNELS;

		u32 const n_buckets = static_cast<u32>(N_HIST_BUCKETS);

		u32 const bucket_width = 20;
		u32 const bucket_spacing = 1;
		u32 const image_width = n_buckets * (bucket_spacing + bucket_width) + bucket_spacing;

		pixel_t white = to_pixel(255, 255, 255);

		make_image(image_dst, image_width, image_height);
		img_fill(image_dst, white);

		u32 max_count = 0;
		for_each_channel_rgb([&](u32 c)
			{
				auto& hist = rgb_stats.stats[c].hist;
				auto max = *std::max_element(hist.begin(), hist.end());

				if (max > max_count)
				{
					max_count = max;
				}
			});

		const auto norm = [&](u32 count)
		{
			return static_cast<u32>(static_cast<r32>(count) / max_count * max_relative_qty);
		};

		for (u32 c = 0; c < RGB_CHANNELS; ++c)
		{
			auto& hist = rgb_stats.stats[c].hist;

			pixel_range_t bar_range = {};
			bar_range.x_begin = bucket_spacing;
			bar_range.x_end = bar_range.x_begin + bucket_width;
			bar_range.y_begin = 0;
			bar_range.y_end = channel_height * (c + 1);

			for (u32 bucket = 0; bucket < n_buckets; ++bucket)
			{
				bar_range.y_begin = bar_range.y_end - norm(hist[bucket]);
				if (bar_range.y_end > bar_range.y_begin)
				{
					u8 shade = 200; // n_buckets* (bucket + 1) - 1;
					pixel_t color = to_pixel(0, 0, 0, 255);
					color.channels[c] = shade;
					auto bar_view = sub_view(image_dst, bar_range);
					img_fill(bar_view, color);
				}

				bar_range.x_begin += (bucket_spacing + bucket_width);
				bar_range.x_end += (bucket_spacing + bucket_width);
			}
		}
	}


	void draw_bar_chart_grouped(grouped_chart_data_t const& data, image_t& image_dst)
	{
		assert(!image_dst.width);
		assert(!image_dst.height);
		assert(!image_dst.data);
		assert(!data.empty());

		u32 const n_buckets = static_cast<u32>(data[0].data.size());
		auto same_size =  std::all_of(data.begin(), data.end(), [&](auto const& v) { return v.data.size() == n_buckets; });
		assert(same_size);
		if (!same_size)
		{
			make_image(image_dst, 13, 13);
			img_fill(image_dst, ugly_yellow);
			return;
		}

		u32 const max_relative_qty = 200;
		u32 const image_height = max_relative_qty + 1;

		u32 const bar_width = 20;
		u32 const bar_spacing = 2;
		u32 const n_groups = static_cast<u32>(data.size());
		u32 const group_spacing = 10;
		u32 const group_width = n_groups * bar_width + (n_groups - 1) * bar_spacing;
		u32 const image_width = n_buckets * group_width + (n_buckets - 1) * group_spacing + 2 * bar_spacing;

		pixel_t white = to_pixel(240, 240, 240);

		make_image(image_dst, image_width, image_height);
		img_fill(image_dst, white);

		r32 max_count = 0.0f;
		for (auto const& item : data)
		{
			auto& vec = item.data;
			auto max = *std::max_element(vec.begin(), vec.end());

			if (max > max_count)
			{
				max_count = max;
			}
		}

		const auto norm = [&](r32 val)
		{
			return static_cast<u32>(val / max_count * max_relative_qty);
		};

		pixel_range_t bar_range = {};
		bar_range.y_end = image_height;

		for (u32 bucket = 0; bucket < n_buckets; ++bucket)
		{
			for (u32 group = 0; group < n_groups; ++group)
			{
				bar_range.x_begin = bar_spacing + bucket * (group_width + group_spacing) + group * (bar_width + bar_spacing);
				bar_range.x_end = bar_range.x_begin + bar_width;

				auto& d = data[group];
				bar_range.y_begin = bar_range.y_end - norm(d.data[bucket]);

				if (bar_range.y_end > bar_range.y_begin)
				{
					auto bar_view = sub_view(image_dst, bar_range);
					img_fill(bar_view, d.color);
				}
			}
		}
	}


	void draw_bar_multi_chart_grouped(grouped_multi_chart_data_t const& data, image_t& image_dst)
	{
		assert(!image_dst.width);
		assert(!image_dst.height);
		assert(!image_dst.data);
		assert(!data.empty());

		u32 const n_charts = static_cast<u32>(data[0].data_list.size());
		auto same_size = std::all_of(data.begin(), data.end(), [&](auto const& v) { return v.data_list.size() == n_charts; });
		assert(same_size);
		if (!same_size)
		{
			make_image(image_dst, 13, 13);
			img_fill(image_dst, ugly_yellow);
			return;
		}

		u32 const n_buckets = static_cast<u32>(data[0].data_list[0].size());
		same_size = std::all_of(data.begin(), data.end(), 
			[&](auto const& v) 
			{ 
				return std::all_of(v.data_list.begin(), v.data_list.end(), [&](auto const& d) { return d.size() == n_buckets;  }); 
			});
		assert(same_size);
		if (!same_size)
		{
			make_image(image_dst, 13, 13);
			img_fill(image_dst, ugly_green);
			return;
		}

		u32 const max_relative_qty = 200;
		u32 const chart_height = max_relative_qty + 1;
		u32 chart_spacing = 10;
		u32 const image_height = n_charts * (max_relative_qty + 1) + (n_charts - 1) * chart_spacing;

		u32 const bar_width = 20;
		u32 const bar_spacing = 2;
		u32 const n_groups = static_cast<u32>(data.size());
		u32 const group_spacing = 10;
		u32 const group_width = n_groups * bar_width + (n_groups - 1) * bar_spacing;
		u32 const image_width = n_buckets * group_width + (n_buckets - 1) * group_spacing + 2 * bar_spacing;

		pixel_t white = to_pixel(240, 240, 240);

		make_image(image_dst, image_width, image_height);
		img_fill(image_dst, white);

		r32 max_count = 0.0f;
		for (auto const& mcd : data)
		{
			for (auto const& vec : mcd.data_list)
			{
				auto max = *std::max_element(vec.begin(), vec.end());

				if (max > max_count)
				{
					max_count = max;
				}
			}
		}

		const auto norm = [&](r32 val)
		{
			return static_cast<u32>(val / max_count * max_relative_qty);
		};

		pixel_range_t bar_range = {};

		for (u32 chart = 0; chart < n_charts; ++chart)
		{
			bar_range.y_end = (chart + 1) * chart_height + chart * chart_spacing;

			for (u32 bucket = 0; bucket < n_buckets; ++bucket)
			{
				for (u32 group = 0; group < n_groups; ++group)
				{
					bar_range.x_begin = bar_spacing + bucket * (group_width + group_spacing) + group * (bar_width + bar_spacing);
					bar_range.x_end = bar_range.x_begin + bar_width;

					auto& mcd = data[group];
					auto val = mcd.data_list[chart][bucket];

					bar_range.y_begin = bar_range.y_end - norm(val);
					if (bar_range.y_end > bar_range.y_begin)
					{
						auto bar_view = sub_view(image_dst, bar_range);
						img_fill(bar_view, mcd.color);
					}
				}
			}
		}
	}


#endif // !LIBIMAGE_NO_COLOR

#ifndef	LIBIMAGE_NO_GRAYSCALE

	void draw_histogram(hist_t const& hist, gray::image_t& image_dst)
	{
		assert(!image_dst.width);
		assert(!image_dst.height);
		assert(!image_dst.data);
		assert(N_HIST_BUCKETS <= CHANNEL_SIZE);
		assert(hist.size() == N_HIST_BUCKETS);

		u32 const max_relative_qty = 200;
		u32 const image_height = max_relative_qty + 1;

		u32 const n_buckets = static_cast<u32>(hist.size());

		u32 const bucket_width = 20;
		u32 const bucket_spacing = 1;
		u32 const image_width = n_buckets * (bucket_spacing + bucket_width) + bucket_spacing;

		u8 const white = 255;

		make_image(image_dst, image_width, image_height);
		img_fill(image_dst, white);

		auto max_count = *std::max_element(hist.begin(), hist.end());

		const auto norm = [&](u32 count)
		{
			return static_cast<u32>(static_cast<r32>(count) / max_count * max_relative_qty);
		};

		pixel_range_t bar_range = {};
		bar_range.x_begin = bucket_spacing;
		bar_range.x_end = (bucket_spacing + bucket_width);
		bar_range.y_begin = 0;
		bar_range.y_end = image_height;

		for (u32 bucket = 0; bucket < n_buckets; ++bucket)
		{
			bar_range.y_begin = bar_range.y_end - norm(hist[bucket]);

			if (bar_range.y_end > bar_range.y_begin)
			{
				u8 shade = 50;// n_buckets* (bucket + 1) - 1;
				auto bar_view = sub_view(image_dst, bar_range);
				img_fill(bar_view, shade);
			}

			bar_range.x_begin += (bucket_spacing + bucket_width);
			bar_range.x_end += (bucket_spacing + bucket_width);
		}

	}

#endif // !LIBIMAGE_NO_GRAYSCALE

}