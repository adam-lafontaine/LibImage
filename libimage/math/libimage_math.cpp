#include "libimage_math.hpp"

#include <numeric>
#include <execution>
#include <cmath>



namespace libimage
{
	constexpr auto HIST_BUCKET_SIZE = CHANNEL_SIZE / N_HIST_BUCKETS;

	typedef struct
	{
		r32 mean;
		hist_t hist;

	} basic_stats_t;


	static r32 calc_std_dev(hist_t const& hist, r32 mean)
	{
		r32 diff_sq_total = 0.0f;
		size_t qty_total = 0;
		for (u32 bucket = 0; bucket < hist.size(); ++bucket)
		{
			auto qty = hist[bucket];

			if (!qty)
				continue;

			qty_total += qty;
			r32 diff = static_cast<r32>(bucket) - mean;

			diff_sq_total += qty * diff * diff;
		}

		return qty_total == 0 ? 0.0f : std::sqrt(diff_sq_total / qty_total);
	}


#ifndef LIBIMAGE_NO_COLOR

	rgb_stats_t calc_stats(image_t const& image)
	{
		std::array<hist_t, RGB_CHANNELS> c_hists = { 0 };
		std::array<r32, RGB_CHANNELS> c_counts = { 0 };

		auto const update = [&](pixel_t const& p)
		{
			for (u32 c = 0; c < RGB_CHANNELS; ++c)
			{
				auto bucket = p.channels[c] / HIST_BUCKET_SIZE;

				++c_hists[c][bucket];
				c_counts[c] += p.channels[c];
			}
		};

		std::for_each(image.begin(), image.end(), update);

		auto num_pixels = static_cast<size_t>(image.width) * image.height;

		rgb_stats_t rgb_stats = {};

		for_each_channel_rgb([&](u32 c) 
		{
			auto mean = c_counts[c] / num_pixels;

			r32 std_dev = calc_std_dev(c_hists[c], mean);
			rgb_stats.stats[c] = { mean, std_dev, c_hists[c] };
		});

		return rgb_stats;
	}
	
	
	rgb_stats_t calc_stats(view_t const& view)
	{
		std::array<hist_t, RGB_CHANNELS> c_hists = { 0 };
		std::array<r32, RGB_CHANNELS> c_counts = { 0 };

		auto const update = [&](pixel_t const& p)
		{
			for_each_channel_rgb([&](u32 c) 
			{
				auto bucket = p.channels[c] / HIST_BUCKET_SIZE;

				++c_hists[c][bucket];
				c_counts[c] += p.channels[c];
			});
		};

		std::for_each(view.begin(), view.end(), update);

		auto num_pixels = static_cast<size_t>(view.width) * view.height;

		rgb_stats_t rgb_stats = {};

		for_each_channel_rgb([&](u32 c) 
		{
			auto mean = c_counts[c] / num_pixels;

			r32 std_dev = calc_std_dev(c_hists[c], mean);
			rgb_stats.stats[c] = { mean, std_dev, c_hists[c] };
		});

		return rgb_stats;
	}


	static basic_stats_t calc_basic_stats(image_t const& image, u32 channel)
	{
		hist_t hist = { 0 };
		r32 count = 0.0f;

		auto const update = [&](pixel_t const& p)
		{
			auto shade = p.channels[channel];
			auto bucket = shade / HIST_BUCKET_SIZE;
			++hist[bucket];
			count += shade;
		};

		std::for_each(image.begin(), image.end(), update);

		auto num_pixels = image.width * image.height;
		r32 mean = count / num_pixels;

		assert(mean >= 0);
		assert(mean < CHANNEL_SIZE);

		return { mean, std::move(hist) };
	}


	static basic_stats_t calc_basic_stats(view_t const& view, u32 channel)
	{
		hist_t hist = { 0 };
		r32 count = 0.0f;

		auto const update = [&](pixel_t const& p)
		{
			auto shade = p.channels[channel];
			auto bucket = shade / HIST_BUCKET_SIZE;
			++hist[bucket];
			count += shade;
		};

		std::for_each(view.begin(), view.end(), update);

		auto num_pixels = view.width * view.height;
		r32 mean = count / num_pixels;

		assert(mean >= 0);
		assert(mean < CHANNEL_SIZE);

		return { mean, std::move(hist) };
	}


	stats_t calc_stats(image_t const& image, Channel ch)
	{
		auto basic = calc_basic_stats(image, to_channel_index(ch));

		r32 std_dev = calc_std_dev(basic.hist, basic.mean);

		return { basic.mean, std_dev, std::move(basic.hist) };
	}


	stats_t calc_stats(view_t const& view, Channel ch)
	{
		auto basic = calc_basic_stats(view, to_channel_index(ch));

		r32 std_dev = calc_std_dev(basic.hist, basic.mean);

		return { basic.mean, std_dev, std::move(basic.hist) };
	}


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
		std::fill(image_dst.begin(), image_dst.end(), white);

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
					std::fill(bar_view.begin(), bar_view.end(), color);
				}

				bar_range.x_begin += (bucket_spacing + bucket_width);
				bar_range.x_end += (bucket_spacing + bucket_width);
			}
		}
	}


	void draw_bar_chart(std::vector<data_color_t> const& data, image_t& image_dst)
	{
		assert(!image_dst.width);
		assert(!image_dst.height);
		assert(!image_dst.data);
		assert(!data.empty());

	    u32 const n_buckets = static_cast<u32>(data[0].data.size());
		for (auto const& item : data)
		{
			if (item.data.size() != n_buckets)
			{
				assert(false);
				return;
			}
		}

		u32 const max_relative_qty = 200;
		u32 const image_height = max_relative_qty + 1;		

		u32 const bar_width = 20;
		u32 const bar_spacing = 2;
		u32 const n_groups = static_cast<u32>(data.size());
		u32 const group_spacing = 10;
		u32 const group_width = n_groups * bar_width + (n_groups - 1) * bar_spacing;
		u32 const image_width = n_buckets * group_width + (n_buckets - 1) * group_spacing + 2 * bar_spacing;

		pixel_t white = to_pixel(255, 255, 255);

		make_image(image_dst, image_width, image_height);
		std::fill(image_dst.begin(), image_dst.end(), white);

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
		bar_range.y_begin = 0;
		bar_range.y_end = image_height;

		for (u32 bucket = 0; bucket < n_buckets; ++bucket)
		{
			for (u32 group = 0; group < n_groups; ++group)
			{
				auto& d = data[group];

				bar_range.x_begin = bar_spacing + bucket * (group_width + group_spacing) + group * (bar_width + bar_spacing);
				bar_range.x_end = bar_range.x_begin + bar_width;
				bar_range.y_begin = bar_range.y_end - norm(d.data[bucket]);

				if (bar_range.y_end > bar_range.y_begin)
				{
					auto bar_view = sub_view(image_dst, bar_range);
					std::fill(bar_view.begin(), bar_view.end(), d.color);
				}
			}
		}
	}

#endif // !LIBIMAGE_NO_COLOR


#ifndef	LIBIMAGE_NO_GRAYSCALE

	static basic_stats_t calc_basic_stats(gray::image_t const& image)
	{
		hist_t hist = { 0 };
		r32 count = 0.0f;

		auto const update = [&](u8 shade)
		{
			auto bucket = shade / HIST_BUCKET_SIZE;
			++hist[bucket];
			count += shade;
		};

		std::for_each(image.begin(), image.end(), update);

		auto num_pixels = image.width * image.height;
		r32 mean = count / num_pixels;

		assert(mean >= 0);
		assert(mean < CHANNEL_SIZE);

		return { mean, hist };
	}


	static basic_stats_t calc_basic_stats(gray::view_t const& view)
	{
		hist_t hist = { 0 };
		r32 count = 0.0f;

		auto const update = [&](u8 shade)
		{
			auto bucket = shade / HIST_BUCKET_SIZE;
			++hist[bucket];
			count += shade;
		};

		std::for_each(view.begin(), view.end(), update);

		auto num_pixels = view.width * view.height;
		r32 mean = count / num_pixels;

		assert(mean >= 0);
		assert(mean < CHANNEL_SIZE);

		return { mean, hist };
	}


	stats_t calc_stats(gray::image_t const& image)
	{
		auto basic = calc_basic_stats(image);

		r32 std_dev = calc_std_dev(basic.hist, basic.mean);

		return { basic.mean, std_dev, std::move(basic.hist) };
	}


	stats_t calc_stats(gray::view_t const& view)
	{
		auto basic = calc_basic_stats(view);

		r32 std_dev = calc_std_dev(basic.hist, basic.mean);

		return { basic.mean, std_dev, std::move(basic.hist) };
	}


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

		make_image(image_dst, image_width, image_height);
		std::fill(image_dst.begin(), image_dst.end(), 255);

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
				std::fill(bar_view.begin(), bar_view.end(), shade);
			}

			bar_range.x_begin += (bucket_spacing + bucket_width);
			bar_range.x_end += (bucket_spacing + bucket_width);
		}

	}


#endif // !LIBIMAGE_NO_GRAYSCALE
	
}