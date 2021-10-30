/*

Copyright (c) 2021 Adam Lafontaine

*/

#include "math.hpp"

#include <numeric>
#include <algorithm>
#include <cmath>

#ifndef LIBIMAGE_NO_PARALLEL
#include <execution>
#endif // !LIBIMAGE_NO_PARALLEL


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

#endif // !LIBIMAGE_NO_GRAYSCALE
	
}