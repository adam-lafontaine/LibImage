#pragma once

#include "image_view.hpp"

#include <array>
#include <algorithm>

using r32 = float;


namespace libimage
{
	constexpr size_t CHANNEL_SIZE = 256; // 8 bit channel
	constexpr size_t N_HIST_BUCKETS = 16;

	using hist_t = std::array<u32, N_HIST_BUCKETS>;


	typedef struct channel_stats_t
	{
		r32 mean;
		r32 std_dev;
		hist_t hist;

	} stats_t;


	typedef union rgb_channel_stats_t
	{
		struct
		{
			stats_t red;
			stats_t green;
			stats_t blue;
		};

		stats_t stats[3];

	} rgb_stats_t;


	rgb_stats_t calc_stats(view_t const& view);

	stats_t calc_stats(gray::view_t const& view);

	void draw_histogram(hist_t const& hist, gray::image_t& image_dst);

	void draw_histogram(rgb_stats_t const& rgb_stats, image_t& image_dst);
}