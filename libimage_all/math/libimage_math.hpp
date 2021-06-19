#pragma once

#include "../libimage.hpp"

#include <array>
#include <algorithm>

// median
// draw stats
// rgba_stats_t, alpha (Grayscale) = 0.299R + 0.587G + 0.114B


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


#ifndef LIBIMAGE_NO_COLOR

	rgb_stats_t calc_stats(image_t const& image);
	
	rgb_stats_t calc_stats(view_t const& view);

	void draw_histogram(rgb_stats_t const& rgb_stats, image_t& image_dst);

#endif // !LIBIMAGE_NO_COLOR

#ifndef	LIBIMAGE_NO_GRAYSCALE

	stats_t calc_stats(gray::image_t const& image);
	
	stats_t calc_stats(gray::view_t const& view);

	void draw_histogram(hist_t const& hist, gray::image_t& image_dst);

#endif // !LIBIMAGE_NO_GRAYSCALE
}