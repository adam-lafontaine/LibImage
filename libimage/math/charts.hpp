#pragma once
/*

Copyright (c) 2021 Adam Lafontaine

*/

#include "../libimage.hpp"


namespace libimage
{
#ifndef LIBIMAGE_NO_COLOR

	typedef struct DataColor
	{
		std::vector<r32> data;
		pixel_t color = to_pixel(0);

	} data_color_t;


	void draw_histogram(rgb_stats_t const& rgb_stats, image_t& image_dst);

	void draw_bar_chart_grouped(std::vector<data_color_t> const& data, image_t& image_dst);


#endif // !LIBIMAGE_NO_COLOR

#ifndef	LIBIMAGE_NO_GRAYSCALE

	void draw_histogram(hist_t const& hist, gray::image_t& image_dst);

#endif // !LIBIMAGE_NO_GRAYSCALE
}