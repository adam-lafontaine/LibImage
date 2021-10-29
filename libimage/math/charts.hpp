#pragma once
/*

Copyright (c) 2021 Adam Lafontaine

*/

#include "../libimage.hpp"

#include <vector>


namespace libimage
{
#ifndef LIBIMAGE_NO_COLOR

	typedef struct ChartData
	{
		std::vector<r32> data;
		pixel_t color = to_pixel(0);

	} chart_data_t;


	using grouped_chart_data_t = std::vector<chart_data_t>;

	
	typedef struct MultiChartData
	{
		std::vector<std::vector<r32>> data_list;
		pixel_t color = to_pixel(0);

	} multi_chart_data_t;

	using grouped_multi_chart_data_t = std::vector<multi_chart_data_t>;


	void draw_histogram(rgb_stats_t const& rgb_stats, image_t& image_dst);

	void draw_bar_chart_grouped(grouped_chart_data_t const& data, image_t& image_dst);

	void draw_bar_multi_chart_grouped(grouped_multi_chart_data_t const& data, image_t& image_dst);


#endif // !LIBIMAGE_NO_COLOR

#ifndef	LIBIMAGE_NO_GRAYSCALE

	void draw_histogram(hist_t const& hist, gray::image_t& image_dst);

#endif // !LIBIMAGE_NO_GRAYSCALE
}