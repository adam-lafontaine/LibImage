#pragma once
/*

Copyright (c) 2021 Adam Lafontaine

*/

#include "../libimage.hpp"

#include <vector>
#include <functional>


namespace libimage
{

#ifndef LIBIMAGE_NO_COLOR	

	rgb_stats_t calc_stats(image_t const& image);
	
	rgb_stats_t calc_stats(view_t const& view);

	stats_t calc_stats(image_t const& image, Channel ch);

	stats_t calc_stats(view_t const& view, Channel ch);
	

#endif // !LIBIMAGE_NO_COLOR

#ifndef	LIBIMAGE_NO_GRAYSCALE

	stats_t calc_stats(gray::image_t const& image);
	
	stats_t calc_stats(gray::view_t const& view);
	

#endif // !LIBIMAGE_NO_GRAYSCALE
}