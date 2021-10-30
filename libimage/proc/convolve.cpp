/*

Copyright (c) 2021 Adam Lafontaine

*/

#ifndef LIBIMAGE_NO_GRAYSCALE

#include "convolve.hpp"
#include "../libimage.hpp"

#include <algorithm>
#include <array>

namespace libimage
{
	static inline void left_2_wide(pixel_range_t& range, u32 x, u32 width)
	{
		// left (2 wide)
		assert(x <= width - 2);
		range.x_begin = x;
		range.x_end = x + 2;
	}


	static inline void right_2_wide(pixel_range_t& range, u32 x, u32 width)
	{
		// right (2 wide)
		assert(x >= 1);
		assert(x <= width - 1);
		range.x_begin = x - 1;
		range.x_end = x + 1;
	}


	static inline void top_2_high(pixel_range_t& range, u32 y, u32 height)
	{
		// top (2 high)
		assert(y <= height - 2);
		range.y_begin = y;
		range.y_end = y + 2;
	}


	static inline void bottom_2_high(pixel_range_t& range, u32 y, u32 height)
	{
		// bottom (2 high)
		assert(y >= 1);
		assert(y <= height - 1);
		range.y_begin = y - 1;
		range.y_end = y + 1;
	}


	static inline void top_or_bottom_3_high(pixel_range_t& range, u32 y, u32 height)
	{
		// top or bottom (3 high)
		assert(y >= 1);
		assert(y <= height - 2);
		range.y_begin = y - 1;
		range.y_end = y + 2;
	}


	static inline void left_or_right_3_wide(pixel_range_t& range, u32 x, u32 width)
	{
		// left or right (3 wide)
		assert(x >= 1);
		assert(x <= width - 2);
		range.x_begin = x - 1;
		range.x_end = x + 2;
	}


	static inline void top_or_bottom_5_high(pixel_range_t& range, u32 y, u32 height)
	{
		// top or bottom (5 high)
		assert(y >= 2);
		assert(y <= height - 3);
		range.y_begin = y - 2;
		range.y_end = y + 3;
	}


	static inline void left_or_right_5_wide(pixel_range_t& range, u32 x, u32 width)
	{
		// left or right (5 wide)
		assert(x >= 2);
		assert(x <= width - 3);
		range.x_begin = x - 2;
		range.x_end = x + 3;
	}


	template<size_t N>
	static r32 apply_weights(gray::view_t const& view, std::array<r32, N> weights)
	{
		assert(static_cast<size_t>(view.width) * view.height == weights.size());

		u32 w = 0;
		r32 total = 0.0f;

		auto const add_weight = [&](u8 p) { total += weights[w++] * p; };

		std::for_each(view.begin(), view.end(), add_weight);

		return total;
	}


	template<class GRAY_IMAGE_T>
	static r32 weighted_center(GRAY_IMAGE_T const& img, u32 x, u32 y, std::array<r32, 9> const& weights)
	{
		pixel_range_t range = {};

		top_or_bottom_3_high(range, y, img.height);

		left_or_right_3_wide(range, x, img.width);

		return apply_weights(sub_view(img, range), weights);
	}


	template<class GRAY_IMAGE_T>
	static r32 weighted_center(GRAY_IMAGE_T const& img, u32 x, u32 y, std::array<r32, 25> const& weights)
	{
		pixel_range_t range = {};

		top_or_bottom_5_high(range, y, img.height);

		left_or_right_5_wide(range, x, img.width);

		return apply_weights(sub_view(img, range), weights);
	}


	template<class GRAY_IMAGE_T>
	static r32 weighted_top_left(GRAY_IMAGE_T const& img, u32 x, u32 y, std::array<r32, 4> const& weights)
	{
		pixel_range_t range = {};

		top_2_high(range, y, img.height);

		left_2_wide(range, x, img.width);

		return apply_weights(sub_view(img, range), weights);
	}


	template<class GRAY_IMAGE_T>
	static r32 weighted_top_right(GRAY_IMAGE_T const& img, u32 x, u32 y, std::array<r32, 4> const& weights)
	{
		pixel_range_t range = {};

		top_2_high(range, y, img.height);

		right_2_wide(range, x, img.width);

		return apply_weights(sub_view(img, range), weights);
	}


	template<class GRAY_IMAGE_T>
	static r32 weighted_bottom_left(GRAY_IMAGE_T const& img, u32 x, u32 y, std::array<r32, 4> const& weights)
	{
		pixel_range_t range = {};

		bottom_2_high(range, y, img.width);

		left_2_wide(range, x, img.width);

		return apply_weights(sub_view(img, range), weights);
	}


	template<class GRAY_IMAGE_T>
	static r32 weighted_bottom_right(GRAY_IMAGE_T const& img, u32 x, u32 y, std::array<r32, 4> const& weights)
	{
		pixel_range_t range = {};

		bottom_2_high(range, y, img.width);

		right_2_wide(range, x, img.width);

		return apply_weights(sub_view(img, range), weights);
	}


	template<class GRAY_IMAGE_T>
	static r32 weighted_top(GRAY_IMAGE_T const& img, u32 x, u32 y, std::array<r32, 6> const& weights)
	{
		pixel_range_t range = {};

		top_2_high(range, y, img.height);

		left_or_right_3_wide(range, x, img.width);

		return apply_weights(sub_view(img, range), weights);
	}


	template<class GRAY_IMAGE_T>
	static r32 weighted_bottom(GRAY_IMAGE_T const& img, u32 x, u32 y, std::array<r32, 6> const& weights)
	{
		pixel_range_t range = {};

		bottom_2_high(range, y, img.width);

		left_or_right_3_wide(range, x, img.width);

		return apply_weights(sub_view(img, range), weights);
	}


	template<class GRAY_IMAGE_T>
	static r32 weighted_left(GRAY_IMAGE_T const& img, u32 x, u32 y, std::array<r32, 6> const& weights)
	{
		pixel_range_t range = {};

		top_or_bottom_3_high(range, y, img.height);

		left_2_wide(range, x, img.width);

		return apply_weights(sub_view(img, range), weights);
	}


	template<class GRAY_IMAGE_T>
	static r32 weighted_right(GRAY_IMAGE_T const& img, u32 x, u32 y, std::array<r32, 6> const& weights)
	{
		pixel_range_t range = {};

		top_or_bottom_3_high(range, y, img.height);

		right_2_wide(range, x, img.width);

		return apply_weights(sub_view(img, range), weights);
	}


	u8 gauss3(gray::image_t const& img, u32 x, u32 y)
	{
		constexpr auto rw = [](u8 w) { return w / 16.0f; };
		constexpr std::array<r32, 9> gauss
		{
			rw(1), rw(2), rw(1),
			rw(2), rw(4), rw(2),
			rw(1), rw(2), rw(1),
		};

		auto p = weighted_center(img, x, y, gauss);

		assert(p >= 0.0f);
		assert(p <= 255.0f);

		return static_cast<u8>(p);
	}


	u8 gauss3(gray::view_t const& view, u32 x, u32 y)
	{
		constexpr auto rw = [](u8 w) { return w / 16.0f; };
		constexpr std::array<r32, 9> gauss
		{
			rw(1), rw(2), rw(1),
			rw(2), rw(4), rw(2),
			rw(1), rw(2), rw(1),
		};

		auto p = weighted_center(view, x, y, gauss);

		assert(p >= 0.0f);
		assert(p <= 255.0f);

		return static_cast<u8>(p);
	}


	u8 gauss5(gray::image_t const& img, u32 x, u32 y)
	{
		constexpr auto rw = [](u8 w) { return w / 256.0f; };
		constexpr std::array<r32, 25> gauss
		{
			rw(1), rw(4),  rw(6),  rw(4),  rw(1),
			rw(4), rw(16), rw(24), rw(16), rw(4),
			rw(6), rw(24), rw(36), rw(24), rw(6),
			rw(4), rw(16), rw(24), rw(16), rw(4),
			rw(1), rw(4),  rw(6),  rw(4),  rw(1),
		};

		auto p = weighted_center(img, x, y, gauss);

		assert(p >= 0.0f);
		assert(p <= 255.0f);

		return static_cast<u8>(p);
	}


	u8 gauss5(gray::view_t const& view, u32 x, u32 y)
	{
		constexpr auto rw = [](u8 w) { return w / 256.0f; };
		constexpr std::array<r32, 25> gauss
		{
			rw(1), rw(4),  rw(6),  rw(4),  rw(1),
			rw(4), rw(16), rw(24), rw(16), rw(4),
			rw(6), rw(24), rw(36), rw(24), rw(6),
			rw(4), rw(16), rw(24), rw(16), rw(4),
			rw(1), rw(4),  rw(6),  rw(4),  rw(1),
		};

		auto p = weighted_center(view, x, y, gauss);

		assert(p >= 0.0f);
		assert(p <= 255.0f);

		return static_cast<u8>(p);
	}


	r32 x_gradient(gray::image_t const& img, u32 x, u32 y)
	{
		constexpr std::array<r32, 9> kernel
		{
			1.0f, 0.0f, -1.0f,
			2.0f, 0.0f, -2.0f,
			1.0f, 0.0f, -1.0f,
		};

		return weighted_center(img, x, y, kernel);
	}


	r32 x_gradient(gray::view_t const& view, u32 x, u32 y)
	{
		constexpr std::array<r32, 9> kernel
		{
			1.0f, 0.0f, -1.0f,
			2.0f, 0.0f, -2.0f,
			1.0f, 0.0f, -1.0f,
		};

		return weighted_center(view, x, y, kernel);
	}


	r32 y_gradient(gray::image_t const& img, u32 x, u32 y)
	{
		constexpr std::array<r32, 9> kernel
		{
			 1.0f,  2.0f,  1.0f,
			 0.0f,  0.0f,  0.0f,
			-1.0f, -2.0f, -1.0f,
		};

		return weighted_center(img, x, y, kernel);
	}


	r32 y_gradient(gray::view_t const& view, u32 x, u32 y)
	{
		constexpr std::array<r32, 9> kernel
		{
			 1.0f,  2.0f,  1.0f,
			 0.0f,  0.0f,  0.0f,
			-1.0f, -2.0f, -1.0f,
		};

		return weighted_center(view, x, y, kernel);
	}
}

#endif