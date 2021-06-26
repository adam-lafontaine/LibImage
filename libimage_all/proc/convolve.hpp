#pragma once

#include "../libimage.hpp"

#include <array>

namespace libimage
{
	template<size_t N>
	u8 apply_weights(gray::view_t const& view, pixel_range_t const& range, std::array<r32, N> weights)
	{
		assert((range.x_end - range.x_begin) * (range.y_end - range.y_begin) == weights.size());
		u32 w = 0;
		r32 p = 0.0f;

		for (u32 vy = range.y_begin; vy < range.y_end; ++vy)
		{
			auto row = view.row_begin(vy);
			for (u32 vx = range.x_begin; vx < range.x_end; ++vx)
			{
				p += weights[w] * row[vx];
				++w;
			}
		}

		if (p < 0.0f)
		{
			p *= -1.0f;
		}

		assert(p <= 255.0f);

		return static_cast<u8>(p);
	}


	void left_2_wide(pixel_range_t& range, u32 x, u32 width)
	{
		// left (2 wide)
		assert(x <= width - 2);
		range.x_begin = x;
		range.x_end = x + 2;
	}


	void right_2_wide(pixel_range_t& range, u32 x, u32 width)
	{
		// right (2 wide)
		assert(x >= 1);
		assert(x <= width - 1);
		range.x_begin = x - 1;
		range.x_end = x + 1;
	}


	void top_2_high(pixel_range_t& range, u32 y, u32 height)
	{
		// top (2 high)
		assert(y <= height - 2);
		range.y_begin = y;
		range.y_end = y + 2;
	}


	void bottom_2_high(pixel_range_t& range, u32 y, u32 height)
	{
		// bottom (2 high)
		assert(y >= 1);
		assert(y <= height - 1);
		range.y_begin = y - 1;
		range.y_end = y + 1;
	}


	void top_or_bottom_3_high(pixel_range_t& range, u32 y, u32 height)
	{
		// top or bottom (3 high)
		assert(y >= 1);
		assert(y <= height - 2);
		range.y_begin = y - 1;
		range.y_end = y + 2;
	}


	void left_or_right_3_wide(pixel_range_t& range, u32 x, u32 width)
	{
		// left or right (3 wide)
		assert(x >= 1);
		assert(x <= width - 2);
		range.x_begin = x - 1;
		range.x_end = x + 2;
	}


	void left_or_right_5_wide(pixel_range_t& range, u32 x, u32 width)
	{
		// left or right (5 wide)
		assert(x >= 2);
		assert(width - x > 2);
		range.x_begin = x - 2;
		range.x_end = x + 3;
	}


	u8 weighted_center(gray::view_t const& view, u32 x, u32 y, std::array<r32, 9> const& weights)
	{
		pixel_range_t range = {};

		top_or_bottom_3_high(range, y, view.height);

		left_or_right_3_wide(range, x, view.width);

		return apply_weights(view, range, weights);
	}


	u8 weighted_center(gray::view_t const& view, u32 x, u32 y, std::array<r32, 25> const& weights)
	{
		pixel_range_t range = {};

		top_or_bottom_3_high(range, y, view.height);

		left_or_right_5_wide(range, x, view.width);

		return apply_weights(view, range, weights);
	}


	u8 weighted_top_left(gray::view_t const& view, u32 x, u32 y, std::array<r32, 4> const& weights)
	{
		pixel_range_t range = {};

		top_2_high(range, y, view.height);

		left_2_wide(range, x, view.width);

		return apply_weights(view, range, weights);
	}


	u8 weighted_top_right(gray::view_t const& view, u32 x, u32 y, std::array<r32, 4> const& weights)
	{
		pixel_range_t range = {};

		top_2_high(range, y, view.height);

		right_2_wide(range, x, view.width);

		return apply_weights(view, range, weights);
	}


	u8 weighted_bottom_left(gray::view_t const& view, u32 x, u32 y, std::array<r32, 4> const& weights)
	{
		pixel_range_t range = {};

		bottom_2_high(range, y, view.width);

		left_2_wide(range, x, view.width);

		return apply_weights(view, range, weights);
	}


	u8 weighted_bottom_right(gray::view_t const& view, u32 x, u32 y, std::array<r32, 4> const& weights)
	{
		pixel_range_t range = {};

		bottom_2_high(range, y, view.width);

		right_2_wide(range, x, view.width);

		return apply_weights(view, range, weights);
	}


	u8 weighted_top(gray::view_t const& view, u32 x, u32 y, std::array<r32, 6> const& weights)
	{
		pixel_range_t range = {};

		top_2_high(range, y, view.height);

		left_or_right_3_wide(range, x, view.width);

		return apply_weights(view, range, weights);
	}


	u8 weighted_bottom(gray::view_t const& view, u32 x, u32 y, std::array<r32, 6> const& weights)
	{
		pixel_range_t range = {};

		bottom_2_high(range, y, view.width);

		left_or_right_3_wide(range, x, view.width);

		return apply_weights(view, range, weights);
	}


	u8 weighted_left(gray::view_t const& view, u32 x, u32 y, std::array<r32, 6> const& weights)
	{
		pixel_range_t range = {};

		top_or_bottom_3_high(range, y, view.height);

		left_2_wide(range, x, view.width);

		return apply_weights(view, range, weights);
	}


	u8 weighted_right(gray::view_t const& view, u32 x, u32 y, std::array<r32, 6> const& weights)
	{
		pixel_range_t range = {};

		top_or_bottom_3_high(range, y, view.height);

		right_2_wide(range, x, view.width);

		return apply_weights(view, range, weights);
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

		return weighted_center(view, x, y, gauss);
	}


	u8 gauss5(gray::view_t const& view, u32 x, u32 y)
	{
		constexpr auto rw = [](u8 w) { return w / 256.0f; };
		constexpr std::array<r32, 25> gauss
		{
			rw(1), rw(4), rw(6), rw(4), rw(1),
			rw(4), rw(16), rw(24), rw(16), rw(4),
			rw(6), rw(24), rw(36), rw(24), rw(6),
			rw(4), rw(16), rw(24), rw(16), rw(4),
			rw(1), rw(4), rw(6), rw(4), rw(1),
		};

		return weighted_center(view, x, y, gauss);
	}
}
