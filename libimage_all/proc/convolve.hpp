#pragma once

#include "../libimage.hpp"

#include <array>

namespace libimage
{
	u8 apply_weights(gray::view_t const& view, pixel_range_t const& range, std::array<r32, 9> weights)
	{
		assert((range.x_end - range.x_begin) * (range.y_end - range.y_begin) == 9);
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


	
	u8 apply_weights(gray::view_t const& view, pixel_range_t const& range, std::array<r32, 25> weights)
	{
		assert((range.x_end - range.x_begin) * (range.y_end - range.y_begin) == 25);
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


	u8 weighted_center(gray::view_t const& view, u32 x, u32 y, std::array<r32, 9> const& weights)
	{
		assert(x < view.width);
		assert(x > 0);
		assert(view.width - x > 1);
		assert(y < view.height);
		assert(y > 0);
		assert(view.height - y > 1);

		pixel_range_t range = {};
		range.y_begin = y - 1;
		range.y_end = y + 2;
		range.x_begin = x - 1;
		range.x_end = x + 2;

		return apply_weights(view, range, weights);
	}


	u8 weighted_center(gray::view_t const& view, u32 x, u32 y, std::array<r32, 25> const& weights)
	{
		assert(x < view.width);
		assert(x > 1);
		assert(view.width - x > 2);
		assert(y < view.height);
		assert(y > 1);
		assert(view.height - y > 2);

		pixel_range_t range = {};
		range.y_begin = y - 2;
		range.y_end = y + 3;
		range.x_begin = x - 2;
		range.x_end = x + 3;

		return apply_weights(view, range, weights);
	}


	/*u8 weighted_top_left(gray::view_t const& view, u32 x, u32 y, std::array<r32, 4> const& weights)
	{

	}*/


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
