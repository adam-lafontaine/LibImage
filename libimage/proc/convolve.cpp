#ifndef LIBIMAGE_NO_GRAYSCALE

#include "convolve.hpp"
#include "../libimage.hpp"

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
	static r32 apply_weights(gray::view_t const& view, std::array<r32, N> const& weights)
	{
		assert((size_t)(view.width) * view.height == weights.size());

		u32 w = 0;
		r32 total = 0.0f;

		auto const add_weight = [&](u8 p) 
		{ 
			total += weights[w++] * p; 
		};

		for_each_pixel(view, add_weight);

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


	constexpr r32 D3 = 16.0f;
	constexpr std::array<r32, 9> GAUSS_3X3
	{
		(1 / D3), (2 / D3), (1 / D3),
		(2 / D3), (4 / D3), (2 / D3),
		(1 / D3), (2 / D3), (1 / D3),
	};

	constexpr r32 D5 = 256.0f;
	constexpr std::array<r32, 25> GAUSS_5X5
	{
		(1 / D5), (4 / D5),  (6 / D5),  (4 / D5),  (1 / D5),
		(4 / D5), (16 / D5), (24 / D5), (16 / D5), (4 / D5),
		(6 / D5), (24 / D5), (36 / D5), (24 / D5), (6 / D5),
		(4 / D5), (16 / D5), (24 / D5), (16 / D5), (4 / D5),
		(1 / D5), (4 / D5),  (6 / D5),  (4 / D5),  (1 / D5),
	};


	constexpr std::array<r32, 9> GRAD_X_3X3
	{
		1.0f, 0.0f, -1.0f,
		2.0f, 0.0f, -2.0f,
		1.0f, 0.0f, -1.0f,
	};
	constexpr std::array<r32, 9> GRAD_Y_3X3
	{
		1.0f,  2.0f,  1.0f,
		0.0f,  0.0f,  0.0f,
		-1.0f, -2.0f, -1.0f,
	};


	u8 gauss3(gray::image_t const& img, u32 x, u32 y)
	{
		auto p = weighted_center(img, x, y, GAUSS_3X3);

		assert(p >= 0.0f);
		assert(p <= 255.0f);

		return (u8)(p);
	}


	u8 gauss3(gray::view_t const& view, u32 x, u32 y)
	{
		auto p = weighted_center(view, x, y, GAUSS_3X3);

		assert(p >= 0.0f);
		assert(p <= 255.0f);

		return (u8)(p);
	}


	u8 gauss5(gray::image_t const& img, u32 x, u32 y)
	{
		auto p = weighted_center(img, x, y, GAUSS_5X5);

		assert(p >= 0.0f);
		assert(p <= 255.0f);

		return (u8)(p);
	}


	u8 gauss5(gray::view_t const& view, u32 x, u32 y)
	{		
		auto p = weighted_center(view, x, y, GAUSS_5X5);

		assert(p >= 0.0f);
		assert(p <= 255.0f);

		return (u8)(p);
	}


	r32 x_gradient(gray::image_t const& img, u32 x, u32 y)
	{
		return weighted_center(img, x, y, GRAD_X_3X3);
	}


	r32 x_gradient(gray::view_t const& view, u32 x, u32 y)
	{
		return weighted_center(view, x, y, GRAD_X_3X3);
	}


	r32 y_gradient(gray::image_t const& img, u32 x, u32 y)
	{
		return weighted_center(img, x, y, GRAD_Y_3X3);
	}


	r32 y_gradient(gray::view_t const& view, u32 x, u32 y)
	{
		return weighted_center(view, x, y, GRAD_Y_3X3);
	}


#ifndef LIBIMAGE_NO_SIMD

	

#endif // !LIBIMAGE_NO_SIMD
}

#endif // !LIBIMAGE_NO_GRAYSCALE