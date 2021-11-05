#pragma once

#ifndef LIBIMAGE_NO_GRAYSCALE

#include "../gray.hpp"

#include <functional>

namespace libimage
{	
	// only used in blur.cpp
	u8 gauss3(gray::image_t const& img, u32 x, u32 y);

	u8 gauss3(gray::view_t const& view, u32 x, u32 y);


	u8 gauss5(gray::image_t const& img, u32 x, u32 y);

	u8 gauss5(gray::view_t const& view, u32 x, u32 y);


	// only used in gradient.cpp
	r32 x_gradient(gray::image_t const& img, u32 x, u32 y);

	r32 x_gradient(gray::view_t const& view, u32 x, u32 y);


	r32 y_gradient(gray::image_t const& img, u32 x, u32 y);

	r32 y_gradient(gray::view_t const& view, u32 x, u32 y);

	namespace simd
	{
		using u8_to_bool_f = std::function<bool(u8)>;


		void inner_gauss(gray::image_t const& src, gray::image_t const& dst);

		void inner_gauss(gray::image_t const& src, gray::view_t const& dst);

		void inner_gauss(gray::view_t const& src, gray::image_t const& dst);

		void inner_gauss(gray::view_t const& src, gray::view_t const& dst);


		void inner_gradients(gray::image_t const& src, gray::image_t const& dst);

		void inner_gradients(gray::image_t const& src, gray::view_t const& dst);

		void inner_gradients(gray::view_t const& src, gray::image_t const& dst);

		void inner_gradients(gray::view_t const& src, gray::view_t const& dst);


		void inner_edges(gray::image_t const& src, gray::image_t const& dst, u8_to_bool_f const& cond);

		void inner_edges(gray::image_t const& src, gray::view_t const& dst, u8_to_bool_f const& cond);

		void inner_edges(gray::view_t const& src, gray::image_t const& dst, u8_to_bool_f const& cond);

		void inner_edges(gray::view_t const& src, gray::view_t const& dst, u8_to_bool_f const& cond);
	}
}

#endif
