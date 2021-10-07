#pragma once

#include "../libimage.hpp"

namespace libimage
{
	u8 gauss3(gray::image_t const& img, u32 x, u32 y);

	u8 gauss3(gray::view_t const& view, u32 x, u32 y);


	u8 gauss5(gray::image_t const& img, u32 x, u32 y);

	u8 gauss5(gray::view_t const& view, u32 x, u32 y);


	r32 x_gradient(gray::image_t const& img, u32 x, u32 y);

	r32 x_gradient(gray::view_t const& view, u32 x, u32 y);


	r32 y_gradient(gray::image_t const& img, u32 x, u32 y);

	r32 y_gradient(gray::view_t const& view, u32 x, u32 y);
}
