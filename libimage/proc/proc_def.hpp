#pragma once
#include "../libimage.hpp"

#include <functional>

namespace libimage
{
	using u8_to_bool_f = std::function<bool(u8)>;

	using pixel_to_pixel_f = std::function<pixel_t(pixel_t const&)>;

	using pixel_to_u8_f = std::function<u8(pixel_t const& p)>;

	using u8_to_u8_f = std::function<u8(u8)>;

	using lookup_table_t = std::array<u8, 256>;
}


