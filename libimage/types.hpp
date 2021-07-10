#pragma once

#include <cstdint>

using u8 = uint8_t;
using u32 = uint32_t;
using u64 = uint64_t;
using r32 = float;
using r64 = double;

namespace libimage
{
	// region of interest in an image
	typedef struct
	{
		u32 x_begin;
		u32 x_end;   // one past last x
		u32 y_begin;
		u32 y_end;   // one past last y

	} pixel_range_t;


	typedef struct
	{
		u32 x;
		u32 y;

	} xy_loc_t;
}
