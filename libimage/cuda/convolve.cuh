#pragma once
/*

Copyright (c) 2021 Adam Lafontaine

*/
#include "cuda_def.cuh"
#include "../types.hpp"

GPU_FUNCTION
inline bool is_outer_top(u32 width, u32 height, u32 i)
{
    // i = y * width + x
    return i < width;
}


GPU_FUNCTION
inline bool is_outer_bottom(u32 width, u32 height, u32 i)
{    
    u32 end = height * width;
    u32 begin = end - width;

    return i >= begin && i < end;
}


GPU_FUNCTION
inline bool is_outer_left(u32 width, u32 height, u32 i)
{
    return i % width == 0;
}


GPU_FUNCTION
inline bool is_outer_right(u32 width, u32 height, u32 i)
{
    return (i + 1) % width == 0;
}


GPU_FUNCTION
inline bool is_inner_top(u32 width, u32 height, u32 i)
{
    u32 begin = width + 1;
    u32 end = 2 * width - 1;

    return i >= begin && i < end;
}


GPU_FUNCTION
inline bool is_inner_bottom(u32 width, u32 height, u32 i)
{
    u32 end = width * (height - 1) - 1;
    u32 begin = end - width + 1;

    return i >= begin && i < end;
}


GPU_FUNCTION
inline bool is_inner_left(u32 width, u32 height, u32 i)
{
    return i > 0 && (i - 1) % width == 0;
}


GPU_FUNCTION
inline bool is_inner_right(u32 width, u32 height, u32 i)
{
    return (i + 2) % width == 0;
}


GPU_FUNCTION
inline bool is_outer_edge(u32 width, u32 height, u32 i)
{
    return is_outer_top(width, height, i) || 
        is_outer_bottom(width, height, i) || 
        is_outer_left(width, height, i) || 
        is_outer_right(width, height, i);
}


GPU_FUNCTION
inline bool is_inner_edge(u32 width, u32 height, u32 i)
{
    return is_inner_top(width, height, i) || 
        is_inner_bottom(width, height, i) || 
        is_inner_left(width, height, i) || 
        is_inner_right(width, height, i);
}


namespace libimage
{
    GPU_FUNCTION
    inline void top_or_bottom_3_high(pixel_range_t& range, u32 y, u32 height)
	{
		range.y_begin = y - 1;
		range.y_end = y + 2;
	}


    GPU_FUNCTION
    inline void left_or_right_3_wide(pixel_range_t& range, u32 x, u32 width)
	{
		range.x_begin = x - 1;
		range.x_end = x + 2;
	}


    GPU_FUNCTION
    inline void top_or_bottom_5_high(pixel_range_t& range, u32 y, u32 height)
	{
		range.y_begin = y - 2;
		range.y_end = y + 3;
	}


    GPU_FUNCTION
	inline void left_or_right_5_wide(pixel_range_t& range, u32 x, u32 width)
	{
		range.x_begin = x - 2;
		range.x_end = x + 3;
	}


    GPU_FUNCTION
    inline r32 apply_weights(u8* data, u32 width, pixel_range_t const& range, r32* weights)
    {
        u32 w = 0;
        r32 total = 0.0f;
        for(u32 y = range.y_begin; y < range.y_end; ++y)
        {
            u8* p = data + y * width;
            for(u32 x = range.x_begin; x < range.x_end; ++x)
            {
                total += p[x] * weights[w++];
            }
        }

        return total;
    }
    

    
}