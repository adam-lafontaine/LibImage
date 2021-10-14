/*

Copyright (c) 2021 Adam Lafontaine

*/
#ifndef LIBIMAGE_NO_GRAYSCALE

#include "cuda_def.cuh"
#include "verify.hpp"
#include "../proc/verify.hpp"

#include <cassert>
#include <array>

constexpr int THREADS_PER_BLOCK = 1024;


GPU_FUNCTION
bool is_outer_top(u32 width, u32 height, u32 i)
{
    // i = y * width + x
    return i < width;
}


GPU_FUNCTION
bool is_outer_bottom(u32 width, u32 height, u32 i)
{    
    u32 end = height * width;
    u32 begin = end - width;

    return i >= begin && i < end;
}


GPU_FUNCTION
bool is_outer_left(u32 width, u32 height, u32 i)
{
    return i % width == 0;
}


GPU_FUNCTION
bool is_outer_right(u32 width, u32 height, u32 i)
{
    return (i + 1) % width == 0;
}


GPU_FUNCTION
bool is_inner_top(u32 width, u32 height, u32 i)
{
    u32 begin = width + 1;
    u32 end = 2 * width - 1;

    return i >= begin && i < end;
}


GPU_FUNCTION
bool is_inner_bottom(u32 width, u32 height, u32 i)
{
    u32 end = width * (height - 1) - 1;
    u32 begin = end - width + 1;

    return i >= begin && i < end;
}


GPU_FUNCTION
bool is_inner_left(u32 width, u32 height, u32 i)
{
    return i > 0 && (i - 1) % width == 0;
}


GPU_FUNCTION
bool is_inner_right(u32 width, u32 height, u32 i)
{
    return (i + 2) % width == 0;
}


GPU_FUNCTION
bool is_outer_edge(u32 width, u32 height, u32 i)
{
    return is_outer_top(width, height, i) || 
        is_outer_bottom(width, height, i) || 
        is_outer_left(width, height, i) || 
        is_outer_right(width, height, i);
}


GPU_FUNCTION
bool is_inner_edge(u32 width, u32 height, u32 i)
{
    return is_inner_top(width, height, i) || 
        is_inner_bottom(width, height, i) || 
        is_inner_left(width, height, i) || 
        is_inner_right(width, height, i);
}


namespace libimage
{
    GPU_FUNCTION
    void top_or_bottom_3_high(pixel_range_t& range, u32 y, u32 height)
	{
		range.y_begin = y - 1;
		range.y_end = y + 2;
	}


    GPU_FUNCTION
    void left_or_right_3_wide(pixel_range_t& range, u32 x, u32 width)
	{
		range.x_begin = x - 1;
		range.x_end = x + 2;
	}


    GPU_FUNCTION
    r32 apply_weights(u8* data, u32 width, pixel_range_t const& range, r32* weights)
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
    

    GPU_FUNCTION
    u8 gauss3(u8* data, u32 width, u32 height, u32 data_index, r32* g3x3_weights)
    {
        pixel_range_t range = {};
        u32 y = data_index / width;
        u32 x = data_index - y * width;

        top_or_bottom_3_high(range, y, height);
		left_or_right_3_wide(range, x, width);

        auto p = apply_weights(data, width, range, g3x3_weights);

        return static_cast<u8>(p);
    }
}



namespace libimage
{
    GPU_KERNAL
    void gpu_blur(u8* src, u8* dst, u32 width, u32 height, r32* g3x3_weights)
    {
        u32 n_elements = width * height;
        u32 i = u32(blockDim.x * blockIdx.x + threadIdx.x);
        if (i >= n_elements)
        {
            return;
        }

        if(is_outer_edge(width, height, i))
        {
            dst[i] = src[i];
        }
        else if(is_inner_edge(width, height, i))
        {
            dst[i] = gauss3(src, width, height, i, g3x3_weights);
        }
        else
        {
            dst[i] = 255;
        }

    }

    namespace cuda
    {
        void blur(gray::image_t const& src, gray::image_t const& dst, DeviceBuffer& d_buffer)
        {
            assert(verify(src, dst));
            assert(verify(d_buffer));
            assert(verify(src, dst, d_buffer));

            u32 n_elements = src.width * src.height;

            constexpr r32 d = 16.0f;
            constexpr u32 gauss3x3_size = 9;
            std::array<r32, gauss3x3_size> gauss3x3
            {
                (1/d), (2/d), (1/d),
                (2/d), (4/d), (2/d),
                (1/d), (2/d), (1/d),
            };

            DeviceArray<u8> d_src;
            DeviceArray<u8> d_dst;
            DeviceArray<r32> d_gauss3x3;

            push_array(d_src, d_buffer, n_elements);
            push_array(d_dst, d_buffer, n_elements);
            push_array(d_gauss3x3, d_buffer, gauss3x3_size);

            copy_to_device(src, d_src);
            copy_to_device(gauss3x3.data(), d_gauss3x3, gauss3x3_size * sizeof(r32));

            int threads_per_block = THREADS_PER_BLOCK;
            int blocks = (n_elements + threads_per_block - 1) / threads_per_block;            

            gpu_blur<<<blocks, threads_per_block>>>(d_src.data, d_dst.data, src.width, src.height, d_gauss3x3.data);

            copy_to_host(d_dst, dst);

            pop_array(d_src, d_buffer);
            pop_array(d_dst, d_buffer);

        }
    }
}

#endif // !LIBIMAGE_NO_GRAYSCALE