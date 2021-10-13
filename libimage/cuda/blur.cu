/*

Copyright (c) 2021 Adam Lafontaine

*/
#ifndef LIBIMAGE_NO_GRAYSCALE

#include "cuda_def.cuh"
#include "verify.hpp"
#include "../proc/verify.hpp"

#include <cassert>

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
    GPU_KERNAL
    void gpu_blur(u8* src, u8* dst, u32 width, u32 height)
    {
        u32 n_elements = width * height;
        u32 i = u32(blockDim.x * blockIdx.x + threadIdx.x);
        if (i >= n_elements)
        {
            return;
        }

        if(is_outer_edge(width, height, i))
        {
            dst[i] = 0; // src[i];
        }
        else if(is_inner_edge(width, height, i))
        {
            dst[i] = 75;
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

            DeviceArray<u8> d_src;
            DeviceArray<u8> d_dst;

            push_array(d_src, d_buffer, n_elements);
            push_array(d_dst, d_buffer, n_elements);

            copy_to_device(src, d_src);

            int threads_per_block = THREADS_PER_BLOCK;
            int blocks = (n_elements + threads_per_block - 1) / threads_per_block;

            gpu_blur<<<blocks, threads_per_block>>>(d_src.data, d_dst.data, src.width, src.height);

            copy_to_host(d_dst, dst);

            pop_array(d_src, d_buffer);
            pop_array(d_dst, d_buffer);

        }
    }
}

#endif // !LIBIMAGE_NO_GRAYSCALE