/*

Copyright (c) 2021 Adam Lafontaine

*/
#ifndef LIBIMAGE_NO_COLOR
#ifndef LIBIMAGE_NO_GRAYSCALE

#include "cuda_def.cuh"
#include "verify.hpp"
#include "../proc/verify.hpp"

#include <cassert>

constexpr int THREADS_PER_BLOCK = 1024;


GPU_FUNCTION
static u8 rgb_grayscale_standard(u8 red, u8 green, u8 blue)
{
    return static_cast<u8>(0.299f * red + 0.587f * green + 0.114f * blue);
}


namespace libimage
{
    GPU_KERNAL
    static void gpu_transform_grayscale(pixel_t* src, u8* dst, int n_elements)
    {
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        if (i >= n_elements)
        {
            return;
        }

        u8 r = src[i].red;
        u8 g = src[i].green;
        u8 b = src[i].blue;

        dst[i] = rgb_grayscale_standard(r, g, b);
    }



    namespace cuda
    {

        void transform_grayscale(image_t const& src, gray::image_t const& dst)
        {
            assert(src.data);
            assert(src.width);
            assert(src.height);
            assert(dst.data);
            assert(dst.width == src.width);
            assert(dst.height == src.height);

            u32 n_elements = src.width * src.height;
            int threads_per_block = THREADS_PER_BLOCK;
            int blocks = (n_elements + threads_per_block - 1) / threads_per_block;

            bool proc;

            DeviceArray<pixel_t> d_src;
            DeviceArray<gray::pixel_t> d_dst;

            DeviceBuffer<pixel_t> pixel_buffer;
            DeviceBuffer<gray::pixel_t> gray_buffer;

            auto max_pixel_bytes = n_elements * sizeof(pixel_t);
            proc = device_malloc(pixel_buffer, max_pixel_bytes);
            assert(proc);

            auto max_gray_bytes = n_elements * sizeof(gray::pixel_t);
            proc = device_malloc(gray_buffer, max_gray_bytes);
            assert(proc);

            proc = push_array(d_src, pixel_buffer, n_elements);
            assert(proc);
            proc = push_array(d_dst, gray_buffer, n_elements);
            assert(proc);

            proc = copy_to_device(src, d_src);
            assert(proc);

            proc = cuda_no_errors();
            assert(proc);

            gpu_transform_grayscale<<<blocks, threads_per_block>>>(
                d_src.data, 
                d_dst.data, 
                n_elements);

            proc = cuda_launch_success();
            assert(proc);

            proc = copy_to_host(d_dst, dst);
            assert(proc);

            proc = device_free(pixel_buffer);
            assert(proc);
            proc = device_free(gray_buffer);
            assert(proc);
        }


        void transform_grayscale(image_t const& src, gray::view_t const& dst);

        void transform_grayscale(view_t const& src, gray::image_t const& dst);

        void transform_grayscale(view_t const& src, gray::view_t const& dst);
    }
}

#endif // !LIBIMAGE_NO_GRAYSCALE
#endif // !LIBIMAGE_NO_COLOR	