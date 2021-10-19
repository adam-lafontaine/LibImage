/*

Copyright (c) 2021 Adam Lafontaine

*/
#ifndef LIBIMAGE_NO_COLOR
#ifndef LIBIMAGE_NO_GRAYSCALE

#include "cuda_def.cuh"
#include "process.hpp"

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

        bool transform_grayscale(DeviceArray<pixel_t> const& src, DeviceArray<gray::pixel_t> const& dst)
        {
            assert(src.data);
            assert(dst.data);
            assert(dst.n_elements == src.n_elements);

            int threads_per_block = THREADS_PER_BLOCK;
            int blocks = (src.n_elements + threads_per_block - 1) / threads_per_block;

            bool proc;

            proc = cuda_no_errors();
            assert(proc); if(!proc) { return false; }

            gpu_transform_grayscale<<<blocks, threads_per_block>>>(
                src.data, 
                dst.data, 
                src.n_elements);

            proc = cuda_launch_success();
            assert(proc); if(!proc) { return false; }

            return true;
        }


        bool transform_grayscale(image_t const& src, gray::image_t const& dst, DeviceBuffer<pixel_t>& c_buffer, DeviceBuffer<gray::pixel_t>& g_buffer)
        {
            assert(src.data);
            assert(src.width);
            assert(src.height);
            assert(dst.data);
            assert(dst.width == src.width);
            assert(dst.height == src.height);
            assert(c_buffer.data);
            assert(c_buffer.total_bytes - c_buffer.offset >= src.width * src.height * sizeof(pixel_t));
            assert(g_buffer.data);
            assert(g_buffer.total_bytes - g_buffer.offset >= dst.width * dst.height * sizeof(gray::pixel_t));

            u32 n_elements = src.width * src.height;

            bool proc;

            DeviceArray<pixel_t> d_src;
            DeviceArray<gray::pixel_t> d_dst;

            proc = push_array(d_src, c_buffer, n_elements);
            assert(proc); if(!proc) { return false; }
            
            proc = push_array(d_dst, g_buffer, n_elements);
            assert(proc); if(!proc) { return false; }

            proc = copy_to_device(src, d_src);
            assert(proc); if(!proc) { return false; }

            proc = transform_grayscale(d_src, d_dst);
            assert(proc); if(!proc) { return false; }

            proc = copy_to_host(d_dst, dst);
            assert(proc); if(!proc) { return false; }

            pop_array(d_dst, g_buffer);
            pop_array(d_src, c_buffer);

            return true;
        }


        void transform_grayscale(image_t const& src, gray::image_t const& dst)
        {
            assert(src.data);
            assert(src.width);
            assert(src.height);
            assert(dst.data);
            assert(dst.width == src.width);
            assert(dst.height == src.height);

            u32 n_elements = src.width * src.height;

            bool proc;

            DeviceBuffer<pixel_t> pixel_buffer;
            DeviceBuffer<gray::pixel_t> gray_buffer;

            auto max_pixel_bytes = n_elements * sizeof(pixel_t);
            proc = device_malloc(pixel_buffer, max_pixel_bytes);
            assert(proc);

            auto max_gray_bytes = n_elements * sizeof(gray::pixel_t);
            proc = device_malloc(gray_buffer, max_gray_bytes);
            assert(proc);

            proc = transform_grayscale(src, dst, pixel_buffer, gray_buffer);
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