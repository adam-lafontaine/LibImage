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
u8 rgb_grayscale_standard(u8 red, u8 green, u8 blue)
{
    return static_cast<u8>(0.299f * red + 0.587f * green + 0.114f * blue);
}


namespace libimage
{
    GPU_KERNAL
    void gpu_transform_grayscale(pixel_t* src, u8* dst, int n_elements)
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

        void transform_grayscale(image_t const& src, gray::image_t const& dst, DeviceBuffer& d_buffer)
        {
            assert(verify(src, dst));
            assert(verify(d_buffer));
            assert(verify(src, dst, d_buffer));

            u32 n_elements = src.width * src.height;

            DeviceArray<pixel_t> d_src;
            DeviceArray<u8> d_dst;

            push_array(d_src, d_buffer, n_elements);
            push_array(d_dst, d_buffer, n_elements);

            copy_to_device(src, d_src);

            int threads_per_block = THREADS_PER_BLOCK;
            int blocks = (n_elements + threads_per_block - 1) / threads_per_block;

            gpu_transform_grayscale<<<blocks, threads_per_block>>>(d_src.data, d_dst.data, n_elements);

            copy_to_host(d_dst, dst);

            pop_array(d_src, d_buffer);
            pop_array(d_dst, d_buffer);
        }


        void transform_grayscale(image_t const& src, gray::view_t const& dst, DeviceBuffer& d_buffer)
        {
            assert(verify(src, dst));
            assert(verify(d_buffer));
            assert(verify(src, dst, d_buffer));

            u32 n_elements = src.width * src.height;

            DeviceArray<pixel_t> d_src;
            DeviceArray<u8> d_dst;

            push_array(d_src, d_buffer, n_elements);
            push_array(d_dst, d_buffer, n_elements);

            copy_to_device(src, d_src);

            int threads_per_block = THREADS_PER_BLOCK;
            int blocks = (n_elements + threads_per_block - 1) / threads_per_block;

            gpu_transform_grayscale<<<blocks, threads_per_block>>>(d_src.data, d_dst.data, n_elements);

            copy_to_host(d_dst, dst);

            pop_array(d_src, d_buffer);
            pop_array(d_dst, d_buffer);
        }


        void transform_grayscale(view_t const& src, gray::image_t const& dst, DeviceBuffer& d_buffer)
        {
            assert(verify(src, dst));
            assert(verify(d_buffer));
            assert(verify(src, dst, d_buffer));

            u32 n_elements = src.width * src.height;

            DeviceArray<pixel_t> d_src;
            DeviceArray<u8> d_dst;

            push_array(d_src, d_buffer, n_elements);
            push_array(d_dst, d_buffer, n_elements);

            copy_to_device(src, d_src);

            int threads_per_block = THREADS_PER_BLOCK;
            int blocks = (n_elements + threads_per_block - 1) / threads_per_block;

            gpu_transform_grayscale<<<blocks, threads_per_block>>>(d_src.data, d_dst.data, n_elements);

            copy_to_host(d_dst, dst);

            pop_array(d_src, d_buffer);
            pop_array(d_dst, d_buffer);
        }


        void transform_grayscale(view_t const& src, gray::view_t const& dst, DeviceBuffer& d_buffer)
        {
            assert(verify(src, dst));
            assert(verify(d_buffer));
            assert(verify(src, dst, d_buffer));

            u32 n_elements = src.width * src.height;

            DeviceArray<pixel_t> d_src;
            DeviceArray<u8> d_dst;

            push_array(d_src, d_buffer, n_elements);
            push_array(d_dst, d_buffer, n_elements);

            copy_to_device(src, d_src);

            int threads_per_block = THREADS_PER_BLOCK;
            int blocks = (n_elements + threads_per_block - 1) / threads_per_block;

            gpu_transform_grayscale<<<blocks, threads_per_block>>>(d_src.data, d_dst.data, n_elements);

            copy_to_host(d_dst, dst);

            pop_array(d_src, d_buffer);
            pop_array(d_dst, d_buffer);
        }


        void transform_grayscale(image_t const& src, gray::image_t const& dst)
        {
            assert(verify(src, dst));

            DeviceBuffer d_buffer;
            device_malloc(d_buffer, bytes(src) + bytes(dst));

            transform_grayscale(src, dst, d_buffer);

            device_free(d_buffer);
        }


        void transform_grayscale(image_t const& src, gray::view_t const& dst)
        {
            assert(verify(src, dst));

            DeviceBuffer d_buffer;
            device_malloc(d_buffer, bytes(src) + bytes(dst));

            transform_grayscale(src, dst, d_buffer);

            device_free(d_buffer);
        }


        void transform_grayscale(view_t const& src, gray::image_t const& dst)
        {
            assert(verify(src, dst));

            DeviceBuffer d_buffer;
            device_malloc(d_buffer, bytes(src) + bytes(dst));

            transform_grayscale(src, dst, d_buffer);

            device_free(d_buffer);
        }


        void transform_grayscale(view_t const& src, gray::view_t const& dst)
        {
            assert(verify(src, dst));

            DeviceBuffer d_buffer;
            device_malloc(d_buffer, bytes(src) + bytes(dst));

            transform_grayscale(src, dst, d_buffer);

            device_free(d_buffer);
        }
    }
}

#endif // !LIBIMAGE_NO_GRAYSCALE
#endif // !LIBIMAGE_NO_COLOR	