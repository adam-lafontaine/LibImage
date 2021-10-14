/*

Copyright (c) 2021 Adam Lafontaine

*/
#ifndef LIBIMAGE_NO_GRAYSCALE


#include "verify.hpp"
#include "../proc/verify.hpp"
#include "convolve.cuh"

#include <cassert>
#include <array>

constexpr int THREADS_PER_BLOCK = 1024;

constexpr r32 d3 = 16.0f;
constexpr u32 GAUSS_3X3_SIZE = 9;
constexpr std::array<r32, GAUSS_3X3_SIZE> gauss3x3
{
    (1/d3), (2/d3), (1/d3),
    (2/d3), (4/d3), (2/d3),
    (1/d3), (2/d3), (1/d3),
};
constexpr u32 GAUSS_3X3_BYTES = GAUSS_3X3_SIZE * sizeof(r32);

constexpr r32 d5 = 256.0f;
constexpr u32 GAUSS_5X5_SIZE = 25;
constexpr std::array<r32, GAUSS_5X5_SIZE> gauss5x5
{
    (1/d5), (4/d5),  (6/d5),  (4/d5),  (1/d5),
    (4/d5), (16/d5), (24/d5), (16/d5), (4/d5),
    (6/d5), (24/d5), (36/d5), (24/d5), (6/d5),
    (4/d5), (16/d5), (24/d5), (16/d5), (4/d5),
    (1/d5), (4/d5),  (6/d5),  (4/d5),  (1/d5),
};
constexpr u32 GAUSS_5X5_BYTES = GAUSS_5X5_SIZE * sizeof(r32);


namespace libimage
{
    GPU_FUNCTION
    inline u8 gauss3(u8* data, u32 width, u32 height, u32 data_index, r32* g3x3_weights)
    {
        pixel_range_t range = {};
        u32 y = data_index / width;
        u32 x = data_index - y * width;

        top_or_bottom_3_high(range, y, height);
		left_or_right_3_wide(range, x, width);

        auto p = apply_weights(data, width, range, g3x3_weights);

        return static_cast<u8>(p);
    }


    GPU_FUNCTION
    inline u8 gauss5(u8* data, u32 width, u32 height, u32 data_index, r32* g5x5_weights)
    {
        pixel_range_t range = {};
        u32 y = data_index / width;
        u32 x = data_index - y * width;

        top_or_bottom_5_high(range, y, height);
		left_or_right_5_wide(range, x, width);

        auto p = apply_weights(data, width, range, g5x5_weights);

        return static_cast<u8>(p);
    }


    GPU_KERNAL
    void gpu_blur(u8* src, u8* dst, u32 width, u32 height, r32* g3x3, r32* g5x5)
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
            dst[i] = gauss3(src, width, height, i, g3x3);
        }
        else
        {
            dst[i] = gauss5(src, width, height, i, g5x5);
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
            DeviceArray<r32> d_gauss3x3;
            DeviceArray<r32> d_gauss5x5;

            push_array(d_src, d_buffer, n_elements);
            push_array(d_dst, d_buffer, n_elements);

            assert(d_buffer.total_bytes - d_buffer.offset >= GAUSS_3X3_SIZE + GAUSS_5X5_SIZE);

            push_array(d_gauss3x3, d_buffer, GAUSS_3X3_SIZE);
            push_array(d_gauss5x5, d_buffer, GAUSS_5X5_SIZE);

            copy_to_device(src, d_src);
            copy_to_device(gauss3x3.data(), d_gauss3x3, GAUSS_3X3_BYTES);
            copy_to_device(gauss5x5.data(), d_gauss5x5, GAUSS_5X5_BYTES);

            int threads_per_block = THREADS_PER_BLOCK;
            int blocks = (n_elements + threads_per_block - 1) / threads_per_block;            

            gpu_blur<<<blocks, threads_per_block>>>(d_src.data, d_dst.data, src.width, src.height, d_gauss3x3.data, d_gauss5x5.data);

            copy_to_host(d_dst, dst);

            pop_array(d_gauss5x5, d_buffer);
            pop_array(d_gauss3x3, d_buffer);
            pop_array(d_src, d_buffer);
            pop_array(d_dst, d_buffer);
        }


        void blur(gray::image_t const& src, gray::view_t const& dst, DeviceBuffer& d_buffer)
        {
            assert(verify(src, dst));
            assert(verify(d_buffer));
            assert(verify(src, dst, d_buffer));

            u32 n_elements = src.width * src.height;

            DeviceArray<u8> d_src;
            DeviceArray<u8> d_dst;
            DeviceArray<r32> d_gauss3x3;
            DeviceArray<r32> d_gauss5x5;

            push_array(d_src, d_buffer, n_elements);
            push_array(d_dst, d_buffer, n_elements);

            assert(d_buffer.total_bytes - d_buffer.offset >= GAUSS_3X3_SIZE + GAUSS_5X5_SIZE);

            push_array(d_gauss3x3, d_buffer, GAUSS_3X3_SIZE);
            push_array(d_gauss5x5, d_buffer, GAUSS_5X5_SIZE);

            copy_to_device(src, d_src);
            copy_to_device(gauss3x3.data(), d_gauss3x3, GAUSS_3X3_BYTES);
            copy_to_device(gauss5x5.data(), d_gauss5x5, GAUSS_5X5_BYTES);

            int threads_per_block = THREADS_PER_BLOCK;
            int blocks = (n_elements + threads_per_block - 1) / threads_per_block;            

            gpu_blur<<<blocks, threads_per_block>>>(d_src.data, d_dst.data, src.width, src.height, d_gauss3x3.data, d_gauss5x5.data);

            copy_to_host(d_dst, dst);

            pop_array(d_gauss5x5, d_buffer);
            pop_array(d_gauss3x3, d_buffer);
            pop_array(d_src, d_buffer);
            pop_array(d_dst, d_buffer);
        }


		void blur(gray::view_t const& src, gray::image_t const& dst, DeviceBuffer& d_buffer)
        {
            assert(verify(src, dst));
            assert(verify(d_buffer));
            assert(verify(src, dst, d_buffer));

            u32 n_elements = src.width * src.height;

            DeviceArray<u8> d_src;
            DeviceArray<u8> d_dst;
            DeviceArray<r32> d_gauss3x3;
            DeviceArray<r32> d_gauss5x5;

            push_array(d_src, d_buffer, n_elements);
            push_array(d_dst, d_buffer, n_elements);

            assert(d_buffer.total_bytes - d_buffer.offset >= GAUSS_3X3_SIZE + GAUSS_5X5_SIZE);

            push_array(d_gauss3x3, d_buffer, GAUSS_3X3_SIZE);
            push_array(d_gauss5x5, d_buffer, GAUSS_5X5_SIZE);

            copy_to_device(src, d_src);
            copy_to_device(gauss3x3.data(), d_gauss3x3, GAUSS_3X3_BYTES);
            copy_to_device(gauss5x5.data(), d_gauss5x5, GAUSS_5X5_BYTES);

            int threads_per_block = THREADS_PER_BLOCK;
            int blocks = (n_elements + threads_per_block - 1) / threads_per_block;            

            gpu_blur<<<blocks, threads_per_block>>>(d_src.data, d_dst.data, src.width, src.height, d_gauss3x3.data, d_gauss5x5.data);

            copy_to_host(d_dst, dst);

            pop_array(d_gauss5x5, d_buffer);
            pop_array(d_gauss3x3, d_buffer);
            pop_array(d_src, d_buffer);
            pop_array(d_dst, d_buffer);
        }


		void blur(gray::view_t const& src, gray::view_t const& dst, DeviceBuffer& d_buffer)
        {
            assert(verify(src, dst));
            assert(verify(d_buffer));
            assert(verify(src, dst, d_buffer));

            u32 n_elements = src.width * src.height;

            DeviceArray<u8> d_src;
            DeviceArray<u8> d_dst;
            DeviceArray<r32> d_gauss3x3;
            DeviceArray<r32> d_gauss5x5;

            push_array(d_src, d_buffer, n_elements);
            push_array(d_dst, d_buffer, n_elements);

            assert(d_buffer.total_bytes - d_buffer.offset >= GAUSS_3X3_SIZE + GAUSS_5X5_SIZE);

            push_array(d_gauss3x3, d_buffer, GAUSS_3X3_SIZE);
            push_array(d_gauss5x5, d_buffer, GAUSS_5X5_SIZE);

            copy_to_device(src, d_src);
            copy_to_device(gauss3x3.data(), d_gauss3x3, GAUSS_3X3_BYTES);
            copy_to_device(gauss5x5.data(), d_gauss5x5, GAUSS_5X5_BYTES);

            int threads_per_block = THREADS_PER_BLOCK;
            int blocks = (n_elements + threads_per_block - 1) / threads_per_block;            

            gpu_blur<<<blocks, threads_per_block>>>(d_src.data, d_dst.data, src.width, src.height, d_gauss3x3.data, d_gauss5x5.data);

            copy_to_host(d_dst, dst);

            pop_array(d_gauss5x5, d_buffer);
            pop_array(d_gauss3x3, d_buffer);
            pop_array(d_src, d_buffer);
            pop_array(d_dst, d_buffer);
        }


        void blur(gray::image_t const& src, gray::image_t const& dst)
        {
            assert(verify(src, dst));

            DeviceBuffer d_buffer;
            u32 bytes_needed = bytes(src) + bytes(dst) + GAUSS_3X3_BYTES + GAUSS_5X5_BYTES;
            device_malloc(d_buffer, bytes_needed);

            blur(src, dst, d_buffer);

            device_free(d_buffer);
        }


		void blur(gray::image_t const& src, gray::view_t const& dst)
        {
            assert(verify(src, dst));

            DeviceBuffer d_buffer;
            u32 bytes_needed = bytes(src) + bytes(dst) + GAUSS_3X3_BYTES + GAUSS_5X5_BYTES;
            device_malloc(d_buffer, bytes_needed);

            blur(src, dst, d_buffer);

            device_free(d_buffer);
        }


		void blur(gray::view_t const& src, gray::image_t const& dst)
        {
            assert(verify(src, dst));

            DeviceBuffer d_buffer;
            u32 bytes_needed = bytes(src) + bytes(dst) + GAUSS_3X3_BYTES + GAUSS_5X5_BYTES;
            device_malloc(d_buffer, bytes_needed);

            blur(src, dst, d_buffer);

            device_free(d_buffer);
        }


		void blur(gray::view_t const& src, gray::view_t const& dst)
        {
            assert(verify(src, dst));

            DeviceBuffer d_buffer;
            u32 bytes_needed = bytes(src) + bytes(dst) + GAUSS_3X3_BYTES + GAUSS_5X5_BYTES;
            device_malloc(d_buffer, bytes_needed);

            blur(src, dst, d_buffer);

            device_free(d_buffer);
        }
    }
}

#endif // !LIBIMAGE_NO_GRAYSCALE