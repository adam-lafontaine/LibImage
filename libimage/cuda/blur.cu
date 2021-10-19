/*

Copyright (c) 2021 Adam Lafontaine

*/
#ifndef LIBIMAGE_NO_GRAYSCALE


#include "convolve.cuh"
#include "process.hpp"

#include <cassert>
#include <array>

constexpr int THREADS_PER_BLOCK = 1024;

constexpr r32 d3 = 16.0f;
constexpr u32 GAUSS_3X3_SIZE = 9;
constexpr std::array<r32, GAUSS_3X3_SIZE> GAUSS_3X3
{
    (1/d3), (2/d3), (1/d3),
    (2/d3), (4/d3), (2/d3),
    (1/d3), (2/d3), (1/d3),
};
constexpr u32 GAUSS_3X3_BYTES = GAUSS_3X3_SIZE * sizeof(r32);

constexpr r32 d5 = 256.0f;
constexpr u32 GAUSS_5X5_SIZE = 25;
constexpr std::array<r32, GAUSS_5X5_SIZE> GAUSS_5X5
{
    (1/d5), (4/d5),  (6/d5),  (4/d5),  (1/d5),
    (4/d5), (16/d5), (24/d5), (16/d5), (4/d5),
    (6/d5), (24/d5), (36/d5), (24/d5), (6/d5),
    (4/d5), (16/d5), (24/d5), (16/d5), (4/d5),
    (1/d5), (4/d5),  (6/d5),  (4/d5),  (1/d5),
};
constexpr u32 GAUSS_5X5_BYTES = GAUSS_5X5_SIZE * sizeof(r32);

template<class T, size_t N>
static bool copy_to_device(std::array<T, N> const& src, DeviceArray<T>& dst)
{
    assert(verify(dst));
    return memcpy_to_device(src.data(), dst);
}


namespace libimage
{
    GPU_KERNAL
    static void gpu_blur(u8* src, u8* dst, u32 width, u32 height, r32* g3x3, r32* g5x5)
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
            dst[i] = convolve_3x3(src, width, height, i, g3x3);
        }
        else
        {
            dst[i] = convolve_5x5(src, width, height, i, g5x5);
        }

    }

    namespace cuda
    {
        void blur(gray::image_t const& src, gray::image_t const& dst)
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

            DeviceArray<gray::pixel_t> d_src;
            DeviceArray<gray::pixel_t> d_dst;
            DeviceArray<r32> d_gauss3x3;
            DeviceArray<r32> d_gauss5x5;

            DeviceBuffer<gray::pixel_t> pixel_buffer;
            DeviceBuffer<r32> r32_buffer;

            auto pixel_bytes = 2 * n_elements * sizeof(gray::pixel_t);
            proc = device_malloc(pixel_buffer, pixel_bytes);
            assert(proc);

            auto r32_bytes = GAUSS_5X5_BYTES + GAUSS_3X3_BYTES;
            proc = device_malloc(r32_buffer, r32_bytes);
            assert(proc);

            proc = push_array(d_src, pixel_buffer, n_elements);
            assert(proc);
            proc = push_array(d_dst, pixel_buffer, n_elements);
            assert(proc);

            proc = push_array(d_gauss3x3, r32_buffer, GAUSS_3X3_SIZE);
            assert(proc);
            proc = push_array(d_gauss5x5, r32_buffer, GAUSS_5X5_SIZE);
            assert(proc);

            proc = copy_to_device(src, d_src);
            assert(proc);
            proc = copy_to_device(GAUSS_3X3, d_gauss3x3);
            assert(proc);
            proc = copy_to_device(GAUSS_5X5, d_gauss5x5);
            assert(proc);

            proc = cuda_no_errors();
            assert(proc);

            gpu_blur<<<blocks, threads_per_block>>>(
                d_src.data, 
                d_dst.data, 
                src.width, 
                src.height, 
                d_gauss3x3.data, 
                d_gauss5x5.data);
            
            proc = cuda_launch_success();
            assert(proc);

            proc = copy_to_host(d_dst, dst);
            assert(proc);

            proc = device_free(pixel_buffer);
            assert(proc);
            proc = device_free(r32_buffer);
            assert(proc);
        }


        void blur(gray::image_t const& src, gray::view_t const& dst);

		void blur(gray::view_t const& src, gray::image_t const& dst);

		void blur(gray::view_t const& src, gray::view_t const& dst);
    }
}

#endif // !LIBIMAGE_NO_GRAYSCALE