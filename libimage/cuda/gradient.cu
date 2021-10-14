/*

Copyright (c) 2021 Adam Lafontaine

*/
#ifndef LIBIMAGE_NO_GRAYSCALE

#include "verify.hpp"
#include "../proc/verify.hpp"
#include "convolve.cuh"

#include <cassert>
#include <array>
#include <cmath>

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


constexpr u32 GRAD_3X3_SIZE = 9;
constexpr std::array<r32, GRAD_3X3_SIZE> GRAD_X_3X3
{
    1.0f, 0.0f, -1.0f,
    2.0f, 0.0f, -2.0f,
    1.0f, 0.0f, -1.0f,
};
constexpr std::array<r32, 9> GRAD_Y_3X3
{
    1.0f,  2.0f,  1.0f,
    0.0f,  0.0f,  0.0f,
    -1.0f, -2.0f, -1.0f,
};
constexpr u32 GRAD_3X3_BYTES = GRAD_3X3_SIZE * sizeof(r32);


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


    GPU_KERNAL
    static void gpu_edges(u8* src, u8* dst, u32 width, u32 height, u8 threshold, r32* grad_x, r32* grad_y)
    {
        u32 n_elements = width * height;
        u32 i = u32(blockDim.x * blockIdx.x + threadIdx.x);
        if (i >= n_elements)
        {
            return;
        }

        if(is_outer_edge(width, height, i))
        {
            dst[i] = 0;
        }
        else
        {
            auto gx = convolve_3x3(src, width, height, i, grad_x);
            auto gy = convolve_3x3(src, width, height, i, grad_y);
            auto g = static_cast<u8>(std::hypot(gx, gy));
            dst[i] = g < threshold ? 0 : 255;
        }
    }


    GPU_KERNAL
    static void gpu_gradients(u8* src, u8* dst, u32 width, u32 height, r32* grad_x, r32* grad_y)
    {
        u32 n_elements = width * height;
        u32 i = u32(blockDim.x * blockIdx.x + threadIdx.x);
        if (i >= n_elements)
        {
            return;
        }

        if(is_outer_edge(width, height, i))
        {
            dst[i] = 0;
        }
        else
        {
            auto gx = convolve_3x3(src, width, height, i, grad_x);
            auto gy = convolve_3x3(src, width, height, i, grad_y);
            dst[i] = static_cast<u8>(std::hypot(gx, gy));
        }
    }


    


    namespace cuda
    {
        static void blur(gray::image_t const& src, DeviceArray<u8>& d_dst, DeviceBuffer& d_buffer)
        {
            assert(verify(d_buffer));

            u32 n_elements = src.width * src.height;
            int threads_per_block = THREADS_PER_BLOCK;
            int blocks = (n_elements + threads_per_block - 1) / threads_per_block;

            DeviceArray<u8> d_src;
            DeviceArray<r32> d_gauss3x3;
            DeviceArray<r32> d_gauss5x5;

            assert(has_bytes(d_buffer, n_elements * sizeof(u8) * 2));

            push_array(d_dst, d_buffer, n_elements);
            push_array(d_src, d_buffer, n_elements);

            assert(has_bytes(d_buffer, GAUSS_3X3_SIZE + GAUSS_5X5_SIZE));

            push_array(d_gauss3x3, d_buffer, GAUSS_3X3_SIZE);
            push_array(d_gauss5x5, d_buffer, GAUSS_5X5_SIZE);

            copy_to_device(src, d_src);
            copy_to_device(GAUSS_3X3.data(), d_gauss3x3, GAUSS_3X3_BYTES);
            copy_to_device(GAUSS_5X5.data(), d_gauss5x5, GAUSS_5X5_BYTES);

            gpu_blur<<<blocks, threads_per_block>>>(
                d_src.data, 
                d_dst.data, 
                src.width, 
                src.height, 
                d_gauss3x3.data, 
                d_gauss5x5.data);

            pop_array(d_gauss5x5, d_buffer);
            pop_array(d_gauss3x3, d_buffer);
            pop_array(d_src, d_buffer);
        }


        void edges(gray::image_t const& src, gray::image_t const& dst, u8 threshold, DeviceBuffer& d_buffer)
        {
            assert(verify(src, dst));
            assert(verify(d_buffer));
            assert(verify(src, dst, d_buffer)); 

            DeviceArray<u8> d_blur;
                     
            blur(src, d_blur, d_buffer);            

            DeviceArray<u8> d_dst;
            DeviceArray<r32> d_grad_x;
            DeviceArray<r32> d_grad_y;

            u32 n_elements = src.width * src.height;
            int threads_per_block = THREADS_PER_BLOCK;
            int blocks = (n_elements + threads_per_block - 1) / threads_per_block;

            assert(has_bytes(d_buffer, n_elements * sizeof(u8) + 2 * GRAD_3X3_BYTES));

            push_array(d_dst, d_buffer, n_elements);
            push_array(d_grad_x, d_buffer, GRAD_3X3_SIZE);
            push_array(d_grad_y, d_buffer, GRAD_3X3_SIZE);

            copy_to_device(GRAD_X_3X3.data(), d_grad_x, GRAD_3X3_BYTES);
            copy_to_device(GRAD_Y_3X3.data(), d_grad_y, GRAD_3X3_BYTES);

            gpu_edges<<<blocks, threads_per_block>>>(
                d_blur.data,
                d_dst.data,
                src.width,
                src.height,
                threshold,
                d_grad_x.data,
                d_grad_y.data);

            copy_to_host(d_dst, dst);            
            
            pop_array(d_grad_y, d_buffer);
            pop_array(d_grad_x, d_buffer);
            pop_array(d_dst, d_buffer);
            pop_array(d_blur, d_buffer);
        }


		void edges(gray::image_t const& src, gray::view_t const& dst, u8 threshold, DeviceBuffer& d_buffer);

		void edges(gray::view_t const& src, gray::image_t const& dst, u8 threshold, DeviceBuffer& d_buffer);

		void edges(gray::view_t const& src, gray::view_t const& dst, u8 threshold, DeviceBuffer& d_buffer);


		void edges(gray::image_t const& src, gray::image_t const& dst, u8 threshold);

		void edges(gray::image_t const& src, gray::view_t const& dst, u8 threshold);

		void edges(gray::view_t const& src, gray::image_t const& dst, u8 threshold);

		void edges(gray::view_t const& src, gray::view_t const& dst, u8 threshold);


        void gradients(gray::image_t const& src, gray::image_t const& dst, DeviceBuffer& d_buffer)
        {
            assert(verify(src, dst));
            assert(verify(d_buffer));
            assert(verify(src, dst, d_buffer)); 

            DeviceArray<u8> d_blur;
                     
            blur(src, d_blur, d_buffer);            

            DeviceArray<u8> d_dst;
            DeviceArray<r32> d_grad_x;
            DeviceArray<r32> d_grad_y;

            u32 n_elements = src.width * src.height;
            int threads_per_block = THREADS_PER_BLOCK;
            int blocks = (n_elements + threads_per_block - 1) / threads_per_block;

            assert(has_bytes(d_buffer, n_elements * sizeof(u8) + 2 * GRAD_3X3_BYTES));

            push_array(d_dst, d_buffer, n_elements);
            push_array(d_grad_x, d_buffer, GRAD_3X3_SIZE);
            push_array(d_grad_y, d_buffer, GRAD_3X3_SIZE);

            copy_to_device(GRAD_X_3X3.data(), d_grad_x, GRAD_3X3_BYTES);
            copy_to_device(GRAD_Y_3X3.data(), d_grad_y, GRAD_3X3_BYTES);

            gpu_gradients<<<blocks, threads_per_block>>>(
                d_blur.data,
                d_dst.data,
                src.width,
                src.height,
                d_grad_x.data,
                d_grad_y.data);

            copy_to_host(d_dst, dst);            
            
            pop_array(d_grad_y, d_buffer);
            pop_array(d_grad_x, d_buffer);
            pop_array(d_dst, d_buffer);
            pop_array(d_blur, d_buffer);
        }


		void gradients(gray::image_t const& src, gray::view_t const& dst, DeviceBuffer& d_buffer);

		void gradients(gray::view_t const& src, gray::image_t const& dst, DeviceBuffer& d_buffer);

		void gradients(gray::view_t const& src, gray::view_t const& dst, DeviceBuffer& d_buffer);


		void gradients(gray::image_t const& src, gray::image_t const& dst);

		void gradients(gray::image_t const& src, gray::view_t const& dst);

		void gradients(gray::view_t const& src, gray::image_t const& dst);

		void gradients(gray::view_t const& src, gray::view_t const& dst);
    }
}

#endif // !LIBIMAGE_NO_GRAYSCALE