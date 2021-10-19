/*

Copyright (c) 2021 Adam Lafontaine

*/
#ifndef LIBIMAGE_NO_GRAYSCALE

#include "process.hpp"
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
constexpr std::array<r32, GRAD_3X3_SIZE> GRAD_Y_3X3
{
    1.0f,  2.0f,  1.0f,
    0.0f,  0.0f,  0.0f,
    -1.0f, -2.0f, -1.0f,
};
constexpr u32 GRAD_3X3_BYTES = GRAD_3X3_SIZE * sizeof(r32);


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

/*
    GPU_KERNAL
    static void gpu_white(u8* dst, u32 width, u32 height)
    {
        u32 n_elements = width * height;
        u32 i = u32(blockDim.x * blockIdx.x + threadIdx.x);
        if (i >= n_elements)
        {
            return;
        }

        dst[i] = 255;
    }*/    


    namespace cuda
    {
        


        void edges(gray::image_t const& src, gray::image_t const& dst, u8 threshold)
        {    
            assert(src.data);
            assert(src.width);
            assert(src.height);
            assert(dst.data);
            assert (dst.width == src.width);
            assert(dst.height == src.height);   

            u32 n_elements = src.width * src.height;
            int threads_per_block = THREADS_PER_BLOCK;
            int blocks = (n_elements + threads_per_block - 1) / threads_per_block;

            bool proc;

            DeviceArray<gray::pixel_t> d_blur;
            DeviceArray<r32> d_gauss3x3;
            DeviceArray<r32> d_gauss5x5;

            DeviceArray<gray::pixel_t> d_edges;
            DeviceArray<r32> d_grad_x;
            DeviceArray<r32> d_grad_y;

            DeviceArray<gray::pixel_t> d_src;

            DeviceBuffer<gray::pixel_t> pixel_buffer;
            DeviceBuffer<r32> r32_buffer;

            auto image_bytes = n_elements * sizeof(gray::pixel_t);

            auto max_pixel_bytes = 2 * image_bytes + GAUSS_3X3_BYTES + GAUSS_5X5_BYTES;
            proc = device_malloc(pixel_buffer, max_pixel_bytes);
            assert(proc);

            auto max_r32_bytes =  GAUSS_3X3_BYTES + GAUSS_5X5_BYTES;
            proc = device_malloc(r32_buffer, max_r32_bytes);
            assert(proc);

            proc = push_array(d_blur, pixel_buffer, n_elements);
            assert(proc);
            proc = push_array(d_src, pixel_buffer, n_elements);
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
                d_blur.data, 
                src.width, 
                src.height, 
                d_gauss3x3.data, 
                d_gauss5x5.data);

            proc = cuda_launch_success();
            assert(proc);

            pop_array(d_gauss5x5, r32_buffer);
            pop_array(d_gauss3x3, r32_buffer);
            pop_array(d_src, pixel_buffer);

            proc = push_array(d_edges, pixel_buffer, n_elements);
            assert(proc);

            proc = push_array(d_grad_x, r32_buffer, GRAD_3X3_SIZE);
            assert(proc);

            proc = push_array(d_grad_y, r32_buffer, GRAD_3X3_SIZE);
            assert(proc);

            proc = copy_to_device(GRAD_X_3X3, d_grad_x);
            assert(proc);

            proc = copy_to_device(GRAD_Y_3X3, d_grad_y);
            assert(proc);

            proc = cuda_no_errors();
            assert(proc);

            gpu_edges<<<blocks, threads_per_block>>>(
                d_blur.data,
                d_edges.data,
                src.width,
                src.height,
                threshold,
                d_grad_x.data,
                d_grad_y.data);

            proc = cuda_launch_success();
            assert(proc);

            proc = copy_to_host(d_edges, dst);
            assert(proc);

            proc = device_free(r32_buffer);
            assert(proc);
            proc = device_free(pixel_buffer);
            assert(proc);
        }


		void edges(gray::image_t const& src, gray::view_t const& dst, u8 threshold);

		void edges(gray::view_t const& src, gray::image_t const& dst, u8 threshold);

		void edges(gray::view_t const& src, gray::view_t const& dst, u8 threshold);


		void gradients(gray::image_t const& src, gray::image_t const& dst)
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

            DeviceArray<gray::pixel_t> d_blur;
            DeviceArray<r32> d_gauss3x3;
            DeviceArray<r32> d_gauss5x5;

            DeviceArray<gray::pixel_t> d_gradients;
            DeviceArray<r32> d_grad_x;
            DeviceArray<r32> d_grad_y;

            DeviceArray<gray::pixel_t> d_src;

            DeviceBuffer<gray::pixel_t> pixel_buffer;
            DeviceBuffer<r32> r32_buffer;

            auto image_bytes = n_elements * sizeof(gray::pixel_t);

            auto max_pixel_bytes = 2 * image_bytes + GAUSS_3X3_BYTES + GAUSS_5X5_BYTES;
            proc = device_malloc(pixel_buffer, max_pixel_bytes);
            assert(proc);

            auto max_r32_bytes =  GAUSS_3X3_BYTES + GAUSS_5X5_BYTES;
            proc = device_malloc(r32_buffer, max_r32_bytes);
            assert(proc);


            proc = push_array(d_blur, pixel_buffer, n_elements);
            assert(proc);            

            proc = push_array(d_src, pixel_buffer, n_elements);
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
                d_blur.data, 
                src.width, 
                src.height, 
                d_gauss3x3.data, 
                d_gauss5x5.data);

            proc = cuda_launch_success();
            assert(proc);

            pop_array(d_gauss5x5, r32_buffer);
            pop_array(d_gauss3x3, r32_buffer);
            pop_array(d_src, pixel_buffer);

            proc = push_array(d_gradients, pixel_buffer, n_elements);
            assert(proc);

            proc = push_array(d_grad_x, r32_buffer, GRAD_3X3_SIZE);
            assert(proc);

            proc = push_array(d_grad_y, r32_buffer, GRAD_3X3_SIZE);
            assert(proc);

            proc = copy_to_device(GRAD_X_3X3, d_grad_x);
            assert(proc);

            proc = copy_to_device(GRAD_Y_3X3, d_grad_y);
            assert(proc);

            proc = cuda_no_errors();
            assert(proc);

            gpu_gradients<<<blocks, threads_per_block>>>(
                d_blur.data,
                d_gradients.data,
                src.width,
                src.height,
                d_grad_x.data,
                d_grad_y.data);

            proc = cuda_launch_success();
            assert(proc);

            proc = copy_to_host(d_gradients, dst);
            assert(proc);

            proc = device_free(r32_buffer);
            assert(proc);
            proc = device_free(pixel_buffer);
            assert(proc);
        }


		void gradients(gray::image_t const& src, gray::view_t const& dst);

		void gradients(gray::view_t const& src, gray::image_t const& dst);

		void gradients(gray::view_t const& src, gray::view_t const& dst);
    }
}

#endif // !LIBIMAGE_NO_GRAYSCALE