/*

Copyright (c) 2021 Adam Lafontaine

*/
#ifndef LIBIMAGE_NO_COLOR

#include "cuda_def.cuh"
#include "process.hpp"

#include <cassert>

constexpr int THREADS_PER_BLOCK = 1024;


GPU_FUNCTION
static u8 blend_linear(u8 s, u8 c, r32 a)
{
    
    auto const sf = static_cast<r32>(s);
    auto const cf = static_cast<r32>(c);

    auto blended = a * cf + (1.0f - a) * sf;

    return static_cast<u8>(blended);
}

namespace libimage
{
    GPU_KERNAL
    static void gpu_alpha_blend_linear(pixel_t* src, pixel_t* current, pixel_t* dst, int n_elements)
    {
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        if (i >= n_elements)
        {
            return;
        }

        auto const a = static_cast<r32>(src[i].alpha) / 255.0f;
        auto red = blend_linear(src[i].red, current[i].red, a);
		auto green = blend_linear(src[i].green, current[i].green, a);
		auto blue = blend_linear(src[i].blue, current[i].blue, a);

        dst[i] = { red, green, blue, 255 };
    }


    bool alpha_blend(DeviceArray<pixel_t> const& src, DeviceArray<pixel_t> const& current, DeviceArray<pixel_t> const& dst)
    {
        assert(src.data);
        assert(current.data);
        assert(dst.data);
        assert(current.n_elements == src.n_elements);
        assert(dst.n_elements == src.n_elements);

        int threads_per_block = THREADS_PER_BLOCK;
        int blocks = (src.n_elements + threads_per_block - 1) / threads_per_block;

        bool proc;

        proc = cuda_no_errors();
        assert(proc); if(!proc) { return false; }

        gpu_alpha_blend_linear<<<blocks, threads_per_block>>>(
            src.data, 
            current.data, 
            dst.data, 
            src.n_elements);

        proc = cuda_launch_success();
        assert(proc); if(!proc) { return false; }

        return true;
    }



    namespace cuda
    {
        void alpha_blend(image_t const& src, image_t const& current, image_t const& dst)
        {
            assert(src.data);
            assert(src.width);
            assert(src.height);
            assert(current.data);
            assert(current.width == src.width);
            assert(current.height == src.height);
            assert(dst.data);
            assert(dst.width == src.width);
            assert(dst.height == src.height);   

            u32 n_elements = src.width * src.height;

            bool proc;

            DeviceArray<pixel_t> d_src;
            DeviceArray<pixel_t> d_cur;
            DeviceArray<pixel_t> d_dst;

            DeviceBuffer<pixel_t> pixel_buffer;

            auto max_pixel_bytes = 3 * n_elements * sizeof(pixel_t);
            proc = device_malloc(pixel_buffer, max_pixel_bytes);
            assert(proc);

            proc = push_array(d_dst, n_elements, pixel_buffer);
            assert(proc);

            proc = push_array(d_src, n_elements, pixel_buffer);
            assert(proc);

            proc = push_array(d_cur, n_elements, pixel_buffer);
            assert(proc);
            
            proc = copy_to_device(src, d_src);
            assert(proc);

            proc = copy_to_device(current, d_cur);
            assert(proc);

            proc = alpha_blend(d_src, d_cur, d_dst);
            assert(proc);

            proc = copy_to_host(d_dst, dst);
            assert(proc);

            proc = device_free(pixel_buffer);
            assert(proc);
        }
        

        void alpha_blend(image_t const& src, image_t const& current, view_t const& dst);

		void alpha_blend(image_t const& src, view_t const& current, image_t const& dst);

		void alpha_blend(image_t const& src, view_t const& current, view_t const& dst);

		void alpha_blend(view_t const& src, image_t const& current, image_t const& dst);

		void alpha_blend(view_t const& src, image_t const& current, view_t const& dst);

		void alpha_blend(view_t const& src, view_t const& current, image_t const& dst);

		void alpha_blend(view_t const& src, view_t const& current, view_t const& dst);


		void alpha_blend(image_t const& src, image_t const& current_dst)
        {
            assert(src.data);
            assert(src.width);
            assert(src.height);
            assert(current_dst.data);
            assert(current_dst.width == src.width);
            assert(current_dst.height == src.height);

            u32 n_elements = src.width * src.height;
            int threads_per_block = THREADS_PER_BLOCK;
            int blocks = (n_elements + threads_per_block - 1) / threads_per_block;

            bool proc;

            DeviceArray<pixel_t> d_src;
            DeviceArray<pixel_t> d_cur_dst;

            DeviceBuffer<pixel_t> pixel_buffer;

            auto max_pixel_bytes = 2 * n_elements * sizeof(pixel_t);
            proc = device_malloc(pixel_buffer, max_pixel_bytes);
            assert(proc);

            proc = push_array(d_src, n_elements, pixel_buffer);
            assert(proc);
            proc = push_array(d_cur_dst, n_elements, pixel_buffer);
            assert(proc);

            proc = copy_to_device(src, d_src);
            assert(proc);
            proc = copy_to_device(current_dst, d_cur_dst);
            assert(proc);

            proc = cuda_no_errors();
            assert(proc);

            gpu_alpha_blend_linear<<<blocks, threads_per_block>>>(
                d_src.data, 
                d_cur_dst.data, 
                d_cur_dst.data, 
                n_elements);

            proc = cuda_launch_success();
            assert(proc);

            proc = copy_to_host(d_cur_dst, current_dst);
            assert(proc);

            proc = device_free(pixel_buffer);
            assert(proc);
        }


		void alpha_blend(image_t const& src, view_t const& current_dst);

		void alpha_blend(view_t const& src, image_t const& current_dst);

		void alpha_blend(view_t const& src, view_t const& current_dst);
    }
}

#endif // !LIBIMAGE_NO_COLOR	