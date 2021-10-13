/*

Copyright (c) 2021 Adam Lafontaine

*/
#ifndef LIBIMAGE_NO_COLOR

#include "cuda_def.cuh"
#include "verify.hpp"
#include "../proc/verify.hpp"

#include <cassert>

constexpr int THREADS_PER_BLOCK = 1024;


GPU_FUNCTION
u8 blend_linear(u8 s, u8 c, r32 a)
{
    
    auto const sf = static_cast<r32>(s);
    auto const cf = static_cast<r32>(c);

    auto blended = a * cf + (1.0f - a) * sf;

    return static_cast<u8>(blended);
}

namespace libimage
{
    GPU_KERNAL
    void gpu_alpha_blend_linear(pixel_t* src, pixel_t* current, pixel_t* dst, int n_elements)
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

    namespace cuda
    {
        void alpha_blend(image_t const& src, image_t const& current, image_t const& dst, DeviceBuffer& d_buffer)
        {
            assert(verify(src, dst));
            assert(verify(d_buffer));
            assert(verify(src, current, dst, d_buffer));

            u32 n_elements = src.width * src.height;

            DeviceArray<pixel_t> d_src;
            DeviceArray<pixel_t> d_cur;
            DeviceArray<pixel_t> d_dst;

            push_array(d_src, d_buffer, n_elements);
            push_array(d_cur, d_buffer, n_elements);
            push_array(d_dst, d_buffer, n_elements);

            copy_to_device(src, d_src);
            copy_to_device(current, d_cur);

            int threads_per_block = THREADS_PER_BLOCK;
            int blocks = (n_elements + threads_per_block - 1) / threads_per_block;

            gpu_alpha_blend_linear<<<blocks, threads_per_block>>>(d_src.data, d_cur.data, d_dst.data, n_elements);

            copy_to_host(d_dst, dst);

            pop_array(d_dst, d_buffer);
            pop_array(d_cur, d_buffer);
            pop_array(d_src, d_buffer);
        }


		void alpha_blend(image_t const& src, image_t const& current, view_t const& dst, DeviceBuffer& d_buffer)
        {
            assert(verify(src, dst));
            assert(verify(d_buffer));
            assert(verify(src, current, dst, d_buffer));

            u32 n_elements = src.width * src.height;

            DeviceArray<pixel_t> d_src;
            DeviceArray<pixel_t> d_cur;
            DeviceArray<pixel_t> d_dst;

            push_array(d_src, d_buffer, n_elements);
            push_array(d_cur, d_buffer, n_elements);
            push_array(d_dst, d_buffer, n_elements);

            copy_to_device(src, d_src);
            copy_to_device(current, d_cur);

            int threads_per_block = THREADS_PER_BLOCK;
            int blocks = (n_elements + threads_per_block - 1) / threads_per_block;

            gpu_alpha_blend_linear<<<blocks, threads_per_block>>>(d_src.data, d_cur.data, d_dst.data, n_elements);

            copy_to_host(d_dst, dst);

            pop_array(d_dst, d_buffer);
            pop_array(d_cur, d_buffer);
            pop_array(d_src, d_buffer);
        }


		void alpha_blend(image_t const& src, view_t const& current, image_t const& dst, DeviceBuffer& d_buffer)
        {
            assert(verify(src, dst));
            assert(verify(d_buffer));
            assert(verify(src, current, dst, d_buffer));

            u32 n_elements = src.width * src.height;

            DeviceArray<pixel_t> d_src;
            DeviceArray<pixel_t> d_cur;
            DeviceArray<pixel_t> d_dst;

            push_array(d_src, d_buffer, n_elements);
            push_array(d_cur, d_buffer, n_elements);
            push_array(d_dst, d_buffer, n_elements);

            copy_to_device(src, d_src);
            copy_to_device(current, d_cur);

            int threads_per_block = THREADS_PER_BLOCK;
            int blocks = (n_elements + threads_per_block - 1) / threads_per_block;

            gpu_alpha_blend_linear<<<blocks, threads_per_block>>>(d_src.data, d_cur.data, d_dst.data, n_elements);

            copy_to_host(d_dst, dst);

            pop_array(d_dst, d_buffer);
            pop_array(d_cur, d_buffer);
            pop_array(d_src, d_buffer);
        }


		void alpha_blend(image_t const& src, view_t const& current, view_t const& dst, DeviceBuffer& d_buffer)
        {
            assert(verify(src, dst));
            assert(verify(d_buffer));
            assert(verify(src, current, dst, d_buffer));

            u32 n_elements = src.width * src.height;

            DeviceArray<pixel_t> d_src;
            DeviceArray<pixel_t> d_cur;
            DeviceArray<pixel_t> d_dst;

            push_array(d_src, d_buffer, n_elements);
            push_array(d_cur, d_buffer, n_elements);
            push_array(d_dst, d_buffer, n_elements);

            copy_to_device(src, d_src);
            copy_to_device(current, d_cur);

            int threads_per_block = THREADS_PER_BLOCK;
            int blocks = (n_elements + threads_per_block - 1) / threads_per_block;

            gpu_alpha_blend_linear<<<blocks, threads_per_block>>>(d_src.data, d_cur.data, d_dst.data, n_elements);

            copy_to_host(d_dst, dst);

            pop_array(d_dst, d_buffer);
            pop_array(d_cur, d_buffer);
            pop_array(d_src, d_buffer);
        }


		void alpha_blend(view_t const& src, image_t const& current, image_t const& dst, DeviceBuffer& d_buffer)
        {
            assert(verify(src, dst));
            assert(verify(d_buffer));
            assert(verify(src, current, dst, d_buffer));

            u32 n_elements = src.width * src.height;

            DeviceArray<pixel_t> d_src;
            DeviceArray<pixel_t> d_cur;
            DeviceArray<pixel_t> d_dst;

            push_array(d_src, d_buffer, n_elements);
            push_array(d_cur, d_buffer, n_elements);
            push_array(d_dst, d_buffer, n_elements);

            copy_to_device(src, d_src);
            copy_to_device(current, d_cur);

            int threads_per_block = THREADS_PER_BLOCK;
            int blocks = (n_elements + threads_per_block - 1) / threads_per_block;

            gpu_alpha_blend_linear<<<blocks, threads_per_block>>>(d_src.data, d_cur.data, d_dst.data, n_elements);

            copy_to_host(d_dst, dst);

            pop_array(d_dst, d_buffer);
            pop_array(d_cur, d_buffer);
            pop_array(d_src, d_buffer);
        }


		void alpha_blend(view_t const& src, image_t const& current, view_t const& dst, DeviceBuffer& d_buffer)
        {
            assert(verify(src, dst));
            assert(verify(d_buffer));
            assert(verify(src, current, dst, d_buffer));

            u32 n_elements = src.width * src.height;

            DeviceArray<pixel_t> d_src;
            DeviceArray<pixel_t> d_cur;
            DeviceArray<pixel_t> d_dst;

            push_array(d_src, d_buffer, n_elements);
            push_array(d_cur, d_buffer, n_elements);
            push_array(d_dst, d_buffer, n_elements);

            copy_to_device(src, d_src);
            copy_to_device(current, d_cur);

            int threads_per_block = THREADS_PER_BLOCK;
            int blocks = (n_elements + threads_per_block - 1) / threads_per_block;

            gpu_alpha_blend_linear<<<blocks, threads_per_block>>>(d_src.data, d_cur.data, d_dst.data, n_elements);

            copy_to_host(d_dst, dst);

            pop_array(d_dst, d_buffer);
            pop_array(d_cur, d_buffer);
            pop_array(d_src, d_buffer);
        }


		void alpha_blend(view_t const& src, view_t const& current, image_t const& dst, DeviceBuffer& d_buffer)
        {
            assert(verify(src, dst));
            assert(verify(d_buffer));
            assert(verify(src, current, dst, d_buffer));

            u32 n_elements = src.width * src.height;

            DeviceArray<pixel_t> d_src;
            DeviceArray<pixel_t> d_cur;
            DeviceArray<pixel_t> d_dst;

            push_array(d_src, d_buffer, n_elements);
            push_array(d_cur, d_buffer, n_elements);
            push_array(d_dst, d_buffer, n_elements);

            copy_to_device(src, d_src);
            copy_to_device(current, d_cur);

            int threads_per_block = THREADS_PER_BLOCK;
            int blocks = (n_elements + threads_per_block - 1) / threads_per_block;

            gpu_alpha_blend_linear<<<blocks, threads_per_block>>>(d_src.data, d_cur.data, d_dst.data, n_elements);

            copy_to_host(d_dst, dst);

            pop_array(d_dst, d_buffer);
            pop_array(d_cur, d_buffer);
            pop_array(d_src, d_buffer);
        }


		void alpha_blend(view_t const& src, view_t const& current, view_t const& dst, DeviceBuffer& d_buffer)
        {
            assert(verify(src, dst));
            assert(verify(d_buffer));
            assert(verify(src, current, dst, d_buffer));

            u32 n_elements = src.width * src.height;

            DeviceArray<pixel_t> d_src;
            DeviceArray<pixel_t> d_cur;
            DeviceArray<pixel_t> d_dst;

            push_array(d_src, d_buffer, n_elements);
            push_array(d_cur, d_buffer, n_elements);
            push_array(d_dst, d_buffer, n_elements);

            copy_to_device(src, d_src);
            copy_to_device(current, d_cur);

            int threads_per_block = THREADS_PER_BLOCK;
            int blocks = (n_elements + threads_per_block - 1) / threads_per_block;

            gpu_alpha_blend_linear<<<blocks, threads_per_block>>>(d_src.data, d_cur.data, d_dst.data, n_elements);

            copy_to_host(d_dst, dst);

            pop_array(d_dst, d_buffer);
            pop_array(d_cur, d_buffer);
            pop_array(d_src, d_buffer);
        }


		void alpha_blend(image_t const& src, image_t const& current_dst, DeviceBuffer& d_buffer)
        {
            assert(verify(src, current_dst));
            assert(verify(d_buffer));
            assert(verify(src, current_dst, d_buffer));

            u32 n_elements = src.width * src.height;

            DeviceArray<pixel_t> d_src;
            DeviceArray<pixel_t> d_cur_dst;

            push_array(d_src, d_buffer, n_elements);
            push_array(d_cur_dst, d_buffer, n_elements);

            copy_to_device(src, d_src);
            copy_to_device(current_dst, d_cur_dst);

            int threads_per_block = THREADS_PER_BLOCK;
            int blocks = (n_elements + threads_per_block - 1) / threads_per_block;

            gpu_alpha_blend_linear<<<blocks, threads_per_block>>>(d_src.data, d_cur_dst.data, d_cur_dst.data, n_elements);

            copy_to_host(d_cur_dst, current_dst);
            
            pop_array(d_cur_dst, d_buffer);
            pop_array(d_src, d_buffer);
        }


		void alpha_blend(image_t const& src, view_t const& current_dst, DeviceBuffer& d_buffer)        
        {
            assert(verify(src, current_dst));
            assert(verify(d_buffer));
            assert(verify(src, current_dst, d_buffer));

            u32 n_elements = src.width * src.height;

            DeviceArray<pixel_t> d_src;
            DeviceArray<pixel_t> d_cur_dst;

            push_array(d_src, d_buffer, n_elements);
            push_array(d_cur_dst, d_buffer, n_elements);

            copy_to_device(src, d_src);
            copy_to_device(current_dst, d_cur_dst);

            int threads_per_block = THREADS_PER_BLOCK;
            int blocks = (n_elements + threads_per_block - 1) / threads_per_block;

            gpu_alpha_blend_linear<<<blocks, threads_per_block>>>(d_src.data, d_cur_dst.data, d_cur_dst.data, n_elements);

            copy_to_host(d_cur_dst, current_dst);
            
            pop_array(d_cur_dst, d_buffer);
            pop_array(d_src, d_buffer);
        }


		void alpha_blend(view_t const& src, image_t const& current_dst, DeviceBuffer& d_buffer)
        {
            assert(verify(src, current_dst));
            assert(verify(d_buffer));
            assert(verify(src, current_dst, d_buffer));

            u32 n_elements = src.width * src.height;

            DeviceArray<pixel_t> d_src;
            DeviceArray<pixel_t> d_cur_dst;

            push_array(d_src, d_buffer, n_elements);
            push_array(d_cur_dst, d_buffer, n_elements);

            copy_to_device(src, d_src);
            copy_to_device(current_dst, d_cur_dst);

            int threads_per_block = THREADS_PER_BLOCK;
            int blocks = (n_elements + threads_per_block - 1) / threads_per_block;

            gpu_alpha_blend_linear<<<blocks, threads_per_block>>>(d_src.data, d_cur_dst.data, d_cur_dst.data, n_elements);

            copy_to_host(d_cur_dst, current_dst);
            
            pop_array(d_cur_dst, d_buffer);
            pop_array(d_src, d_buffer);
        }


		void alpha_blend(view_t const& src, view_t const& current_dst, DeviceBuffer& d_buffer)
        {
            assert(verify(src, current_dst));
            assert(verify(d_buffer));
            assert(verify(src, current_dst, d_buffer));

            u32 n_elements = src.width * src.height;

            DeviceArray<pixel_t> d_src;
            DeviceArray<pixel_t> d_cur_dst;

            push_array(d_src, d_buffer, n_elements);
            push_array(d_cur_dst, d_buffer, n_elements);

            copy_to_device(src, d_src);
            copy_to_device(current_dst, d_cur_dst);

            int threads_per_block = THREADS_PER_BLOCK;
            int blocks = (n_elements + threads_per_block - 1) / threads_per_block;

            gpu_alpha_blend_linear<<<blocks, threads_per_block>>>(d_src.data, d_cur_dst.data, d_cur_dst.data, n_elements);

            copy_to_host(d_cur_dst, current_dst);
            
            pop_array(d_cur_dst, d_buffer);
            pop_array(d_src, d_buffer);
        }        


        void alpha_blend(image_t const& src, image_t const& current, image_t const& dst)
        {
            assert(verify(src, current));
            assert(verify(src, dst));

            DeviceBuffer d_buffer;
            device_malloc(d_buffer, bytes(src) + bytes(current) + bytes(dst));

            alpha_blend(src, current, dst, d_buffer);

            device_free(d_buffer);            
        }


		void alpha_blend(image_t const& src, image_t const& current, view_t const& dst)
        {
            assert(verify(src, current));
            assert(verify(src, dst));

            DeviceBuffer d_buffer;
            device_malloc(d_buffer, bytes(src) + bytes(current) + bytes(dst));

            alpha_blend(src, current, dst, d_buffer);

            device_free(d_buffer);            
        }


		void alpha_blend(image_t const& src, view_t const& current, image_t const& dst)
        {
            assert(verify(src, current));
            assert(verify(src, dst));

            DeviceBuffer d_buffer;
            device_malloc(d_buffer, bytes(src) + bytes(current) + bytes(dst));

            alpha_blend(src, current, dst, d_buffer);

            device_free(d_buffer);            
        }


		void alpha_blend(image_t const& src, view_t const& current, view_t const& dst)
        {
            assert(verify(src, current));
            assert(verify(src, dst));

            DeviceBuffer d_buffer;
            device_malloc(d_buffer, bytes(src) + bytes(current) + bytes(dst));

            alpha_blend(src, current, dst, d_buffer);

            device_free(d_buffer);            
        }


		void alpha_blend(view_t const& src, image_t const& current, image_t const& dst)
        {
            assert(verify(src, current));
            assert(verify(src, dst));

            DeviceBuffer d_buffer;
            device_malloc(d_buffer, bytes(src) + bytes(current) + bytes(dst));

            alpha_blend(src, current, dst, d_buffer);

            device_free(d_buffer);            
        }


		void alpha_blend(view_t const& src, image_t const& current, view_t const& dst)
        {
            assert(verify(src, current));
            assert(verify(src, dst));

            DeviceBuffer d_buffer;
            device_malloc(d_buffer, bytes(src) + bytes(current) + bytes(dst));

            alpha_blend(src, current, dst, d_buffer);

            device_free(d_buffer);            
        }


		void alpha_blend(view_t const& src, view_t const& current, image_t const& dst)
        {
            assert(verify(src, current));
            assert(verify(src, dst));

            DeviceBuffer d_buffer;
            device_malloc(d_buffer, bytes(src) + bytes(current) + bytes(dst));

            alpha_blend(src, current, dst, d_buffer);

            device_free(d_buffer);            
        }


		void alpha_blend(view_t const& src, view_t const& current, view_t const& dst)
        {
            assert(verify(src, current));
            assert(verify(src, dst));

            DeviceBuffer d_buffer;
            device_malloc(d_buffer, bytes(src) + bytes(current) + bytes(dst));

            alpha_blend(src, current, dst);

            device_free(d_buffer);            
        }


		void alpha_blend(image_t const& src, image_t const& current_dst)
        {
            assert(verify(src, current_dst));

            DeviceBuffer d_buffer;
            device_malloc(d_buffer, bytes(src) + bytes(current_dst));

            alpha_blend(src, current_dst, d_buffer);

            device_free(d_buffer);
        }


		void alpha_blend(image_t const& src, view_t const& current_dst)
        {
            assert(verify(src, current_dst));

            DeviceBuffer d_buffer;
            device_malloc(d_buffer, bytes(src) + bytes(current_dst));

            alpha_blend(src, current_dst, d_buffer);

            device_free(d_buffer);
        }


		void alpha_blend(view_t const& src, image_t const& current_dst)
        {
            assert(verify(src, current_dst));

            DeviceBuffer d_buffer;
            device_malloc(d_buffer, bytes(src) + bytes(current_dst));

            alpha_blend(src, current_dst, d_buffer);

            device_free(d_buffer);
        }


		void alpha_blend(view_t const& src, view_t const& current_dst)
        {
            assert(verify(src, current_dst));

            DeviceBuffer d_buffer;
            device_malloc(d_buffer, bytes(src) + bytes(current_dst));

            alpha_blend(src, current_dst, d_buffer);

            device_free(d_buffer);
        }
    }
}

#endif // !LIBIMAGE_NO_COLOR	