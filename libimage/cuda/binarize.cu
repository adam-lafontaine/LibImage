/*

Copyright (c) 2021 Adam Lafontaine

*/
#ifndef LIBIMAGE_NO_GRAYSCALE

#include "cuda_def.cuh"
#include "verify.hpp"
#include "../proc/verify.hpp"

#include <cassert>


constexpr int THREADS_PER_BLOCK = 1024;

namespace libimage
{
    GPU_KERNAL
    void gpu_binarize(u8* src, u8* dst, u8 threshold, int n_elements)
    {
        int i = blockDim.x * blockIdx.x + threadIdx.x;

        if (i < n_elements)
        {
            dst[i] = src[i] >= threshold ? 255 : 0;
        }
    }

    namespace cuda
    {
        void binarize(gray::image_t const& src, gray::image_t const& dst, u8 min_threshold)
        {
            assert(verify(src, dst));

            u32 n_elements = src.width * src.height;

            DeviceBuffer d_buffer;
            device_malloc(d_buffer, bytes(src) + bytes(dst));

            DeviceArray<gray::pixel_t> d_src;
            DeviceArray<u8> d_dst;

            push_array(d_src, d_buffer, n_elements);
            push_array(d_dst, d_buffer, n_elements);

            copy_to_device(src, d_src);

            int threads_per_block = THREADS_PER_BLOCK;
            int blocks = (n_elements + threads_per_block - 1) / threads_per_block;

            gpu_binarize<<<blocks, threads_per_block>>>(d_src.data, d_dst.data, min_threshold, n_elements);

            copy_to_host(d_dst, dst);
            device_free(d_buffer);
        }


		void binarize(gray::image_t const& src, gray::view_t const& dst, u8 min_threshold)
        {
            assert(verify(src, dst));

            u32 n_elements = src.width * src.height;

            DeviceBuffer d_buffer;
            device_malloc(d_buffer, bytes(src) + bytes(dst));

            DeviceArray<gray::pixel_t> d_src;
            DeviceArray<u8> d_dst;

            push_array(d_src, d_buffer, n_elements);
            push_array(d_dst, d_buffer, n_elements);

            copy_to_device(src, d_src);

            int threads_per_block = THREADS_PER_BLOCK;
            int blocks = (n_elements + threads_per_block - 1) / threads_per_block;

            gpu_binarize<<<blocks, threads_per_block>>>(d_src.data, d_dst.data, min_threshold, n_elements);

            copy_to_host(d_dst, dst);
            device_free(d_buffer);
        }


		void binarize(gray::view_t const& src, gray::image_t const& dst, u8 min_threshold)
        {
            assert(verify(src, dst));

            u32 n_elements = src.width * src.height;

            DeviceBuffer d_buffer;
            device_malloc(d_buffer, bytes(src) + bytes(dst));

            DeviceArray<gray::pixel_t> d_src;
            DeviceArray<u8> d_dst;

            push_array(d_src, d_buffer, n_elements);
            push_array(d_dst, d_buffer, n_elements);

            copy_to_device(src, d_src);

            int threads_per_block = THREADS_PER_BLOCK;
            int blocks = (n_elements + threads_per_block - 1) / threads_per_block;

            gpu_binarize<<<blocks, threads_per_block>>>(d_src.data, d_dst.data, min_threshold, n_elements);

            copy_to_host(d_dst, dst);
            device_free(d_buffer);
        }


		void binarize(gray::view_t const& src, gray::view_t const& dst, u8 min_threshold)
        {
            assert(verify(src, dst));

            u32 n_elements = src.width * src.height;

            DeviceBuffer d_buffer;
            device_malloc(d_buffer, bytes(src) + bytes(dst));

            DeviceArray<gray::pixel_t> d_src;
            DeviceArray<u8> d_dst;

            push_array(d_src, d_buffer, n_elements);
            push_array(d_dst, d_buffer, n_elements);

            copy_to_device(src, d_src);

            int threads_per_block = THREADS_PER_BLOCK;
            int blocks = (n_elements + threads_per_block - 1) / threads_per_block;

            gpu_binarize<<<blocks, threads_per_block>>>(d_src.data, d_dst.data, min_threshold, n_elements);

            copy_to_host(d_dst, dst);
            device_free(d_buffer);
        }


        void binarize(gray::image_t const& src, gray::image_t const& dst, u8 min_threshold, DeviceBuffer& d_buffer)
        {
            assert(verify(src, dst));
            assert(verify(d_buffer));
            assert(verify(src, dst, d_buffer));

            u32 n_elements = src.width * src.height;

            DeviceArray<gray::pixel_t> d_src;
            DeviceArray<u8> d_dst;

            push_array(d_src, d_buffer, n_elements);
            push_array(d_dst, d_buffer, n_elements);

            copy_to_device(src, d_src);

            int threads_per_block = THREADS_PER_BLOCK;
            int blocks = (n_elements + threads_per_block - 1) / threads_per_block;

            gpu_binarize<<<blocks, threads_per_block>>>(d_src.data, d_dst.data, min_threshold, n_elements);

            copy_to_host(d_dst, dst);

            pop_array(d_src, d_buffer);
            pop_array(d_dst, d_buffer);
        }


		void binarize(gray::image_t const& src, gray::view_t const& dst, u8 min_threshold, DeviceBuffer& d_buffer)
        {
            assert(verify(src, dst));
            assert(verify(d_buffer));
            assert(verify(src, dst, d_buffer));

            u32 n_elements = src.width * src.height;

            DeviceArray<gray::pixel_t> d_src;
            DeviceArray<u8> d_dst;

            push_array(d_src, d_buffer, n_elements);
            push_array(d_dst, d_buffer, n_elements);

            copy_to_device(src, d_src);

            int threads_per_block = THREADS_PER_BLOCK;
            int blocks = (n_elements + threads_per_block - 1) / threads_per_block;

            gpu_binarize<<<blocks, threads_per_block>>>(d_src.data, d_dst.data, min_threshold, n_elements);

            copy_to_host(d_dst, dst);

            pop_array(d_src, d_buffer);
            pop_array(d_dst, d_buffer);
        }


		void binarize(gray::view_t const& src, gray::image_t const& dst, u8 min_threshold, DeviceBuffer& d_buffer)
        {
            assert(verify(src, dst));
            assert(verify(d_buffer));
            assert(verify(src, dst, d_buffer));

            u32 n_elements = src.width * src.height;

            DeviceArray<gray::pixel_t> d_src;
            DeviceArray<u8> d_dst;

            push_array(d_src, d_buffer, n_elements);
            push_array(d_dst, d_buffer, n_elements);

            copy_to_device(src, d_src);

            int threads_per_block = THREADS_PER_BLOCK;
            int blocks = (n_elements + threads_per_block - 1) / threads_per_block;

            gpu_binarize<<<blocks, threads_per_block>>>(d_src.data, d_dst.data, min_threshold, n_elements);

            copy_to_host(d_dst, dst);

            pop_array(d_src, d_buffer);
            pop_array(d_dst, d_buffer);
        }


		void binarize(gray::view_t const& src, gray::view_t const& dst, u8 min_threshold, DeviceBuffer& d_buffer)
        {
            assert(verify(src, dst));
            assert(verify(d_buffer));
            assert(verify(src, dst, d_buffer));

            u32 n_elements = src.width * src.height;

            DeviceArray<gray::pixel_t> d_src;
            DeviceArray<u8> d_dst;

            push_array(d_src, d_buffer, n_elements);
            push_array(d_dst, d_buffer, n_elements);

            copy_to_device(src, d_src);

            int threads_per_block = THREADS_PER_BLOCK;
            int blocks = (n_elements + threads_per_block - 1) / threads_per_block;

            gpu_binarize<<<blocks, threads_per_block>>>(d_src.data, d_dst.data, min_threshold, n_elements);

            copy_to_host(d_dst, dst);

            pop_array(d_src, d_buffer);
            pop_array(d_dst, d_buffer);
        }
        
    }
}

#endif // !LIBIMAGE_NO_GRAYSCALE