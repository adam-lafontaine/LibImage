/*

Copyright (c) 2021 Adam Lafontaine

*/
#ifndef LIBIMAGE_NO_GRAYSCALE

#include "cuda_def.cuh"
#include "process.hpp"

#include <cassert>


constexpr int THREADS_PER_BLOCK = 1024;

namespace libimage
{
    GPU_KERNAL
    static void gpu_binarize(u8* src, u8* dst, u8 threshold, int n_elements)
    {
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        if (i >= n_elements)
        {
            return;
        }

        dst[i] = src[i] >= threshold ? 255 : 0;
    }


    bool abinarize(DeviceArray<gray::pixel_t> const& src, DeviceArray<gray::pixel_t> const& dst, u8 min_threshold) // TODO
    {
        assert(src.data);
        assert(dst.data);
        assert(dst.n_elements == src.n_elements);

        int threads_per_block = THREADS_PER_BLOCK;
        int blocks = (src.n_elements + threads_per_block - 1) / threads_per_block;

        bool proc;

        proc = cuda_no_errors();
        assert(proc); if(!proc) { return false; }

        gpu_binarize<<<blocks, threads_per_block>>>(
            src.data, 
            dst.data,
            min_threshold, 
            src.n_elements);

        proc = cuda_launch_success();
        assert(proc); if(!proc) { return false; }

        return true;
    }

    namespace cuda
    {
        void binarize(gray::image_t const& src, gray::image_t const& dst, u8 min_threshold)
        {
            assert(src.data);
            assert(src.width);
            assert(src.height);
            assert(dst.data);
            assert(dst.width == src.width);
            assert(dst.height == src.height);   

            u32 n_elements = src.width * src.height;

            bool proc;

            DeviceArray<gray::pixel_t> d_src;
            DeviceArray<gray::pixel_t> d_dst;

            DeviceBuffer<gray::pixel_t> pixel_buffer;

            auto max_pixel_bytes = 2 * n_elements * sizeof(gray::pixel_t);
            proc = device_malloc(pixel_buffer, max_pixel_bytes);
            assert(proc);

            proc = push_array(d_src, n_elements, pixel_buffer);
            assert(proc);

            proc = push_array(d_dst, n_elements, pixel_buffer);
            assert(proc);

            proc = copy_to_device(src, d_src);
            assert(proc);

            proc = abinarize(d_src, d_dst, min_threshold);
            assert(proc);

            proc = copy_to_host(d_dst, dst);
            assert(proc);

            proc = device_free(pixel_buffer);
            assert(proc);
        }


		void binarize(gray::image_t const& src, gray::view_t const& dst, u8 min_threshold);

		void binarize(gray::view_t const& src, gray::image_t const& dst, u8 min_threshold);

		void binarize(gray::view_t const& src, gray::view_t const& dst, u8 min_threshold);

    }
}

#endif // !LIBIMAGE_NO_GRAYSCALE