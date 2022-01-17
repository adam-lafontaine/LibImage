#include "cuda_def.cuh"
#include "process.hpp"
#include "convolve.cuh"

#include <cassert>

constexpr int THREADS_PER_BLOCK = 1024;

namespace libimage
{
    
#ifndef LIBIMAGE_NO_COLOR

GPU_FUNCTION
static u8 blend_linear(u8 s, u8 c, r32 a)
{
    
    auto const sf = static_cast<r32>(s);
    auto const cf = static_cast<r32>(c);

    auto blended = a * cf + (1.0f - a) * sf;

    return static_cast<u8>(blended);
}


GPU_KERNAL
static void gpu_alpha_blend_linear(pixel_t* src, pixel_t* current, pixel_t* dst, u32 n_elements)
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



#endif // !LIBIMAGE_NO_COLOR

#ifndef LIBIMAGE_NO_GRAYSCALE

GPU_CONSTEXPR_FUNCTION r32 div16(int i) { return i / 16.0f; }

GPU_GLOBAL_CONSTANT r32 GAUSS_3X3[]
{
    div16(1), div16(2), div16(1),
    div16(2), div16(4), div16(2),
    div16(1), div16(2), div16(1),
};


GPU_CONSTEXPR_FUNCTION r32 div256(int i) { return i / 256.0f; }

GPU_GLOBAL_CONSTANT r32 GAUSS_5X5[]
{
    div256(1), div256(4),  div256(6),  div256(4),  div256(1),
    div256(4), div256(16), div256(24), div256(16), div256(4),
    div256(6), div256(24), div256(36), div256(24), div256(6),
    div256(4), div256(16), div256(24), div256(16), div256(4),
    div256(1), div256(4),  div256(6),  div256(4),  div256(1),
};


GPU_GLOBAL_CONSTANT r32 GRAD_X_3X3[]
{
    1.0f, 0.0f, -1.0f,
    2.0f, 0.0f, -2.0f,
    1.0f, 0.0f, -1.0f,
};


GPU_GLOBAL_CONSTANT r32 GRAD_Y_3X3[]
{
    1.0f,  2.0f,  1.0f,
    0.0f,  0.0f,  0.0f,
    -1.0f, -2.0f, -1.0f,
};


GPU_KERNAL
static void gpu_binarize(u8* src, u8* dst, u8 threshold, u32 n_elements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n_elements)
    {
        return;
    }

    dst[i] = src[i] >= threshold ? 255 : 0;
}


GPU_KERNAL
static void gpu_blur(u8* src, u8* dst, u32 width, u32 height)
{
    u32 n_elements = width * height;
    u32 i = u32(blockDim.x * blockIdx.x + threadIdx.x);
    if (i >= n_elements)
    {
        return;
    }

    auto g3x3 = (r32*)GAUSS_3X3;
    auto g5x5 = (r32*)GAUSS_5X5;

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
static void gpu_edges(u8* src, u8* dst, u32 width, u32 height, u8 threshold)
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
        auto gx = convolve_3x3(src, width, height, i, GRAD_X_3X3);
        auto gy = convolve_3x3(src, width, height, i, GRAD_Y_3X3);
        auto g = static_cast<u8>(std::hypot(gx, gy));
        dst[i] = g < threshold ? 0 : 255;
    }
}


GPU_KERNAL
static void gpu_gradients(u8* src, u8* dst, u32 width, u32 height)
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
        auto gx = convolve_3x3(src, width, height, i, GRAD_X_3X3);
        auto gy = convolve_3x3(src, width, height, i, GRAD_Y_3X3);
        dst[i] = static_cast<u8>(std::hypot(gx, gy));
    }
}


GPU_FUNCTION
static u8 lerp_clamp(u8 src_low, u8 src_high, u8 dst_low, u8 dst_high, u8 val)
{
    if (val < src_low)
    {
        return dst_low;
    }
    else if (val > src_high)
    {
        return dst_high;
    }

    auto const ratio = (static_cast<r64>(val) - src_low) / (src_high - src_low);

    assert(ratio >= 0.0);
    assert(ratio <= 1.0);

    auto const diff = ratio * (dst_high - dst_low);

    return dst_low + static_cast<u8>(diff);
}


GPU_KERNAL
static void gpu_transform_contrast(u8* src, u8* dst, u8 src_low, u8 src_high, u32 n_elements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n_elements)
    {
        return;
    }

    u8 dst_low = 0;
	u8 dst_high = 255;

    dst[i] = lerp_clamp(src_low, src_high, dst_low, dst_high, src[i]);
}

#endif // !LIBIMAGE_NO_GRAYSCALE

#ifndef LIBIMAGE_NO_COLOR
#ifndef LIBIMAGE_NO_GRAYSCALE

GPU_FUNCTION
static u8 rgb_grayscale_standard(u8 red, u8 green, u8 blue)
{
    return static_cast<u8>(0.299f * red + 0.587f * green + 0.114f * blue);
}


GPU_KERNAL
static void gpu_transform_grayscale(pixel_t* src, u8* dst, u32 n_elements)
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

#endif // !LIBIMAGE_NO_GRAYSCALE
#endif // !LIBIMAGE_NO_COLOR	

}


namespace libimage
{

#ifndef LIBIMAGE_NO_COLOR

    bool alpha_blend(device_image_t const& src, device_image_t const& current, device_image_t const& dst)
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
        int threads_per_block = THREADS_PER_BLOCK;
        int blocks = (n_elements + threads_per_block - 1) / threads_per_block;

        bool proc;

        proc = cuda_no_errors();
        assert(proc);

        gpu_alpha_blend_linear<<<blocks, threads_per_block>>>(
            src.data, 
            current.data, 
            dst.data, 
            n_elements);

        proc &= cuda_launch_success();
        assert(proc);

        return proc;
    }


    bool alpha_blend(device_image_t const& src, device_image_t const& current_dst)
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

        proc = cuda_no_errors();
        assert(proc);

        gpu_alpha_blend_linear<<<blocks, threads_per_block>>>(
            src.data, 
            current_dst.data, 
            current_dst.data, 
            n_elements);

        proc &= cuda_launch_success();
        assert(proc);

        return proc;
    }


#endif // !LIBIMAGE_NO_COLOR	

#ifndef LIBIMAGE_NO_GRAYSCALE

    bool binarize(gray::device_image_t const& src, gray::device_image_t const& dst, u8 min_threshold)
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

        proc = cuda_no_errors();
        assert(proc);

        gpu_binarize<<<blocks, threads_per_block>>>(
            src.data, 
            dst.data,
            min_threshold, 
            n_elements);

        proc &= cuda_launch_success();
        assert(proc);

        return proc;
    }


    bool blur(gray::device_image_t const& src, gray::device_image_t const& dst)
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

        proc = cuda_no_errors();
        assert(proc);

        gpu_blur<<<blocks, threads_per_block>>>(
            src.data, 
            dst.data, 
            src.width, 
            src.height);
        
        proc &= cuda_launch_success();
        assert(proc);

        return proc;
    }


    bool edges(gray::device_image_t const& src, gray::device_image_t const& dst, u8 threshold, gray::device_image_t const& temp)
    {
        assert(src.data);
        assert(src.width);
        assert(src.height);
        assert(dst.data);
        assert(dst.width == src.width);
        assert(dst.height == src.height);
        assert(temp.data);
        assert(temp.width == src.width);
        assert(temp.height == src.height);

        u32 n_elements = src.width * src.height;
        int threads_per_block = THREADS_PER_BLOCK;
        int blocks = (n_elements + threads_per_block - 1) / threads_per_block;

        bool proc;

        proc = cuda_no_errors();
        assert(proc);

        gpu_blur<<<blocks, threads_per_block>>>(
            src.data, 
            temp.data,
            src.width, 
            src.height);
        
        proc &= cuda_launch_success();
        assert(proc);

        proc &= cuda_no_errors();
        assert(proc);

        gpu_edges<<<blocks, threads_per_block>>>(
            temp.data,
            dst.data,
            src.width,
            src.height,
            threshold);

        proc &= cuda_launch_success();
        assert(proc);

        return proc;
    }   


    bool gradients(gray::device_image_t const& src, gray::device_image_t const& dst, gray::device_image_t const& temp)
    {
        assert(src.data);
        assert(src.width);
        assert(src.height);
        assert(dst.data);
        assert(dst.width == src.width);
        assert(dst.height == src.height);
        assert(temp.data);
        assert(temp.width == src.width);
        assert(temp.height == src.height);

        u32 n_elements = src.width * src.height;
        int threads_per_block = THREADS_PER_BLOCK;
        int blocks = (n_elements + threads_per_block - 1) / threads_per_block;

        bool proc;

        proc = cuda_no_errors();
        assert(proc);

        gpu_blur<<<blocks, threads_per_block>>>(
            src.data, 
            temp.data,
            src.width, 
            src.height);
        
        proc &= cuda_launch_success();
        assert(proc);

        proc &= cuda_no_errors();
        assert(proc);

        gpu_gradients<<<blocks, threads_per_block>>>(
            temp.data,
            dst.data,
            src.width,
            src.height);

        proc &= cuda_launch_success();
        assert(proc);

        return proc;
    }


    bool contrast(gray::device_image_t const& src, gray::device_image_t const& dst, u8 src_low, u8 src_high)
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

        proc = cuda_no_errors();
        assert(proc);

        gpu_transform_contrast<<<blocks, threads_per_block>>>(
            src.data,
            dst.data,
            src_low,
            src_high,
            n_elements);

        proc &= cuda_launch_success();
        assert(proc);

        return proc;
    }
        

#endif // !LIBIMAGE_NO_GRAYSCALE


#ifndef LIBIMAGE_NO_COLOR
#ifndef LIBIMAGE_NO_GRAYSCALE
        

    bool grayscale(device_image_t const& src, gray::device_image_t const& dst)
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

        proc = cuda_no_errors();
        assert(proc);

        gpu_transform_grayscale<<<blocks, threads_per_block>>>(
            src.data, 
            dst.data, 
            n_elements);

        proc &= cuda_launch_success();
        assert(proc);

        return proc;
    }

#endif // !LIBIMAGE_NO_GRAYSCALE
#endif // !LIBIMAGE_NO_COLOR	
    
}