#include "device_image.hpp"

#include <cassert>


namespace libimage
{

#ifndef LIBIMAGE_NO_COLOR

    bool copy_to_device(image_t const& src, device_image_t const& dst)
    {
        assert(src.data);
        assert(src.width);
        assert(src.height);
        assert(dst.data);
        assert(dst.width == src.width);
        assert(dst.height == src.height);

        auto bytes = src.width * src.height * sizeof(pixel_t);

        return cuda_memcpy_to_device(src.data, dst.data, bytes);
    }


    bool copy_to_host(device_image_t const& src, image_t const& dst)
    {
        assert(src.data);
        assert(src.width);
        assert(src.height);
        assert(dst.data);
        assert(dst.width == src.width);
        assert(dst.height == src.height);

        auto bytes = src.width * src.height * sizeof(pixel_t);

        return cuda_memcpy_to_host(src.data, dst.data, bytes);
    }


    bool copy_to_device(view_t const& src, device_image_t const& dst)
    {
        assert(src.image_data);
        assert(src.width);
        assert(src.height);
        assert(dst.data);
        assert(dst.width == src.width);
        assert(dst.height == src.height);

        u32 row_bytes = src.width * sizeof(pixel_t);
        for(u32 y = 0; y < src.height; ++y)
        {
            auto src_p = src.row_begin(y);
            auto dst_p = (u8*)dst.data + y * row_bytes;

            if(!cuda_memcpy_to_device(src_p, dst_p, row_bytes))
            {
                return false;
            }
        }

        return true;
    }


    bool copy_to_host(device_image_t const& src, view_t const& dst)
    {
        assert(src.data);
        assert(src.width);
        assert(src.height);
        assert(dst.image_data);
        assert(dst.width == src.width);
        assert(dst.height == src.height);

        u32 row_bytes = dst.width * sizeof(pixel_t);
        for(u32 y = 0; y < dst.height; ++y)
        {
            auto src_p = (u8*)src.data + y * row_bytes;
            auto dst_p = dst.row_begin(y);

            if(!cuda_memcpy_to_host(src_p, dst_p, row_bytes))
            {
                return false;
            }
        }

        return true;
    }

#endif // !LIBIMAGE_NO_COLOR

#ifndef LIBIMAGE_NO_GRAYSCALE


    bool copy_to_device(gray::image_t const& src, gray::device_image_t const& dst)
    {
        assert(src.data);
        assert(src.width);
        assert(src.height);
        assert(dst.data);
        assert(dst.width == src.width);
        assert(dst.height == src.height);

        auto bytes = src.width * src.height * sizeof(gray::pixel_t);

        return cuda_memcpy_to_device(src.data, dst.data, bytes);
    }


    bool copy_to_host(gray::device_image_t const& src, gray::image_t const& dst)
    {
        assert(src.data);
        assert(src.width);
        assert(src.height);
        assert(dst.data);
        assert(dst.width == src.width);
        assert(dst.height == src.height);

        auto bytes = src.width * src.height * sizeof(gray::pixel_t);

        return cuda_memcpy_to_host(src.data, dst.data, bytes);
    }


    bool copy_to_device(gray::view_t const& src, gray::device_image_t const& dst)
    {
        assert(src.image_data);
        assert(src.width);
        assert(src.height);
        assert(dst.data);
        assert(dst.width == src.width);
        assert(dst.height == src.height);

        u32 row_bytes = src.width * sizeof(gray::pixel_t);
        for(u32 y = 0; y < src.height; ++y)
        {
            auto src_p = src.row_begin(y);
            auto dst_p = (u8*)dst.data + y * row_bytes;

            if(!cuda_memcpy_to_device(src_p, dst_p, row_bytes))
            {
                return false;
            }
        }

        return true;
    }


    bool copy_to_host(gray::device_image_t const& src, gray::view_t const& dst)
    {
        assert(src.data);
        assert(src.width);
        assert(src.height);
        assert(dst.image_data);
        assert(dst.width == src.width);
        assert(dst.height == src.height);

        u32 row_bytes = dst.width * sizeof(gray::pixel_t);
        for(u32 y = 0; y < dst.height; ++y)
        {
            auto src_p = (u8*)src.data + y * row_bytes;
            auto dst_p = dst.row_begin(y);

            if(!cuda_memcpy_to_host(src_p, dst_p, row_bytes))
            {
                return false;
            }
        }

        return true;
    }

#endif // !LIBIMAGE_NO_GRAYSCALE        
    
}


namespace device
{
#ifndef LIBIMAGE_NO_COLOR

    bool push(MemoryBuffer& buffer, libimage::device_image_t& image, u32 width, u32 height)
    {
        assert(is_valid(buffer));        
        assert(width);
        assert(height);
        assert(!image.data);

        auto n_bytes = sizeof(libimage::pixel_t) * width * height;

        auto ptr = push(buffer, n_bytes);
        if(!ptr)
        {
            return false;
        }

        image.width = width;
        image.height = height;
        image.data = (libimage::pixel_t*)ptr;

        return true;
    }


    bool push(MemoryBuffer& buffer, libimage::device_image_t& image)
    {
        auto width = image.width;
        auto height = image.height;

        assert(is_valid(buffer));        
        assert(width);
        assert(height);
        assert(!image.data);

        auto n_bytes = sizeof(libimage::pixel_t) * width * height;

        auto ptr = push(buffer, n_bytes);
        if(!ptr)
        {
            return false;
        }

        image.width = width;
        image.height = height;
        image.data = (libimage::pixel_t*)ptr;

        return true;
    }

#endif // !LIBIMAGE_NO_COLOR

#ifndef LIBIMAGE_NO_GRAYSCALE

    bool push(MemoryBuffer& buffer, libimage::gray::device_image_t& image, u32 width, u32 height)
    {
        assert(is_valid(buffer));        
        assert(width);
        assert(height);
        assert(!image.data);

        auto n_bytes = sizeof(libimage::gray::pixel_t) * width * height;

        auto ptr = push(buffer, n_bytes);
        if(!ptr)
        {
            return false;
        }

        image.width = width;
        image.height = height;
        image.data = (libimage::gray::pixel_t*)ptr;

        return true;
    }


    bool push(MemoryBuffer& buffer, libimage::gray::device_image_t& image)
    {
        auto width = image.width;
        auto height = image.height;

        assert(is_valid(buffer));        
        assert(width);
        assert(height);
        assert(!image.data);

        auto n_bytes = sizeof(libimage::gray::pixel_t) * width * height;

        auto ptr = push(buffer, n_bytes);
        if(!ptr)
        {
            return false;
        }

        image.width = width;
        image.height = height;
        image.data = (libimage::gray::pixel_t*)ptr;

        return true;
    }

#endif // !LIBIMAGE_NO_GRAYSCALE
}