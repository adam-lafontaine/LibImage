#include "device_image.hpp"

#include <cassert>


namespace libimage
{

#ifndef LIBIMAGE_NO_COLOR

    bool make_image(device_image_t& image, u32 width, u32 height, DeviceBuffer& buffer)
    {
        assert(buffer.data);
        auto bytes = width * height * sizeof(pixel_t);

        bool result = buffer.total_bytes - buffer.offset >= bytes;
        if(result)
        {
            image.width = width;
            image.height = height;
            image.data = (pixel_t*)((u8*)buffer.data + buffer.offset);
            buffer.offset += bytes;
        }

        return result;
    }


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


    bool make_image(gray::device_image_t& image, u32 width, u32 height, DeviceBuffer& buffer)
    {
        assert(buffer.data);
        auto bytes = width * height * sizeof(gray::pixel_t);

        bool result = buffer.total_bytes - buffer.offset >= bytes;
        if(result)
        {
            image.width = width;
            image.height = height;
            image.data = (gray::pixel_t*)((u8*)buffer.data + buffer.offset);
            buffer.offset += bytes;
        }

        return result;
    }


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