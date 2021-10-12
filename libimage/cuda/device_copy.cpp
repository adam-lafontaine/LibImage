#include "process.hpp"


namespace libimage
{
    namespace cuda
    {

#ifndef LIBIMAGE_NO_COLOR

        bool copy_to_device(image_t const& src, DeviceArray<pixel_t> const& dst)
        {
            u32 bytes = src.width * src.height * sizeof(pixel_t);
            return copy_to_device(src.data, dst, bytes);
        }


        bool copy_to_host(DeviceArray<pixel_t> const& src, image_t const& dst)
        {
            u32 bytes = dst.width * dst.height * sizeof(pixel_t);
            return copy_to_host(src, dst.data, bytes);
        }


        bool copy_to_device(view_t const& src, DeviceArray<pixel_t> const& dst)
        {
            u32 row_bytes = src.width * sizeof(pixel_t);
            for(u32 y = 0; y < src.height; ++y)
            {
                auto src_p = src.row_begin(y);
                auto dst_p = (u8*)dst.data + y * row_bytes;

                if(!memcpy_to_device(src_p, dst_p, row_bytes))
                {
                    return false;
                }
            }

            return true;
        }


        bool copy_to_host(DeviceArray<pixel_t> const& src, view_t const& dst)
        {
            u32 row_bytes = dst.width * sizeof(pixel_t);
            for(u32 y = 0; y < dst.height; ++y)
            {
                auto src_p = (u8*)src.data + y * row_bytes;
                auto dst_p = dst.row_begin(y);

                if(!memcpy_to_host(src_p, dst_p, row_bytes))
                {
                    return false;
                }
            }

            return true;
        }

#endif // !LIBIMAGE_NO_COLOR

#ifndef LIBIMAGE_NO_GRAYSCALE

        bool copy_to_device(gray::image_t const& src, DeviceArray<gray::pixel_t> const& dst)
        {
            u32 bytes = src.width * src.height * sizeof(gray::pixel_t);
            return copy_to_device(src.data, dst, bytes);
        }


        bool copy_to_host(DeviceArray<gray::pixel_t> const& src, gray::image_t const& dst)
        {
            u32 bytes = dst.width * dst.height * sizeof(gray::pixel_t);
            return copy_to_host(src, dst.data, bytes);
        }


        bool copy_to_device(gray::view_t const& src, DeviceArray<gray::pixel_t> const& dst)
        {
            u32 row_bytes = src.width * sizeof(gray::pixel_t);
            for(u32 y = 0; y < src.height; ++y)
            {
                auto src_p = src.row_begin(y);
                auto dst_p = (u8*)dst.data + y * row_bytes;

                if(!memcpy_to_device(src_p, dst_p, row_bytes))
                {
                    return false;
                }
            }

            return true;
        }


        bool copy_to_host(DeviceArray<gray::pixel_t> const& src, gray::view_t const& dst)
        {
            u32 row_bytes = dst.width * sizeof(gray::pixel_t);
            for(u32 y = 0; y < dst.height; ++y)
            {
                auto src_p = (u8*)src.data + y * row_bytes;
                auto dst_p = dst.row_begin(y);

                if(!memcpy_to_host(src_p, dst_p, row_bytes))
                {
                    return false;
                }
            }

            return true;
        }

#endif // !LIBIMAGE_NO_GRAYSCALE        
    }
}