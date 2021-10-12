#pragma once
/*

Copyright (c) 2021 Adam Lafontaine

*/

#include "process.hpp"


namespace libimage
{
    inline u32 bytes(image_t const& img)
    {
        return img.width * img.height * sizeof(pixel_t);
    }


    inline u32 bytes(view_t const& img)
    {
        return img.width * img.height * sizeof(pixel_t);
    }


    inline u32 bytes(gray::image_t const& img)
    {
        return img.width * img.height * sizeof(gray::pixel_t);
    }


    inline u32 bytes(gray::view_t const& img)
    {
        return img.width * img.height * sizeof(gray::pixel_t);
    }


    template <class SRC_IMG_T, class DST_IMG_T>
    bool verify(SRC_IMG_T const& src, DST_IMG_T const& dst, DeviceBuffer const& buffer)
    {
        u32 src_bytes = bytes(src);
        u32 dst_bytes = bytes(dst);

        return (src_bytes + dst_bytes) <= buffer.total_bytes;
    }
    
}
