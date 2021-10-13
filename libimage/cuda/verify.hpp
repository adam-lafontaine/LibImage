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


    template <class IMG_A_T, class IMG_B_T>
    bool verify(IMG_A_T const& a, IMG_B_T const& b, DeviceBuffer const& buffer)
    {
        return (bytes(a) + bytes(b)) <= buffer.total_bytes;
    }


    template <class IMG_A_T, class IMG_B_T, class IMG_C_T>
    bool verify(IMG_A_T const& a, IMG_B_T const& b, IMG_C_T const& c, DeviceBuffer const& buffer)
    {
        return (bytes(a) + bytes(b) + bytes(c)) <= buffer.total_bytes;
    }
    
}
