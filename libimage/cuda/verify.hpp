#pragma once
/*

Copyright (c) 2021 Adam Lafontaine

*/

#include "process.hpp"

#include "../proc/verify.hpp"


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


    
    
}
