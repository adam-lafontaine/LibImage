#pragma once
/*

Copyright (c) 2021 Adam Lafontaine

*/

#include "../rgba.hpp"
#include "../gray.hpp"

namespace libimage
{

#ifndef LIBIMAGE_NO_COLOR

    class DeviceRGBAImage
    {
    public:

        u32 width;
        u32 height;

        pixel_t* data;
    };

    using device_image_t = DeviceRGBAImage;


    bool copy_to_device(image_t const& src, device_image_t const& dst);

    bool copy_to_host(device_image_t const& src, image_t const& dst);

    bool copy_to_device(view_t const& src, device_image_t const& dst);

    bool copy_to_host(device_image_t const& src, view_t const& dst);

#endif // !LIBIMAGE_NO_COLOR

#ifndef LIBIMAGE_NO_GRAYSCALE

    namespace gray
    {
        class DeviceGrayImage
        {
        public:
            u32 width;
            u32 height;

            pixel_t* data;
        };

        using device_image_t = DeviceGrayImage;
    }
    

    bool copy_to_device(gray::image_t const& src, gray::device_image_t const& dst);

    bool copy_to_host(gray::device_image_t const& src, gray::image_t const& dst);

    bool copy_to_device(gray::view_t const& src, gray::device_image_t const& dst);

    bool copy_to_host(gray::device_image_t const& src, gray::view_t const& dst);

#endif // !LIBIMAGE_NO_GRAYSCALE
}