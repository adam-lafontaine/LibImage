#pragma once

#include "../rgba.hpp"
#include "../gray.hpp"
#include "device.hpp"

namespace libimage
{

#ifndef LIBIMAGE_NO_COLOR

    class DeviceRGBAImage
    {
    public:

        u32 width = 0;
        u32 height = 0;

        pixel_t* data = nullptr;
    };

    using device_image_t = DeviceRGBAImage;


    bool make_image(device_image_t& image, u32 width, u32 height, DeviceBuffer& buffer);


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
            u32 width = 0;
            u32 height = 0;

            pixel_t* data = nullptr;
        };

        using device_image_t = DeviceGrayImage;
    }


    bool make_image(gray::device_image_t& image, u32 width, u32 height, DeviceBuffer& buffer);
    

    bool copy_to_device(gray::image_t const& src, gray::device_image_t const& dst);

    bool copy_to_host(gray::device_image_t const& src, gray::image_t const& dst);

    bool copy_to_device(gray::view_t const& src, gray::device_image_t const& dst);

    bool copy_to_host(gray::device_image_t const& src, gray::view_t const& dst);

#endif // !LIBIMAGE_NO_GRAYSCALE
}


namespace device
{
#ifndef LIBIMAGE_NO_COLOR

    bool push(MemoryBuffer& buffer, libimage::device_image_t& image, u32 width, u32 height);

#endif // !LIBIMAGE_NO_COLOR

#ifndef LIBIMAGE_NO_GRAYSCALE

    bool push(MemoryBuffer& buffer, libimage::gray::device_image_t& image, u32 width, u32 height);

#endif // !LIBIMAGE_NO_GRAYSCALE
}