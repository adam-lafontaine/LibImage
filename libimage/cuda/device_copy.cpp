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

#endif // !LIBIMAGE_NO_GRAYSCALE        
    }
}