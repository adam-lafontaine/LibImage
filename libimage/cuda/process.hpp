#pragma once
/*

Copyright (c) 2021 Adam Lafontaine

*/
#include "device_image.hpp"


namespace libimage
{
    namespace cuda
    {

#ifndef LIBIMAGE_NO_COLOR

        void alpha_blend(device_image_t const& src, device_image_t const& current, device_image_t const& dst);

		void alpha_blend(device_image_t const& src, device_image_t const& current_dst);


#endif // !LIBIMAGE_NO_COLOR	

#ifndef LIBIMAGE_NO_GRAYSCALE

        void binarize(gray::device_image_t const& src, gray::device_image_t const& dst, u8 min_threshold);

        void blur(gray::device_image_t const& src, gray::device_image_t const& dst);

		void edges(gray::device_image_t const& src, gray::device_image_t const& dst, u8 threshold);

		void gradients(gray::device_image_t const& src, gray::device_image_t const& dst);
        

#endif // !LIBIMAGE_NO_GRAYSCALE


#ifndef LIBIMAGE_NO_COLOR
#ifndef LIBIMAGE_NO_GRAYSCALE

        void transform_grayscale(device_image_t const& src, gray::device_image_t const& dst);

#endif // !LIBIMAGE_NO_GRAYSCALE
#endif // !LIBIMAGE_NO_COLOR	
    }
}