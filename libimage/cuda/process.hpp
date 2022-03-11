#pragma once

#include "device_image.hpp"


namespace libimage
{
    

#ifndef LIBIMAGE_NO_COLOR

	bool alpha_blend(device_image_t const& src, device_image_t const& current, device_image_t const& dst);

	bool alpha_blend(device_image_t const& src, device_image_t const& current_dst);


	bool rotate(device_image_t const& src, device_image_t const& dst, u32 origin_x, u32 origin_y, r32 theta);


#endif // !LIBIMAGE_NO_COLOR	

#ifndef LIBIMAGE_NO_GRAYSCALE

	bool binarize(gray::device_image_t const& src, gray::device_image_t const& dst, u8 min_threshold);

	bool blur(gray::device_image_t const& src, gray::device_image_t const& dst);

	bool edges(gray::device_image_t const& src, gray::device_image_t const& dst, u8 threshold, gray::device_image_t const& temp);

	bool edges(gray::device_image_t const& src, gray::device_image_t const& dst, u8 threshold);

	bool gradients(gray::device_image_t const& src, gray::device_image_t const& dst, gray::device_image_t const& temp);

	bool gradients(gray::device_image_t const& src, gray::device_image_t const& dst);

	bool contrast(gray::device_image_t const& src, gray::device_image_t const& dst, u8 src_low, u8 src_high);

	        

#endif // !LIBIMAGE_NO_GRAYSCALE


#ifndef LIBIMAGE_NO_COLOR
#ifndef LIBIMAGE_NO_GRAYSCALE

    bool grayscale(device_image_t const& src, gray::device_image_t const& dst);

#endif // !LIBIMAGE_NO_GRAYSCALE
#endif // !LIBIMAGE_NO_COLOR	
    
}