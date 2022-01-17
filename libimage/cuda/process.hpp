#pragma once

#include "device_image.hpp"


namespace libimage
{
    

#ifndef LIBIMAGE_NO_COLOR

	bool alpha_blend(device_image_t const& src, device_image_t const& current, device_image_t const& dst);

	bool alpha_blend(device_image_t const& src, device_image_t const& current_dst);


#endif // !LIBIMAGE_NO_COLOR	

#ifndef LIBIMAGE_NO_GRAYSCALE
/*
	class BlurKernels
	{
	public:
		DeviceArray<r32> kernel_3x3;
		DeviceArray<r32> kernel_5x5;
	};


	class GradientKernels
	{
	public:
		DeviceArray<r32> kernel_x_3x3;
		DeviceArray<r32> kernel_y_3x3;
	};


	bool make_blur_kernels(BlurKernels& blur_k, DeviceBuffer& buffer);

	bool make_gradient_kernels(GradientKernels& grad_k, DeviceBuffer& buffer);
	*/

	bool binarize(gray::device_image_t const& src, gray::device_image_t const& dst, u8 min_threshold);

	bool blur(gray::device_image_t const& src, gray::device_image_t const& dst/*, BlurKernels const& blur_k*/);

	bool edges(gray::device_image_t const& src, gray::device_image_t const& dst, u8 threshold, gray::device_image_t const& temp/*, BlurKernels const& blur_k, GradientKernels const& grad_k*/);

	bool gradients(gray::device_image_t const& src, gray::device_image_t const& dst, gray::device_image_t const& temp/*, BlurKernels const& blur_k, GradientKernels const& grad_k*/);

	bool contrast(gray::device_image_t const& src, gray::device_image_t const& dst, u8 src_low, u8 src_high);

	        

#endif // !LIBIMAGE_NO_GRAYSCALE


#ifndef LIBIMAGE_NO_COLOR
#ifndef LIBIMAGE_NO_GRAYSCALE

    bool grayscale(device_image_t const& src, gray::device_image_t const& dst);

#endif // !LIBIMAGE_NO_GRAYSCALE
#endif // !LIBIMAGE_NO_COLOR	
    
}