#pragma once
/*

Copyright (c) 2021 Adam Lafontaine

*/
#include "../rgba.hpp"
#include "../gray.hpp"
#include "device.hpp"


namespace libimage
{
    namespace cuda
    {
#ifndef LIBIMAGE_NO_COLOR

        bool copy_to_device(image_t const& src, DeviceArray<pixel_t> const& dst);

        bool copy_to_host(DeviceArray<pixel_t> const& src, image_t const& dst);

        bool copy_to_device(view_t const& src, DeviceArray<pixel_t> const& dst);

        bool copy_to_host(DeviceArray<pixel_t> const& src, view_t const& dst);

#endif // !LIBIMAGE_NO_COLOR

#ifndef LIBIMAGE_NO_GRAYSCALE

        bool copy_to_device(gray::image_t const& src, DeviceArray<gray::pixel_t> const& dst);

        bool copy_to_host(DeviceArray<gray::pixel_t> const& src, gray::image_t const& dst);

        bool copy_to_device(gray::view_t const& src, DeviceArray<gray::pixel_t> const& dst);

        bool copy_to_host(DeviceArray<gray::pixel_t> const& src, gray::view_t const& dst);

#endif // !LIBIMAGE_NO_GRAYSCALE
    }
}


namespace libimage
{
    namespace cuda
    {

#ifndef LIBIMAGE_NO_COLOR


        void alpha_blend(image_t const& src, image_t const& current, image_t const& dst, DeviceBuffer& d_buffer);

		void alpha_blend(image_t const& src, image_t const& current, view_t const& dst, DeviceBuffer& d_buffer);

		void alpha_blend(image_t const& src, view_t const& current, image_t const& dst, DeviceBuffer& d_buffer);

		void alpha_blend(image_t const& src, view_t const& current, view_t const& dst, DeviceBuffer& d_buffer);

		void alpha_blend(view_t const& src, image_t const& current, image_t const& dst, DeviceBuffer& d_buffer);

		void alpha_blend(view_t const& src, image_t const& current, view_t const& dst, DeviceBuffer& d_buffer);

		void alpha_blend(view_t const& src, view_t const& current, image_t const& dst, DeviceBuffer& d_buffer);

		void alpha_blend(view_t const& src, view_t const& current, view_t const& dst, DeviceBuffer& d_buffer);


		void alpha_blend(image_t const& src, image_t const& current_dst, DeviceBuffer& d_buffer);

		void alpha_blend(image_t const& src, view_t const& current_dst, DeviceBuffer& d_buffer);

		void alpha_blend(view_t const& src, image_t const& current_dst, DeviceBuffer& d_buffer);

		void alpha_blend(view_t const& src, view_t const& current_dst, DeviceBuffer& d_buffer);


        void alpha_blend(image_t const& src, image_t const& current, image_t const& dst);

		void alpha_blend(image_t const& src, image_t const& current, view_t const& dst);

		void alpha_blend(image_t const& src, view_t const& current, image_t const& dst);

		void alpha_blend(image_t const& src, view_t const& current, view_t const& dst);

		void alpha_blend(view_t const& src, image_t const& current, image_t const& dst);

		void alpha_blend(view_t const& src, image_t const& current, view_t const& dst);

		void alpha_blend(view_t const& src, view_t const& current, image_t const& dst);

		void alpha_blend(view_t const& src, view_t const& current, view_t const& dst);


		void alpha_blend(image_t const& src, image_t const& current_dst);

		void alpha_blend(image_t const& src, view_t const& current_dst);

		void alpha_blend(view_t const& src, image_t const& current_dst);

		void alpha_blend(view_t const& src, view_t const& current_dst);


#endif // !LIBIMAGE_NO_COLOR	

#ifndef LIBIMAGE_NO_GRAYSCALE

        void binarize(gray::image_t const& src, gray::image_t const& dst, u8 min_threshold, DeviceBuffer& d_buffer);

		void binarize(gray::image_t const& src, gray::view_t const& dst, u8 min_threshold, DeviceBuffer& d_buffer);

		void binarize(gray::view_t const& src, gray::image_t const& dst, u8 min_threshold, DeviceBuffer& d_buffer);

		void binarize(gray::view_t const& src, gray::view_t const& dst, u8 min_threshold, DeviceBuffer& d_buffer);


        void binarize(gray::image_t const& src, gray::image_t const& dst, u8 min_threshold);

		void binarize(gray::image_t const& src, gray::view_t const& dst, u8 min_threshold);

		void binarize(gray::view_t const& src, gray::image_t const& dst, u8 min_threshold);

		void binarize(gray::view_t const& src, gray::view_t const& dst, u8 min_threshold);
        

#endif // !LIBIMAGE_NO_GRAYSCALE


#ifndef LIBIMAGE_NO_COLOR
#ifndef LIBIMAGE_NO_GRAYSCALE

        void transform_grayscale(image_t const& src, gray::image_t const& dst, DeviceBuffer& d_buffer);

        void transform_grayscale(image_t const& src, gray::view_t const& dst, DeviceBuffer& d_buffer);

        void transform_grayscale(view_t const& src, gray::image_t const& dst, DeviceBuffer& d_buffer);

        void transform_grayscale(view_t const& src, gray::view_t const& dst, DeviceBuffer& d_buffer);


        void transform_grayscale(image_t const& src, gray::image_t const& dst);

        void transform_grayscale(image_t const& src, gray::view_t const& dst);

        void transform_grayscale(view_t const& src, gray::image_t const& dst);

        void transform_grayscale(view_t const& src, gray::view_t const& dst);        

#endif // !LIBIMAGE_NO_GRAYSCALE
#endif // !LIBIMAGE_NO_COLOR	
    }
}