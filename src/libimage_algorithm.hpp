#pragma once
/*

Copyright (c) 2021 Adam Lafontaine

*/

#include "image_view.hpp"

#include <functional>
#include <algorithm>
#include <execution>

namespace libimage
{
	// for_each
	using fe_ref_t = std::function<void(pixel_t& p)>;
	using fe_cref_t = std::function<void(pixel_t const& p)>;

	// transform
	using tf_1src_func_t = std::function<pixel_t(pixel_t& p)>;
	using tf_2src_func_t = std::function<pixel_t(pixel_t& p1, pixel_t& p2)>;

	namespace seq
	{
		template<typename F>
		inline void for_each_pixel(image_t& image, F const& func)
		{
			std::for_each(image.begin(), image.end(), func);
		}


		template<typename F>
		inline void for_each_pixel(view_t& view, F const& func)
		{
			std::for_each(view.begin(), view.end(), func);
		}


		inline void transform_pixels(view_t& src, view_t& dst, tf_1src_func_t const& func)
		{
			assert(dst.width >= src.width);
			assert(dst.height >= src.height);

			std::transform(src.begin(), src.end(), dst.begin(), func);
		}


		inline void transform_pixels(view_t& src1, view_t& src2, view_t& dst, tf_2src_func_t const& func)
		{
			assert(src1.width == src2.width);
			assert(src1.height == src2.height);
			assert(dst.width >= src1.width);
			assert(dst.height >= src1.height);

			std::transform(src1.begin(), src1.end(), src2.begin(), dst.begin(), func);
		}
	}


	namespace par
	{
		template<typename F>
		inline void for_each_pixel(image_t& image, F const& func)
		{
			std::for_each(std::execution::par, image.begin(), image.end(), func);
		}


		template<typename F>
		inline void for_each_pixel(view_t& view, F const& func)
		{
			std::for_each(std::execution::par, view.begin(), view.end(), func);
		}

		
		inline void transform_pixels(view_t& src, view_t& dst, tf_1src_func_t const& func)
		{
			assert(dst.width >= src.width);
			assert(dst.height >= src.height);

			std::transform(std::execution::par, src.begin(), src.end(), dst.begin(), func);
		}


		inline void transform_pixels(view_t& src1, view_t& src2, view_t& dst, tf_2src_func_t const& func)
		{
			assert(src1.width == src2.width);
			assert(src1.height == src2.height);
			assert(dst.width >= src1.width);
			assert(dst.height >= src1.height);

			std::transform(std::execution::par, src1.begin(), src1.end(), src2.begin(), dst.begin(), func);
		}
	}


	//======= GRAYSCALE OVERLOADS ================

	namespace gray
	{
		// for_each
		using fe_ref_t = std::function<void(pixel_t& p)>;
		using fe_cref_t = std::function<void(pixel_t const& p)>;
		using fe_xy_t = std::function<void(u32 x, u32 y)>;


		using tf_1src_func_t = std::function<pixel_t(pixel_t& p)>;
		using tf_2src_func_t = std::function<pixel_t(pixel_t& p1, pixel_t& p2)>;
	}


	namespace seq
	{
		template<typename F>
		inline void for_each_pixel(gray::image_t& image, F const& func)
		{
			std::for_each(image.begin(), image.end(), func);
		}


		template<typename F>
		inline void for_each_pixel(gray::view_t& view, F const& func)
		{
			std::for_each(view.begin(), view.end(), func);
		}

		template<typename F>
		inline void for_each_pixel(gray::view_t const& view, F const& func)
		{
			std::for_each(view.cbegin(), view.cend(), func);
		}


		inline void transform_pixels(gray::view_t& src, gray::view_t& dst, gray::tf_1src_func_t const& func)
		{
			assert(dst.width >= src.width);
			assert(dst.height >= src.height);

			std::transform(src.begin(), src.end(), dst.begin(), func);
		}


		inline void transform_pixels(gray::view_t& src1, gray::view_t& src2, gray::view_t& dst, gray::tf_2src_func_t const& func)
		{
			assert(src1.width == src2.width);
			assert(src1.height == src2.height);
			assert(dst.width >= src1.width);
			assert(dst.height >= src1.height);

			std::transform(src1.begin(), src1.end(), src2.begin(), dst.begin(), func);
		}
		
	}


	namespace par
	{
		template<typename F>
		inline void for_each_pixel(gray::image_t& image, F const& func)
		{
			std::for_each(std::execution::par, image.begin(), image.end(), func);
		}


		template<typename F>
		inline void for_each_pixel(gray::view_t& view, F const& func)
		{
			std::for_each(std::execution::par, view.begin(), view.end(), func);
		}


		inline void transform_pixels(gray::view_t& src, gray::view_t& dst, gray::tf_1src_func_t const& func)
		{
			assert(dst.width >= src.width);
			assert(dst.height >= src.height);

			std::transform(std::execution::par, src.begin(), src.end(), dst.begin(), func);
		}


		inline void transform_pixels(gray::view_t& src1, gray::view_t& src2, gray::view_t& dst, gray::tf_2src_func_t const& func)
		{
			assert(src1.width == src2.width);
			assert(src1.height == src2.height);
			assert(dst.width >= src1.width);
			assert(dst.height >= src1.height);

			std::transform(std::execution::par, src1.begin(), src1.end(), src2.begin(), dst.begin(), func);
		}
	}
}
