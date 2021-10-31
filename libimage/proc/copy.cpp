/*

Copyright (c) 2021 Adam Lafontaine

*/

#include "process.hpp"
#include "verify.hpp"

#include <algorithm>

#ifndef LIBIMAGE_NO_PARALLEL
#include <execution>
#endif // !LIBIMAGE_NO_PARALLEL

namespace libimage
{
#ifndef LIBIMAGE_NO_PARALLEL


#ifndef LIBIMAGE_NO_COLOR

	void copy(image_t const& src, image_t const& dst)
	{
		assert(verify(src, dst));
		std::copy(std::execution::par, src.begin(), src.end(), dst.begin());
	}


	void copy(image_t const& src, view_t const& dst)
	{
		assert(verify(src, dst));
		std::copy(std::execution::par, src.begin(), src.end(), dst.begin());
	}


	void copy(view_t const& src, image_t const& dst)
	{
		assert(verify(src, dst));
		std::copy(std::execution::par, src.begin(), src.end(), dst.begin());
	}


	void copy(view_t const& src, view_t const& dst)
	{
		assert(verify(src, dst));
		std::copy(std::execution::par, src.begin(), src.end(), dst.begin());
	}


#endif // !LIBIMAGE_NO_COLOR


#ifndef LIBIMAGE_NO_GRAYSCALE

	void copy(gray::image_t const& src, gray::image_t const& dst)
	{
		assert(verify(src, dst));
		std::copy(std::execution::par, src.begin(), src.end(), dst.begin());
	}


	void copy(gray::image_t const& src, gray::view_t const& dst)
	{
		assert(verify(src, dst));
		std::copy(std::execution::par, src.begin(), src.end(), dst.begin());
	}


	void copy(gray::view_t const& src, gray::image_t const& dst)
	{
		assert(verify(src, dst));
		std::copy(std::execution::par, src.begin(), src.end(), dst.begin());
	}


	void copy(gray::view_t const& src, gray::view_t const& dst)
	{
		assert(verify(src, dst));
		std::copy(std::execution::par, src.begin(), src.end(), dst.begin());
	}

#endif // !LIBIMAGE_NO_GRAYSCALE

#endif // !LIBIMAGE_NO_PARALLEL
	namespace seq
	{

#ifndef LIBIMAGE_NO_COLOR

		void copy(image_t const& src, image_t const& dst)
		{
			assert(verify(src, dst));
			std::copy(src.begin(), src.end(), dst.begin());
		}


		void copy(image_t const& src, view_t const& dst)
		{
			assert(verify(src, dst));
			std::copy(src.begin(), src.end(), dst.begin());
		}


		void copy(view_t const& src, image_t const& dst)
		{
			assert(verify(src, dst));
			std::copy(src.begin(), src.end(), dst.begin());
		}


		void copy(view_t const& src, view_t const& dst)
		{
			assert(verify(src, dst));
			std::copy(src.begin(), src.end(), dst.begin());
		}

#endif // !LIBIMAGE_NO_COLOR


#ifndef LIBIMAGE_NO_GRAYSCALE


		void copy(gray::image_t const& src, gray::image_t const& dst)
		{
			assert(verify(src, dst));
			std::copy(src.begin(), src.end(), dst.begin());
		}


		void copy(gray::image_t const& src, gray::view_t const& dst)
		{
			assert(verify(src, dst));
			std::copy(src.begin(), src.end(), dst.begin());
		}


		void copy(gray::view_t const& src, gray::image_t const& dst)
		{
			assert(verify(src, dst));
			std::copy(src.begin(), src.end(), dst.begin());
		}


		void copy(gray::view_t const& src, gray::view_t const& dst)
		{
			assert(verify(src, dst));
			std::copy(src.begin(), src.end(), dst.begin());
		}


#endif // !LIBIMAGE_NO_GRAYSCALE
	}
}

